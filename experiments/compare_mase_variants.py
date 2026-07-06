"""Compare the three MASE definitions on ONE GiftEval dataset with ONE model.

For a single (dataset, term) and a single TSFM, this runs the model once at a
chosen context window and scores the *same* forecasts under three MASE variants:

  1. ``mase``          — the project's own metric: ``global_MAE / one pooled
                         training-set seasonal-naive MAE``, with the custom
                         seasonality map (``D->7, W->52, ...``).
  2. ``mase_gluonts``  — this repo's *port* of the leaderboard definition
                         (``experiments.gifteval_mase``): the mean over instances
                         of ``mean_h(|y-yhat|) / seasonal_error_instance``, with the
                         gluonts seasonality map (``D->1, W->1, ...``) and the
                         per-instance seasonal error taken from each series' own
                         context.
  3. ``mase_real``     — the *actual* gluonts machinery the GiftEval leaderboard
                         runs: ``gluonts.model.evaluate_forecasts`` with
                         ``gluonts.ev.metrics.MASE``, fed the same forecasts as
                         gluonts Forecast objects over the real test instances.
                         This is the ground truth that (2) is meant to reproduce;
                         (2) ≈ (3) validates the port.

The whole point is that all three consume the *identical* model predictions, so
the numbers differ only by MASE definition — not by any difference in the
forecasts. The instance set is the ablation's kept+served set (labels long enough
for the horizon, context long enough for the window), so the three are directly
comparable to each other; ``mase_real`` here is the leaderboard *definition*
applied to that set, not a full-test-split leaderboard submission.

Run on the SERVER (needs the TSFMs, GIFT_EVAL data, and gluonts):

    python -m experiments.compare_mase_variants --dataset ETTm1-15T --term short \
        --model Chronos2-Small --window 512
    python -m experiments.compare_mase_variants --dataset JenaWeather-H --term medium \
        --model Moirai2-Small --full
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from gift_eval.data import Dataset as GiftEvalDataset

from experiments import models_config
from experiments.gifteval_mase import get_seasonality
from experiments.test_window_ablation_gifteval_v5 import (
    DATASETS,
    GiftEvalCache,
    compute_all_metrics,
    load_handle,
    _forecast_cell,
)


# ----------------------------------------------------------------------------
#  Lookups
# ----------------------------------------------------------------------------

def resolve_dataset(display: str, term: str):
    """Find the DATASETS row for a (display, term) pair -> (ge_name, to_univariate)."""
    for ge_name, t, disp, to_univariate in DATASETS:
        if disp == display and t == term:
            return ge_name, to_univariate
    avail = sorted({d[2] for d in DATASETS})
    raise SystemExit(
        f"No dataset '{display}' with term '{term}' in DATASETS.\n"
        f"Available displays: {', '.join(avail)}"
    )


def resolve_model(display: str):
    """Find the models_config catalog row for a display -> (model_id, family)."""
    for spec in models_config.CATALOG:
        if spec.display == display:
            return spec.model_id, spec.family
    avail = ", ".join(s.display for s in models_config.CATALOG)
    raise SystemExit(f"No model '{display}'. Available: {avail}")


# ----------------------------------------------------------------------------
#  The real (leaderboard) gluonts MASE
# ----------------------------------------------------------------------------

def build_gluonts_forecasts(fr, starts, freq):
    """Turn a ForecastResult (+ per-instance start Periods) into gluonts Forecast
    objects — QuantileForecast when the model emits quantiles, else SampleForecast.

    ``fr.median`` is (N, H); ``fr.quantiles`` (N, Q, H); ``fr.samples`` (N, S, H).
    ``starts[i]`` is the pandas Period at which instance i's forecast begins.
    """
    from gluonts.model.forecast import QuantileForecast, SampleForecast

    median = fr.median.detach().cpu().float().numpy()
    n, h = median.shape
    forecasts = []

    if fr.quantiles is not None and fr.quantile_levels is not None:
        q = fr.quantiles.detach().cpu().float().numpy()          # (N, Q, H)
        keys = [str(float(l)) for l in fr.quantile_levels]
        has_median = any(abs(float(l) - 0.5) < 1e-9 for l in fr.quantile_levels)
        for i in range(n):
            arrays = q[i]                                         # (Q, H)
            fkeys = list(keys)
            if not has_median:
                arrays = np.concatenate([arrays, median[i][None, :]], axis=0)
                fkeys = fkeys + ["0.5"]
            forecasts.append(QuantileForecast(
                forecast_arrays=arrays, start_date=starts[i],
                forecast_keys=fkeys, item_id=str(i),
            ))
    elif fr.samples is not None:
        s = fr.samples.detach().cpu().float().numpy()            # (N, S, H)
        for i in range(n):
            forecasts.append(SampleForecast(
                samples=s[i], start_date=starts[i], item_id=str(i)))
    else:
        # Median only: a single-sample forecast whose 0.5 quantile is the median.
        for i in range(n):
            forecasts.append(SampleForecast(
                samples=median[i][None, :], start_date=starts[i], item_id=str(i)))

    return forecasts


class _ListTestData:
    """Minimal stand-in for gluonts' ``split.TestData``: ``evaluate_forecasts``
    only needs ``.input`` and ``.label`` (each a re-iterable of entry dicts) and
    to zip them together. Lists satisfy the re-iteration it does internally
    (seasonal error over ``.input``, then labels over ``.label``)."""

    def __init__(self, inputs, labels):
        self.input = inputs
        self.label = labels

    def __iter__(self):
        return zip(self.input, self.label)

    def __len__(self):
        return len(self.input)


def real_gluonts_mase(cache, valid_indices, starts, fr, freq):
    """Score ``fr`` with the actual gluonts leaderboard machinery.

    Builds a filtered TestData (one input/label per served instance, in forecast
    order) and runs ``evaluate_forecasts`` with ``MASE`` and the gluonts
    seasonality. Returns (mase_value, column_name, season) or raises on import
    failure.
    """
    from gluonts.model import evaluate_forecasts
    from gluonts.ev.metrics import MASE

    forecasts = build_gluonts_forecasts(fr, starts, freq)

    # Per served instance: input target = the series' FULL context (gluonts derives
    # the per-instance seasonal error from it), label target = the horizon ground
    # truth — exactly the pieces GiftEvalCache already holds.
    inputs, labels = [], []
    for k, i in enumerate(valid_indices):
        ctx = np.asarray(cache.contexts[i], dtype=np.float64)
        lbl = np.asarray(cache.labels_np[i], dtype=np.float64)
        inputs.append({"start": starts[k] - len(ctx), "target": ctx})
        labels.append({"start": starts[k], "target": lbl})
    test_data = _ListTestData(inputs, labels)

    season = get_seasonality(freq)
    df = evaluate_forecasts(
        forecasts,
        test_data=test_data,
        metrics=[MASE()],
        axis=None,
        seasonality=season,
    )
    # evaluate_forecasts returns a 1-row DataFrame with a column like "MASE[0.5]".
    col = next((c for c in df.columns if str(c).upper().startswith("MASE")), None)
    if col is None:
        raise RuntimeError(f"No MASE column in evaluate_forecasts output: {list(df.columns)}")
    return float(np.asarray(df[col]).ravel()[0]), str(col), season


# ----------------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset display, e.g. ETTm1-15T")
    ap.add_argument("--term", default="short", choices=["short", "medium", "long"])
    ap.add_argument("--model", required=True, help="model display, e.g. Chronos2-Small")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--window", type=int, default=None, help="context window size")
    grp.add_argument("--full", action="store_true",
                     help="use each dataset's max available context")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ge_name, to_univariate = resolve_dataset(args.dataset, args.term)
    model_id, family = resolve_model(args.model)

    print(f"Dataset : {args.dataset} (ge_name={ge_name}, term={args.term})")
    print(f"Model   : {args.model} (family={family}, id={model_id})")

    ge_dataset = GiftEvalDataset(name=ge_name, term=args.term, to_univariate=to_univariate)
    cache = GiftEvalCache(ge_dataset, args.dataset)
    freq, horizon = cache.freq, cache.horizon
    print(f"freq={freq}  horizon={horizon}  n_instances={cache.n_total}  "
          f"context: min={cache.min_context} max={cache.max_context}")

    window = cache.max_context if args.full else (args.window or min(cache.max_context, 1024))
    window = min(window, cache.max_context)
    print(f"window  = {window}\n")

    # Per-instance forecast-start Periods, captured in GiftEvalCache's iteration
    # order (same len(label) >= horizon filter), so they align 1:1 with cache rows.
    starts = []
    for _, test_label in ge_dataset.test_data:
        lbl = test_label["target"]
        if len(lbl) < horizon:            # same drop rule GiftEvalCache applies
            continue
        starts.append(test_label["start"])
    if len(starts) != cache.n_total:
        raise RuntimeError(
            f"start capture ({len(starts)}) != cache instances ({cache.n_total}); "
            "iteration order mismatch.")

    # ---- Run the model once (skip-mode batching, same as the ablation) --------
    batches, all_x, all_y, valid_indices = cache.build_batches(
        window, args.batch_size, args.device, pin_memory=(args.device == "cuda"))
    handle = load_handle(family, model_id, args.device)
    fr, tgts = _forecast_cell(family, handle, model_id, batches, window, horizon,
                              args.device, args.batch_size)
    starts_served = [starts[i] for i in valid_indices]
    print(f"served instances (context >= {window}): {len(valid_indices)}/{cache.n_total}\n")

    # ---- (1) + (2): project mase and the ported gluonts mase ------------------
    se_served = cache.seasonal_errors_gluonts[valid_indices]
    metrics = compute_all_metrics(fr, tgts, cache.naive_seasonal_mae_train,
                                  seasonal_errors=se_served)
    mase_project = metrics["mase"]
    mase_port = metrics["mase_gluonts"]

    # ---- (3): the real gluonts leaderboard machinery --------------------------
    real_val, real_col, real_season = None, None, None
    try:
        real_val, real_col, real_season = real_gluonts_mase(
            cache, valid_indices, starts_served, fr, freq)
    except ImportError as e:
        print(f"[mase_real] gluonts not importable here ({e}); skipping the "
              "leaderboard-machinery value.", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 - surface but don't crash the comparison
        print(f"[mase_real] evaluate_forecasts failed: {e}", file=sys.stderr)

    # ---- Report ---------------------------------------------------------------
    print("=" * 64)
    print(f"MASE comparison — {args.model} on {args.dataset}/{args.term} @ w={window}")
    print("=" * 64)
    print(f"  season (project map)  : {cache.season}")
    print(f"  season (gluonts map)  : {get_seasonality(freq)}")
    print(f"  naive_seasonal_mae    : {cache.naive_seasonal_mae_train:.6f}")
    print("-" * 64)
    print(f"  1. mase        (project custom)      : {mase_project:.6f}")
    print(f"  2. mase_gluonts(repo port)           : {mase_port:.6f}")
    if real_val is not None:
        print(f"  3. mase_real   (gluonts {real_col:<10}) : {real_val:.6f}")
        diff = abs(real_val - mase_port)
        rel = diff / abs(real_val) if real_val else float("nan")
        print("-" * 64)
        print(f"  |port - real| = {diff:.3e}   (rel {rel:.2%})   "
              f"{'OK ✓' if rel < 0.01 else 'CHECK'}")
    else:
        print("  3. mase_real   (gluonts machinery)   : <unavailable>")
    print("=" * 64)


if __name__ == "__main__":
    main()
