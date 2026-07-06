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

Multiple (dataset, term) configs can be scored at once and aggregated across
them. The per-config MASE is aggregated by GEOMETRIC mean (the GiftEval
leaderboard's aggregation — each config weighted equally), with the arithmetic
mean shown alongside.

Run on the SERVER (needs the TSFMs, GIFT_EVAL data, and gluonts):

    # single config
    python -m experiments.compare_mase_variants --dataset ETTm1-15T --term short \
        --model Chronos2-Small --window 512

    # several datasets x terms (cartesian product), aggregated
    python -m experiments.compare_mase_variants \
        --dataset ETTm1-15T ETTm2-15T JenaWeather-H --term short medium long \
        --model Moirai2-Small --full

    # explicit config list
    python -m experiments.compare_mase_variants \
        --configs ETTm1-15T/short JenaWeather-H/medium BizITObsApp/long \
        --model Chronos2-Small --window 512
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
#  Per-config evaluation
# ----------------------------------------------------------------------------

def evaluate_config(display, term, model_id, family, handle, args):
    """Run the model on one (dataset display, term) and return the three MASE
    variants for it (plus book-keeping). ``handle`` is the already-loaded base
    model, reused across datasets. Returns a result dict."""
    ge_name, to_univariate = resolve_dataset(display, term)
    ge_dataset = GiftEvalDataset(name=ge_name, term=term, to_univariate=to_univariate)
    cache = GiftEvalCache(ge_dataset, display)
    freq, horizon = cache.freq, cache.horizon

    window = cache.max_context if args.full else (args.window or min(cache.max_context, 1024))
    window = min(window, cache.max_context)

    # Per-instance forecast-start Periods, captured in GiftEvalCache's iteration
    # order (same len(label) >= horizon filter), so they align 1:1 with cache rows.
    starts = []
    for _, test_label in ge_dataset.test_data:
        if len(test_label["target"]) < horizon:   # same drop rule GiftEvalCache applies
            continue
        starts.append(test_label["start"])
    if len(starts) != cache.n_total:
        raise RuntimeError(
            f"start capture ({len(starts)}) != cache instances ({cache.n_total}).")

    batches, all_x, all_y, valid_indices = cache.build_batches(
        window, args.batch_size, args.device, pin_memory=(args.device == "cuda"))
    fr, tgts = _forecast_cell(family, handle, model_id, batches, window, horizon,
                              args.device, args.batch_size)
    starts_served = [starts[i] for i in valid_indices]

    se_served = cache.seasonal_errors_gluonts[valid_indices]
    metrics = compute_all_metrics(fr, tgts, cache.naive_seasonal_mae_train,
                                  seasonal_errors=se_served)

    real_val = None
    if not args.no_real:
        try:
            real_val, _, _ = real_gluonts_mase(cache, valid_indices, starts_served, fr, freq)
        except ImportError as e:
            print(f"[mase_real] gluonts not importable ({e}); skipping.", file=sys.stderr)
            args.no_real = True     # don't retry per config
        except Exception as e:      # noqa: BLE001 - surface but keep going
            print(f"[mase_real] {display}/{term}: evaluate_forecasts failed: {e}",
                  file=sys.stderr)

    return {
        "display": display, "term": term, "freq": freq, "window": window,
        "n_total": cache.n_total, "n_served": len(valid_indices),
        "season_proj": cache.season, "season_gl": get_seasonality(freq),
        "mase": metrics["mase"], "mase_gluonts": metrics["mase_gluonts"],
        "mase_real": real_val,
    }


# ----------------------------------------------------------------------------
#  Config list + aggregation helpers
# ----------------------------------------------------------------------------

def build_config_list(args):
    """Resolve the requested (display, term) configs. Prefers explicit
    ``--configs display/term`` tokens; otherwise takes the cartesian product of
    ``--dataset`` x ``--term``, silently dropping combos absent from DATASETS."""
    valid = {(d[2], d[1]) for d in DATASETS}
    configs = []
    if args.configs:
        for tok in args.configs:
            if "/" not in tok:
                raise SystemExit(f"--configs token '{tok}' must be 'Display/term'.")
            disp, term = tok.rsplit("/", 1)
            if (disp, term) not in valid:
                raise SystemExit(f"Config '{disp}/{term}' not in DATASETS.")
            configs.append((disp, term))
    else:
        for disp in args.dataset:
            for term in args.term:
                if (disp, term) in valid:
                    configs.append((disp, term))
                else:
                    print(f"[skip] {disp}/{term} not in DATASETS.", file=sys.stderr)
    if not configs:
        raise SystemExit("No valid (dataset, term) configs to evaluate.")
    # De-dup while preserving order.
    seen, out = set(), []
    for c in configs:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def _gmean(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v) and v > 0],
                   dtype=np.float64)
    return float(np.exp(np.log(a).mean())) if a.size else float("nan")


def _amean(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
    return float(a.mean()) if a.size else float("nan")


# ----------------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=[],
                    help="one or more dataset displays, e.g. ETTm1-15T JenaWeather-H")
    ap.add_argument("--term", nargs="+", default=["short"],
                    choices=["short", "medium", "long"],
                    help="term(s); applied to every --dataset (cartesian product)")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="explicit 'Display/term' tokens (overrides --dataset/--term)")
    ap.add_argument("--model", required=True, help="model display, e.g. Chronos2-Small")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--window", type=int, default=None, help="context window size")
    grp.add_argument("--full", action="store_true",
                     help="use each dataset's max available context")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-real", action="store_true",
                    help="skip the gluonts-machinery mase_real value")
    args = ap.parse_args()

    if not args.dataset and not args.configs:
        raise SystemExit("Provide --dataset (one or more) or --configs.")

    configs = build_config_list(args)
    model_id, family = resolve_model(args.model)
    print(f"Model   : {args.model} (family={family}, id={model_id})")
    print(f"Configs : {len(configs)}  ->  " +
          ", ".join(f"{d}/{t}" for d, t in configs) + "\n")

    # Load the base model once and reuse it across all datasets.
    handle = load_handle(family, model_id, args.device)

    results = []
    for i, (disp, term) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {disp}/{term} ...", flush=True)
        try:
            results.append(evaluate_config(disp, term, model_id, family, handle, args))
        except Exception as e:  # noqa: BLE001 - one bad config shouldn't sink the run
            print(f"  ! failed: {e}", file=sys.stderr)

    if not results:
        raise SystemExit("No config produced a result.")

    # ---- Per-dataset table ----------------------------------------------------
    have_real = any(r["mase_real"] is not None for r in results)
    print("\n" + "=" * 92)
    print(f"MASE comparison — {args.model}   (window: "
          f"{'full' if args.full else (args.window or 'auto<=1024')})")
    print("=" * 92)
    hdr = f"{'dataset/term':<26}{'served':>10}{'mase':>12}{'mase_gluonts':>14}"
    hdr += f"{'mase_real':>12}" if have_real else ""
    print(hdr)
    print("-" * 92)
    for r in results:
        line = (f"{r['display'] + '/' + r['term']:<26}"
                f"{r['n_served']:>4}/{r['n_total']:<5}"
                f"{r['mase']:>12.6f}{r['mase_gluonts']:>14.6f}")
        if have_real:
            rv = r["mase_real"]
            line += f"{rv:>12.6f}" if rv is not None else f"{'n/a':>12}"
        print(line)

    # ---- Aggregates across datasets ------------------------------------------
    # The GiftEval leaderboard aggregates per-config MASE by GEOMETRIC mean (each
    # config weighted equally); the arithmetic mean is shown alongside.
    print("-" * 92)
    for label, fn in (("geomean (leaderboard)", _gmean), ("arithmetic mean", _amean)):
        line = f"{label:<26}{'':>10}"
        line += f"{fn([r['mase'] for r in results]):>12.6f}"
        line += f"{fn([r['mase_gluonts'] for r in results]):>14.6f}"
        if have_real:
            line += f"{fn([r['mase_real'] for r in results]):>12.6f}"
        print(line)
    print("=" * 92)

    if have_real:
        # port-vs-real fidelity on the geomeans.
        g_port = _gmean([r["mase_gluonts"] for r in results])
        g_real = _gmean([r["mase_real"] for r in results])
        rel = abs(g_port - g_real) / g_real if g_real else float("nan")
        print(f"geomean |port - real| rel = {rel:.2%}   "
              f"{'OK ✓' if rel < 0.01 else 'CHECK'}")


if __name__ == "__main__":
    main()
