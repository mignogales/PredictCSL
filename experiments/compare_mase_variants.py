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

    # reproduce the leaderboard exactly (every instance at its own full context,
    # nothing dropped) -> mase_real is directly comparable to the published number
    python -m experiments.compare_mase_variants --dataset ETTm1-15T --term short \
        --model Chronos2-Small --leaderboard

    # same setup but sweep the input context length (the effect you asked about)
    python -m experiments.compare_mase_variants --dataset ETTm1-15T --term short \
        --model Chronos2-Small --leaderboard --context-length 512
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
from experiments.gifteval_mase import get_seasonality, gluonts_leaderboard_mase
from experiments.build_context_length_dataset import WINDOW_GRID
from experiments.test_window_ablation_gifteval_v5 import (
    DATASETS,
    GiftEvalCache,
    compute_all_metrics,
    load_handle,
    _forecast_cell,
    _merge_grouped,
    SUNDIAL_MAX_CONTEXT,
    TIMEMOE_MAX_TOTAL,
    TOTO_MAX_CONTEXT,
    FLOWSTATE_MAX_CONTEXT,
    TIREX_MAX_CONTEXT,
)

# Practical ceiling for "full native context" in --leaderboard mode: the ablation
# grid's max, and a common TSFM context limit (PatchTST-FM is fixed at 8192).
# Keeps us from feeding a 69k-step series to a model that would OOM — the real
# leaderboard likewise runs each model at its own (bounded) default context.
LEADERBOARD_CONTEXT_CEILING = 8192


def model_context_cap(family: str, horizon: int, requested: int) -> int:
    """Clamp a requested context length to what ``family`` actually accepts,
    mirroring the stage-3 ablation's per-model caps (Sundial 2880, TimeMoE
    ctx+h<=4096, Toto/FlowState 4096, TiRex 2048). Uncapped families keep
    ``requested`` (already bounded by the ceiling / instance length)."""
    cap = requested
    hard = {
        "sundial": SUNDIAL_MAX_CONTEXT,
        "toto": TOTO_MAX_CONTEXT,
        "flowstate": FLOWSTATE_MAX_CONTEXT,
        "tirex": TIREX_MAX_CONTEXT,
    }
    if family == "timemoe":
        cap = min(cap, TIMEMOE_MAX_TOTAL - horizon)
    if family in hard:
        cap = min(cap, hard[family])
    return max(int(cap), 1)


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

def real_gluonts_mase(cache, valid_indices, starts, fr, freq):
    """Score ``fr`` with the actual gluonts leaderboard machinery (shared with the
    stage-3 ``mase_gluonts_real`` column via ``gifteval_mase``).

    ``valid_indices`` are the served instances (rows of ``fr``); ``starts`` their
    forecast-start Periods. Returns (mase_value, "MASE", season) or raises on
    import failure.
    """
    median = fr.median.detach().cpu().float().numpy()
    quantiles = (fr.quantiles.detach().cpu().float().numpy()
                 if fr.quantiles is not None else None)
    samples = (fr.samples.detach().cpu().float().numpy()
               if fr.samples is not None else None)
    contexts = [cache.contexts[i] for i in valid_indices]
    labels = cache.labels_np[np.asarray(valid_indices)]
    val = gluonts_leaderboard_mase(
        median, starts, contexts, labels, freq,
        quantiles=quantiles, quantile_levels=fr.quantile_levels, samples=samples)
    return val, "MASE", get_seasonality(freq)


# ----------------------------------------------------------------------------
#  Per-config evaluation
# ----------------------------------------------------------------------------

def _forecast_padded(cache, family, handle, model_id, cap, horizon, args):
    """Pad-mode (leaderboard-faithful) forecasting: EVERY instance is served at
    its own genuine context, truncated only to ``cap`` — none dropped. Reuses the
    ablation's width-bucketed batching + merge. Returns (fr, tgts) over all
    ``cache.n_total`` instances in 0..n-1 order."""
    # Bucket widths down to the grid to bound the number of per-width runners, but
    # inject the cap itself so every instance long enough uses EXACTLY the cap (not
    # the grid point below it) — the context sweep must hit the requested length.
    grid = sorted(set(WINDOW_GRID) | {int(cap)})
    groups = cache.build_batches_padded(
        cap, args.batch_size, args.device,
        pin_memory=(args.device == "cuda"), window_grid=grid)
    results = []
    for L, batches_L, _ax, _ay, idx_L in groups:
        fr_L, tgts_L = _forecast_cell(family, handle, model_id, batches_L, L,
                                      horizon, args.device, args.batch_size,
                                      flowstate_scale=cache.flowstate_scale)
        results.append((idx_L, fr_L, tgts_L))
    return _merge_grouped(results, cache.n_total, horizon, args.device)


def evaluate_config(display, term, model_id, family, handle, args):
    """Run the model on one (dataset display, term) and return the three MASE
    variants for it (plus book-keeping). ``handle`` is the already-loaded base
    model, reused across datasets. Returns a result dict.

    Two setups:
      * default (ablation): skip-mode batching at a FIXED window — instances with
        context < window are dropped, and context is truncated to ``window``.
      * ``--leaderboard``: pad-mode — every instance forecast at its OWN full
        context (capped to the model's limit / ``--context-length``), nothing
        dropped. This reproduces the GiftEval setup, so ``mase_real`` here is
        directly comparable to the leaderboard number.
    """
    ge_name, to_univariate = resolve_dataset(display, term)
    ge_dataset = GiftEvalDataset(name=ge_name, term=term, to_univariate=to_univariate)
    cache = GiftEvalCache(ge_dataset, display)
    freq, horizon = cache.freq, cache.horizon

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

    if args.leaderboard:
        # Context ceiling: --context-length if given, else the full-native ceiling.
        # Clamp to the model's own cap and to the longest available context.
        requested = args.context_length or LEADERBOARD_CONTEXT_CEILING
        cap = model_context_cap(family, horizon, min(requested, cache.max_context))
        fr, tgts = _forecast_padded(cache, family, handle, model_id, cap, horizon, args)
        served_indices = np.arange(cache.n_total)
        window = cap                    # reported context ceiling (per-instance <= this)
    else:
        window = (cache.max_context if args.full
                  else (args.window or min(cache.max_context, 1024)))
        window = min(window, cache.max_context)
        batches, _ax, _ay, valid_indices = cache.build_batches(
            window, args.batch_size, args.device, pin_memory=(args.device == "cuda"))
        fr, tgts = _forecast_cell(family, handle, model_id, batches, window, horizon,
                                  args.device, args.batch_size,
                                  flowstate_scale=cache.flowstate_scale)
        served_indices = valid_indices

    starts_served = [starts[i] for i in served_indices]
    se_served = cache.seasonal_errors_gluonts[np.asarray(served_indices)]
    metrics = compute_all_metrics(fr, tgts, cache.naive_seasonal_mae_train,
                                  seasonal_errors=se_served)

    real_val = None
    if not args.no_real:
        try:
            real_val, _, _ = real_gluonts_mase(
                cache, np.asarray(served_indices), starts_served, fr, freq)
        except ImportError as e:
            print(f"[mase_real] gluonts not importable ({e}); skipping.", file=sys.stderr)
            args.no_real = True     # don't retry per config
        except Exception as e:      # noqa: BLE001 - surface but keep going
            print(f"[mase_real] {display}/{term}: evaluate_forecasts failed: {e}",
                  file=sys.stderr)

    return {
        "display": display, "term": term, "freq": freq, "window": window,
        "n_total": cache.n_total, "n_served": len(served_indices),
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
    grp.add_argument("--window", type=int, default=None,
                     help="ablation mode: fixed context window (instances shorter "
                          "than it are dropped)")
    grp.add_argument("--full", action="store_true",
                     help="ablation mode: use each dataset's max available context")
    ap.add_argument("--leaderboard", action="store_true",
                    help="reproduce the GiftEval setup: every instance forecast at "
                         "its OWN full context (no dropping), so mase_real matches "
                         "the leaderboard. Combine with --context-length to sweep.")
    ap.add_argument("--context-length", type=int, default=None,
                    help="(--leaderboard only) cap each instance's context at this "
                         "many steps — the 'slightly different input context length' "
                         "knob. Omit for full native context (<= model cap).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-real", action="store_true",
                    help="skip the gluonts-machinery mase_real value")
    args = ap.parse_args()

    if args.context_length is not None and not args.leaderboard:
        raise SystemExit("--context-length only applies with --leaderboard.")

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
    if args.leaderboard:
        ctx_desc = (f"leaderboard / ctx<={args.context_length}" if args.context_length
                    else "leaderboard / full native ctx")
    else:
        ctx_desc = "full" if args.full else (args.window or "auto<=1024")
    print("\n" + "=" * 92)
    print(f"MASE comparison — {args.model}   (context: {ctx_desc})")
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
