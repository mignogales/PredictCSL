"""
period_window_eval.py  --  "2x strongest period" context-length heuristic.

A fourth context-selection strategy for the PredictCSL pipeline, complementing
the grid strategies produced by ``test_window_ablation_gifteval_v5.py`` (full /
best / predictor).  Instead of choosing a window from the predictor's discrete
grid, this method picks, *per test instance*, the context length

    L_i = max( 2 x strongest_period_i ,  horizon )

where ``strongest_period_i`` is detected from that instance's own context via a
combined FFT + autocorrelation analysis (see ``detect_period``).  The intuition:
a foundation model needs at least two full cycles of the dominant seasonality as
context, but never less than the horizon it has to predict.

Because each instance gets its OWN window, ``L_i`` is generally NOT one of the
ablation-grid windows -- so we cannot just read a cached cell.  We evaluate the
model directly at each per-series length here (grouping instances that share a
length into one forward pass, exactly like v5's pad-mode width groups), then
aggregate one MASE per (model, dataset, term) -- directly comparable to the grid
strategies' aggregate MASE (same instances, same naive-seasonal MASE
denominator).

Outputs (one per (model, dataset, term)) are written as sidecars NEXT TO v5's
comparison artefacts, so ``compare_window_strategies_gifteval.py`` can pick them
up and surface ``period`` as a fourth strategy:

    <run_dir>/models/<model_short>/compare_real_vs_predicted/
        period_<dataset>_t<term>_<model_short>.json    aggregate metrics + window stats
        period_<dataset>_t<term>_<model_short>_win.npz  per-instance windows + periods

Re-running is safe: a (model, dataset, term) whose sidecar JSON already exists is
skipped unless --force is given.

Usage
-----
    python -m experiments.period_window_eval --models Chronos2-Small \
        --run-dir logs/experiments/window_ablation_gifteval/general

Heavy machinery (dataset cache, model loaders, per-group forecasting, metrics)
is imported from test_window_ablation_gifteval_v5 so the two stay in lock-step.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from colorama import Fore

from gift_eval.data import Dataset as GiftEvalDataset

# Reuse v5's catalog + machinery verbatim so this method evaluates models exactly
# the way the grid ablation does (same loaders, same forward passes, same metric
# + naive-seasonal denominator).  Importing v5 is side-effect-light (it only
# defines globals and calls load_dotenv()).
from experiments.test_window_ablation_gifteval_v5 import (
    MODELS,
    DATASETS,
    GiftEvalCache,
    ForecastResult,            # noqa: F401  (re-exported for type clarity)
    compute_all_metrics,
    _forecast_cell,
    _merge_grouped,
    load_chronos_bolt,
    load_chronos2,
    load_moirai_module,
    load_moirai_1_1_module,
    load_patchtst_fm,
    load_sundial,
    load_timemoe,
    SUNDIAL_MAX_CONTEXT,
    TIMEMOE_MAX_TOTAL,
)


# ==============================================================================
#  PERIOD DETECTION  (per-series "strongest period")
# ==============================================================================

def _fft_period(x: np.ndarray, min_period: int, max_period: int) -> Tuple[Optional[float], float]:
    """Dominant period via the periodogram peak within [min_period, max_period].

    Returns (period, salience) where salience = peak_power / median_power (>1
    means the peak stands above the spectral floor). period is None if no
    in-range frequency exists.
    """
    n = x.size
    if n < 2 * min_period + 1:
        return None, 0.0
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n)
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    mask = (periods >= min_period) & (periods <= max_period)
    if not mask.any():
        return None, 0.0
    cand = np.flatnonzero(mask)
    best = cand[int(np.argmax(power[cand]))]
    floor = float(np.median(power[1:])) if n > 2 else 0.0
    salience = float(power[best] / floor) if floor > 0 else float("inf")
    return float(periods[best]), salience


def _acf_period(x: np.ndarray, min_period: int, max_period: int) -> Tuple[Optional[float], float]:
    """Dominant period via the first strong autocorrelation peak in range.

    Returns (period, strength) where strength is the ACF value at the peak lag
    (normalised so lag-0 = 1). period is None if no positive in-range peak.
    """
    n = x.size
    if n < 2 * min_period + 1:
        return None, 0.0
    # Biased ACF via FFT (fast); normalise so r[0] == 1.
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if acf[0] <= 0:
        return None, 0.0
    acf = acf / acf[0]
    hi = min(max_period, n - 1)
    if hi < min_period:
        return None, 0.0
    seg = acf[min_period:hi + 1]
    if seg.size == 0:
        return None, 0.0
    # Prefer a local maximum; fall back to the arg-max of the segment.
    lag = int(np.argmax(seg)) + min_period
    return float(lag), float(acf[lag])


def detect_period(
    x: np.ndarray,
    min_period: int,
    max_period: int,
    season_fallback: int,
    fft_salience_min: float = 3.0,
    fft_strong_salience_min: float = 40.0,
    acf_strength_min: float = 0.2,
) -> Tuple[float, str]:
    """Estimate a single series' strongest period.

    Strategy: linearly detrend, then take the FFT periodogram peak and the ACF
    peak.  Prefer the FFT peak when it is salient AND the ACF agrees it is a real
    correlation; otherwise fall back to the ACF peak, then to the frequency-based
    seasonality (``season_fallback``).  Returns (period, method).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2 * min_period + 1:
        return float(max(season_fallback, min_period)), "fallback_short"

    # Linear detrend + de-mean so trend/level don't dominate the low-freq bins.
    t = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(t, x, 1)
    x = x - (slope * t + intercept)
    if not np.any(np.abs(x) > 1e-12):
        return float(max(season_fallback, min_period)), "fallback_flat"

    fft_p, fft_sal = _fft_period(x, min_period, max_period)
    acf_p, acf_str = _acf_period(x, min_period, max_period)

    # FFT peak, confirmed by a non-trivial ACF at a compatible lag.
    if fft_p is not None and fft_sal >= fft_salience_min and acf_str >= acf_strength_min:
        return fft_p, "fft"
    if acf_p is not None and acf_str >= acf_strength_min:
        return acf_p, "acf"
    # ACF-unconfirmed FFT peak: accept ONLY when the spectral peak is very strong
    # (a clean periodic series sits >100x the spectral floor; white noise peaks
    # stay below ~20x), otherwise fall back to the frequency-based seasonality.
    if fft_p is not None and fft_sal >= fft_strong_salience_min:
        return fft_p, "fft_strong"
    return float(max(season_fallback, min_period)), "fallback_season"


# ==============================================================================
#  WINDOW QUANTIZATION  (bound the number of distinct forward-pass widths)
# ==============================================================================

def quantize_windows(L: np.ndarray, mode: str, n_buckets: int) -> np.ndarray:
    """Snap per-instance window lengths onto a small grid to limit distinct widths.

    Per-series windows would otherwise force models that recompile per context
    length (moirai, timesfm) to rebuild once per unique value.  We snap each
    length DOWN to the nearest bucket edge (never granting a model more context
    than its L_i dictates), mirroring v5's pad-mode snap-down philosophy.

    mode='none' returns L unchanged (exact per-series widths).
    mode='log'  uses ``n_buckets`` log-spaced edges between min(L) and max(L).
    """
    L = np.asarray(L, dtype=np.int64)
    if mode == "none" or L.size == 0:
        return L
    lo, hi = int(L.min()), int(L.max())
    if lo >= hi:
        return L
    edges = np.unique(
        np.round(np.exp(np.linspace(np.log(lo), np.log(hi), n_buckets))).astype(np.int64)
    )
    # Largest edge <= each L (snap down); clip guards L below the smallest edge.
    pos = np.clip(np.searchsorted(edges, L, side="right") - 1, 0, None)
    return edges[pos]


# ==============================================================================
#  MODEL CONTEXT CAPS  (mirror v5's per-family skips, applied per instance)
# ==============================================================================

def _family_cap(model_family: str, horizon: int) -> int:
    """Largest context this family can serve for the given horizon (inclusive)."""
    if model_family == "sundial":
        return SUNDIAL_MAX_CONTEXT
    if model_family == "timemoe":
        return max(0, TIMEMOE_MAX_TOTAL - horizon)
    return 1 << 30  # effectively unbounded; real_len is the real limit


# ==============================================================================
#  PER-(model, dataset, term) EVALUATION
# ==============================================================================

def _compare_dir(run_dir: str, model_short: str) -> str:
    return os.path.join(run_dir, "models", model_short, "compare_real_vs_predicted")


def _sidecar_paths(run_dir: str, dataset_display: str, term: str, model_short: str) -> Tuple[str, str]:
    cdir = _compare_dir(run_dir, model_short)
    base = f"period_{dataset_display}_t{term}_{model_short}"
    return os.path.join(cdir, base + ".json"), os.path.join(cdir, base + "_win.npz")


def _full_window_cap(
    run_dir: str, dataset_display: str, term: str, model_short: str
) -> Optional[int]:
    """Largest VALID ablation-grid window for this (model, dataset, term).

    This is the same quantity ``compare_window_strategies_gifteval.py`` treats as
    the ``full`` strategy (largest grid window with a non-NaN real MASE), read
    from v5's ``compare_<dataset>_t<term>_<model>.npz``.  The period strategy is
    capped at this so the model is never given more context than the full window
    — keeping the methods comparable and the period/full FLOPs ratio <= 1.

    Returns None when the v5 npz is absent/unreadable (no cap can be derived).
    """
    npz_path = os.path.join(
        _compare_dir(run_dir, model_short),
        f"compare_{dataset_display}_t{term}_{model_short}.npz",
    )
    if not os.path.isfile(npz_path):
        return None
    try:
        data = np.load(npz_path)
        window_grid = np.asarray(data["window_grid"])
        real_curve = np.asarray(data["real_curve"])
    except Exception:
        return None
    valid = np.flatnonzero(~np.isnan(real_curve))
    if valid.size == 0:
        return None
    return int(window_grid[valid[-1]])


def evaluate_one(
    cache: GiftEvalCache,
    model_id: str,
    model_family: str,
    model_short: str,
    ensure_handle,
    args,
    device: str,
    full_window: Optional[int] = None,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Run the period-window strategy for one (model, dataset, term).

    Returns (metrics_dict, per_instance_windows, per_instance_periods). Instances
    the family cannot serve at any length (e.g. TimeMoE when horizon alone
    exhausts its budget) are dropped from the aggregate.

    ``full_window`` (when given) is the largest valid ablation-grid window; the
    per-series period window is never allowed to exceed it, because the model is
    never fed more context than the full-window strategy uses.
    """
    horizon = cache.horizon
    n_total = cache.n_total
    cap = _family_cap(model_family, horizon)

    max_period_cap = max(args.min_period, int(args.max_period_frac * cache.max_context))

    periods = np.empty(n_total, dtype=np.float64)
    methods: List[str] = []
    raw_L = np.empty(n_total, dtype=np.int64)
    for i in range(n_total):
        ctx = cache.contexts[i]
        real_len = int(cache.context_lengths[i])
        p, method = detect_period(
            ctx,
            min_period=args.min_period,
            max_period=min(max_period_cap, max(args.min_period, real_len // 2)),
            season_fallback=cache.season,
        )
        periods[i] = p
        methods.append(method)
        raw_L[i] = max(int(round(2.0 * p)), horizon)

    # Clamp to each instance's genuine context and the family's serving cap.
    eff_L = np.minimum(raw_L, cache.context_lengths.astype(np.int64))
    eff_L = np.minimum(eff_L, cap)
    # Never exceed the full-window grid ceiling: the model cannot ingest more
    # than the full strategy does, so capping here keeps period comparable and
    # bounds its FLOPs ratio vs full at <= 1.
    if full_window is not None and full_window > 0:
        eff_L = np.minimum(eff_L, np.int64(full_window))

    valid = eff_L >= 1
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        raise RuntimeError(
            f"{model_short}/{cache.dataset_display}/t{cache.horizon}: no instance "
            f"servable by family cap={cap} (horizon={horizon})."
        )

    # Quantize the servable windows onto a bounded grid, then group identical
    # widths into single forward passes.
    W = quantize_windows(eff_L[valid_idx], args.quantize, args.n_buckets)
    # Guard: never exceed the instance's real length after snap-down rounding.
    W = np.minimum(W, cache.context_lengths[valid_idx].astype(np.int64))

    t0 = time.perf_counter()
    results = []   # (compact_indices, ForecastResult, tgts)
    labels_valid = cache.labels_np[valid_idx]
    for w in np.unique(W):
        w = int(w)
        grp = np.flatnonzero(W == w)            # positions into valid_idx
        g = grp.size
        x_np = np.empty((g, w), dtype=np.float32)
        for j, pos in enumerate(grp):
            x_np[j] = cache.contexts[valid_idx[pos]][-w:]
        y_np = labels_valid[grp]

        all_x = torch.from_numpy(x_np).unsqueeze(-1)
        all_y = torch.from_numpy(y_np).unsqueeze(-1)
        if device == "cuda":
            all_x = all_x.pin_memory()
            all_y = all_y.pin_memory()
        batches = []
        for start in range(0, g, args.batch_size):
            stop = min(start + args.batch_size, g)
            batches.append({"x": all_x[start:stop], "y": all_y[start:stop]})

        fr_w, tgts_w = _forecast_cell(
            model_family, ensure_handle(), model_id, batches,
            w, horizon, device, args.batch_size)
        results.append((grp, fr_w, tgts_w))

    fr, tgts = _merge_grouped(results, valid_idx.size, horizon, device)
    elapsed = time.perf_counter() - t0

    metrics = compute_all_metrics(fr, tgts, cache.naive_seasonal_mae_train)
    metrics["elapsed_seconds"] = round(elapsed, 3)
    metrics["horizon"] = horizon

    # Window stats are over the actually-served (quantized) widths.
    win_full = np.full(n_total, -1, dtype=np.int64)
    win_full[valid_idx] = W
    method_counts: Dict[str, int] = {}
    for m in methods:
        method_counts[m] = method_counts.get(m, 0) + 1

    summary = {
        "dataset_display": cache.dataset_display,
        "term": None,                      # filled by caller (term not on cache)
        "model_short": model_short,
        "model": model_id,
        "model_family": model_family,
        "horizon": horizon,
        "n_total": int(n_total),
        "n_instances": int(valid_idx.size),
        "full_window_cap": (int(full_window) if full_window else None),
        "period_mae": float(metrics.get("mae", float("nan"))),
        "period_mase": float(metrics.get("mase", float("nan"))),
        "period_elapsed_s": float(elapsed),
        "window_mean": float(W.mean()),
        "window_median": float(np.median(W)),
        "window_min": int(W.min()),
        "window_max": int(W.max()),
        "n_distinct_windows": int(np.unique(W).size),
        "period_mean": float(periods[valid_idx].mean()),
        "period_median": float(np.median(periods[valid_idx])),
        "method_counts": method_counts,
        "all_metrics": {k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in metrics.items()},
    }
    return summary, win_full, periods


# ==============================================================================
#  DRIVER
# ==============================================================================

def _make_ensure_handle(model_id: str, model_family: str, device: str):
    handle = [None]

    def ensure_handle():
        if handle[0] is None:
            if model_family == "chronos_bolt":
                handle[0] = load_chronos_bolt(model_id, device)
            elif model_family == "chronos2":
                handle[0] = load_chronos2(model_id, device)
            elif model_family == "moirai":
                handle[0] = load_moirai_module(model_id)
            elif model_family == "moirai_1_1":
                handle[0] = load_moirai_1_1_module(model_id)
            elif model_family == "patchtst_fm":
                handle[0] = load_patchtst_fm(model_id, device)
            elif model_family == "sundial":
                handle[0] = load_sundial(model_id, device)
            elif model_family == "timemoe":
                handle[0] = load_timemoe(model_id, device)
        return handle[0]

    return ensure_handle, handle


def run(args, device: str) -> None:
    models = [m for m in MODELS if (args.models is None or m[2] in args.models)]
    datasets = [d for d in DATASETS if (args.datasets is None or d[2] in args.datasets)]
    if not models:
        raise SystemExit(f"No models selected. Available: {[m[2] for m in MODELS]}")

    print(Fore.CYAN
          + f"Period-window eval  |  device={device}  |  quantize={args.quantize}"
          + (f" (n_buckets={args.n_buckets})" if args.quantize != "none" else "")
          + f"  |  models={[m[2] for m in models]}"
          + Fore.RESET)

    ge_cache: Dict[Tuple[str, str], GiftEvalCache] = {}

    for model_id, model_family, model_short in models:
        print(Fore.CYAN + "\n" + "=" * 78
              + f"\n  MODEL: {model_id}  ({model_family})\n" + "=" * 78 + Fore.RESET)
        ensure_handle, handle = _make_ensure_handle(model_id, model_family, device)

        for ge_name, term, dataset_display, to_univariate in datasets:
            json_path, npz_path = _sidecar_paths(args.run_dir, dataset_display, term, model_short)
            if os.path.isfile(json_path) and not args.force:
                print(Fore.WHITE
                      + f"  CACHED  {model_short} | {dataset_display} | t={term}  -> skip"
                      + Fore.RESET)
                continue

            ds_key = (ge_name, term)
            if ds_key not in ge_cache:
                print(Fore.CYAN + f"\n  Loading GiftEval: {ge_name}  term={term}" + Fore.RESET)
                ge_dataset = GiftEvalDataset(name=ge_name, term=term, to_univariate=to_univariate)
                ge_cache[ds_key] = GiftEvalCache(ge_dataset, dataset_display)
            cache = ge_cache[ds_key]

            # Full-window ceiling (largest valid grid window) from v5's npz, so
            # period never feeds the model more context than the full strategy.
            full_w = _full_window_cap(args.run_dir, dataset_display, term, model_short)
            if full_w is None:
                print(Fore.YELLOW
                      + f"    WARN: no v5 npz for {dataset_display} t={term}; "
                        "period window left uncapped by full window."
                      + Fore.RESET)

            tag = f"{model_short} | {dataset_display} | t={term} | h={cache.horizon}"
            print(Fore.YELLOW + f"\n  > {tag}  (n={cache.n_total}"
                  + (f", full_cap={full_w}" if full_w else "") + ")" + Fore.RESET)
            try:
                summary, windows, periods = evaluate_one(
                    cache, model_id, model_family, model_short,
                    ensure_handle, args, device, full_window=full_w)
            except RuntimeError as exc:
                print(Fore.RED + f"    SKIP: {exc}" + Fore.RESET)
                continue

            summary["term"] = term
            print(Fore.GREEN
                  + f"    period_mase={summary['period_mase']:.6f}  "
                  + f"window: mean={summary['window_mean']:.0f} "
                  + f"median={summary['window_median']:.0f} "
                  + f"[{summary['window_min']},{summary['window_max']}]  "
                  + f"({summary['n_distinct_windows']} widths, {summary['period_elapsed_s']:.1f}s)"
                  + Fore.RESET)

            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=2)
            np.savez_compressed(npz_path, windows=windows, periods=periods)

            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        handle[0] = None
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print(Fore.GREEN + "\nPeriod-window eval done." + Fore.RESET)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-series 2x-strongest-period context-length strategy.")
    p.add_argument("--models", type=str, nargs="+", default=None,
                   help="Restrict to these model_short names (default: all in v5's MODELS).")
    p.add_argument("--datasets", type=str, nargs="+", default=None,
                   help="Restrict to these dataset_display names (default: all).")
    p.add_argument("--run-dir", type=str, required=True,
                   help="v5 run dir (the 'general' folder). Sidecars are written under "
                        "<run-dir>/models/<model>/compare_real_vs_predicted/.")
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--force", action="store_true",
                   help="Recompute even when a sidecar JSON already exists.")
    # Period detection knobs.
    p.add_argument("--min-period", type=int, default=2,
                   help="Smallest detectable period (samples).")
    p.add_argument("--max-period-frac", type=float, default=0.5,
                   help="Largest detectable period as a fraction of a dataset's max context.")
    # Window quantization (bounds distinct forward-pass widths -> moirai/timesfm recompiles).
    p.add_argument("--quantize", choices=["log", "none"], default="log",
                   help="'log': snap per-series windows down onto n_buckets log-spaced widths "
                        "(default; bounds recompiles). 'none': exact per-series widths.")
    p.add_argument("--n-buckets", type=int, default=48,
                   help="Number of log-spaced window buckets when --quantize log.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    run(args, device)


if __name__ == "__main__":
    main()
