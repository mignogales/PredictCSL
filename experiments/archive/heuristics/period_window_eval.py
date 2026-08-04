"""
period_window_eval.py  --  GiftEval-cadence period context heuristics.

A fourth context-selection strategy for the PredictCSL pipeline, complementing
the grid strategies produced by ``test_window_ablation_gifteval_v5.py`` (full /
best / predictor).  Instead of choosing a window from the predictor's discrete
grid, this method picks, *per test instance*, the context length

    L_i = max( k x strongest_period_i ,  horizon ),  k in {2, 3}

where ``strongest_period_i`` is selected only from the sampling cadences used by
GiftEval (10S, 5T, 10T, 15T, H, D, W, M, Q, A).  Each cadence is translated to
samples at the dataset's sampling rate.  The context is split into consecutive
non-overlapping chunks of that size and the candidate with the greatest mean
correlation between adjacent, per-chunk-normalised windows wins.  We report both
two-cycle and three-cycle contexts, never shorter than the forecast horizon.
Every requested width is capped by that model/horizon's true full-native limit,
not by the largest numeric ablation-grid point.

Because each instance gets its OWN window, ``L_i`` is generally NOT one of the
ablation-grid windows -- so we cannot just read a cached cell.  We evaluate the
model directly at each per-series length here (grouping instances that share a
length into one forward pass, exactly like v5's pad-mode width groups), then
aggregate one MASE per (model, dataset, term) -- directly comparable to the grid
strategies' aggregate MASE (same instances, same naive-seasonal MASE
denominator).

Outputs (one per (model, dataset, term)) are written to a dedicated strategy
tree, so ``compare_window_strategies_gifteval.py`` can pick them up and surface
``period`` as a first-class strategy:

    <run_dir>/models/<model_short>/compare_real_vs_predicted/
        period_<dataset>_t<term>_<model_short>.json      2-cycle strategy
        period_<dataset>_t<term>_<model_short>_win.npz
        period3_<dataset>_t<term>_<model_short>.json     3-cycle strategy
        period3_<dataset>_t<term>_<model_short>_win.npz

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
import glob
import json
import os
import shutil
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from colorama import Fore

from gift_eval.data import Dataset as GiftEvalDataset
from experiments.archive.heuristics.period_detection import (
    GIFT_EVAL_PERIOD_LABELS, detect_period,
)

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
    compute_per_sample_metrics,
    cell_mase_gluonts_real,
    _forecast_cell,
    _merge_grouped,
    load_chronos_bolt,
    load_chronos2,
    load_moirai_module,
    load_moirai_1_1_module,
    load_patchtst_fm,
    load_sundial,
    load_timemoe,
    load_toto,
    load_flowstate,
    load_tirex,
    _full_native_context_cap,
)


PERIOD_POLICY_VERSION = 2  # v2 uses full-native caps, never a grid ceiling
PERIOD_DETECTION_CACHE_VERSION = 1


def _detection_cache_path(run_dir: str, dataset_display: str, term: str) -> str:
    """Model-independent cadence cache shared by 2xP, 3xP, and all models."""
    return os.path.join(
        run_dir, "period_detection_cache",
        f"periods_{dataset_display}_t{term}.npz")


def _load_aligned_detection(path: str, cache: GiftEvalCache):
    try:
        with np.load(path, allow_pickle=False) as data:
            periods = np.asarray(data["periods"], dtype=np.float64)
            labels = np.asarray(data["period_labels"]).astype(str)
            similarities = np.asarray(data["period_similarity"], dtype=np.float64)
            lengths = np.asarray(data["context_lengths"], dtype=np.int64)
            version = int(data["cache_version"])
            freq = str(data["sampling_frequency"])
        if (version != PERIOD_DETECTION_CACHE_VERSION
                or freq != str(cache.freq)
                or periods.shape != (cache.n_total,)
                or labels.shape != (cache.n_total,)
                or similarities.shape != (cache.n_total,)
                or not np.array_equal(lengths, cache.context_lengths.astype(np.int64))):
            return None
        return periods, labels.tolist(), similarities
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _get_period_detections(cache: GiftEvalCache, run_dir: str, term: str):
    """Load aligned detections, seed from old sidecars, or compute exactly once."""
    cache_path = _detection_cache_path(run_dir, cache.dataset_display, term)
    loaded = _load_aligned_detection(cache_path, cache)
    if loaded is not None:
        print(Fore.WHITE + f"    cadence cache: {cache_path}" + Fore.RESET)
        return loaded

    # Completed 2xP/3xP sidecars already contain the same model-independent
    # detection.  Promote one rather than repeating work after an upgrade.
    pattern = os.path.join(
        run_dir, "models", "*", "compare_real_vs_predicted",
        f"period*_{cache.dataset_display}_t{term}_*_win.npz")
    for old_path in glob.glob(pattern):
        try:
            with np.load(old_path, allow_pickle=False) as data:
                periods = np.asarray(data["periods"], dtype=np.float64)
                labels = np.asarray(data["period_labels"]).astype(str)
                similarities = np.asarray(data["period_similarity"], dtype=np.float64)
            if all(a.shape == (cache.n_total,) for a in
                   (periods, labels, similarities)):
                loaded = (periods, labels.tolist(), similarities)
                print(Fore.WHITE + f"    cadence reuse: {old_path}" + Fore.RESET)
                break
        except (OSError, ValueError, KeyError):
            continue
    else:
        periods = np.empty(cache.n_total, dtype=np.float64)
        labels = []
        similarities = np.full(cache.n_total, np.nan, dtype=np.float64)
        for i, context in enumerate(cache.contexts_raw):
            p, method, scores = detect_period(
                context, sampling_freq=cache.freq,
                season_fallback=cache.season)
            periods[i] = p
            labels.append(method)
            if method in scores and np.isfinite(scores[method]):
                similarities[i] = scores[method]
        loaded = (periods, labels, similarities)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        cache_version=np.asarray(PERIOD_DETECTION_CACHE_VERSION),
        sampling_frequency=np.asarray(str(cache.freq)),
        context_lengths=cache.context_lengths.astype(np.int64),
        periods=loaded[0], period_labels=np.asarray(loaded[1]),
        period_similarity=loaded[2])
    return loaded


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
    # Keep the period policy on exactly the same native serving limits as the
    # full-native baseline, including horizon-dependent Moirai/TimeMoE caps and
    # the registered Toto/FlowState/TiRex limits.
    return _full_native_context_cap(model_family, horizon, 1 << 30)


# ==============================================================================
#  PER-(model, dataset, term) EVALUATION
# ==============================================================================

def _compare_dir(run_dir: str, model_short: str) -> str:
    return os.path.join(run_dir, "models", model_short, "compare_real_vs_predicted")


def _sidecar_paths(
    run_dir: str,
    dataset_display: str,
    term: str,
    model_short: str,
    period_multiple: int = 2,
) -> Tuple[str, str]:
    cdir = _compare_dir(run_dir, model_short)
    prefix = "period" if period_multiple == 2 else f"period{period_multiple}"
    base = f"{prefix}_{dataset_display}_t{term}_{model_short}"
    return os.path.join(cdir, base + ".json"), os.path.join(cdir, base + "_win.npz")


def _full_window_cap(
    run_dir: str, dataset_display: str, term: str, model_short: str
) -> Optional[int]:
    """Largest valid grid window, used only to verify comparison-cell presence.

    The authoritative baseline is now ``full_native``.  This legacy grid ceiling
    remains useful as a cheap existence/scope check, but must not cap a series:
    doing so can truncate below the native baseline when a model cap is off-grid.

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
    period_detections=None,
) -> Tuple[dict, np.ndarray, np.ndarray, dict]:
    """Run the period-window strategy for one (model, dataset, term).

    Returns (metrics_dict, per_instance_windows, per_instance_periods,
    per_instance_metrics). Instances the family cannot serve at any length
    (e.g. TimeMoE when horizon alone exhausts its budget) are dropped from the
    aggregate.

    ``full_window`` is retained as comparison metadata for compatibility.  The
    serving ceiling is the model's true native cap, not the largest numeric grid
    point, so off-grid native capacity remains available to the period policy.
    """
    horizon = cache.horizon
    n_total = cache.n_total
    cap = _family_cap(model_family, horizon)

    if period_detections is None:
        periods = np.empty(n_total, dtype=np.float64)
        methods: List[str] = []
        similarities = np.full(n_total, np.nan, dtype=np.float64)
        for i in range(n_total):
            p, method, scores = detect_period(
                cache.contexts_raw[i], sampling_freq=cache.freq,
                season_fallback=cache.season)
            periods[i] = p
            methods.append(method)
            if method in scores and np.isfinite(scores[method]):
                similarities[i] = scores[method]
    else:
        periods, methods, similarities = period_detections
    raw_L = np.empty(n_total, dtype=np.int64)
    for i in range(n_total):
        raw_L[i] = max(int(round(args.period_multiple * periods[i])), horizon)

    # Clamp to each instance's genuine context and the family's serving cap.
    eff_L = np.minimum(raw_L, cache.context_lengths.astype(np.int64))
    eff_L = np.minimum(eff_L, cap)
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
            w, horizon, device, args.batch_size,
            flowstate_scale=cache.flowstate_scale)
        results.append((grp, fr_w, tgts_w))

    fr, tgts = _merge_grouped(results, valid_idx.size, horizon, device)
    elapsed = time.perf_counter() - t0

    # Gluonts metrics too: the merged forecast covers valid_idx in row order, so
    # the per-instance seasonal errors (port) and the start Periods / raw contexts
    # (machinery) align 1:1. Without these the period strategy would fall out of
    # the gluonts-scored comparisons (or worse, mix in the custom `mase`).
    metrics = compute_all_metrics(fr, tgts, cache.naive_seasonal_mae_train,
                                  seasonal_errors=cache.seasonal_errors_gluonts[valid_idx])
    metrics["mase_gluonts_real"] = cell_mase_gluonts_real(fr, cache, valid_idx)
    metrics["elapsed_seconds"] = round(elapsed, 3)
    metrics["horizon"] = horizon
    per_sample = compute_per_sample_metrics(
        fr, tgts, cache.naive_seasonal_mae_train,
        seasonal_errors=cache.seasonal_errors_gluonts[valid_idx])
    per_sample["served_index"] = valid_idx.astype(np.int32, copy=False)

    # Window stats are over the actually-served (quantized) widths.
    win_full = np.full(n_total, -1, dtype=np.int64)
    win_full[valid_idx] = W
    method_counts: Dict[str, int] = {}
    for m in methods:
        method_counts[m] = method_counts.get(m, 0) + 1

    summary = {
        "period_policy_version": PERIOD_POLICY_VERSION,
        "dataset_display": cache.dataset_display,
        "term": None,                      # filled by caller (term not on cache)
        "model_short": model_short,
        "model": model_id,
        "model_family": model_family,
        "horizon": horizon,
        "sampling_frequency": cache.freq,
        "period_multiple": int(args.period_multiple),
        "period_candidates": list(GIFT_EVAL_PERIOD_LABELS),
        "n_total": int(n_total),
        "n_instances": int(valid_idx.size),
        "full_window_cap": int(cap),  # compatibility: now the true native cap
        "comparison_grid_ceiling": (int(full_window) if full_window else None),
        "native_context_cap": int(cap),
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
        "cadence_counts": method_counts,
        "similarity_mean": (
            float(np.nanmean(similarities[valid_idx]))
            if np.isfinite(similarities[valid_idx]).any() else float("nan")
        ),
        "all_metrics": {k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in metrics.items()},
    }
    per_sample["period_labels"] = np.asarray(methods)
    per_sample["period_similarity"] = similarities
    return summary, win_full, periods, per_sample


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
            elif model_family == "toto":
                handle[0] = load_toto(model_id, device)
            elif model_family == "flowstate":
                handle[0] = load_flowstate(model_id, device)
            elif model_family == "tirex":
                handle[0] = load_tirex(model_id, device)
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
          + f"  |  multiples={args.period_multiples}"
          + f"  |  models={[m[2] for m in models]}"
          + Fore.RESET)

    ge_cache: Dict[Tuple[str, str], GiftEvalCache] = {}

    for model_id, model_family, model_short in models:
        print(Fore.CYAN + "\n" + "=" * 78
              + f"\n  MODEL: {model_id}  ({model_family})\n" + "=" * 78 + Fore.RESET)
        ensure_handle, handle = _make_ensure_handle(model_id, model_family, device)

        for ge_name, term, dataset_display, to_univariate in datasets:
            pending: List[Tuple[int, str, str]] = []
            for multiple in args.period_multiples:
                json_path, npz_path = _sidecar_paths(
                    args.run_dir, dataset_display, term, model_short, multiple)
                # One-time migration from the historical layout where period
                # sidecars lived inside the base predictor tree.
                if (os.path.normpath(args.comparison_run_dir)
                        != os.path.normpath(args.run_dir)
                        and not os.path.exists(json_path)
                        and not os.path.exists(npz_path)):
                    legacy_json, legacy_npz = _sidecar_paths(
                        args.comparison_run_dir, dataset_display, term,
                        model_short, multiple)
                    if os.path.isfile(legacy_json) and os.path.isfile(legacy_npz):
                        os.makedirs(os.path.dirname(json_path), exist_ok=True)
                        shutil.copy2(legacy_json, json_path)
                        shutil.copy2(legacy_npz, npz_path)
                        print(Fore.WHITE
                              + f"  MIGRATED {model_short} | {dataset_display} | "
                                f"t={term} | {multiple}xP -> {args.run_dir}"
                              + Fore.RESET)
                per_instance_cached = False
                if os.path.isfile(json_path) and os.path.isfile(npz_path):
                    try:
                        with open(json_path) as metadata_file:
                            cached_meta = json.load(metadata_file)
                        with np.load(npz_path) as cached:
                            per_instance_cached = {
                                "windows", "periods", "period_labels",
                                "period_similarity", "mase_gluonts", "served_index",
                            }.issubset(cached.files) and (
                                cached_meta.get("period_policy_version")
                                == PERIOD_POLICY_VERSION)
                    except (OSError, ValueError):
                        per_instance_cached = False
                if per_instance_cached and not args.force:
                    print(Fore.WHITE
                          + f"  CACHED  {model_short} | {dataset_display} | "
                            f"t={term} | {multiple}xP  -> skip"
                          + Fore.RESET)
                    continue
                if os.path.isfile(json_path) and not args.force:
                    print(Fore.YELLOW
                          + f"  STALE   {model_short} | {dataset_display} | t={term} | "
                            f"{multiple}xP: sidecar uses the old detector or lacks "
                            "aligned per-instance MASE; recomputing."
                          + Fore.RESET)
                pending.append((multiple, json_path, npz_path))
            if not pending:
                continue

            ds_key = (ge_name, term)
            if ds_key not in ge_cache:
                print(Fore.CYAN + f"\n  Loading GiftEval: {ge_name}  term={term}" + Fore.RESET)
                ge_dataset = GiftEvalDataset(name=ge_name, term=term, to_univariate=to_univariate)
                ge_cache[ds_key] = GiftEvalCache(ge_dataset, dataset_display)
            cache = ge_cache[ds_key]

            # The grid ceiling is only a comparison-cell presence check/metadata
            # field. evaluate_one applies the authoritative native model cap.
            comparison_ceiling = _full_window_cap(
                args.comparison_run_dir, dataset_display, term, model_short)
            if comparison_ceiling is None:
                if args.require_comparison:
                    print(Fore.WHITE
                          + f"  SKIP  {model_short} | {dataset_display} | t={term}: "
                            "no matching v5 comparison cell." + Fore.RESET)
                    continue
                print(Fore.YELLOW
                      + f"    WARN: no v5 npz for {dataset_display} t={term}; "
                        "running with the native model cap only."
                      + Fore.RESET)

            period_detections = _get_period_detections(cache, args.run_dir, term)

            for multiple, json_path, npz_path in pending:
                args.period_multiple = multiple
                tag = (f"{model_short} | {dataset_display} | t={term} | "
                       f"h={cache.horizon} | {multiple}xP")
                print(Fore.YELLOW + f"\n  > {tag}  (n={cache.n_total}"
                      + (f", grid_ceiling={comparison_ceiling}"
                         if comparison_ceiling else "") + ")" + Fore.RESET)
                try:
                    summary, windows, periods, per_sample = evaluate_one(
                        cache, model_id, model_family, model_short,
                        ensure_handle, args, device,
                        full_window=comparison_ceiling,
                        period_detections=period_detections)
                except RuntimeError as exc:
                    print(Fore.RED + f"    SKIP: {exc}" + Fore.RESET)
                    continue

                summary["term"] = term
                print(Fore.GREEN
                      + f"    {multiple}xP_mase={summary['period_mase']:.6f}  "
                      + f"window: mean={summary['window_mean']:.0f} "
                      + f"median={summary['window_median']:.0f} "
                      + f"[{summary['window_min']},{summary['window_max']}]  "
                      + f"({summary['n_distinct_windows']} widths, "
                        f"{summary['period_elapsed_s']:.1f}s)"
                      + Fore.RESET)

                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, "w") as f:
                    json.dump(summary, f, indent=2)
                np.savez_compressed(
                    npz_path, windows=windows, periods=periods, **per_sample)

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
        description="Per-series GiftEval-cadence 2x/3x-period context strategies.")
    p.add_argument("--models", type=str, nargs="+", default=None,
                   help="Restrict to these model_short names (default: all in v5's MODELS).")
    p.add_argument("--datasets", type=str, nargs="+", default=None,
                   help="Restrict to these dataset_display names (default: all).")
    p.add_argument("--run-dir", type=str, required=True,
                   help="Dedicated period-strategy output tree. Sidecars are written under "
                        "<run-dir>/models/<model>/compare_real_vs_predicted/.")
    p.add_argument("--comparison-run-dir", type=str, default=None,
                   help="Base v5 tree used to verify matching comparison cells "
                        "(default: --run-dir, for backward compatibility).")
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--force", action="store_true",
                   help="Recompute even when a sidecar JSON already exists.")
    p.add_argument("--require-comparison", action="store_true",
                   help="Skip dataset cells without a matching v5 comparison NPZ. "
                        "Used by the master so smoke/partial runs stay in scope.")
    p.add_argument("--period-multiples", type=int, nargs="+", default=[2, 3],
                   help="Numbers of detected cycles to use as context (default: 2 3).")
    # Window quantization (bounds distinct forward-pass widths -> moirai/timesfm recompiles).
    p.add_argument("--quantize", choices=["log", "none"], default="none",
                   help="'log': snap per-series windows down onto n_buckets log-spaced widths "
                        "(bounds recompiles). 'none': exact 2x/3x cadence widths (default).")
    p.add_argument("--n-buckets", type=int, default=48,
                   help="Number of log-spaced window buckets when --quantize log.")
    args = p.parse_args()
    if args.comparison_run_dir is None:
        args.comparison_run_dir = args.run_dir
    if any(k < 1 for k in args.period_multiples):
        p.error("--period-multiples values must be positive integers")
    args.period_multiples = list(dict.fromkeys(args.period_multiples))
    return args


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
