"""
Robust wall-clock benchmark for the strategy-selected windows (the "clock-wise"
comparison's measurement stage).

The window-ablation stage (``test_window_ablation_gifteval_v5.py``) records a
*single-shot* ``elapsed_seconds`` per cell — fine for a coarse cost axis, but a
single timing is noisy (kernel-launch jitter, cuDNN autotune, CUDA-context
warm-up on the first call). This stage produces a *robust* forward-pass timing
for only the windows the strategy comparison actually consumes:

  * For each ``(model, dataset, term)`` it reads the strategy comparison's
    ``comparison.csv`` and collects the union of the on-grid windows chosen by the
    strategies — ``full_window``, ``best_window``, ``pred_window`` and every
    predictor-variant ``*_window`` column. The period strategy is per-series /
    off-grid (its window is not a single grid value), so it is NOT timed here and
    keeps its single-shot sidecar timing.

  * For each such ``(model, dataset, term, window)`` it deterministically samples
    up to ``--max-series`` series across the context-length distribution, pays
    ALL per-window setup
    ONCE — batch building plus (via ``build_forward``) the weight load + compile
    that timesfm, and the forecaster build that moirai, would otherwise re-run on
    every call — then runs ``--warmup`` discarded forward passes to absorb
    remaining one-time init, then ``--repeats`` timed passes. Each timed pass is
    bracketed by ``torch.cuda.synchronize()`` so the timer measures *completed*
    GPU work, not just enqueued kernels. The timed unit is the PURE forward pass
    (inference only) — the cost the strategy comparison's FLOPs axis represents —
    not the reload/compile. (The ablation's single-shot ``elapsed_seconds``, by
    contrast, includes per-cell setup for timesfm/moirai; the robust mean is the
    cleaner number and is preferred by the comparison when present.)

  * ``full_native`` is timed separately under ``wfull_native`` with every sampled
    series receiving its genuine available history (subject to the model cap).
    It is never approximated by a fixed numeric cap.

Output: a schema-versioned ``timing.json`` sidecar written into the SAME per-cell directory as the
ablation's ``metrics.json`` (``_cache_dir``), holding
``{timing_kind, n_timed_series, max_series, batch_size, n_warmup, n_repeats,
samples_s, mean_s, std_s, min_s, median_s, cv, cuda_*_gb}``. CUDA memory fields report the post-warmup peak allocated and
reserved memory (including resident weights and prepared inputs), the baseline
immediately before timed forwards, and the incremental allocated peak. Because
the v3/v4 variant trees symlink their ``datasets/`` cell cache to the shared
``general/datasets/``, writing here means every variant's comparison sees the
robust timing for free — the TSFM forward pass is predictor-independent, so it is
measured once on the base ``general/`` tree. A run-level ``timing_summary.csv`` is
also emitted at the cache root.

The run-level summary reports seconds per measured series, which downstream
rollup scales by the policy's assignment counts. Multi-GPU uses identical
dataset-level sharding to the v5 ablation — one worker per
visible CUDA device drains a round-robin slice of datasets, the coordinator then
aggregates the summary CSV. Re-runs are safe only for sidecars matching the
current schema, sample cap, timing kind, repeat count, and memory fields.

NOTE: robust timing currently covers ``skip`` short-context mode (the default
pipeline tree). ``pad`` mode lives under a separate ``general_pad`` tree and is
not timed here.

Usage (run on the SERVER)
-------------------------
    python -m experiments.benchmark_window_timing_gifteval \
        --run-dir logs/experiments/window_ablation_gifteval/general
    python -m experiments.benchmark_window_timing_gifteval --models Chronos2-Small
    python -m experiments.benchmark_window_timing_gifteval --repeats 10 --warmup 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from colorama import Fore

# Reuse the ablation's heavy machinery verbatim: dataset cache, the per-family
# load dispatch (load_handle), the forecast wrapper, the per-cell path helper,
# the catalogs and the model context caps. _cache_dir reads wab.CACHE_ROOT, so we
# set that global (as run_ablation does) before using it.
import experiments.test_window_ablation_gifteval_v5 as wab
from experiments.test_window_ablation_gifteval_v5 import (
    GiftEvalCache,
    GiftEvalDataset,
    MODELS,
    DATASETS,
    load_handle,
    build_forward,
    _cache_dir,
    SUNDIAL_MAX_CONTEXT,
    TIMEMOE_MAX_TOTAL,
    TOTO_MAX_CONTEXT,
    FLOWSTATE_MAX_CONTEXT,
    TIREX_MAX_CONTEXT,
)

DEFAULT_STRATEGY_SUBDIRS = [
    "strategy_comparison_v3",
    "strategy_comparison_v4",
]
TIMING_SCHEMA_VERSION = 2
FULL_NATIVE = wab.FULL_NATIVE_WINDOW


def _sync(device: str) -> None:
    """Drain the CUDA queue so the timer brackets completed work (no-op on CPU)."""
    if device == "cuda":
        torch.cuda.synchronize()


# ==============================================================================
#  SELECTED-WINDOW DISCOVERY  (from the strategy comparison)
# ==============================================================================

def _selected_windows(comparison_root: str, display: str,
                      strategy_subdirs: List[str]
                      ) -> Dict[Tuple[str, str], Set[int]]:
    """Union selected windows from one or more strategy-comparison variants.

    Collects every ``*_window`` column except the period ones (period is
    per-series / off-grid and not a single grid window). Empty dict when none of
    the requested comparisons has been produced for this model yet.
    """
    out: Dict[Tuple[str, str], Set[int]] = {}
    found: List[str] = []
    missing: List[str] = []
    for subdir in strategy_subdirs:
        csv_path = os.path.join(
            comparison_root, display, subdir, "comparison.csv")
        if not os.path.isfile(csv_path):
            missing.append(csv_path)
            continue
        found.append(subdir)
        df = pd.read_csv(csv_path)
        win_cols = [
            c for c in df.columns
            if c.endswith("_window") and not c.startswith("period")
        ]
        for _, row in df.iterrows():
            key = (str(row["dataset_display"]), str(row["term"]))
            wins: Set[int] = set()
            for column in win_cols:
                value = row[column]
                if pd.notna(value):
                    wins.add(int(round(float(value))))
            if wins:
                out.setdefault(key, set()).update(wins)
    if found:
        print(Fore.CYAN
              + f"  [{display}] strategy comparisons: {', '.join(found)}"
              + Fore.RESET)
    else:
        print(Fore.YELLOW
              + f"  [{display}] none of the requested comparison files exists:\n    "
              + "\n    ".join(missing) + Fore.RESET)
    return out


def _comparison_horizons(
    comparison_root: str,
    displays: List[str],
    strategy_subdirs: List[str],
) -> Dict[Tuple[str, str], int]:
    """Read dataset horizons without loading GiftEval or any model."""
    out: Dict[Tuple[str, str], int] = {}
    for display in displays:
        for subdir in strategy_subdirs:
            csv_path = os.path.join(
                comparison_root, display, subdir, "comparison.csv")
            if not os.path.isfile(csv_path):
                continue
            df = pd.read_csv(csv_path, usecols=[
                "dataset_display", "term", "horizon"])
            for row in df.itertuples(index=False):
                key = (str(row.dataset_display), str(row.term))
                horizon = int(row.horizon)
                previous = out.setdefault(key, horizon)
                if previous != horizon:
                    raise ValueError(
                        f"Conflicting horizons for {key}: {previous} vs {horizon}")
    return out


def _selection_csv_windows(
    csv_paths: List[str], display: str, methods: Optional[Set[str]] = None,
) -> Tuple[Dict[Tuple[str, str], Set[int]], Dict[Tuple[str, str], int]]:
    """Read per-instance policy window histograms emitted by risk evaluation."""
    selected: Dict[Tuple[str, str], Set[int]] = {}
    horizons: Dict[Tuple[str, str], int] = {}
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Selection histogram does not exist: {csv_path}")
        frame = pd.read_csv(csv_path)
        model_column = "model" if "model" in frame.columns else "model_short"
        dataset_column = (
            "dataset" if "dataset" in frame.columns else "dataset_display")
        required = {model_column, dataset_column, "term", "window_size", "horizon"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
        frame = frame[frame[model_column].astype(str) == display]
        if methods and "method" in frame.columns:
            frame = frame[frame["method"].astype(str).isin(methods)]
        # Full-native is not a fixed numeric window. It is discovered
        # separately and timed with each series' genuine available history.
        if "method" in frame.columns:
            frame = frame[frame["method"].astype(str) != "full_native"]
        for row in frame.itertuples(index=False):
            values = row._asdict()
            key = (str(values[dataset_column]), str(values["term"]))
            selected.setdefault(key, set()).add(int(values["window_size"]))
            horizon = int(values["horizon"])
            previous = horizons.setdefault(key, horizon)
            if previous != horizon:
                raise ValueError(
                    f"Conflicting horizons in selection CSVs for {key}: "
                    f"{previous} vs {horizon}")
    return selected, horizons


def _selection_csv_full_native(
    csv_paths: List[str], display: str, methods: Optional[Set[str]] = None,
) -> Tuple[Set[Tuple[str, str]], Dict[Tuple[str, str], int]]:
    """Find dataset/term cells that request the genuine full-native baseline."""
    selected: Set[Tuple[str, str]] = set()
    horizons: Dict[Tuple[str, str], int] = {}
    if methods is not None and "full_native" not in methods:
        return selected, horizons
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Selection histogram does not exist: {csv_path}")
        frame = pd.read_csv(csv_path)
        model_column = "model" if "model" in frame.columns else "model_short"
        dataset_column = (
            "dataset" if "dataset" in frame.columns else "dataset_display")
        required = {model_column, dataset_column, "term", "horizon", "method"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
        frame = frame[
            (frame[model_column].astype(str) == display)
            & (frame["method"].astype(str) == "full_native")
        ]
        for row in frame.itertuples(index=False):
            values = row._asdict()
            key = (str(values[dataset_column]), str(values["term"]))
            selected.add(key)
            horizon = int(values["horizon"])
            previous = horizons.setdefault(key, horizon)
            if previous != horizon:
                raise ValueError(
                    f"Conflicting full-native horizons for {key}: "
                    f"{previous} vs {horizon}")
    return selected, horizons


# ==============================================================================
#  TIMING SIDECAR I/O
# ==============================================================================

def _timing_path(dataset_display: str, model_short: str, term: str,
                 window_size: Union[int, str]) -> str:
    return os.path.join(
        _cache_dir(dataset_display, model_short, term, window_size), "timing.json")


def _timing_done(dataset_display: str, model_short: str, term: str,
                 window_size: Union[int, str], repeats: int, require_memory: bool,
                 timing_kind: str, max_series: int) -> bool:
    p = _timing_path(dataset_display, model_short, term, window_size)
    if not os.path.isfile(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        enough_repeats = int(d.get("n_repeats", 0)) >= repeats
        has_memory = not require_memory or (
            "cuda_peak_allocated_gb" in d
            and "cuda_peak_reserved_gb" in d
            and "cuda_incremental_peak_allocated_gb" in d
        )
        return (
            int(d.get("timing_schema_version", 0)) == TIMING_SCHEMA_VERSION
            and d.get("timing_kind") == timing_kind
            and int(d.get("max_series", -1)) == max_series
            and int(d.get("n_timed_series", 0)) > 0
            and enough_repeats and has_memory
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return False


def _save_timing(dataset_display: str, model_short: str, term: str,
                 window_size: Union[int, str], payload: dict) -> None:
    d = _cache_dir(dataset_display, model_short, term, window_size)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "timing.json"), "w") as f:
        json.dump(payload, f, indent=2)


def _summarize(times: List[float], warmup: int, batch_size: int,
               n_timed_series: int, max_series: int, timing_kind: str,
               device: str, memory: Optional[dict] = None) -> dict:
    arr = np.asarray(times, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {
        "timing_schema_version": TIMING_SCHEMA_VERSION,
        "timing_kind": timing_kind,
        "batch_size": int(batch_size),
        "n_timed_series": int(n_timed_series),
        "max_series": int(max_series),
        "n_warmup":  int(warmup),
        "n_repeats": int(arr.size),
        "samples_s": [round(t, 6) for t in times],
        "mean_s":    round(mean, 6),
        "std_s":     round(std, 6),
        "min_s":     round(float(arr.min()), 6),
        "median_s":  round(float(np.median(arr)), 6),
        "cv":        round(std / mean, 6) if mean > 0 else float("nan"),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
    } | (memory or {})


# ==============================================================================
#  MEASUREMENT
# ==============================================================================

def _model_cap_skip(model_family: str, window_size: int, horizon: int,
                    cache: GiftEvalCache) -> Optional[str]:
    """Return a skip reason string if this window can't (or shouldn't) be run for
    this model/dataset, else None. Mirrors the v5 ablation's guards."""
    if not cache.can_serve(window_size):
        return f"max_context={cache.max_context} < ws"
    if model_family == "sundial" and window_size > SUNDIAL_MAX_CONTEXT:
        return f"Sundial max context={SUNDIAL_MAX_CONTEXT} < ws"
    if model_family == "timemoe" and window_size + horizon > TIMEMOE_MAX_TOTAL:
        return f"TimeMoE ws+h={window_size + horizon} > {TIMEMOE_MAX_TOTAL}"
    if model_family == "toto" and window_size > TOTO_MAX_CONTEXT:
        return f"Toto max context={TOTO_MAX_CONTEXT} < ws"
    if model_family == "flowstate" and window_size > FLOWSTATE_MAX_CONTEXT:
        return f"FlowState max context={FLOWSTATE_MAX_CONTEXT} < ws"
    if model_family == "tirex" and window_size > TIREX_MAX_CONTEXT:
        return f"TiRex max context={TIREX_MAX_CONTEXT} < ws"
    return None


def _stratified_indices(lengths: np.ndarray, max_series: int) -> np.ndarray:
    """Deterministically span the effective-context distribution."""
    lengths = np.asarray(lengths)
    n = int(lengths.size)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if max_series <= 0 or n <= max_series:
        return np.arange(n, dtype=np.int64)
    order = np.argsort(lengths, kind="stable")
    positions = np.rint(np.linspace(0, n - 1, max_series)).astype(np.int64)
    return order[positions]


def _batches_from_tensors(
    x: torch.Tensor, y: torch.Tensor, indices: np.ndarray, batch_size: int,
    device: str,
) -> List[Dict[str, torch.Tensor]]:
    chosen = torch.as_tensor(indices, dtype=torch.long)
    sx, sy = x.index_select(0, chosen), y.index_select(0, chosen)
    if device == "cuda":
        sx, sy = sx.pin_memory(), sy.pin_memory()
    return [
        {"x": sx[start:start + batch_size], "y": sy[start:start + batch_size]}
        for start in range(0, len(indices), batch_size)
    ]


def _time_forward(
    forward: Callable[[], object], teardown: Callable[[], None], device: str,
    warmup: int, repeats: int,
) -> Tuple[List[float], dict]:
    try:
        for _ in range(warmup):
            forward()
        _sync(device)

        memory: dict = {}
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            baseline_allocated = torch.cuda.memory_allocated()
            baseline_reserved = torch.cuda.memory_reserved()

        times: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            forward()
            _sync(device)
            times.append(time.perf_counter() - t0)
        if device == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            gib = float(1024 ** 3)
            memory = {
                "cuda_baseline_allocated_gb": round(baseline_allocated / gib, 6),
                "cuda_baseline_reserved_gb": round(baseline_reserved / gib, 6),
                "cuda_peak_allocated_gb": round(peak_allocated / gib, 6),
                "cuda_peak_reserved_gb": round(peak_reserved / gib, 6),
                "cuda_incremental_peak_allocated_gb": round(
                    max(0, peak_allocated - baseline_allocated) / gib, 6),
            }
        return times, memory
    finally:
        teardown()


def _measure_window(cache: GiftEvalCache, model_family: str, ensure_handle,
                    model_id: str, window_size: int, horizon: int, device: str,
                    batch_size: int, warmup: int, repeats: int, max_series: int,
                    ) -> Tuple[List[float], dict, int]:
    """Time ``repeats`` PURE forward passes (after ``warmup`` discarded ones) for
    one window, each GPU-synced.

    Batch building, weight loading, and persistent-wrapper construction happen
    outside the timed loop. TimesFM is the exception: its official GiftEval
    ``forecast()`` recipe recompiles from each outer batch's maximum context, so
    that required dispatch remains inside its timed call. The model built for a
    window is freed afterwards via the teardown to avoid GPU accumulation."""
    _batches, all_x, all_y, valid_indices = cache.build_batches(
        window_size, batch_size, device, pin_memory=False,
        preserve_missing=(model_family == "timesfm"))
    del _batches
    sampled = _stratified_indices(
        cache.context_lengths[valid_indices], max_series)
    batches = _batches_from_tensors(
        all_x, all_y, sampled, batch_size, device)

    forward, teardown = build_forward(
        model_family, ensure_handle(), model_id, batches, window_size, horizon,
        device, batch_size, flowstate_scale=cache.flowstate_scale)
    times, memory = _time_forward(forward, teardown, device, warmup, repeats)
    return times, memory, len(sampled)


def _measure_full_native(
    cache: GiftEvalCache, model_family: str, ensure_handle, model_id: str,
    horizon: int, device: str, batch_size: int, warmup: int, repeats: int,
    max_series: int,
) -> Tuple[List[float], dict, int, int]:
    """Time sampled genuine-history inference; return timing and context cap."""
    cap = wab._full_native_context_cap(model_family, horizon, cache.max_context)
    effective_lengths = np.maximum(
        1, np.minimum(cache.context_lengths, cap)).astype(np.int32)
    sampled = _stratified_indices(effective_lengths, max_series)
    if sampled.size == 0:
        raise RuntimeError("Full-native timing has no series to sample")

    variable_families = {"timesfm", "chronos_bolt", "patchtst_fm", "sundial"}
    if model_family in variable_families:
        contexts = [
            np.asarray(cache.contexts_raw[int(i)][-cap:], dtype=np.float32)
            for i in sampled
        ]
        targets = cache.labels_np[sampled]

        def forward():
            return wab.predict_official_full_contexts(
                model_family, ensure_handle(), contexts, targets, horizon,
                device, batch_size=min(batch_size, len(contexts)))

        teardown = lambda: None
    else:
        source = (cache.contexts_raw if wab.preserves_missing(model_family)
                  else cache.contexts)
        forwards: List[Callable[[], object]] = []
        teardowns: List[Callable[[], None]] = []
        for width in np.unique(effective_lengths[sampled]):
            width = int(width)
            group_indices = sampled[effective_lengths[sampled] == width]
            x_np = np.stack([
                np.asarray(source[int(i)][-width:], dtype=np.float32)
                for i in group_indices
            ])
            y_np = cache.labels_np[group_indices]
            all_x = torch.from_numpy(x_np).unsqueeze(-1)
            all_y = torch.from_numpy(y_np).unsqueeze(-1)
            local_indices = np.arange(len(group_indices), dtype=np.int64)
            batches = _batches_from_tensors(
                all_x, all_y, local_indices, batch_size, device)
            group_forward, group_teardown = build_forward(
                model_family, ensure_handle(), model_id, batches, width, horizon,
                device, batch_size, flowstate_scale=cache.flowstate_scale)
            forwards.append(group_forward)
            teardowns.append(group_teardown)

        def forward():
            return [fn() for fn in forwards]

        def teardown():
            for fn in reversed(teardowns):
                fn()

    times, memory = _time_forward(forward, teardown, device, warmup, repeats)
    return times, memory, len(sampled), cap


# ==============================================================================
#  RUN-LEVEL SUMMARY
# ==============================================================================

def _write_summary(
    models, selected_by_model: Dict[str, Dict[Tuple[str, str], Set[int]]],
    full_native_by_model: Dict[str, Set[Tuple[str, str]]],
    horizons: Dict[Tuple[str, str], int], cache_root: str, repeats: int,
    max_series: int,
) -> None:
    """Scan every selected cell's timing.json and write timing_summary.csv at the
    cache root. Pure file-read aggregation (no GPU)."""
    rows = []
    missing = []
    for _model_id, _family, model_short in models:
        requested = [
            (dataset_display, term, window_size, "numeric_window")
            for (dataset_display, term), windows in sorted(
                selected_by_model.get(model_short, {}).items())
            for window_size in sorted(windows)
        ] + [
            (dataset_display, term, FULL_NATIVE, "full_native")
            for dataset_display, term in sorted(
                full_native_by_model.get(model_short, set()))
        ]
        for dataset_display, term, window_size, timing_kind in requested:
            valid = _timing_done(
                dataset_display, model_short, term, window_size, repeats,
                require_memory=False, timing_kind=timing_kind,
                max_series=max_series)
            p = _timing_path(dataset_display, model_short, term, window_size)
            if not valid:
                missing.append(
                    f"{model_short}/{dataset_display}/{term}/{window_size}")
                continue
            try:
                with open(p) as f:
                    d = json.load(f)
            except (OSError, json.JSONDecodeError):
                missing.append(
                    f"{model_short}/{dataset_display}/{term}/{window_size}")
                continue
            rows.append({
                    "model_short":     model_short,
                    "dataset_display": dataset_display,
                    "term":            term,
                    "horizon":         horizons.get((dataset_display, term), -1),
                    "window_size":     window_size,
                    "timing_schema_version": d.get("timing_schema_version"),
                    "timing_kind":     d.get("timing_kind"),
                    "batch_size":      d.get("batch_size"),
                    "n_timed_series":  d.get("n_timed_series"),
                    "max_series":      d.get("max_series"),
                    "n_warmup":        d.get("n_warmup"),
                    "n_repeats":       d.get("n_repeats"),
                    "mean_s":          d.get("mean_s"),
                    "std_s":           d.get("std_s"),
                    "min_s":           d.get("min_s"),
                    "median_s":        d.get("median_s"),
                    "cv":              d.get("cv"),
                    "per_series_s": (
                        float(d["mean_s"]) / int(d["n_timed_series"])),
                    "context_cap":      d.get("context_cap"),
                    "device":           d.get("device"),
                    "gpu_name":         d.get("gpu_name"),
                    "cuda_baseline_allocated_gb": d.get("cuda_baseline_allocated_gb"),
                    "cuda_baseline_reserved_gb": d.get("cuda_baseline_reserved_gb"),
                    "cuda_peak_allocated_gb": d.get("cuda_peak_allocated_gb"),
                    "cuda_peak_reserved_gb": d.get("cuda_peak_reserved_gb"),
                    "cuda_incremental_peak_allocated_gb": d.get(
                        "cuda_incremental_peak_allocated_gb"),
            })
    if missing:
        preview = "\n  ".join(missing[:20])
        suffix = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise RuntimeError(
            f"Timing coverage is incomplete ({len(missing)} cells):\n  "
            f"{preview}{suffix}")
    out = os.path.join(cache_root, "timing_summary.csv")
    os.makedirs(cache_root, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(Fore.GREEN + f"\nSaved {len(rows)} timed cells -> {out}" + Fore.RESET)


# ==============================================================================
#  MAIN BENCHMARK LOOP
# ==============================================================================

def run_timing(args, device: str, shard_id: Optional[int] = None,
               num_shards: int = 1) -> None:
    wab.CACHE_ROOT = args.cache_root  # _cache_dir reads this module global

    comparison_root = args.comparison_root or os.path.dirname(
        os.path.normpath(args.cache_root))

    models = [m for m in MODELS if (args.models is None or m[2] in args.models)]
    if args.models is not None and not models:
        known = [m[2] for m in MODELS]
        raise SystemExit(
            Fore.RED + f"--models {args.models} matched none of the known models.\n"
            + f"  Known: {known}" + Fore.RESET)

    # Per-model selected windows from the strategy comparison.
    selected_by_model: Dict[str, Dict[Tuple[str, str], Set[int]]] = {
        m[2]: _selected_windows(
            comparison_root, m[2], args.strategy_subdirs) for m in models}
    full_native_by_model: Dict[str, Set[Tuple[str, str]]] = {
        m[2]: set() for m in models}
    selection_horizons: Dict[Tuple[str, str], int] = {}
    requested_methods = set(args.selection_methods or []) or None
    for _model_id, _family, model_short in models:
        extra, extra_horizons = _selection_csv_windows(
            args.selection_csvs, model_short, requested_methods)
        native, native_horizons = _selection_csv_full_native(
            args.selection_csvs, model_short, requested_methods)
        full_native_by_model[model_short].update(native)
        for key, windows in extra.items():
            selected_by_model[model_short].setdefault(key, set()).update(windows)
        for key, horizon in extra_horizons.items():
            previous = selection_horizons.setdefault(key, horizon)
            if previous != horizon:
                raise ValueError(
                    f"Conflicting selection CSV horizons for {key}: "
                    f"{previous} vs {horizon}")
        for key, horizon in native_horizons.items():
            previous = selection_horizons.setdefault(key, horizon)
            if previous != horizon:
                raise ValueError(
                    f"Conflicting selection CSV horizons for {key}: "
                    f"{previous} vs {horizon}")
    if not any(selected_by_model.values()) and not any(full_native_by_model.values()):
        raise SystemExit(
            "No selected strategy windows were found; refusing to write an "
            "empty timing summary. Check --comparison-root and "
            "--strategy-subdirs."
        )

    horizons = _comparison_horizons(
        comparison_root,
        [model_short for _model_id, _family, model_short in models],
        args.strategy_subdirs,
    )
    for key, horizon in selection_horizons.items():
        previous = horizons.setdefault(key, horizon)
        if previous != horizon:
            raise ValueError(
                f"Comparison/selection horizon mismatch for {key}: "
                f"{previous} vs {horizon}")
    if args.summary_only:
        _write_summary(
            models, selected_by_model, full_native_by_model, horizons,
            args.cache_root, args.repeats, args.max_series)
        return
    ge_cache: Dict[Tuple[str, str], GiftEvalCache] = {}

    repeats, warmup = args.repeats, args.warmup
    print(Fore.CYAN
          + f"Robust timing: warmup={warmup}, repeats={repeats}, device={device}, "
            f"max_series={args.max_series}\n"
          + f"  cache-root={args.cache_root}\n"
          + f"  comparison-root={comparison_root}" + Fore.RESET)

    for model_id, model_family, model_short in models:
        sel = selected_by_model.get(model_short, {})
        native_sel = full_native_by_model.get(model_short, set())
        if not sel and not native_sel:
            continue
        print(Fore.CYAN + "\n" + "=" * 78 + Fore.RESET)
        print(Fore.CYAN + f"  MODEL: {model_id}  ({model_family})  "
              f"— {sum(len(w) for w in sel.values())} numeric + "
              f"{len(native_sel)} full-native cells" + Fore.RESET)
        print(Fore.CYAN + "=" * 78 + Fore.RESET)

        _handle = [None]

        def ensure_handle():
            if _handle[0] is None:
                _handle[0] = load_handle(model_family, model_id, device)
            return _handle[0]

        # Iterate DATASETS in catalog order for a stable, shard-safe index.
        for d_idx, (ge_name, term, dataset_display, to_univariate) in enumerate(DATASETS):
            key = (dataset_display, term)
            if key not in sel and key not in native_sel:
                continue
            if num_shards > 1 and d_idx % num_shards != shard_id:
                continue

            windows = sorted(sel.get(key, set()))
            wants_native = key in native_sel
            # Skip the whole dataset load if every selected window is already timed.
            pending = [w for w in windows
                       if args.force or not _timing_done(
                           dataset_display, model_short, term, w, repeats,
                           require_memory=(device == "cuda"),
                           timing_kind="numeric_window",
                           max_series=args.max_series)]
            native_pending = wants_native and (
                args.force or not _timing_done(
                    dataset_display, model_short, term, FULL_NATIVE, repeats,
                    require_memory=(device == "cuda"),
                    timing_kind="full_native", max_series=args.max_series))
            if not pending and not native_pending:
                print(Fore.WHITE
                      + f"  CACHED  {model_short} | {dataset_display} | t={term} "
                        f"({len(windows)} numeric"
                        + (" + full-native" if wants_native else "") + ")"
                      + Fore.RESET)
                continue

            if key not in ge_cache:
                print(Fore.CYAN + f"\n  Loading GiftEval: {ge_name}  term={term}" + Fore.RESET)
                ge_dataset = GiftEvalDataset(name=ge_name, term=term,
                                             to_univariate=to_univariate)
                ge_cache[key] = GiftEvalCache(ge_dataset, dataset_display)
                horizons[key] = ge_cache[key].horizon
            cache = ge_cache[key]
            horizon = cache.horizon

            for window_size in pending:
                tag = (f"{model_short} | {dataset_display} | t={term} | "
                       f"h={horizon} | w={window_size}")
                reason = _model_cap_skip(model_family, window_size, horizon, cache)
                if reason is not None:
                    raise RuntimeError(f"Required timing cell is invalid: {tag} ({reason})")
                try:
                    times, memory, n_timed = _measure_window(
                        cache, model_family, ensure_handle, model_id, window_size,
                        horizon, device, args.batch_size, warmup, repeats,
                        args.max_series)
                except RuntimeError as exc:
                    raise RuntimeError(f"Timing failed for {tag}") from exc
                payload = _summarize(
                    times, warmup, args.batch_size, n_timed, args.max_series,
                    "numeric_window", device, memory)
                _save_timing(dataset_display, model_short, term, window_size, payload)
                print(Fore.MAGENTA
                      + f"  TIMED   {tag}  mean={payload['mean_s']:.4f}s "
                        f"std={payload['std_s']:.4f}s "
                        f"min={payload['min_s']:.4f}s cv={payload['cv']:.3f}"
                        + (f" peak={payload['cuda_peak_allocated_gb']:.3f}GiB"
                           if 'cuda_peak_allocated_gb' in payload else "")
                      + Fore.RESET)

            if native_pending:
                tag = (f"{model_short} | {dataset_display} | t={term} | "
                       f"h={horizon} | w={FULL_NATIVE}")
                try:
                    times, memory, n_timed, cap = _measure_full_native(
                        cache, model_family, ensure_handle, model_id, horizon,
                        device, args.batch_size, warmup, repeats, args.max_series)
                except RuntimeError as exc:
                    raise RuntimeError(f"Timing failed for {tag}") from exc
                payload = _summarize(
                    times, warmup, args.batch_size, n_timed, args.max_series,
                    "full_native", device, memory)
                payload["context_cap"] = int(cap)
                _save_timing(
                    dataset_display, model_short, term, FULL_NATIVE, payload)
                print(Fore.MAGENTA
                      + f"  TIMED   {tag} cap={cap} n={n_timed} "
                        f"mean={payload['mean_s']:.4f}s "
                        f"std={payload['std_s']:.4f}s cv={payload['cv']:.3f}"
                      + Fore.RESET)

    # Only the single-process run or the coordinator's aggregation pass writes the
    # summary (workers would race on the same file).
    if shard_id is None:
        _write_summary(
            models, selected_by_model, full_native_by_model, horizons,
            args.cache_root, args.repeats, args.max_series)


def _run_coordinator(args, device: str, n_gpus: int, n_visible: int) -> None:
    """Spawn one worker per GPU (each owns a dataset shard), wait, then aggregate
    the summary in this process (all cells cached -> measurement is skipped)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    phys = visible.split(",") if visible else [str(j) for j in range(n_visible)]
    print(Fore.CYAN
          + f"Coordinator: sharding timing across {n_gpus} GPU(s) "
          + f"(physical {phys[:n_gpus]}), by dataset." + Fore.RESET)

    procs = []
    for i in range(n_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = phys[i]
        cmd = [sys.executable, "-m",
               "experiments.benchmark_window_timing_gifteval",
               *sys.argv[1:],
               "--shard-id", str(i), "--num-shards", str(n_gpus)]
        procs.append(subprocess.Popen(cmd, env=env))

    rcs = [p.wait() for p in procs]
    failed = [i for i, rc in enumerate(rcs) if rc != 0]
    if failed:
        raise SystemExit(
            Fore.RED + f"timing worker(s) on GPU index {failed} failed "
            + "(see output above); summary not written." + Fore.RESET)

    print(Fore.CYAN + "Coordinator: all shards done — aggregating summary." + Fore.RESET)
    run_timing(args, device, shard_id=None, num_shards=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Robust forward-pass timing for strategy-selected windows.")
    p.add_argument("--run-dir", type=str, default=None,
                   help="Ablation tree holding the per-cell datasets/ cache "
                        "(default: <ABLATION_ROOT>/general).")
    p.add_argument("--cache-root", type=str, default=None,
                   help="Alias for --run-dir (the cell cache root). One of "
                        "--run-dir / --cache-root must resolve.")
    p.add_argument("--comparison-root", type=str, default=None,
                   help="Root holding <display>/<strategy-subdir>/comparison.csv "
                        "(default: parent dir of the cache root).")
    p.add_argument(
        "--strategy-subdirs", nargs="+", default=DEFAULT_STRATEGY_SUBDIRS,
        help=("Strategy comparison variants whose selected windows are unioned "
              "for timing (default: strategy_comparison_v3 "
              "strategy_comparison_v4)."),
    )
    p.add_argument(
        "--selection-csvs", nargs="*", default=[],
        help=("Additional selected_window_histograms.csv files emitted by "
              "calibrated-risk evaluation. Their windows are unioned with the "
              "strategy-comparison windows."),
    )
    p.add_argument(
        "--selection-methods", nargs="*", default=[],
        help=("Optional method filter for --selection-csvs, for example "
              "balanced aggressive max_efficiency full_native."),
    )
    p.add_argument("--models", type=str, nargs="+", default=None,
                   help="Restrict to these model_short names (default: all known).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--max-series", type=int, default=64,
        help=("Maximum deterministically sampled series per timing cell. "
              "The summary reports measured per-series throughput (default 64)."),
    )
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    p.add_argument("--repeats", type=int, default=10,
                   help="Timed forward passes per cell (default 10).")
    p.add_argument("--warmup", type=int, default=3,
                   help="Discarded warm-up forward passes per cell (default 3).")
    p.add_argument("--force", action="store_true",
                   help="Re-time cells even when a complete timing.json exists.")
    p.add_argument(
        "--summary-only", action="store_true",
        help=("Only rebuild timing_summary.csv from existing timing.json files; "
              "does not load GiftEval or any forecasting model."),
    )
    p.add_argument("--num-gpus", type=int, default=0,
                   help="GPUs to shard across. 0 = auto (all visible); 1/CPU = "
                        "single process.")
    p.add_argument("--shard-id", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    args = p.parse_args()

    # Resolve the cell cache root from --run-dir / --cache-root, defaulting to the
    # shared general tree (wab.CACHE_ROOT is the ablation ROOT; cells live under
    # its general/ subtree).
    resolved = args.run_dir or args.cache_root
    if resolved is None:
        resolved = os.path.join(wab.CACHE_ROOT, "general")
    args.cache_root = resolved
    return args


def main() -> None:
    args = parse_args()

    # Summary reconstruction is inference-free. Keep it single-process even
    # when multiple GPUs remain visible after a sharded timing run; spawning
    # workers here would make them race on the same timing_summary.csv.
    if args.summary_only:
        torch.set_grad_enabled(False)
        run_timing(args, "cpu", shard_id=None, num_shards=1)
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    is_worker = args.shard_id is not None
    n_visible = torch.cuda.device_count() if device == "cuda" else 0
    n_gpus = n_visible if args.num_gpus == 0 else min(args.num_gpus, n_visible)

    if device == "cuda" and n_gpus > 1 and not is_worker:
        _run_coordinator(args, device, n_gpus, n_visible)
    else:
        run_timing(args, device,
                   shard_id=args.shard_id,
                   num_shards=(args.num_shards if is_worker else 1))


if __name__ == "__main__":
    main()
