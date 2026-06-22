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

  * For each such ``(model, dataset, term, window)`` it pays ALL per-window setup
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

Output: a ``timing.json`` sidecar written into the SAME per-cell directory as the
ablation's ``metrics.json`` (``_cache_dir``), holding
``{n_warmup, n_repeats, samples_s, mean_s, std_s, min_s, median_s, cv}``. Because
the v3/v4 variant trees symlink their ``datasets/`` cell cache to the shared
``general/datasets/``, writing here means every variant's comparison sees the
robust timing for free — the TSFM forward pass is predictor-independent, so it is
measured once on the base ``general/`` tree. A run-level ``timing_summary.csv`` is
also emitted at the cache root.

Multi-GPU: identical dataset-level sharding to the v5 ablation — one worker per
visible CUDA device drains a round-robin slice of datasets, the coordinator then
aggregates the summary CSV. Re-runs are safe: a cell whose ``timing.json`` already
has ``n_repeats >= --repeats`` is skipped unless ``--force``.

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
from typing import Dict, List, Optional, Set, Tuple

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

STRATEGY_SUBDIR = "strategy_comparison"


def _sync(device: str) -> None:
    """Drain the CUDA queue so the timer brackets completed work (no-op on CPU)."""
    if device == "cuda":
        torch.cuda.synchronize()


# ==============================================================================
#  SELECTED-WINDOW DISCOVERY  (from the strategy comparison)
# ==============================================================================

def _selected_windows(comparison_root: str, display: str
                      ) -> Dict[Tuple[str, str], Set[int]]:
    """Read <comparison_root>/<display>/strategy_comparison/comparison.csv and
    return {(dataset_display, term): {on-grid windows chosen by any strategy}}.

    Collects every ``*_window`` column except the period ones (period is
    per-series / off-grid and not a single grid window). Empty dict when the
    comparison has not been produced for this model yet.
    """
    csv_path = os.path.join(comparison_root, display, STRATEGY_SUBDIR, "comparison.csv")
    if not os.path.isfile(csv_path):
        print(Fore.YELLOW
              + f"  [{display}] no comparison.csv at {csv_path} — run stage 4 first; "
                "nothing to time." + Fore.RESET)
        return {}
    df = pd.read_csv(csv_path)
    win_cols = [c for c in df.columns
                if c.endswith("_window") and not c.startswith("period")]
    out: Dict[Tuple[str, str], Set[int]] = {}
    for _, row in df.iterrows():
        key = (str(row["dataset_display"]), str(row["term"]))
        wins: Set[int] = set()
        for c in win_cols:
            v = row[c]
            if pd.notna(v):
                wins.add(int(round(float(v))))
        if wins:
            out.setdefault(key, set()).update(wins)
    return out


# ==============================================================================
#  TIMING SIDECAR I/O
# ==============================================================================

def _timing_path(dataset_display: str, model_short: str, term: str, window_size: int) -> str:
    return os.path.join(
        _cache_dir(dataset_display, model_short, term, window_size), "timing.json")


def _timing_done(dataset_display: str, model_short: str, term: str,
                 window_size: int, repeats: int) -> bool:
    p = _timing_path(dataset_display, model_short, term, window_size)
    if not os.path.isfile(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        return int(d.get("n_repeats", 0)) >= repeats
    except (OSError, json.JSONDecodeError, ValueError):
        return False


def _save_timing(dataset_display: str, model_short: str, term: str,
                 window_size: int, payload: dict) -> None:
    d = _cache_dir(dataset_display, model_short, term, window_size)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "timing.json"), "w") as f:
        json.dump(payload, f, indent=2)


def _summarize(times: List[float], warmup: int) -> dict:
    arr = np.asarray(times, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {
        "n_warmup":  int(warmup),
        "n_repeats": int(arr.size),
        "samples_s": [round(t, 6) for t in times],
        "mean_s":    round(mean, 6),
        "std_s":     round(std, 6),
        "min_s":     round(float(arr.min()), 6),
        "median_s":  round(float(np.median(arr)), 6),
        "cv":        round(std / mean, 6) if mean > 0 else float("nan"),
    }


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


def _measure_window(cache: GiftEvalCache, model_family: str, ensure_handle,
                    model_id: str, window_size: int, horizon: int, device: str,
                    batch_size: int, warmup: int, repeats: int) -> List[float]:
    """Time ``repeats`` PURE forward passes (after ``warmup`` discarded ones) for
    one window, each GPU-synced.

    All per-window setup is paid ONCE outside the timed loop: batch building plus,
    via ``build_forward``, the weight load + compile (timesfm) or forecaster build
    (moirai) that ``_forecast_cell`` would otherwise re-run on every call. The
    timed unit is therefore the inference itself, not the reload/compile — which is
    the cost the strategy comparison's FLOPs axis represents. The model built for
    this window is freed afterwards via the teardown to avoid GPU accumulation."""
    batches, _ax, _ay, _idx = cache.build_batches(
        window_size, batch_size, device, pin_memory=True)

    forward, teardown = build_forward(
        model_family, ensure_handle(), model_id, batches, window_size, horizon,
        device, batch_size)
    try:
        for _ in range(warmup):
            forward()
        _sync(device)

        times: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            forward()
            _sync(device)
            times.append(time.perf_counter() - t0)
    finally:
        teardown()
    return times


# ==============================================================================
#  RUN-LEVEL SUMMARY
# ==============================================================================

def _write_summary(models, selected_by_model: Dict[str, Dict[Tuple[str, str], Set[int]]],
                   horizons: Dict[Tuple[str, str], int], cache_root: str) -> None:
    """Scan every selected cell's timing.json and write timing_summary.csv at the
    cache root. Pure file-read aggregation (no GPU)."""
    rows = []
    for _model_id, _family, model_short in models:
        sel = selected_by_model.get(model_short, {})
        for (dataset_display, term), windows in sorted(sel.items()):
            for window_size in sorted(windows):
                p = _timing_path(dataset_display, model_short, term, window_size)
                if not os.path.isfile(p):
                    continue
                try:
                    with open(p) as f:
                        d = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                rows.append({
                    "model_short":     model_short,
                    "dataset_display": dataset_display,
                    "term":            term,
                    "horizon":         horizons.get((dataset_display, term), -1),
                    "window_size":     window_size,
                    "n_warmup":        d.get("n_warmup"),
                    "n_repeats":       d.get("n_repeats"),
                    "mean_s":          d.get("mean_s"),
                    "std_s":           d.get("std_s"),
                    "min_s":           d.get("min_s"),
                    "median_s":        d.get("median_s"),
                    "cv":              d.get("cv"),
                })
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
        m[2]: _selected_windows(comparison_root, m[2]) for m in models}

    horizons: Dict[Tuple[str, str], int] = {}
    ge_cache: Dict[Tuple[str, str], GiftEvalCache] = {}

    repeats, warmup = args.repeats, args.warmup
    print(Fore.CYAN
          + f"Robust timing: warmup={warmup}, repeats={repeats}, device={device}\n"
          + f"  cache-root={args.cache_root}\n"
          + f"  comparison-root={comparison_root}" + Fore.RESET)

    for model_id, model_family, model_short in models:
        sel = selected_by_model.get(model_short, {})
        if not sel:
            continue
        print(Fore.CYAN + "\n" + "=" * 78 + Fore.RESET)
        print(Fore.CYAN + f"  MODEL: {model_id}  ({model_family})  "
              f"— {sum(len(w) for w in sel.values())} cells" + Fore.RESET)
        print(Fore.CYAN + "=" * 78 + Fore.RESET)

        _handle = [None]

        def ensure_handle():
            if _handle[0] is None:
                _handle[0] = load_handle(model_family, model_id, device)
            return _handle[0]

        # Iterate DATASETS in catalog order for a stable, shard-safe index.
        for d_idx, (ge_name, term, dataset_display, to_univariate) in enumerate(DATASETS):
            key = (dataset_display, term)
            if key not in sel:
                continue
            if num_shards > 1 and d_idx % num_shards != shard_id:
                continue

            windows = sorted(sel[key])
            # Skip the whole dataset load if every selected window is already timed.
            pending = [w for w in windows
                       if args.force or not _timing_done(
                           dataset_display, model_short, term, w, repeats)]
            if not pending:
                print(Fore.WHITE
                      + f"  CACHED  {model_short} | {dataset_display} | t={term} "
                        f"({len(windows)} windows)" + Fore.RESET)
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
                    print(Fore.RED + f"  SKIP    {tag}  ({reason})" + Fore.RESET)
                    continue
                try:
                    times = _measure_window(
                        cache, model_family, ensure_handle, model_id, window_size,
                        horizon, device, args.batch_size, warmup, repeats)
                except RuntimeError as exc:
                    print(Fore.RED + f"  SKIP    {tag}  ({exc})" + Fore.RESET)
                    continue
                payload = _summarize(times, warmup)
                _save_timing(dataset_display, model_short, term, window_size, payload)
                print(Fore.MAGENTA
                      + f"  TIMED   {tag}  mean={payload['mean_s']:.4f}s "
                        f"std={payload['std_s']:.4f}s "
                        f"min={payload['min_s']:.4f}s cv={payload['cv']:.3f}"
                      + Fore.RESET)

    # Only the single-process run or the coordinator's aggregation pass writes the
    # summary (workers would race on the same file).
    if shard_id is None:
        _write_summary(models, selected_by_model, horizons, args.cache_root)


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
                   help="Root holding <display>/strategy_comparison/comparison.csv "
                        "(default: parent dir of the cache root).")
    p.add_argument("--models", type=str, nargs="+", default=None,
                   help="Restrict to these model_short names (default: all known).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    p.add_argument("--repeats", type=int, default=10,
                   help="Timed forward passes per cell (default 10).")
    p.add_argument("--warmup", type=int, default=3,
                   help="Discarded warm-up forward passes per cell (default 3).")
    p.add_argument("--force", action="store_true",
                   help="Re-time cells even when a complete timing.json exists.")
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
