"""
Attention-masking counterpart of the GiftEval window ablation
(``test_window_ablation_gifteval_v5``).

The v5 ablation shortens context by *slicing*: at window ``L`` it feeds each
GiftEval instance its last ``L`` genuine timesteps. This script produces the
*masking* curve instead: it feeds the SAME full-window input at every grid point
and attention-masks everything older than the last ``L`` timesteps
(``experiments.context_attention_mask``), so only the attention span shrinks while
normalization + positions stay over the full window. It then overlays the two
curves per (dataset, model, term) — the GiftEval analog of
``masking_vs_slicing.py`` — so you can see, on real data, whether the saturation
effect is purely attention span (curves coincide) or driven by the
normalization / positional change that slicing also induces.

Reuses v5 wholesale (``import ... as abl``): ``GiftEvalCache`` + batching, the
per-family ``load_handle`` / ``_forecast_cell`` / ``_merge_grouped`` dispatch, the
metric functions, seasonal-error caches, and the MODELS/DATASETS catalogs. Only
transformer families with maskable attention run (PatchTST-FM, Sundial, TimeMoE —
see ``context_attention_mask.SUPPORTED_FAMILIES``); everything else is skipped.

The **sliced "real" curve** is read from the v5 ablation cache when present
(``--ablation-cache-root``, no re-inference), and computed on the fly otherwise —
so a machine that has already run v5 gets the overlay almost for free.

Runs on the SERVER (TSFM + GPU). Multi-GPU dataset-sharding mirrors v5.

ENV: Sundial / TimeMoE need the legacy ``transformers==4.40.1`` env
(``TSFM_sundial_patch``) — the main env's newer transformers breaks their
``trust_remote_code`` mask build (``_prepare_4d_causal_attention_mask`` shape
error) even before any masking runs. Chronos-2 uses the main env; PatchTST-FM
uses the leaderboard-compatible ``predictcsl-patchtst`` env.

    python -m experiments.test_window_masking_gifteval --models PatchTST-FM-R1
    python -m experiments.test_window_masking_gifteval --models Chronos2-Small
    # legacy env for the decoders:
    python -m experiments.test_window_masking_gifteval --models Sundial-Base-128M TimeMoE-200M \
        --datasets ETTh1 --num-gpus 1

Output under ``logs/experiments/window_masking_gifteval/``:
    datasets/<ds>/<model>/t<term>/w<L>/metrics_masked.json   (per-cell, resumable)
    models/<model>/overlay/compare_<ds>_t<term>_<model>.png  (+ .npz per curve)
    results_masking.csv
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from colorama import Fore

import experiments.test_window_ablation_gifteval_v5 as abl
from experiments.build_context_length_dataset import WINDOW_GRID
from experiments.context_attention_mask import SUPPORTED_FAMILIES, context_attention_mask

GiftEvalDataset = abl.GiftEvalDataset
GiftEvalCache = abl.GiftEvalCache

MASK_CACHE_ROOT = "logs/experiments/window_masking_gifteval"
PATCHTST_FM_CTX = 8192          # PatchTST-FM's fixed native context
# Chronos-2 right-truncates context to its native context_length; the masked full
# window must stay <= this so the context-patch count is exact (no truncation).
# NOTE: verify against pipeline.model.config on the server (8192 is the assumption).
CHRONOS2_MAX_CONTEXT = 8192
# Moirai2-R-small: max_seq_len(512 tokens) x patch(16) = 8192 for context+horizon.
MOIRAI_MAX_TOTAL = 8192

# Curves we build + overlay. mase_gluonts_real needs gluonts; it degrades to NaN
# (v5's cell_mase_gluonts_real warns once) so it never sinks the run.
CURVE_METRICS = ["mae", "mase", "mase_gluonts", "mase_gluonts_real"]


# ==============================================================================
#  Full-window sizing (per family) + cache paths
# ==============================================================================

def family_full_window(family: str, horizon: int, window_sizes: List[int],
                       max_context: int) -> Optional[int]:
    """Largest grid window we can feed as the *full* input for this family.

    Bucketed down to a grid value so ``build_batches_padded`` groups line up. The
    per-family context caps mirror v5's skip guards (Sundial 2880; TimeMoE
    context+horizon <= 4096); PatchTST-FM is capped by its native 8192."""
    cap = min(max(window_sizes), int(max_context))
    if family == "sundial":
        cap = min(cap, abl.SUNDIAL_MAX_CONTEXT)
    elif family == "timemoe":
        cap = min(cap, abl.TIMEMOE_MAX_TOTAL - int(horizon))
    elif family == "moirai":
        cap = min(cap, MOIRAI_MAX_TOTAL - int(horizon))
    elif family == "patchtst_fm":
        cap = min(cap, PATCHTST_FM_CTX)
    elif family == "chronos2":
        cap = min(cap, CHRONOS2_MAX_CONTEXT)
    elif family == "toto":
        cap = min(cap, abl.TOTO_MAX_CONTEXT)
    # chronos_bolt / moirai: no v5 skip constant; rely on the model's own context
    # (their native context_length truncates internally). NOTE verify on server.
    grid = [w for w in sorted(set(window_sizes)) if w <= cap]
    return grid[-1] if grid else None


def _mask_cell_dir(dataset_display: str, model_short: str, term: str, L: int) -> str:
    return os.path.join(MASK_CACHE_ROOT, "datasets", dataset_display, model_short,
                        f"t{term}", f"w{L}")


def _masked_cached(dataset_display, model_short, term, L) -> Optional[dict]:
    path = os.path.join(_mask_cell_dir(dataset_display, model_short, term, L),
                        "metrics_masked.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            m = json.load(f)
        if m.get("mae") is None or (isinstance(m["mae"], float) and np.isnan(m["mae"])):
            return None
        return m
    except (json.JSONDecodeError, OSError):
        return None


def _save_masked(dataset_display, model_short, term, L, metrics: dict) -> None:
    d = _mask_cell_dir(dataset_display, model_short, term, L)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "metrics_masked.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def _load_sliced_metrics(ablation_root: str, dataset_display, model_short, term, L
                         ) -> Optional[dict]:
    """Read the v5 ablation cell (the sliced 'real' curve point) if it exists."""
    path = os.path.join(ablation_root, "datasets", dataset_display, model_short,
                        f"t{term}", f"w{L}", "metrics.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ==============================================================================
#  Forecast helpers (reuse v5 dispatch)
# ==============================================================================

def _cell_metrics(fr, tgts, cache, served_idx, se_cell) -> dict:
    m = abl.compute_all_metrics(fr, tgts, cache.naive_seasonal_mae_train,
                                seasonal_errors=se_cell)
    m["mase_gluonts_real"] = abl.cell_mase_gluonts_real(fr, cache, served_idx)
    return m


def masked_forecast(family, handle, model_id, groups_full, L, cache, horizon,
                    device, batch_size) -> dict:
    """Full-window forecast with attention restricted to the last ``L`` timesteps.

    ``groups_full`` are the padded width-groups at the full window (built once).
    For each group we mask down to ``min(L, group_width)`` and run the ordinary
    per-family forward; results are stitched back into instance order (0..n-1)."""
    results = []
    for W_g, batches_g, _ax, _ay, idx_g in groups_full:
        L_vis = min(int(L), int(W_g))
        with context_attention_mask(
                family, handle, L_vis, int(W_g), horizon=horizon):
            fr_g, tg_g = abl._forecast_cell(
                family, handle, model_id, batches_g, int(W_g), horizon,
                device, batch_size)
        results.append((idx_g, fr_g, tg_g))
    fr, tgts = abl._merge_grouped(results, cache.n_total, horizon, device)
    served = np.arange(cache.n_total)
    return _cell_metrics(fr, tgts, cache, served, cache.seasonal_errors_gluonts)


def sliced_forecast(family, handle, model_id, cache, L, window_sizes, horizon,
                    device, batch_size) -> dict:
    """Pad-mode sliced forecast (feed last min(L, ctx)) — the v5 curve, computed
    here when the ablation cache doesn't already have this cell."""
    groups = cache.build_batches_padded(
        L, batch_size, device, pin_memory=(device == "cuda"),
        window_grid=window_sizes)
    results = []
    for W_g, batches_g, _ax, _ay, idx_g in groups:
        fr_g, tg_g = abl._forecast_cell(
            family, handle, model_id, batches_g, int(W_g), horizon,
            device, batch_size)
        results.append((idx_g, fr_g, tg_g))
    fr, tgts = abl._merge_grouped(results, cache.n_total, horizon, device)
    served = np.arange(cache.n_total)
    return _cell_metrics(fr, tgts, cache, served, cache.seasonal_errors_gluonts)


# ==============================================================================
#  Overlay plot
# ==============================================================================

def plot_overlay(window_sizes, sliced_curve, masked_curve, metric, model_short,
                 dataset_display, term, horizon, full_window, save_dir) -> None:
    os.makedirs(save_dir, exist_ok=True)
    ws = np.asarray(window_sizes)
    s = np.asarray(sliced_curve, dtype=np.float64)
    m = np.asarray(masked_curve, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sv = ~np.isnan(s)
    mv = ~np.isnan(m)
    if sv.any():
        ax.plot(ws[sv], s[sv], "o-", lw=2.0, color="#2ca02c",
                label="sliced (feed last L)")
        ax.axvline(ws[int(np.nanargmin(s))], color="#2ca02c", ls=":", alpha=0.6,
                   label=f"sliced argmin = {ws[int(np.nanargmin(s))]}")
    if mv.any():
        ax.plot(ws[mv], m[mv], "s--", lw=2.0, color="#d62728",
                label="masked (full window, attend last L)")
        ax.axvline(ws[int(np.nanargmin(m))], color="#d62728", ls=":", alpha=0.6,
                   label=f"masked argmin = {ws[int(np.nanargmin(m))]}")

    ax.set_xscale("log", base=2)
    ax.set_xticks(ws)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("effective context L (timesteps)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{dataset_display} (term={term}, h={horizon}, full={full_window}) "
                 f"-- {model_short}  [{metric}]", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    path = os.path.join(save_dir,
                        f"compare_{dataset_display}_t{term}_{model_short}_{metric}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ==============================================================================
#  Driver
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Attention-masking analog of the GiftEval window ablation.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--models", type=str, nargs="+", default=None)
    p.add_argument("--datasets", type=str, nargs="+", default=None)
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    p.add_argument("--window-grid", type=int, nargs="+", default=None,
                   help="Override the ablation window grid (default: WINDOW_GRID).")
    p.add_argument("--ablation-cache-root", type=str, default=abl.CACHE_ROOT,
                   help="v5 ablation root to read the sliced 'real' curve from.")
    p.add_argument("--mask-cache-root", type=str, default=MASK_CACHE_ROOT)
    p.add_argument("--no-compute-sliced", action="store_true",
                   help="Only read the sliced curve from the ablation cache; leave "
                        "NaN when a cell is missing (don't re-infer it here).")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--test-datasets", type=int, default=None)
    p.add_argument("--test-datasets-seed", type=int, default=0)
    p.add_argument("--num-gpus", type=int, default=0)
    p.add_argument("--shard-id", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    return p.parse_args()


def _run_coordinator(args, n_gpus: int, n_visible: int) -> None:
    """One worker per GPU (dataset-sharded), then a single aggregation pass —
    mirrors v5's coordinator."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    phys = visible.split(",") if visible else [str(j) for j in range(n_visible)]
    print(Fore.CYAN + f"Coordinator: sharding masking run across {n_gpus} GPU(s) "
          f"(physical {phys[:n_gpus]}), by dataset." + Fore.RESET)
    procs = []
    for i in range(n_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = phys[i]
        cmd = [sys.executable, "-m", "experiments.test_window_masking_gifteval",
               *sys.argv[1:], "--shard-id", str(i), "--num-shards", str(n_gpus)]
        procs.append(subprocess.Popen(cmd, env=env))
    rcs = [p.wait() for p in procs]
    failed = [i for i, rc in enumerate(rcs) if rc != 0]
    if failed:
        raise SystemExit(Fore.RED + f"masking worker(s) on GPU index {failed} "
                         "failed; aggregation skipped." + Fore.RESET)
    print(Fore.CYAN + "Coordinator: all shards done — aggregating." + Fore.RESET)
    run_masking(args, "cuda", shard_id=None, num_shards=1)


def main():
    args = parse_args()
    global MASK_CACHE_ROOT
    MASK_CACHE_ROOT = args.mask_cache_root

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    is_worker = args.shard_id is not None
    n_visible = torch.cuda.device_count() if device == "cuda" else 0
    n_gpus = n_visible if args.num_gpus == 0 else min(args.num_gpus, n_visible)

    if device == "cuda" and n_gpus > 1 and not is_worker:
        _run_coordinator(args, n_gpus, n_visible)
    else:
        run_masking(args, device, shard_id=args.shard_id,
                    num_shards=(args.num_shards if is_worker else 1))


def run_masking(args, device: str, shard_id: Optional[int] = None,
                num_shards: int = 1) -> None:
    global MASK_CACHE_ROOT
    MASK_CACHE_ROOT = args.mask_cache_root

    window_sizes = sorted(set(args.window_grid or WINDOW_GRID))

    # Only maskable transformer families run.
    models = [m for m in abl.MODELS
              if (args.models is None or m[2] in args.models)
              and m[1] in SUPPORTED_FAMILIES]
    if not models:
        raise SystemExit(Fore.RED + "No maskable models selected. Supported "
                         f"families: {sorted(SUPPORTED_FAMILIES)}. "
                         f"(Requested: {args.models})" + Fore.RESET)
    datasets = [d for d in abl.DATASETS
                if (args.datasets is None or d[2] in args.datasets)]
    if args.test_datasets is not None and args.test_datasets < len(datasets):
        rng = random.Random(args.test_datasets_seed)
        datasets = sorted(rng.sample(datasets, args.test_datasets),
                          key=lambda d: abl.DATASETS.index(d))

    print(Fore.CYAN + f"Device: {device}  |  windows: {window_sizes}\n"
          f"Models (maskable): {[m[2] for m in models]}\n"
          f"Sliced curve source: {args.ablation_cache_root} "
          f"(compute missing: {not args.no_compute_sliced})" + Fore.RESET)

    ge_cache: Dict[Tuple[str, str], GiftEvalCache] = {}
    all_rows: List[dict] = []

    for model_id, family, model_short in models:
        print(Fore.CYAN + f"\n{'=' * 78}\n  MODEL: {model_id} ({family})\n{'=' * 78}"
              + Fore.RESET)
        _handle = [None]

        def ensure_handle():
            if _handle[0] is None:
                _handle[0] = abl.load_handle(family, model_id, device)
            return _handle[0]

        for d_idx, (ge_name, term, dataset_display, to_univariate) in enumerate(datasets):
            if num_shards > 1 and d_idx % num_shards != shard_id:
                continue
            ds_key = (ge_name, term)
            if ds_key not in ge_cache:
                try:
                    ge_dataset = GiftEvalDataset(name=ge_name, term=term,
                                                 to_univariate=to_univariate)
                    ge_cache[ds_key] = GiftEvalCache(ge_dataset, dataset_display)
                except Exception as exc:  # noqa: BLE001
                    print(Fore.RED + f"    SKIP {ge_name} t={term}: {exc}" + Fore.RESET)
                    continue
            cache = ge_cache[ds_key]
            horizon = cache.horizon

            full_window = family_full_window(
                family, horizon, window_sizes, cache.max_context)
            if full_window is None:
                print(Fore.RED + f"  SKIP {dataset_display} t={term}: no servable "
                      f"window (max_context={cache.max_context})." + Fore.RESET)
                continue

            # Build the full-window padded groups ONCE; every masked window reuses
            # them (only the attention mask changes across L).
            groups_full = cache.build_batches_padded(
                full_window, args.batch_size, device,
                pin_memory=(device == "cuda"), window_grid=window_sizes)

            sliced_curve = {k: np.full(len(window_sizes), np.nan) for k in CURVE_METRICS}
            masked_curve = {k: np.full(len(window_sizes), np.nan) for k in CURVE_METRICS}

            for j, L in enumerate(window_sizes):
                if L > full_window:
                    continue  # can't feed/mask beyond the full window for this family
                tag = f"{model_short} | {dataset_display} | t={term} | w={L}"

                # ---- masked cell (cache/resume) -----------------------------
                mm = _masked_cached(dataset_display, model_short, term, L)
                if mm is None:
                    print(Fore.YELLOW + f"  > MASK  {tag}" + Fore.RESET)
                    t0 = time.perf_counter()
                    mm = masked_forecast(
                        family, ensure_handle(), model_id, groups_full, L, cache,
                        horizon, device, args.batch_size)
                    mm["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
                    mm["horizon"] = horizon
                    mm["full_window"] = int(full_window)
                    _save_masked(dataset_display, model_short, term, L, mm)
                    print(Fore.MAGENTA + f"    masked MAE={mm['mae']:.6f} "
                          f"({mm['elapsed_seconds']}s)" + Fore.RESET)
                else:
                    print(Fore.WHITE + f"  CACHED MASK {tag} MAE={mm['mae']:.6f}"
                          + Fore.RESET)
                for k in CURVE_METRICS:
                    masked_curve[k][j] = mm.get(k, np.nan)

                # ---- sliced 'real' cell (read v5 cache; else compute) --------
                sm = _load_sliced_metrics(
                    args.ablation_cache_root, dataset_display, model_short, term, L)
                if sm is None and not args.no_compute_sliced:
                    sm = sliced_forecast(
                        family, ensure_handle(), model_id, cache, L, window_sizes,
                        horizon, device, args.batch_size)
                if sm is not None:
                    for k in CURVE_METRICS:
                        sliced_curve[k][j] = sm.get(k, np.nan)

                all_rows.append({
                    "model": model_id, "model_short": model_short,
                    "model_family": family, "dataset": ge_name,
                    "dataset_display": dataset_display, "term": term,
                    "horizon": horizon, "window_size": L, "full_window": full_window,
                    **{f"masked_{k}": masked_curve[k][j] for k in CURVE_METRICS},
                    **{f"sliced_{k}": sliced_curve[k][j] for k in CURVE_METRICS},
                })

            # ---- per (dataset, model, term) overlay + npz -------------------
            overlay_dir = os.path.join(MASK_CACHE_ROOT, "models", model_short, "overlay")
            os.makedirs(overlay_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(overlay_dir,
                             f"compare_{dataset_display}_t{term}_{model_short}.npz"),
                window_grid=np.asarray(window_sizes), full_window=full_window,
                horizon=horizon,
                **{f"masked_{k}": masked_curve[k] for k in CURVE_METRICS},
                **{f"sliced_{k}": sliced_curve[k] for k in CURVE_METRICS})
            if not args.no_plots:
                for k in CURVE_METRICS:
                    if np.all(np.isnan(masked_curve[k])):
                        continue
                    plot_overlay(window_sizes, sliced_curve[k], masked_curve[k], k,
                                 model_short, dataset_display, term, horizon,
                                 full_window, overlay_dir)

        _handle[0] = None
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if shard_id is not None:
        print(Fore.GREEN + f"Worker shard {shard_id}/{num_shards} done." + Fore.RESET)
        return

    if all_rows:
        csv_path = os.path.join(MASK_CACHE_ROOT, "results_masking.csv")
        os.makedirs(MASK_CACHE_ROOT, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        print(Fore.GREEN + f"\n  Masking results CSV: {csv_path}" + Fore.RESET)
    else:
        print(Fore.RED + "No results produced." + Fore.RESET)


if __name__ == "__main__":
    main()
