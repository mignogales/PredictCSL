"""
Synthetic context-length ablation dataset builder.

Purpose
-------
Labeling stage for a zero-shot "useful context length" predictor. The idea:
when forecasting with a TS foundation model, more context is not always
better -- stale history from an old regime can actively hurt. We want a model
that, given a series, predicts how much of the tail context is actually useful.

There is no generative parameter for "useful context length": it is an
*operational* quantity, defined as the input window that minimizes a given
TSFM's forecast error. So we MEASURE it -- this script is the measurement.

Pipeline
--------
1. Generate N synthetic series of length MAX_WINDOW + max(HORIZON_GRID). Each
   series is built segment-by-segment with injected non-stationarity (regime
   changes, level shifts, variance shifts, per-segment trend/AR/seasonality).
   This is what makes context length matter: without change points every
   ablation curve is monotone and there is nothing to learn.
2. For each series, forecast max(HORIZON_GRID) steps from the last W context
   samples, for every W in WINDOW_GRID, using a chosen TSFM (Chronos-2 first).
   Per-window inference is done once at the longest horizon; the prefix
   property of left-to-right forecasting lets us slice the prediction at every
   h in HORIZON_GRID and get the same answer as if we'd called the model with
   prediction_length=h directly.
3. Save the full per-series error-vs-(context, horizon) surface (MAE and MSE
   for every (W, h) pair). Horizon is a first-class axis because the useful
   context length depends on how far ahead you're forecasting.

The downstream predictor (experiments/predict_context_length.py) consumes
`contexts.npy` (its input) and `curves_mae.npy` / `curves_mse.npy` (its label,
shape (N, n_windows, n_horizons)).

Parallelism
-----------
Series generation runs once in the parent (CPU-bound, seconds). The expensive
labeling stage is sharded across GPUs: one worker process per CUDA device
drains a shared queue of series-shards, loading the TSFM into VRAM exactly
once. Each completed shard is written to disk; the parent merges shards into
the top-level curve arrays. Runs are resumable per shard via --resume.

Outputs
-------
logs/experiments/context_length_dataset/<run>/
    contexts.npy      (N, MAX_WINDOW)            float32  -- predictor input
    targets.npy       (N, MAX_HORIZON)           float32  -- forecast target
    n_segments.npy    (N,)                       int32    -- regime count per series
    curves_mae.npy    (N, n_windows, n_horizons) float32  -- MAE per (W, h)
    curves_mse.npy    (N, n_windows, n_horizons) float32  -- MSE per (W, h)
    shards/shard_NNN/                            -- per-shard curves (resume unit)
    meta.json                                    -- grid, horizon grid, model, seeds
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from queue import Empty
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from colorama import Fore
from scipy.signal import lfilter
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()


# ==============================================================================
#  CONFIGURATION
# ==============================================================================

# -- Series geometry ----------------------------------------------------------
MAX_WINDOW   = 8192          # length of the context pool (largest ablation window)
PERIOD_MIN   = 16            # seasonal period bounds (samples)
PERIOD_MAX   = 2048

# -- Ablation grid (suffix lengths of the context pool) -----------------------
# Identical to test_window_ablation_gifteval_v4.py so the predictor's output
# is directly comparable to real GiftEval ablation curves.
WINDOW_GRID = [
    32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
    1024, 1536, 2048, 2560, 3072, 4096, 6144, 8192,
]

# -- Horizon grid (forecast lengths labeled per series) -----------------------
# Each series gets one curve per horizon. The useful context length depends on
# horizon, so this is a first-class label axis. We forecast once at the longest
# horizon and slice — left-to-right autoregressive output of the first h steps
# is identical regardless of the total prediction_length requested.
HORIZON_GRID = [8, 16, 32, 64, 96]
MAX_HORIZON  = max(HORIZON_GRID)

# -- Dataset size -------------------------------------------------------------
N_SERIES   = 10_000          # scale up for the real run; lower with --n-series
SEED       = 42
BATCH_SIZE = 32              # TSFM inference batch size
SHARD_SIZE = 500             # series per shard (parallelism + resume granularity)

# -- Non-stationarity controls ------------------------------------------------
# Number of regimes per series. Biased toward 1-2 so the dataset contains both
# stationary series (curve monotone -> label = max window) and series where
# truncation helps (curve has an interior optimum).
N_SEGMENT_CHOICES = [1, 1, 1, 2, 2, 3]
MIN_SEGMENT_LEN   = 512      # shortest regime; also keeps the horizon clear of cuts
LEVEL_SHIFT_STD   = 1.5      # std of the cumulative mean jump at each regime change
SEG_AMP_RANGE     = (0.6, 1.6)   # per-segment amplitude scale (variance regimes)

# -- Multi-GPU ----------------------------------------------------------------
DEVICES = None               # None -> all visible CUDA devices (or CPU)

# -- Models. Only the first uncommented entry is used. ------------------------
#    (model_id, family, display_name)
MODELS = [
    ("autogluon/chronos-2-small",       "chronos2",     "Chronos2-Small"),
    # ("amazon/chronos-bolt-small",       "chronos_bolt", "ChronosBolt-Small"),
    # ("Salesforce/moirai-2.0-R-small",   "moirai",       "Moirai2-Small"),
    # ("google/timesfm-2.5-200m-pytorch", "timesfm",      "TimesFM2.5-200M"),
    # ("ibm-research/patchtst-fm-r1",     "patchtst_fm",  "PatchTST-FM-R1"),
]

# -- Quantile bookkeeping (per model family) ----------------------------------
MOIRAI2_MEDIAN_IDX            = 4
PATCHTST_FM_MEDIAN_QUANTILE_IDX = 49

OUTPUT_ROOT = "logs/experiments/context_length_dataset"


# ==============================================================================
#  DEVICE RESOLUTION
# ==============================================================================

def resolve_devices(force: Optional[str]) -> List[str]:
    if force == "cpu":
        return ["cpu"]
    if not torch.cuda.is_available():
        return ["cpu"]
    if DEVICES is not None:
        return list(DEVICES)
    n = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(n)] if n > 0 else ["cpu"]


# ==============================================================================
#  SYNTHETIC GENERATOR
# ==============================================================================

def _generate_segment(rng: np.random.RandomState, length: int) -> np.ndarray:
    """One stationary regime: periodic components + optional AR + trend + noise.

    Not standardized here -- amplitude scaling and level shifts are applied at
    the series level so regimes differ in scale and mean.
    """
    t = np.arange(length, dtype=np.float32)
    seg = np.zeros(length, dtype=np.float32)

    # -- Periodic components --------------------------------------------------
    n_periodic = int(rng.randint(1, 4))
    log_lo, log_hi = math.log(PERIOD_MIN), math.log(PERIOD_MAX)
    periods = np.exp(rng.uniform(log_lo, log_hi, size=n_periodic))
    amplitudes = rng.uniform(0.5, 2.0, size=n_periodic)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_periodic)
    for amp, T_p, ph in zip(amplitudes, periods, phases):
        omega = 2.0 * np.pi / float(T_p)
        if rng.uniform() < 0.5:
            seg += amp * np.sin(omega * t + ph)
        else:
            seg += amp * np.cos(omega * t + ph)

    # -- Optional AR(p) (stable; lfilter for speed) ---------------------------
    if rng.uniform() < 0.5:
        p = int(rng.randint(1, 4))
        coeffs = rng.uniform(-0.3 / p, 0.3 / p, size=p)
        innov = rng.normal(0.0, 0.3, size=length).astype(np.float32)
        # x_t = sum_k coeffs_k x_{t-k} + innov  ->  a = [1, -c1, ..., -cp]
        a = np.concatenate([[1.0], -coeffs])
        seg += lfilter([1.0], a, innov).astype(np.float32)

    # -- Optional polynomial trend --------------------------------------------
    if rng.uniform() < 0.5:
        deg = int(rng.randint(1, 4))
        t_norm = (t / float(max(length, 1))).astype(np.float32)
        poly_coeffs = rng.uniform(-1.0, 1.0, size=deg + 1) * 0.5
        seg += np.polyval(poly_coeffs[::-1], t_norm).astype(np.float32)

    # -- Gaussian noise -------------------------------------------------------
    noise_std = float(rng.uniform(0.05, 0.30))
    seg += rng.normal(0.0, noise_std, size=length).astype(np.float32)

    return seg.astype(np.float32, copy=False)


def _sample_cut_points(
    rng: np.random.RandomState, n_cuts: int, lo: int, hi: int, min_gap: int
) -> List[int]:
    """n_cuts sorted change points in [lo, hi] with pairwise spacing >= min_gap."""
    for _ in range(200):
        cuts = sorted(int(c) for c in rng.randint(lo, hi + 1, size=n_cuts))
        ok = all(cuts[i + 1] - cuts[i] >= min_gap for i in range(len(cuts) - 1))
        if ok:
            return cuts
    # Fallback: evenly spaced.
    step = (hi - lo) / (n_cuts + 1)
    return [int(lo + step * (i + 1)) for i in range(n_cuts)]


def _generate_synthetic_series(
    rng: np.random.RandomState, total_length: int
) -> Tuple[np.ndarray, int]:
    """Build one synthetic series with injected non-stationarity.

    Composed of 1-3 regimes. Change points fall strictly inside the context
    pool (the final HORIZON samples stay within the last regime, with a buffer)
    so a small window can isolate the most recent regime while a large window
    drags in stale history.

    Returns:
        series: float32 (total_length,), standardized by context-pool stats.
        n_segments: number of regimes.
    """
    n_seg = int(rng.choice(N_SEGMENT_CHOICES))

    if n_seg == 1:
        series = _generate_segment(rng, total_length)
    else:
        # Cuts inside the pool; last regime must hold HORIZON + a buffer.
        lo = MIN_SEGMENT_LEN
        hi = MAX_WINDOW - MIN_SEGMENT_LEN
        cuts = _sample_cut_points(rng, n_seg - 1, lo, hi, MIN_SEGMENT_LEN)
        bounds = [0] + cuts + [total_length]

        parts: List[np.ndarray] = []
        level = 0.0
        for i in range(n_seg):
            seg_len = bounds[i + 1] - bounds[i]
            seg = _generate_segment(rng, seg_len)
            seg = seg * float(rng.uniform(*SEG_AMP_RANGE))
            if i > 0:
                level += float(rng.normal(0.0, LEVEL_SHIFT_STD))
            parts.append(seg + level)
        series = np.concatenate(parts).astype(np.float32)

    # Standardize using ONLY the observed context pool (no horizon leakage).
    ctx = series[:MAX_WINDOW]
    mu, sigma = float(ctx.mean()), float(ctx.std())
    series = (series - mu) / (sigma + 1e-8)
    return series.astype(np.float32, copy=False), n_seg


def generate_dataset(
    n_series: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize the synthetic pool (parent process, serial).

    Returns:
        contexts:   (n_series, MAX_WINDOW)   float32  -- predictor input
        targets:    (n_series, MAX_HORIZON)  float32  -- forecast target
        n_segments: (n_series,)              int32
    """
    total_length = MAX_WINDOW + MAX_HORIZON
    contexts = np.empty((n_series, MAX_WINDOW), dtype=np.float32)
    targets = np.empty((n_series, MAX_HORIZON), dtype=np.float32)
    n_segments = np.empty((n_series,), dtype=np.int32)

    for i in tqdm(range(n_series), desc="  Generating series", leave=False):
        rng = np.random.RandomState(seed + i)
        series, n_seg = _generate_synthetic_series(rng, total_length)
        contexts[i] = series[:MAX_WINDOW]
        targets[i] = series[MAX_WINDOW:MAX_WINDOW + MAX_HORIZON]
        n_segments[i] = n_seg

    return contexts, targets, n_segments


# ==============================================================================
#  MODEL LOADERS + MEDIAN-FORECAST WRAPPERS
# ==============================================================================
#  Each predict_* wrapper takes a CPU tensor x of shape (B, W, 1) and returns
#  the point/median forecast as (B, horizon) on `device`, where `horizon` is
#  the value passed in (we call these at MAX_HORIZON and slice).

def _is_cuda(device: str) -> bool:
    return device.startswith("cuda")


def load_chronos2(model_id: str, device: str):
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(
        model_id, device_map=device,
        torch_dtype=torch.bfloat16 if _is_cuda(device) else torch.float32,
    )


def predict_chronos2(pipeline, x: torch.Tensor, horizon: int, device: str) -> torch.Tensor:
    context = x.permute(0, 2, 1)                       # (B, 1, W)
    samples = pipeline.predict(inputs=context, prediction_length=horizon)
    if isinstance(samples, list):
        samples = torch.stack(samples, dim=0).squeeze(1)
    if samples.dim() == 4:
        samples = samples.squeeze(2)                   # (B, Q, H)
    samples = samples.to(device=device, dtype=torch.float32)
    return torch.median(samples, dim=1).values         # (B, H)


def load_chronos_bolt(model_id: str, device: str):
    from chronos import ChronosBoltPipeline
    return ChronosBoltPipeline.from_pretrained(
        model_id, device_map=device,
        torch_dtype=torch.bfloat16 if _is_cuda(device) else torch.float32,
    )


def predict_chronos_bolt(pipeline, x: torch.Tensor, horizon: int, device: str) -> torch.Tensor:
    samples = pipeline.predict(inputs=x[:, :, 0], prediction_length=horizon)
    samples = samples.to(device=device, dtype=torch.float32)
    return torch.median(samples, dim=1).values


def load_moirai_module(model_id: str):
    from uni2ts.model.moirai2 import Moirai2Module
    return Moirai2Module.from_pretrained(model_id)


def _build_moirai(module, horizon: int, window: int, device: str):
    from uni2ts.model.moirai2 import Moirai2Forecast
    return Moirai2Forecast(
        module=module, prediction_length=horizon, context_length=window,
        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
    ).to(device)


def predict_moirai(model, x: torch.Tensor, horizon: int, device: str) -> torch.Tensor:
    x_cpu = x[:, :, 0].numpy()
    context_list = [x_cpu[i] for i in range(x_cpu.shape[0])]
    forecast = model.predict(past_target=context_list)
    forecast_t = torch.as_tensor(
        forecast[:, :, :horizon], dtype=torch.float32, device=device)
    return forecast_t[:, MOIRAI2_MEDIAN_IDX, :]


def load_timesfm(model_id: str, window: int, horizon: int, batch_size: int):
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
    model.compile(
        timesfm.ForecastConfig(
            max_context=window, max_horizon=horizon,
            normalize_inputs=True, use_continuous_quantile_head=True,
            force_flip_invariance=True, per_core_batch_size=batch_size,
            infer_is_positive=True, fix_quantile_crossing=True,
        )
    )
    return model


def predict_timesfm(model, x: torch.Tensor, horizon: int, device: str) -> torch.Tensor:
    PATCH = 32
    bs, ws, _ = x.shape
    x_np = x[:, :, 0].numpy()
    pad_len = (PATCH - ws % PATCH) % PATCH
    if pad_len > 0:
        padding = np.zeros((bs, pad_len), dtype=np.float32)
        padded = np.concatenate([padding, x_np], axis=1)
        masks_np = np.concatenate(
            [np.ones((bs, pad_len), bool), np.zeros((bs, ws), bool)], axis=1)
    else:
        padded, masks_np = x_np, np.zeros((bs, ws), bool)
    values = [padded[i] for i in range(bs)]
    masks = [masks_np[i] for i in range(bs)]
    point_forecast, _ = model.compiled_decode(horizon, values, masks)
    return torch.as_tensor(
        point_forecast[:, :horizon], dtype=torch.float32, device=device)


def load_patchtst_fm(model_id: str, device: str):
    from tsfm_public import PatchTSTFMForPrediction
    model = PatchTSTFMForPrediction.from_pretrained(model_id, device_map=device)
    model.eval()
    return model


def predict_patchtst_fm(model, x: torch.Tensor, horizon: int, device: str) -> torch.Tensor:
    past_values = x.to(device, non_blocking=True).squeeze(-1)
    raw = model(inputs=past_values, prediction_length=horizon)[0]
    if raw.dim() == 4:                                 # (B, Q, H, 1)
        qf = raw[:, :, :horizon, 0]
        return qf[:, PATCHTST_FM_MEDIAN_QUANTILE_IDX, :].to(torch.float32)
    if raw.dim() == 3:                                 # (B, Q, H) or (B, 1, H)
        if raw.shape[1] == 1:
            return raw[:, 0, :horizon].to(torch.float32)
        return torch.median(raw[:, :, :horizon], dim=1).values.to(torch.float32)
    return raw[:, :horizon].to(torch.float32)          # (B, H)


# ==============================================================================
#  ABLATION DRIVER
# ==============================================================================

def setup_model(family: str, model_id: str, device: str):
    """Load whatever persists across windows. Per-window objects built later."""
    if family == "chronos2":
        return load_chronos2(model_id, device)
    if family == "chronos_bolt":
        return load_chronos_bolt(model_id, device)
    if family == "moirai":
        return load_moirai_module(model_id)            # per-window forecast obj
    if family == "timesfm":
        return None                                    # recompiled per window
    if family == "patchtst_fm":
        return load_patchtst_fm(model_id, device)
    raise ValueError(f"Unknown model family: {family}")


def forecast_window(
    family: str,
    base,
    model_id: str,
    contexts: np.ndarray,
    window: int,
    horizon: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Forecast the horizon for every series, using the last `window` samples.

    Returns the median forecast (N, horizon) on `device`.
    """
    x_all = torch.from_numpy(
        np.ascontiguousarray(contexts[:, -window:])).unsqueeze(-1)  # (N, W, 1)
    n = x_all.shape[0]

    # Per-window model objects.
    if family == "moirai":
        runner = _build_moirai(base, horizon, window, device)
    elif family == "timesfm":
        runner = load_timesfm(model_id, window, horizon, batch_size)
    else:
        runner = base

    medians: List[torch.Tensor] = []
    for start in range(0, n, batch_size):
        xb = x_all[start:start + batch_size]
        if family == "chronos2":
            m = predict_chronos2(runner, xb, horizon, device)
        elif family == "chronos_bolt":
            m = predict_chronos_bolt(runner, xb, horizon, device)
        elif family == "moirai":
            m = predict_moirai(runner, xb, horizon, device)
        elif family == "timesfm":
            m = predict_timesfm(runner, xb, horizon, device)
        elif family == "patchtst_fm":
            m = predict_patchtst_fm(runner, xb, horizon, device)
        else:
            raise ValueError(f"Unknown model family: {family}")
        medians.append(m)

    if family in ("moirai", "timesfm"):
        del runner
        if _is_cuda(device):
            torch.cuda.empty_cache()

    return torch.cat(medians, dim=0)                   # (N, H)


# ==============================================================================
#  GPU WORKER  (one process per device, drains a shard queue)
# ==============================================================================

def _shard_dir(run_dir: str, shard_id: int) -> str:
    return os.path.join(run_dir, "shards", f"shard_{shard_id:03d}")


def gpu_worker(
    worker_id: int,
    device: str,
    shard_queue: "mp.Queue",
    result_queue: "mp.Queue",
    run_dir: str,
    contexts_path: str,
    targets_path: str,
    model_id: str,
    family: str,
    max_horizon: int,
    batch_size: int,
    win_indices: List[int],
) -> None:
    """Load the TSFM once, then label series-shards until the queue drains.

    Each window is forecast once at ``max_horizon``; per-(window, horizon)
    errors are computed by slicing the prediction at every h in HORIZON_GRID.
    """
    if _is_cuda(device):
        torch.cuda.set_device(torch.device(device))
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    n_win = len(WINDOW_GRID)
    n_h = len(HORIZON_GRID)

    try:
        base = setup_model(family, model_id, device)
        print(Fore.CYAN + f"  [{device}] worker {worker_id} ready." + Fore.RESET)
    except Exception as exc:
        print(Fore.RED + f"  [{device}] worker {worker_id} model load failed: "
              + f"{type(exc).__name__}: {exc}" + Fore.RESET)
        while True:
            spec = shard_queue.get()
            if spec is None:
                break
            result_queue.put({"shard_id": spec[0], "status": "model_load_failed"})
        return

    while True:
        spec = shard_queue.get()
        if spec is None:
            break
        shard_id, start, end = spec
        t0 = time.perf_counter()
        try:
            ctx = np.ascontiguousarray(
                np.load(contexts_path, mmap_mode="r")[start:end])
            tgt = np.ascontiguousarray(
                np.load(targets_path, mmap_mode="r")[start:end])
            tgt_t = torch.from_numpy(tgt).to(device)        # (B, MAX_HORIZON)

            cm = np.full((end - start, n_win, n_h), np.nan, dtype=np.float32)
            cs = np.full((end - start, n_win, n_h), np.nan, dtype=np.float32)
            for w_idx in win_indices:
                w = WINDOW_GRID[w_idx]
                medians = forecast_window(
                    family, base, model_id, ctx, w, max_horizon,
                    batch_size, device)                     # (B, MAX_HORIZON)
                for h_idx, h in enumerate(HORIZON_GRID):
                    err = medians[:, :h] - tgt_t[:, :h]
                    cm[:, w_idx, h_idx] = err.abs().mean(dim=1).cpu().numpy()
                    cs[:, w_idx, h_idx] = err.pow(2).mean(dim=1).cpu().numpy()

            sd = _shard_dir(run_dir, shard_id)
            os.makedirs(sd, exist_ok=True)
            np.save(os.path.join(sd, "curves_mae.npy"), cm)
            np.save(os.path.join(sd, "curves_mse.npy"), cs)
            with open(os.path.join(sd, "done.json"), "w") as f:
                json.dump({"shard_id": shard_id, "start": start, "end": end,
                           "window_indices": win_indices,
                           "horizon_grid": HORIZON_GRID}, f)

            elapsed = time.perf_counter() - t0
            result_queue.put({"shard_id": shard_id, "status": "ok",
                              "elapsed": elapsed})
            print(Fore.YELLOW + f"  [{device}] shard {shard_id:03d} "
                  + f"[{start}:{end}] done ({elapsed:.1f}s)" + Fore.RESET)
        except Exception as exc:
            print(Fore.RED + f"  [{device}] shard {shard_id:03d} FAILED: "
                  + f"{type(exc).__name__}: {exc}" + Fore.RESET)
            result_queue.put({"shard_id": shard_id,
                              "status": f"error:{type(exc).__name__}"})
        if _is_cuda(device):
            torch.cuda.empty_cache()

    print(Fore.CYAN + f"  [{device}] worker {worker_id} exited." + Fore.RESET)


# ==============================================================================
#  SHARD MERGE + SANITY
# ==============================================================================

def merge_shards(run_dir: str, n_series: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Concatenate completed shard curves into top-level (N, n_win, n_h) arrays.

    Missing/incomplete shards leave NaN rows so curves stay index-aligned with
    contexts.npy. Returns (curves_mae, curves_mse, n_shards_done).
    """
    n_win = len(WINDOW_GRID)
    n_h = len(HORIZON_GRID)
    cm = np.full((n_series, n_win, n_h), np.nan, dtype=np.float32)
    cs = np.full((n_series, n_win, n_h), np.nan, dtype=np.float32)

    sdir = os.path.join(run_dir, "shards")
    n_done = 0
    if os.path.isdir(sdir):
        for name in sorted(os.listdir(sdir)):
            dpath = os.path.join(sdir, name, "done.json")
            if not os.path.isfile(dpath):
                continue
            with open(dpath) as f:
                d = json.load(f)
            s, e = d["start"], d["end"]
            cm[s:e] = np.load(os.path.join(sdir, name, "curves_mae.npy"))
            cs[s:e] = np.load(os.path.join(sdir, name, "curves_mse.npy"))
            n_done += 1

    np.save(os.path.join(run_dir, "curves_mae.npy"), cm)
    np.save(os.path.join(run_dir, "curves_mse.npy"), cs)
    return cm, cs, n_done


def _print_data_sanity(curves_mae: np.ndarray, n_segments: np.ndarray) -> None:
    """Report whether the generated data is actually context-sensitive.

    curves_mae shape: (N, n_windows, n_horizons). Reports per-horizon stats so
    we can see how the useful window changes with forecast length.
    """
    valid = ~np.isnan(curves_mae).any(axis=(1, 2))
    if not valid.any():
        print(Fore.RED + "  No completed curves to summarize." + Fore.RESET)
        return
    cm_v = curves_mae[valid]                                # (V, n_win, n_h)
    n_win = cm_v.shape[1]
    win_arr = np.array(WINDOW_GRID)
    print(Fore.GREEN + "  Data sanity (MAE curves):" + Fore.RESET)
    print(f"    valid curves: {int(valid.sum())}/{len(curves_mae)}")
    for h_idx, h in enumerate(HORIZON_GRID):
        cm_h = cm_v[:, :, h_idx]                            # (V, n_win)
        argmin = cm_h.argmin(axis=1)
        interior = float((argmin < n_win - 1).mean())
        opt_w = win_arr[argmin]
        print(f"    h={h:>3}: interior_opt={interior:5.1%}  "
              f"median_opt_win={int(np.median(opt_w)):>5}  "
              f"mean={opt_w.mean():.0f}")
    for s in sorted(set(n_segments[valid].tolist())):
        frac = float((n_segments[valid] == s).mean())
        print(f"    regimes={s}: {frac:.1%} of valid series")


# ==============================================================================
#  MAIN
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic context-length ablation dataset.")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume an existing run directory.")
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"],
                   help="Force device. Default: all CUDA devices, else CPU.")
    p.add_argument("--n-series", type=int, default=N_SERIES,
                   help="Number of synthetic series.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help="TSFM inference batch size.")
    p.add_argument("--shard-size", type=int, default=SHARD_SIZE,
                   help="Series per shard (parallelism + resume granularity).")
    p.add_argument("--windows", type=int, nargs="+", default=None,
                   help="Subset of WINDOW_GRID to run (quick smoke tests).")
    return p.parse_args()


def main() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    devices = resolve_devices(args.device)
    model_id, family, display = MODELS[0]

    grid = [w for w in WINDOW_GRID if args.windows is None or w in args.windows]
    if any(w > MAX_WINDOW for w in grid):
        raise ValueError(f"WINDOW_GRID exceeds MAX_WINDOW={MAX_WINDOW}.")
    win_indices = [WINDOW_GRID.index(w) for w in grid]

    print(Fore.CYAN + f"Devices: {devices}  |  Model: {display} ({model_id})" + Fore.RESET)
    print(Fore.CYAN + f"HORIZON_GRID={HORIZON_GRID}  MAX_WINDOW={MAX_WINDOW}  "
          + f"windows={grid}  shard_size={args.shard_size}" + Fore.RESET)

    # ---------- Run directory + series pool (generated once) ----------------
    if args.resume:
        run_dir = args.resume
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"--resume dir not found: {run_dir}")
        print(Fore.CYAN + f"Resuming run: {run_dir}" + Fore.RESET)
    else:
        run_dir = os.path.join(OUTPUT_ROOT, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(run_dir, exist_ok=True)
        print(Fore.CYAN + f"Run directory: {run_dir}" + Fore.RESET)

    contexts_path = os.path.join(run_dir, "contexts.npy")
    targets_path = os.path.join(run_dir, "targets.npy")
    nseg_path = os.path.join(run_dir, "n_segments.npy")

    if os.path.isfile(contexts_path):
        n_series = int(np.load(contexts_path, mmap_mode="r").shape[0])
        n_segments = np.load(nseg_path)
        print(Fore.CYAN + f"  Found existing pool: {n_series} series" + Fore.RESET)
    else:
        n_series = args.n_series
        contexts, targets, n_segments = generate_dataset(n_series, SEED)
        np.save(contexts_path, contexts)
        np.save(targets_path, targets)
        np.save(nseg_path, n_segments)
        del contexts, targets
        print(Fore.GREEN + f"  Generated synthetic pool: {n_series} series" + Fore.RESET)

    # ---------- Build shard list (skip completed) ---------------------------
    pending: List[Tuple[int, int, int]] = []
    n_completed = 0
    for shard_id, start in enumerate(range(0, n_series, args.shard_size)):
        end = min(start + args.shard_size, n_series)
        if os.path.isfile(os.path.join(_shard_dir(run_dir, shard_id), "done.json")):
            n_completed += 1
        else:
            pending.append((shard_id, start, end))
    print(Fore.CYAN + f"  Shards: {n_completed} cached, {len(pending)} pending"
          + Fore.RESET)

    # ---------- Spawn workers + dispatch shards -----------------------------
    if pending:
        ctx = mp.get_context("spawn")
        shard_queue = ctx.Queue()
        result_queue = ctx.Queue()
        for spec in pending:
            shard_queue.put(spec)
        for _ in devices:
            shard_queue.put(None)

        workers: List[mp.Process] = []
        for i, dev in enumerate(devices):
            p = ctx.Process(
                target=gpu_worker,
                args=(i, dev, shard_queue, result_queue, run_dir,
                      contexts_path, targets_path, model_id, family,
                      MAX_HORIZON, args.batch_size, win_indices),
                name=f"gpu_worker_{i}_{dev.replace(':', '')}",
            )
            p.start()
            workers.append(p)

        t0 = time.perf_counter()
        n_received = 0
        while n_received < len(pending):
            try:
                result_queue.get(timeout=3600)
            except Empty:
                if not any(p.is_alive() for p in workers):
                    print(Fore.RED + f"  All workers died with "
                          + f"{n_received}/{len(pending)} shards." + Fore.RESET)
                    break
                continue
            n_received += 1

        for p in workers:
            p.join(timeout=120)
            if p.is_alive():
                print(Fore.RED + f"  Worker {p.name} hung — terminating." + Fore.RESET)
                p.terminate()
                p.join(timeout=10)
        print(Fore.MAGENTA + f"  Labeling wall-clock: "
              + f"{time.perf_counter() - t0:.1f}s" + Fore.RESET)

    # ---------- Merge shards + metadata -------------------------------------
    curves_mae, _, n_done = merge_shards(run_dir, n_series)
    total_shards = (n_series + args.shard_size - 1) // args.shard_size

    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({
            "model_id": model_id, "model_family": family, "model_display": display,
            "window_grid": WINDOW_GRID, "horizon_grid": HORIZON_GRID,
            "max_horizon": MAX_HORIZON, "max_window": MAX_WINDOW,
            "n_series": int(n_series), "seed": SEED,
            "shard_size": args.shard_size,
            "shards_done": n_done, "shards_total": total_shards,
            "devices": devices,
            "period_min": PERIOD_MIN, "period_max": PERIOD_MAX,
            "n_segment_choices": N_SEGMENT_CHOICES,
            "min_segment_len": MIN_SEGMENT_LEN,
            "level_shift_std": LEVEL_SHIFT_STD,
            "created": datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2)

    _print_data_sanity(curves_mae, n_segments)
    if n_done < total_shards:
        print(Fore.YELLOW + f"  {total_shards - n_done} shard(s) incomplete — "
              + f"re-run with --resume {run_dir}" + Fore.RESET)
    print(Fore.GREEN + f"\nDataset written to: {run_dir}" + Fore.RESET)


if __name__ == "__main__":
    main()
