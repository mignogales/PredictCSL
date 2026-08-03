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
   series is built segment-by-segment with injected non-stationarity: regime
   changes, level shifts, variance shifts, per-segment trend/AR/seasonality,
   wave-type variety (sin/cos/sawtooth/square), and random spikes. Change-point
   transitions are optionally gradual (linear crossfade). Within multi-regime
   series the dominant seasonal period can shift at each regime boundary.
2. For each series, forecast max(HORIZON_GRID) steps from the last W context
   samples, for every W in WINDOW_GRID, using a chosen TSFM. Per-window
   inference is done once at the longest horizon; left-to-right prefix property
   lets us slice at every h in HORIZON_GRID for free.
3. Save the full per-series error-vs-(context, horizon) surface (MAE and MSE
   for every (W, h) pair). By default all models in MODELS are labeled in
   sequence, sharing the same series pool. Use --model-idx to run one model.

Parallelism
-----------
Series generation runs once in the parent (CPU-bound, seconds). The labeling
stage is sharded across GPUs: one worker per CUDA device drains a shared queue
of series-shards, loading the TSFM into VRAM once. Each shard is written to
disk; the parent merges shards. Completed shards are skipped automatically on
re-runs (safe to interrupt and restart).

Outputs
-------
logs/experiments/context_length_dataset/
    contexts.npy             (N, MAX_WINDOW)            float32  -- shared pool
    targets.npy              (N, MAX_HORIZON)           float32
    n_segments.npy           (N,)                       int32
    meta.json                series-pool config (written once)
    <family>/
        curves_mae.npy       (N, n_windows, n_horizons) float32
        curves_mse.npy       (N, n_windows, n_horizons) float32
        shards/shard_NNN/    per-shard curves (resume unit)
        meta.json            per-model run metadata
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from datetime import datetime
from queue import Empty
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from colorama import Fore
from scipy.signal import lfilter
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

from experiments import models_config
from experiments.gifteval_inference_recipes import inference_recipe
from experiments.timesfm_gifteval import (
    forecast_quantiles as forecast_timesfm_quantiles,
    load_model as load_timesfm_official,
)
from experiments.patchtst_gifteval import forecast_patchtst_quantiles_official


# ==============================================================================
#  CONFIGURATION
# ==============================================================================

# -- Series geometry ----------------------------------------------------------
MAX_WINDOW   = 15360         # shared pool; TimesFM official GiftEval cap
PERIOD_MIN   = 16            # seasonal period bounds (samples)
# Preserve dense coverage of the historical 16..2,048 range while explicitly
# teaching every predictor about partial-cycle / almost-full-context cases such
# as BizITObs Application and Service (daily period = 8,640 ten-second samples).
# A single log-uniform 16..15,360 draw would starve the old short-period regime,
# so periods are sampled from a two-component mixture instead.
PERIOD_CORE_MAX = 2048
PERIOD_MAX   = MAX_WINDOW
LONG_PERIOD_PROB = 0.25
SYNTHETIC_POOL_VERSION = 2


def synthetic_pool_signature() -> dict:
    """Versioned generator fields that make cached pools/labels incompatible."""
    return {
        "version": SYNTHETIC_POOL_VERSION,
        "period_min": PERIOD_MIN,
        "period_core_max": PERIOD_CORE_MAX,
        "period_max": PERIOD_MAX,
        "long_period_probability": LONG_PERIOD_PROB,
    }

# -- Ablation grid (suffix lengths of the context pool) -----------------------
# The shared union includes every historical ablation point plus any registered
# family cap that falls off that grid.  A family only receives its own off-grid
# cap (not another family's), which makes the final synthetic action a genuine
# native/full-context label without adding irrelevant classes to other models.
BASE_WINDOW_GRID = [
    32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
    1024, 1536, 2048, 2560, 3072, 4096, 6144, 8192, 12288, 15360,
]
_OFF_GRID_FAMILY_CAPS = {
    int(cap) for cap in models_config.FAMILY_CONTEXT_LIMIT.values()
    if 1 <= int(cap) <= MAX_WINDOW and int(cap) not in BASE_WINDOW_GRID
}
# Append instead of inserting so the load-bearing global indices of every
# historical window remain stable in existing shard done-markers.  Per-family
# public grids are sorted below.
WINDOW_GRID = BASE_WINDOW_GRID + sorted(_OFF_GRID_FAMILY_CAPS)

# Smoke-test mode (PREDICTCSL_TEST=1, set by experiments/run_all.py --test):
# collapse the ablation grid to just its smallest and largest window so the
# whole pipeline can be exercised end-to-end in minutes. Resolved from the env
# at import time on purpose — `mp spawn` workers re-import this module, so a
# main()-level override would not reach them, but an inherited env var does.
TEST_MODE = os.environ.get("PREDICTCSL_TEST") == "1"


def window_grid_for_family(family: str) -> List[int]:
    """Candidate windows supported by ``family``.

    Historical points are retained up to the registered cap.  When the cap is
    off-grid (currently Sundial at 2,880), it is appended only for that family.
    Therefore the last output always represents the model's native cap, while
    smaller models do not carry classes they can never select.
    """
    cap = models_config.context_limit(family)
    grid = [w for w in BASE_WINDOW_GRID if w <= cap]
    if cap <= MAX_WINDOW and cap not in grid:
        grid.append(cap)
        grid.sort()
    if not grid:
        raise ValueError(f"No WINDOW_GRID point is <= context cap {cap} for {family}.")
    if TEST_MODE:
        grid = [grid[0], grid[-1]] if len(grid) > 1 else grid
    return grid

# -- Horizon grid (forecast lengths labeled per series) -----------------------
HORIZON_GRID = [16, 32, 64, 128, 512, 1024]
MAX_HORIZON  = max(HORIZON_GRID)

# -- Dataset size -------------------------------------------------------------
N_SERIES   = 50_000
SEED       = 42
BATCH_SIZE = 32              # TSFM inference batch size
SHARD_SIZE = 500             # series per shard (parallelism + resume granularity)
DYNAMIC_BATCH_REFERENCE_CONTEXT = 8192
DYNAMIC_BATCH_MAX_SIZE = 4096

# -- Short / padded series ----------------------------------------------------
# A fraction of series mimic real-world short inputs: the genuine signal occupies
# only the last `real_len` samples of the context pool and the leading
# MAX_WINDOW - real_len samples are left-padded with zeros. This puts padded
# inputs into the predictor's training distribution so it is robust at inference
# to series shorter than MAX_WINDOW (which are left-padded identically).
#   - real_len ~ Uniform[MIN_REAL_LEN, MAX_WINDOW)  (i.e. padding amount uniform)
#   - ablation windows W > real_len are served model-aware, NOT with raw padding:
#       * variable-length models (chronos, timesfm, moirai, sundial, timemoe) are
#         fed ONLY the min(W, real_len) genuine samples (the leading zero-padding
#         is stripped, never shown to the model),
#       * fixed-context models (patchtst_fm) are fed the genuine samples NaN-padded
#         to their native length (NaN is the model's missing-value indicator).
#     The label curve therefore flattens beyond real_len (extra context carries no
#     signal, so it cannot reduce error) and stays NaN-free.
PAD_FRAC     = 0.10          # fraction of series that are short / left-padded
MIN_REAL_LEN = 500           # shortest genuine-signal length for a padded series

# -- Non-stationarity controls ------------------------------------------------
# Rebalanced: 50% single-regime removed in favour of richer multi-regime series.
N_SEGMENT_CHOICES = [1, 2, 2, 3, 3, 4]   # 17% × 1, 33% × 2, 33% × 3, 17% × 4
MIN_SEGMENT_LEN   = 512      # shortest regime; also keeps the horizon clear of cuts
LEVEL_SHIFT_STD   = 1.5      # std of the cumulative mean jump at each regime change
SEG_AMP_RANGE     = (0.6, 1.6)   # per-segment amplitude scale (variance regimes)

# -- Seasonal-shift controls --------------------------------------------------
SEASONAL_SHIFT_PROB = 0.5    # probability of dominant-period change at each regime boundary

# -- Gradual-transition controls ----------------------------------------------
TRANSITION_MAX = 64          # max half-width of linear blend zone at cut points (samples)
                             # 0 = hard cut; uniformly sampled in [0, TRANSITION_MAX]

# -- Spike (outlier impulse) controls -----------------------------------------
SPIKE_PROB       = 0.30      # fraction of segments that contain impulse spikes
SPIKE_RATE_RANGE = (0.002, 0.015)  # Poisson rate as fraction of segment length
SPIKE_STD        = 3.0       # spike amplitude std (post-standardization scale)

# -- Multi-GPU ----------------------------------------------------------------
DEVICES = None               # None -> all visible CUDA devices (or CPU)

# -- Models  (model_id, family, display_name) ---------------------------------
# Full loader catalog, indexed by --model-idx. Defined in experiments.models_config
# (single source of truth); see the per-model loader/predict wrappers below for the
# family-specific gotchas (Toto's toto2 module, FlowState's r1.1 pin, TiRex's
# genuine-only handling, etc.). APPEND-ONLY there — --model-idx positions are stable.
MODELS = models_config.catalog()

# -- Quantile bookkeeping (per model family) ----------------------------------
MOIRAI2_MEDIAN_IDX              = 4
PATCHTST_FM_QUANTILE_LEVELS     = [i / 10.0 for i in range(1, 10)]
PATCHTST_FM_MEDIAN_QUANTILE_IDX = 4
SUNDIAL_NUM_SAMPLES             = 100
SUNDIAL_MAX_CONTEXT             = 2880
TIMEMOE_MAX_TOTAL               = 4096   # context + horizon must not exceed this
TOTO_NUM_QUANTILES              = 9      # Toto 2.0 quantile head: [0.1..0.9]
TOTO_MEDIAN_QUANTILE_IDX        = 4      # middle of the 9 quantiles (0.5)
# Conservative context cap: Toto 2.0 handles long history but full-grid windows
# (6144/8192) blow up the attention memory. Windows above this are skipped
# (curve flattens past it), mirroring SUNDIAL_MAX_CONTEXT. Raise if your GPU has
# the headroom.
TOTO_MAX_CONTEXT                = 4096
# Toto 2.0 patches the time axis into fixed chunks; the context length fed to
# forecast() must be a multiple of this patch size (verified from the einops
# error on the 313m checkpoint). Non-multiple windows (e.g. grid point 48) and
# short genuine series are left-padded up to the next multiple, with the padded
# steps masked out. If you switch size tiers, confirm this against the model's
# patch_embed config.
TOTO_PATCH_SIZE                 = 32
FLOWSTATE_REVISION              = "r1.1"  # GIFT-Eval checkpoint (top of leaderboard)
FLOWSTATE_NUM_QUANTILES         = 9
FLOWSTATE_MEDIAN_QUANTILE_IDX   = 4       # middle of the 9 output quantiles
# scale_factor relates the data's seasonality to FlowState's pretraining base
# (24 steps): scale = 24 / seasonality_steps. Synthetic series carry no fixed
# sampling rate, so 1.0 (treat input cadence as the base) is the neutral choice
# HERE ONLY. On real data this must be per-dataset — stage 3 computes it from
# the GiftEval frequency/domain via flowstate_scale_factor() (IBM's leaderboard
# recipe); running GiftEval at 1.0 measures a handicapped FlowState.
FLOWSTATE_SCALE_FACTOR          = 1.0
# Pretraining context is 4096 (r1.1). FlowState is a *linear-cost* SSM so longer
# windows won't blow up memory, but fidelity past the trained context is
# unverified — cap here to keep labels trustworthy. Curve flattens past it, like
# SUNDIAL/TOTO. Raise (or remove the skip in gpu_worker) to probe longer context.
FLOWSTATE_MAX_CONTEXT           = 4096
TIREX_NUM_QUANTILES             = 9      # TiRex quantile head: [0.1..0.9]
TIREX_MEDIAN_QUANTILE_IDX       = 4      # middle of the 9 quantiles (0.5)
# TiRex is xLSTM (recurrent, LINEAR cost) so long windows won't blow up memory,
# but fidelity past its trained context is unverified — cap here to keep labels
# trustworthy (curve flattens past it), like SUNDIAL/FLOWSTATE. TiRex2's
# pretraining context length is 8192 samples (TiRex-1 was 2048).
TIREX_MAX_CONTEXT               = 8192

# Output root. Overridable via env so run_all.py --test can redirect the whole
# smoke run into a throwaway tree (which it deletes afterwards) without ever
# touching the real datasets. Used only in the parent process (workers receive
# explicit paths), but env-resolution keeps it consistent regardless.
OUTPUT_ROOT = os.environ.get(
    "PREDICTCSL_DATASET_ROOT", "logs/experiments/context_length_dataset")


# ==============================================================================
#  DEVICE RESOLUTION
# ==============================================================================

# Snapshot the GPU ordering the user requested via CUDA_VISIBLE_DEVICES *before*
# any worker re-masks it. torch enumerates only the listed GPUs and re-indexes
# them from 0, so the logical label cuda:i maps positionally onto this list
# (e.g. cuda:1 under CUDA_VISIBLE_DEVICES=0,2 is physical GPU "2").
_ORIG_VISIBLE_DEVICES: Optional[List[str]] = (
    [d.strip() for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if d.strip()]
    if os.environ.get("CUDA_VISIBLE_DEVICES")
    else None
)


def _physical_gpu_id(device: str) -> str:
    """Map a torch logical device label (``cuda:i``) to its physical GPU id.

    A worker that re-masks ``CUDA_VISIBLE_DEVICES`` to pin itself must use the
    original physical id, not the logical index -- otherwise ``cuda:1`` under
    ``CUDA_VISIBLE_DEVICES=0,2`` wrongly selects physical GPU 1 instead of 2.
    """
    logical = device.split(":")[-1] if ":" in device else "0"
    if _ORIG_VISIBLE_DEVICES is None:
        return logical
    try:
        return _ORIG_VISIBLE_DEVICES[int(logical)]
    except (ValueError, IndexError):
        return logical


def resolve_devices(force: Optional[str]) -> List[str]:
    if force == "cpu":
        return ["cpu"]
    if force == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was explicitly requested for Stage 1, but PyTorch reports "
            "torch.cuda.is_available() == False. Refusing to silently run the "
            "50,000-series TSFM labeling workload on CPU. Check the active "
            "conda environment, CUDA_VISIBLE_DEVICES, NVIDIA driver, and the "
            "installed PyTorch CUDA build; use --device cpu only intentionally."
        )
    if not torch.cuda.is_available():
        return ["cpu"]
    if DEVICES is not None:
        return list(DEVICES)
    n = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(n)] if n > 0 else ["cpu"]


# ==============================================================================
#  SYNTHETIC GENERATOR
# ==============================================================================

def _sample_periods(
    rng: np.random.RandomState, size: Optional[int] = None,
) -> np.ndarray:
    """Draw seasonal periods without losing the historical short-cycle density.

    ``LONG_PERIOD_PROB`` of draws cover 2,048..MAX_WINDOW, including the
    one-cycle/partial-cycle cases absent from the original generator.  The
    remaining draws retain the exact original log-uniform 16..2,048 support.
    """
    shape = () if size is None else (size,)
    use_long = rng.uniform(size=shape) < LONG_PERIOD_PROB
    core = np.exp(rng.uniform(
        math.log(PERIOD_MIN), math.log(PERIOD_CORE_MAX), size=shape))
    long = np.exp(rng.uniform(
        math.log(PERIOD_CORE_MAX), math.log(PERIOD_MAX), size=shape))
    return np.where(use_long, long, core)

def _sample_ar_coefficients(rng: np.random.RandomState) -> Optional[np.ndarray]:
    """
    Sample AR(p) coefficients with diverse behavior.

    Returns stable AR coefficients by sampling roots and converting, ensuring
    variety across (p=1, 2, 3) with different dynamics:
    - AR(1): persistent (0.4-0.95), weak (0-0.3), or oscillatory damped (-0.7-0)
    - AR(2): oscillatory complex pairs or mixed real roots
    - AR(3): diverse multi-root combinations
    """
    # 50% of the time, return None (no AR for that segment)
    if rng.uniform() < 0.5:
        return None

    # Choose AR order: bias toward AR(1) and AR(2) for interpretability
    p = int(rng.choice([1, 2, 3], p=[0.50, 0.35, 0.15]))

    if p == 1:
        # AR(1): φ ∈ [-1, 1] for stability; weight toward persistence
        if rng.uniform() < 0.60:
            # Persistent: φ ∈ [0.4, 0.95] (slow mean reversion)
            phi = rng.uniform(0.4, 0.95)
        elif rng.uniform() < 0.50:
            # Weak positive: φ ∈ [0.0, 0.3]
            phi = rng.uniform(0.0, 0.3)
        else:
            # Negative damped oscillation: φ ∈ [-0.7, 0.0]
            phi = rng.uniform(-0.7, 0.0)
        coeffs = np.array([phi])

    elif p == 2:
        # AR(2): Sample roots and convert to [c1, c2] coefficients
        if rng.uniform() < 0.55:
            # Oscillatory: complex conjugate pair e^(±iω)
            # Magnitude r ∈ [0.7, 0.98], frequency ω ∈ [0.1π, 0.9π]
            r = rng.uniform(0.7, 0.98)
            omega = rng.uniform(0.1 * np.pi, 0.9 * np.pi)
            # From (1 - r*e^(iω)*B)(1 - r*e^(-iω)*B) = 1 - 2r*cos(ω)*B + r^2*B^2
            c1 = 2 * r * np.cos(omega)
            c2 = -(r ** 2)
            coeffs = np.array([c1, c2])
        else:
            # Real roots: two independent persistence levels.
            # For poles p1,p2 the stable AR coeffs are c1=p1+p2, c2=-p1*p2
            # (filter denom a=[1,-c1,-c2] -> char poly z^2 - c1 z - c2 with
            # roots p1,p2). The previous signs put a pole outside the unit
            # circle (|pole|>1), causing exponential blow-up over the series.
            r1 = rng.uniform(0.2, 0.95)
            r2 = rng.uniform(0.2, 0.95)
            c1 = r1 + r2
            c2 = -(r1 * r2)
            coeffs = np.array([c1, c2])

    else:  # p == 3
        # AR(3): Complex dynamics; mix of real and complex roots
        if rng.uniform() < 0.50:
            # One real root + one complex conjugate pair r_complex*e^(±iω).
            # Stable coeffs from elementary symmetric fns of the poles:
            #   c1 = e1, c2 = -e2, c3 = e3.
            r_real = rng.uniform(0.3, 0.90)
            r_complex = rng.uniform(0.6, 0.95)
            omega = rng.uniform(0.1 * np.pi, 0.8 * np.pi)
            two_re = 2 * r_complex * np.cos(omega)
            r2c = r_complex ** 2
            c1 = r_real + two_re
            c2 = -(r_real * two_re + r2c)
            c3 = r_real * r2c
            coeffs = np.array([c1, c2, c3])
        else:
            # Three real roots: c1=Σp, c2=-Σpairwise, c3=Πp (stable poles).
            r1 = rng.uniform(0.3, 0.90)
            r2 = rng.uniform(0.3, 0.90)
            r3 = rng.uniform(0.3, 0.90)
            c1 = r1 + r2 + r3
            c2 = -(r1*r2 + r1*r3 + r2*r3)
            c3 = r1 * r2 * r3
            coeffs = np.array([c1, c2, c3])

    return coeffs


def _generate_segment(
    rng: np.random.RandomState,
    length: int,
    force_period: Optional[float] = None,
) -> np.ndarray:
    """One stationary regime: periodic components + optional AR + trend + noise.

    Supports sin, cos, sawtooth, and square waveforms. Optionally forces the
    dominant (first) periodic component to `force_period` so that the caller
    can control seasonal-period changes across regime boundaries.

    Not standardized here -- amplitude scaling and level shifts are applied at
    the series level so regimes differ in scale and mean.
    """
    t = np.arange(length, dtype=np.float32)
    seg = np.zeros(length, dtype=np.float32)

    # -- Periodic components --------------------------------------------------
    n_periodic = int(rng.randint(1, 4))
    periods = _sample_periods(rng, size=n_periodic)
    if force_period is not None:
        periods[0] = force_period   # dominant period is caller-controlled
    amplitudes = rng.uniform(0.5, 2.0, size=n_periodic)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_periodic)

    for amp, T_p, ph in zip(amplitudes, periods, phases):
        omega = 2.0 * np.pi / float(T_p)
        wtype = int(rng.randint(0, 4))   # 0=sin, 1=cos, 2=sawtooth, 3=square
        if wtype == 0:
            seg += amp * np.sin(omega * t + ph)
        elif wtype == 1:
            seg += amp * np.cos(omega * t + ph)
        elif wtype == 2:
            # Sawtooth in [-1, 1]: 2 * frac(t/T + ph/(2π)) - 1
            seg += amp * (
                2.0 * ((t / float(T_p) + ph / (2.0 * np.pi)) % 1.0) - 1.0
            ).astype(np.float32)
        else:
            seg += amp * np.sign(np.sin(omega * t + ph)).astype(np.float32)

    # -- Optional AR(p) with diverse roots (stable; lfilter for speed) --------
    coeffs = _sample_ar_coefficients(rng)
    if coeffs is not None:
        innov = rng.normal(0.0, 0.5, size=length).astype(np.float32)
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

    # -- Random spikes (Poisson-arrival impulses) -----------------------------
    if rng.uniform() < SPIKE_PROB:
        rate = float(rng.uniform(*SPIKE_RATE_RANGE))
        n_spikes = int(rng.poisson(rate * length))
        if n_spikes > 0:
            locs = rng.randint(0, length, size=n_spikes)
            amps = rng.normal(0.0, SPIKE_STD, size=n_spikes).astype(np.float32)
            np.add.at(seg, locs, amps)

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
    rng: np.random.RandomState, total_length: int, pad_len: int = 0
) -> Tuple[np.ndarray, int]:
    """Build one synthetic series with injected non-stationarity.

    Composed of 1-4 regimes. Change points fall strictly inside the context
    pool. Within each multi-regime series:
      - The dominant seasonal period can shift at each regime boundary
        (with probability SEASONAL_SHIFT_PROB).
      - Hard cuts are replaced by a linear crossfade of half-width T (sampled
        uniformly in [0, TRANSITION_MAX]) that preserves high-frequency
        structure while smoothing the mean-level jump.

    If ``pad_len > 0`` the series is treated as a short input: the genuine
    signal occupies only the last ``MAX_WINDOW - pad_len`` context samples
    (plus the horizon) and the leading ``pad_len`` samples are zeroed out as
    left-padding before standardization (so they map to a constant, matching
    the inference path).

    Returns:
        series: float32 (total_length,), standardized by the full-context
            stats; left-padded when pad_len > 0.
        n_segments: number of regimes.
    """
    n_seg = int(rng.choice(N_SEGMENT_CHOICES))

    if n_seg == 1:
        series = _generate_segment(rng, total_length)
    else:
        lo = MIN_SEGMENT_LEN
        hi = MAX_WINDOW - MIN_SEGMENT_LEN
        cuts = _sample_cut_points(rng, n_seg - 1, lo, hi, MIN_SEGMENT_LEN)
        bounds = [0] + cuts + [total_length]

        # Initial dominant period; may shift at each boundary.
        dominant_period = float(_sample_periods(rng))

        parts: List[np.ndarray] = []
        level = 0.0
        for i in range(n_seg):
            seg_len = bounds[i + 1] - bounds[i]
            seg = _generate_segment(rng, seg_len, force_period=dominant_period)
            seg = seg * float(rng.uniform(*SEG_AMP_RANGE))
            if i > 0:
                level += float(rng.normal(0.0, LEVEL_SHIFT_STD))
                if rng.uniform() < SEASONAL_SHIFT_PROB:
                    dominant_period = float(_sample_periods(rng))
            parts.append((seg + level).astype(np.float32))

        # Gradual transitions: smear the level jump at each regime boundary.
        # The crossfade adds a ramp that rises from 0 to +jump over the T
        # samples before the cut (in the old regime) and a matching ramp that
        # falls from -jump back to 0 over the T samples after the cut (in the
        # new regime). Net effect: continuity at the cut point, no change
        # outside the [cut-T, cut+T] window.
        for i in range(len(parts) - 1):
            T = int(rng.randint(0, TRANSITION_MAX + 1))
            T = min(T, len(parts[i]), len(parts[i + 1]))
            if T == 0:
                continue
            jump = float(parts[i + 1][0] - parts[i][-1])
            ramp = np.linspace(0.0, jump, 2 * T, dtype=np.float32)
            ramp[T:] -= jump                         # right half: ≈ -jump/2 → 0
            parts[i][-T:]    = parts[i][-T:]    + ramp[:T]   # tail of old regime
            parts[i + 1][:T] = parts[i + 1][:T] + ramp[T:]  # head of new regime

        series = np.concatenate(parts).astype(np.float32)

    # Zero the left-padding (if any) in raw space, then standardize using the
    # full observed context (no horizon leakage). This matches the inference
    # path exactly — test_window_ablation_gifteval_v5._prepare_predictor_inputs
    # zero-pads then computes mu/sigma over the whole context_length vector, so
    # the padded region maps to the constant -mu/sigma rather than exactly 0.
    if pad_len > 0:
        series[:pad_len] = 0.0
    # Guard against rare AR/lfilter blow-ups: sanitize non-finite values and
    # compute the standardization stats in float64 so squaring large-but-finite
    # values does not overflow float32 (which would yield an all-NaN series).
    series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
    ctx = series[:MAX_WINDOW].astype(np.float64)
    mu, sigma = float(ctx.mean()), float(ctx.std())
    series = (series - mu) / (sigma + 1e-8)
    return series.astype(np.float32, copy=False), n_seg


def generate_dataset(
    n_series: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialize the synthetic pool (parent process, serial).

    A PAD_FRAC fraction of series are short / left-padded: their genuine signal
    occupies only the last `real_len` context samples (real_len uniform in
    [MIN_REAL_LEN, MAX_WINDOW)) and the leading samples are zeroed.

    Returns:
        contexts:     (n_series, MAX_WINDOW)   float32  -- predictor input
        targets:      (n_series, MAX_HORIZON)  float32  -- forecast target
        n_segments:   (n_series,)              int32
        real_lengths: (n_series,)              int32    -- genuine-signal length
                          in the context (MAX_WINDOW for non-padded series)
    """
    total_length = MAX_WINDOW + MAX_HORIZON
    contexts = np.empty((n_series, MAX_WINDOW), dtype=np.float32)
    targets = np.empty((n_series, MAX_HORIZON), dtype=np.float32)
    n_segments = np.empty((n_series,), dtype=np.int32)
    real_lengths = np.empty((n_series,), dtype=np.int32)

    for i in tqdm(range(n_series), desc="  Generating series", leave=False):
        rng = np.random.RandomState(seed + i)
        # Decide short/padded status first so the RNG stream is deterministic.
        if rng.uniform() < PAD_FRAC:
            real_len = int(rng.randint(MIN_REAL_LEN, MAX_WINDOW))
        else:
            real_len = MAX_WINDOW
        pad_len = MAX_WINDOW - real_len
        series, n_seg = _generate_synthetic_series(rng, total_length, pad_len=pad_len)
        contexts[i] = series[:MAX_WINDOW]
        targets[i] = series[MAX_WINDOW:MAX_WINDOW + MAX_HORIZON]
        n_segments[i] = n_seg
        real_lengths[i] = real_len

    return contexts, targets, n_segments, real_lengths


# ==============================================================================
#  MODEL LOADERS + MEDIAN-FORECAST WRAPPERS
# ==============================================================================
#  Each predict_* wrapper takes a CPU tensor x of shape (B, W, 1) and returns
#  the point/median forecast as (B, horizon) on `device`.

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
    # Chronos2Pipeline otherwise defaults to an internal batch_size=256,
    # silently undoing the caller's dynamic microbatch selection. Each call
    # here already contains exactly one tuned microbatch, so forward its size.
    samples = pipeline.predict(
        inputs=context,
        prediction_length=horizon,
        batch_size=int(context.shape[0]),
        # Keep every time series independent. Chronos2 calls this cross-learning;
        # enabling it would share information across series in the same batch and
        # make forecasts depend on batch composition.
        cross_learning=False,
    )
    if isinstance(samples, list):
        samples = torch.stack(samples, dim=0).squeeze(1)
    if samples.dim() == 4:
        samples = samples.squeeze(2)                   # (B, Q, H)
    # Chronos2 deliberately returns predictions on CPU. Keep the quantile
    # reduction there, then copy only the median back because forecast_window
    # scatters every family result into a device-side output buffer.
    samples = samples.to(dtype=torch.float32)
    return torch.median(samples, dim=1).values.to(
        device=device, non_blocking=True)              # (B, H)


def load_chronos_bolt(model_id: str, device: str):
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(model_id, device_map=device)


def predict_chronos_bolt(pipeline, x: torch.Tensor, horizon: int, device: str) -> torch.Tensor:
    context = [row for row in x[:, :, 0]]
    quantiles = torch.as_tensor(
        pipeline.predict(context, prediction_length=horizon),
        device=device, dtype=torch.float32)
    levels = [float(q) for q in pipeline.quantiles]
    median_idx = min(range(len(levels)), key=lambda i: abs(levels[i] - 0.5))
    return quantiles[:, median_idx, :horizon]


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


def predict_timesfm(model, x: torch.Tensor, horizon: int, device: str) -> torch.Tensor:
    contexts = [np.asarray(row, dtype=np.float32) for row in x[:, :, 0].numpy()]
    quantiles = forecast_timesfm_quantiles(
        model, contexts, horizon, batch_size=len(contexts))
    return torch.as_tensor(
        quantiles[:, 4, :horizon], dtype=torch.float32, device=device)


def load_patchtst_fm(model_id: str, device: str):
    from tsfm_public import PatchTSTFMForPrediction
    model = PatchTSTFMForPrediction.from_pretrained(model_id, device_map=device)
    model.eval()
    return model


def predict_patchtst_fm(model, x: torch.Tensor, horizon: int, device: str) -> torch.Tensor:
    qf = forecast_patchtst_quantiles_official(
        model,
        list(x[:, :, 0]),
        horizon,
        device,
        PATCHTST_FM_QUANTILE_LEVELS,
    )
    return qf[:, PATCHTST_FM_MEDIAN_QUANTILE_IDX, :]


def _patch_dynamic_cache_seen_tokens() -> None:
    """Restore legacy ``DynamicCache`` APIs for old trust_remote_code models.

    Sundial's (and TimeMoE's) remote modeling code predates several transformers
    Cache-API changes and calls members that newer versions deprecated/removed:

      * ``seen_tokens``    — attribute removed in >=4.41 (now ``get_seq_length()``)
      * ``get_max_length`` — method deprecated in 4.41, removed ~4.48 (now
                             ``get_max_cache_shape()``); always ``None`` for the
                             unbounded ``DynamicCache``.
      * ``get_usable_length`` — companion of ``get_max_length``, removed in the
                             same cleanup; restored with its original logic.

    Without these, ``model.generate`` raises ``AttributeError`` mid-decode. We
    re-expose them as thin shims so the cap-free forecast path works unchanged.
    Each is added only if missing, so this is a no-op on versions that still
    provide them and never shadows native behavior.
    """
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return

    if not hasattr(DynamicCache, "seen_tokens"):
        def _seen_tokens(self):
            val = getattr(self, "_seen_tokens", None)
            return val if val is not None else self.get_seq_length()
        DynamicCache.seen_tokens = property(_seen_tokens)

    if not hasattr(DynamicCache, "get_max_length"):
        def _get_max_length(self):
            shape_fn = getattr(self, "get_max_cache_shape", None)
            return shape_fn() if shape_fn is not None else None
        DynamicCache.get_max_length = _get_max_length

    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_length, layer_idx=0):
            max_length = self.get_max_length()
            previous_seq_length = self.get_seq_length(layer_idx)
            if (max_length is not None
                    and previous_seq_length + new_seq_length > max_length):
                return max_length - new_seq_length
            return previous_seq_length
        DynamicCache.get_usable_length = _get_usable_length


def load_sundial(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, set_seed
    set_seed(1)
    _patch_dynamic_cache_seen_tokens()
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.to(device).eval()
    return model


def predict_sundial(model, x: torch.Tensor, horizon: int, device: str) -> torch.Tensor:
    from contextlib import nullcontext
    from gluonts.transform import LastValueImputation
    rows = []
    for row in x[:, :, 0].numpy():
        rows.append(LastValueImputation()(row) if np.isnan(row).any() else row)
    seqs = torch.as_tensor(np.vstack(rows), dtype=torch.float32, device=device)
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if _is_cuda(device) else nullcontext())
    with amp:
        samples = model.generate(
            seqs, max_new_tokens=horizon, revin=True,
            num_samples=SUNDIAL_NUM_SAMPLES,
        )                                              # (B, S, H)
    samples = samples[:, :, :horizon].to(torch.float32)
    return torch.median(samples, dim=1).values         # (B, H)


def load_timemoe(model_id: str, device: str):
    from transformers import AutoModelForCausalLM
    _patch_dynamic_cache_seen_tokens()
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.to(device).eval()
    return model


def predict_timemoe(model, x: torch.Tensor, horizon: int, device: str) -> torch.Tensor:
    seqs = x[:, :, 0].to(device, non_blocking=True)
    mean = seqs.mean(dim=-1, keepdim=True)
    std = seqs.std(dim=-1, keepdim=True)
    normed = (seqs - mean) / (std + 1e-8)
    out = model.generate(normed, max_new_tokens=horizon)            # (B, C+H)
    preds = out[:, -horizon:].to(torch.float32)
    return preds * std + mean                                       # (B, H)


def load_toto(model_id: str, device: str):
    from toto2 import Toto2Model
    model = Toto2Model.from_pretrained(model_id)
    model.to(device).eval()
    return model


def predict_toto(model, x: torch.Tensor, horizon: int, device: str) -> torch.Tensor:
    """Quantile forecast (Toto 2.0); return the per-step median (B, horizon).

    Each of the B univariate series is fed as one batch element with a single
    variate, so there is no cross-series attention bleed to guard against. The
    2.0 `forecast()` returns quantiles of shape (9, batch, n_variates, horizon);
    we take the 0.5 row (index 4) and squeeze the singleton variate.

    NOTE: the `toto2` API shapes below (target/target_mask/series_ids and the
    `forecast()` kwargs) were reconstructed from Datadog's docs, not the package
    source — verify against the installed `toto2` on the server before a full run.
    """
    seqs = x[:, :, 0].to(device, non_blocking=True)        # (B, L)
    b, length = seqs.shape
    # The patcher requires L to be a multiple of TOTO_PATCH_SIZE. Left-pad the
    # oldest steps so the genuine context (and the forecast origin at the right)
    # is untouched, then mask the pad as missing so it can't affect the forecast.
    pad = (-length) % TOTO_PATCH_SIZE
    if pad:
        seqs = torch.cat([seqs.new_zeros((b, pad)), seqs], dim=1)  # (B, L+pad)
    target = seqs.unsqueeze(1)                             # (B, 1, L')
    target_mask = torch.ones_like(target, dtype=torch.bool)
    if pad:
        target_mask[:, :, :pad] = False
    series_ids = torch.arange(b, device=device, dtype=torch.long).unsqueeze(1)  # (B, 1)
    quantiles = model.forecast(
        {"target": target, "target_mask": target_mask, "series_ids": series_ids},
        horizon=horizon,
        has_missing_values=bool(pad),
    )                                                      # (9, B, 1, horizon)
    median = quantiles[TOTO_MEDIAN_QUANTILE_IDX, :, 0, :horizon]
    return median.to(torch.float32)                        # (B, horizon)


def load_flowstate(model_id: str, device: str):
    from tsfm_public import FlowStateForPrediction
    model = FlowStateForPrediction.from_pretrained(
        model_id, revision=FLOWSTATE_REVISION).to(device)
    model.eval()
    return model


def predict_flowstate(model, x: torch.Tensor, horizon: int, device: str) -> torch.Tensor:
    """Quantile forecast; return the per-step median (B, horizon).

    ``x`` is (B, W, 1) == (batch, context, channels), which is exactly FlowState's
    ``batch_first=True`` layout, so it's passed through unreshaped. The median is
    read from ``quantile_outputs`` (B, Q=9, H, C), index 4 — NOT from
    ``prediction_outputs``: on current tsfm_public the r1.1 config's
    ``prediction_type='quantile'`` is deprecated and coerced to ``'mean'``, so
    ``prediction_outputs`` is a 3-D quantile-weighted MEAN there, not the median
    every other wrapper contributes. Older tsfm_public (no ``quantile_outputs``
    attr) exposed the quantiles as a 4-D ``prediction_outputs`` (the model card's
    ``(32, 9, 48, 1)``) — kept as the fallback; a 3-D output on that path is a
    true point forecast.
    """
    series = x.to(device, non_blocking=True)               # (B, W, 1)
    out = model(
        series,
        scale_factor=FLOWSTATE_SCALE_FACTOR,
        prediction_length=horizon,
        batch_first=True,
    )
    qf = getattr(out, "quantile_outputs", None)
    if qf is not None:                                     # (B, Q, H, C)
        med = qf[:, FLOWSTATE_MEDIAN_QUANTILE_IDX, :horizon, 0]
    else:
        pf = out.prediction_outputs
        if pf.dim() == 4:                                  # (B, Q, H, C)
            med = pf[:, FLOWSTATE_MEDIAN_QUANTILE_IDX, :horizon, 0]
        else:                                              # (B, H, C)
            med = pf[:, :horizon, 0]
    return med.to(torch.float32)                           # (B, horizon)


def load_tirex(model_id: str, device: str):
    # TiRex2 lives in the `tirex2` package (TiRex-1 was `tirex`); load_model takes
    # the device up front and handles placement internally. Its API accepts only
    # bare device types ("cuda"/"cpu"); workers already mask each process to one
    # visible GPU, so "cuda" still selects the intended card.
    from experiments.tirex_compat import configure_tirex_backend
    from tirex2 import load_model
    tirex_device = "cuda" if _is_cuda(device) else "cpu"
    backend = configure_tirex_backend(tirex_device)
    print(f"  TiRex FlashRNN backend: {backend}", flush=True)
    return load_model(model_id, device=tirex_device)


def predict_tirex(model, x: torch.Tensor, horizon: int, device: str) -> torch.Tensor:
    """Quantile forecast (TiRex2); return the per-step median (B, horizon).

    TiRex2's API differs from TiRex-1: it takes a LIST of ``TimeseriesType``
    objects (one per series, ``target`` shaped (n_variates, length)) rather than
    a batched ``context=`` tensor, and each returned forecast is shaped
    (n_targets, 9 quantiles, prediction_length) — the quantile axis is the MIDDLE
    one now (TiRex-1 was (B, horizon, 9)). The 9 levels are still [0.1..0.9], so
    index 4 (0.5) is the point forecast. ``output_type="numpy"`` is the documented
    return format; we convert back to a `device` torch tensor because
    forecast_window scatters the median into a `device`-side buffer. TiRex2 caps
    one forecast call at ``model.future_len`` (320 for the current checkpoint),
    so longer horizons roll forward in chunks using each preceding median as
    context.
    """
    seqs = x[:, :, 0].to(device, non_blocking=True)        # (B, L)
    from experiments.tirex_compat import forecast_tirex_medians
    medians = forecast_tirex_medians(
        model, seqs, horizon, TIREX_MEDIAN_QUANTILE_IDX
    )                                                       # (B, horizon)
    return torch.as_tensor(medians, dtype=torch.float32, device=device)


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
        return load_timesfm_official(model_id)
    if family == "patchtst_fm":
        return load_patchtst_fm(model_id, device)
    if family == "sundial":
        return load_sundial(model_id, device)
    if family == "timemoe":
        return load_timemoe(model_id, device)
    if family == "toto":
        return load_toto(model_id, device)
    if family == "flowstate":
        return load_flowstate(model_id, device)
    if family == "tirex":
        return load_tirex(model_id, device)
    raise ValueError(f"Unknown model family: {family}")


def _forecast_uniform(
    family: str,
    base,
    model_id: str,
    x_all: torch.Tensor,            # (n, width, 1) on CPU
    width: int,
    horizon: int,
    batch_size: int,
    device: str,
    dynamic_batching: bool = False,
    max_batch_size: Optional[int] = None,
    tuned_batch_sizes: Optional[dict] = None,
) -> torch.Tensor:
    """Forecast a uniform-width batch; return median (n, horizon) on `device`.

    Moirai rebuilds its forecast object against ``width``. TimesFM keeps one
    official checkpoint handle and its public forecast method performs the
    recipe's required compile for each batch."""
    n = int(x_all.shape[0])
    max_batch_size = max(batch_size, int(max_batch_size or batch_size))
    # Kept as a compatibility argument for older callers; upward probing and the
    # cross-shard tuned-size cache are intentionally disabled. Start immediately
    # at the context-scaled heuristic size and only reduce after a real OOM.
    del tuned_batch_sizes
    current_batch = min(batch_size, max_batch_size, n)
    if family == "moirai":
        runner = _build_moirai(base, horizon, width, device)
    else:
        runner = base

    medians: List[torch.Tensor] = []
    start = 0
    while start < n:
        size = min(current_batch, n - start)
        xb = x_all[start:start + size]
        try:
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
            elif family == "sundial":
                m = predict_sundial(runner, xb, horizon, device)
            elif family == "timemoe":
                m = predict_timemoe(runner, xb, horizon, device)
            elif family == "toto":
                m = predict_toto(runner, xb, horizon, device)
            elif family == "flowstate":
                m = predict_flowstate(runner, xb, horizon, device)
            elif family == "tirex":
                m = predict_tirex(runner, xb, horizon, device)
            else:
                raise ValueError(f"Unknown model family: {family}")
        except Exception as exc:
            if (not dynamic_batching or not _is_cuda_oom(exc) or size <= 1):
                raise
            del exc
            torch.cuda.empty_cache()
            current_batch = max(1, size // 2)
            print(Fore.YELLOW + f"  dynamic batch [{family}] W={width}: "
                  + f"OOM at {size}; using {current_batch}" + Fore.RESET,
                  flush=True)
            continue
        medians.append(m)
        start += size

    if family == "moirai":
        del runner

    return torch.cat(medians, dim=0)                   # (n, H)


def forecast_window(
    family: str,
    base,
    model_id: str,
    contexts: np.ndarray,
    window: int,
    real_lengths: np.ndarray,
    horizon: int,
    batch_size: int,
    device: str,
    dynamic_batching: bool = False,
    batch_reference_context: int = DYNAMIC_BATCH_REFERENCE_CONTEXT,
    max_batch_size: int = DYNAMIC_BATCH_MAX_SIZE,
    tuned_batch_sizes: Optional[dict] = None,
) -> torch.Tensor:
    """Forecast the horizon for every series using only its *genuine* context.

    Each series is fed its last ``min(window, real_len)`` genuine samples — the
    pool left-pads short series, so the genuine signal is the suffix and slicing
    ``[-L:]`` never touches the artificial padding. Variable-length models receive
    exactly those samples; PatchTST-FM is NaN-padded to its fixed context inside
    ``predict_patchtst_fm``. Effective lengths are bucketed *down* to WINDOW_GRID
    so models that recompile per context width (timesfm, moirai) build at most one
    runner per distinct grid width. The curve flattens past real_len and the
    labels are NaN-free.

    Returns the median forecast (N, horizon) on `device`.
    """
    n = contexts.shape[0]
    eff = np.minimum(int(window), np.asarray(real_lengths))      # genuine width / series
    grid = np.asarray(sorted(set(WINDOW_GRID)))
    # Largest grid width <= eff; per-series min() guards the rare eff < grid[0].
    eff_buck = np.minimum(
        eff, grid[np.clip(np.searchsorted(grid, eff, side="right") - 1, 0, None)]
    )

    out = torch.empty((n, horizon), device=device, dtype=torch.float32)
    for L in np.unique(eff_buck):
        idx = np.flatnonzero(eff_buck == L)
        x_grp = torch.from_numpy(
            np.ascontiguousarray(contexts[idx, -int(L):])).unsqueeze(-1)  # (g, L, 1)
        effective_batch = batch_size_for_context(
            batch_size, int(L), len(idx), dynamic_batching,
            batch_reference_context, max_batch_size)
        med = _forecast_uniform(
            family, base, model_id, x_grp, int(L), horizon, effective_batch,
            device, dynamic_batching, max_batch_size, tuned_batch_sizes)
        out[torch.as_tensor(idx, device=device, dtype=torch.long)] = med

    return out                                          # (N, H)


def batch_size_for_context(base_batch_size: int, context_length: int,
                           n_samples: int, dynamic_batching: bool,
                           reference_context: int,
                           max_batch_size: int) -> int:
    """Token-budgeted initial microbatch for a Stage-1 context width."""
    size = max(1, int(base_batch_size))
    if dynamic_batching:
        scale = max(1, int(reference_context) // max(1, int(context_length)))
        size = min(max(size, int(max_batch_size)), size * scale)
    return max(1, min(size, int(n_samples)))


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# ==============================================================================
#  GPU WORKER  (one process per device, drains a shard queue)
# ==============================================================================

def _model_dir(display: str) -> str:
    # Key the output folder on the unique display name (e.g. "Chronos2-Small"),
    # not the family — families are shared across model variants (e.g.
    # chronos-2-small and chronos-2-synth both map to family "chronos2") and
    # would collide on disk.
    return os.path.join(OUTPUT_ROOT, display)


def _shard_dir(model_dir: str, shard_id: int) -> str:
    return os.path.join(model_dir, "shards", f"shard_{shard_id:03d}")


def gpu_worker(
    worker_id: int,
    device: str,
    shard_queue: "mp.Queue",
    result_queue: "mp.Queue",
    model_dir: str,
    contexts_path: str,
    targets_path: str,
    real_lengths_path: str,
    model_id: str,
    family: str,
    max_horizon: int,
    batch_size: int,
    win_indices: List[int],
    dynamic_batching: bool,
    batch_reference_context: int,
    max_batch_size: int,
) -> None:
    """Load the TSFM once, then label series-shards until the queue drains.

    Each window is forecast once at ``max_horizon``; per-(window, horizon)
    errors are computed by slicing the prediction at every h in HORIZON_GRID.
    """
    dev_label = device
    if _is_cuda(device):
        # Pin this worker to its physical GPU *before* CUDA initializes. Some
        # backends (e.g. TimesFM) hardcode cuda:0 and ignore the requested
        # device, which makes every worker pile onto GPU 0 while the others sit
        # idle. Masking to one visible GPU forces correct placement regardless;
        # inside this process that GPU is then always addressed as cuda:0.
        os.environ["CUDA_VISIBLE_DEVICES"] = _physical_gpu_id(device)
        device = "cuda:0"
        torch.cuda.set_device(torch.device(device))
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    n_win = len(WINDOW_GRID)
    n_h = len(HORIZON_GRID)
    tb_shown = False        # print a full traceback only for the first shard failure

    try:
        base = setup_model(family, model_id, device)
        print(Fore.CYAN + f"  [{dev_label}] worker {worker_id} ready." + Fore.RESET)
    except Exception as exc:
        print(Fore.RED + f"  [{dev_label}] worker {worker_id} model load failed: "
              + f"{type(exc).__name__}: {exc}" + Fore.RESET)
        while True:
            spec = shard_queue.get()
            if spec is None:
                break
            result_queue.put({"shard_id": spec[0], "status": "model_load_failed"})
        return

    # Open each read-only array once per worker. Reopening three .npy memmaps for
    # every shard added filesystem/metadata overhead and repeatedly rebuilt the
    # mapping objects. Slices are still copied before torch conversion because
    # the underlying mappings are read-only.
    contexts_mm = np.load(contexts_path, mmap_mode="r")
    targets_mm = np.load(targets_path, mmap_mode="r")
    real_lengths_mm = np.load(real_lengths_path, mmap_mode="r")

    while True:
        spec = shard_queue.get()
        if spec is None:
            break
        shard_id, start, end = spec
        t0 = time.perf_counter()
        try:
            # Read-only memmap slices must be copied before torch.from_numpy.
            ctx = np.array(contexts_mm[start:end], copy=True)
            tgt = np.array(targets_mm[start:end], copy=True)
            real_len = np.array(real_lengths_mm[start:end], copy=True)  # (B,)
            tgt_t = torch.from_numpy(tgt).to(device)        # (B, MAX_HORIZON)

            cm = np.full((end - start, n_win, n_h), np.nan, dtype=np.float32)
            cs = np.full((end - start, n_win, n_h), np.nan, dtype=np.float32)
            for w_idx in win_indices:
                w = WINDOW_GRID[w_idx]
                if family == "sundial" and w > SUNDIAL_MAX_CONTEXT:
                    continue
                if family == "timemoe" and w + max_horizon > TIMEMOE_MAX_TOTAL:
                    continue
                if family == "toto" and w > TOTO_MAX_CONTEXT:
                    continue
                if family == "flowstate" and w > FLOWSTATE_MAX_CONTEXT:
                    continue
                if family == "tirex" and w > TIREX_MAX_CONTEXT:
                    continue
                medians = forecast_window(
                    family, base, model_id, ctx, w, real_len, max_horizon,
                    batch_size, device, dynamic_batching,
                    batch_reference_context, max_batch_size,
                    None)                                   # (B, MAX_HORIZON)
                for h_idx, h in enumerate(HORIZON_GRID):
                    err = medians[:, :h] - tgt_t[:, :h]
                    cm[:, w_idx, h_idx] = err.abs().mean(dim=1).cpu().numpy()
                    cs[:, w_idx, h_idx] = err.pow(2).mean(dim=1).cpu().numpy()
                # No NaN-masking for short series: forecast_window already feeds
                # each series only its min(w, real_len) genuine samples, so the
                # error is real and finite. The curve flattens past real_len.

            sd = _shard_dir(model_dir, shard_id)
            os.makedirs(sd, exist_ok=True)
            np.save(os.path.join(sd, "curves_mae.npy"), cm)
            np.save(os.path.join(sd, "curves_mse.npy"), cs)
            with open(os.path.join(sd, "done.json"), "w") as f:
                json.dump({"shard_id": shard_id, "start": start, "end": end,
                           "window_indices": win_indices,
                           "horizon_grid": HORIZON_GRID,
                           "synthetic_pool_signature": synthetic_pool_signature(),
                           "inference_recipe": inference_recipe(family)}, f)

            elapsed = time.perf_counter() - t0
            result_queue.put({"shard_id": shard_id, "status": "ok",
                              "elapsed": elapsed})
            print(Fore.YELLOW + f"  [{dev_label}] shard {shard_id:03d} "
                  + f"[{start}:{end}] done ({elapsed:.1f}s)" + Fore.RESET)
        except Exception as exc:
            print(Fore.RED + f"  [{dev_label}] shard {shard_id:03d} FAILED: "
                  + f"{type(exc).__name__}: {exc}" + Fore.RESET)
            if not tb_shown:
                # First failure on this worker: dump the full traceback so the
                # offending modeling line (not just the message) is in the log.
                print(Fore.RED + f"  [{dev_label}] shard {shard_id:03d} "
                      + "traceback (shown once per worker):" + Fore.RESET)
                traceback.print_exc()
                tb_shown = True
            result_queue.put({"shard_id": shard_id,
                              "status": f"error:{type(exc).__name__}"})
    del contexts_mm, targets_mm, real_lengths_mm, base
    if _is_cuda(device):
        # The persistent model is leaving scope: this is an appropriate allocator
        # teardown boundary, unlike the previous per-shard flush.
        torch.cuda.empty_cache()
    print(Fore.CYAN + f"  [{dev_label}] worker {worker_id} exited." + Fore.RESET)


# ==============================================================================
#  SHARD MERGE + SANITY
# ==============================================================================

def merge_shards(
    model_dir: str, n_series: int, window_indices: List[int], family: str
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Concatenate completed shard curves into top-level (N, n_win, n_h) arrays.

    Missing/incomplete shards leave NaN rows so curves stay index-aligned with
    contexts.npy. Returns (curves_mae, curves_mse, n_shards_done).
    """
    n_win = len(WINDOW_GRID)
    n_h = len(HORIZON_GRID)
    cm = np.full((n_series, n_win, n_h), np.nan, dtype=np.float32)
    cs = np.full((n_series, n_win, n_h), np.nan, dtype=np.float32)

    sdir = os.path.join(model_dir, "shards")
    n_done = 0
    if os.path.isdir(sdir):
        for name in sorted(os.listdir(sdir)):
            dpath = os.path.join(sdir, name, "done.json")
            if not os.path.isfile(dpath):
                continue
            with open(dpath) as f:
                d = json.load(f)
            if (d.get("window_indices") != window_indices
                    or d.get("horizon_grid") != HORIZON_GRID
                    or d.get("synthetic_pool_signature")
                    != synthetic_pool_signature()
                    or d.get("inference_recipe") != inference_recipe(family)):
                continue
            s, e = d["start"], d["end"]
            shard_mae = np.load(os.path.join(sdir, name, "curves_mae.npy"))
            shard_mse = np.load(os.path.join(sdir, name, "curves_mse.npy"))
            if shard_mae.shape[1:] == (n_win, n_h):
                cm[s:e] = shard_mae
                cs[s:e] = shard_mse
            elif (shard_mae.shape[1:] == (len(BASE_WINDOW_GRID), n_h)
                  and shard_mse.shape == shard_mae.shape
                  and all(idx < len(BASE_WINDOW_GRID) for idx in window_indices)):
                # The off-grid native-cap union was appended without shifting any
                # historical index. Reuse pre-union shards for families that do
                # not request the appended action; Sundial's changed win_indices
                # deliberately fail this branch and are recomputed.
                cm[s:e, :len(BASE_WINDOW_GRID)] = shard_mae
                cs[s:e, :len(BASE_WINDOW_GRID)] = shard_mse
            else:
                continue
            n_done += 1

    # Shards use the global union grid so their schema is spawn-stable.  The
    # public per-model arrays expose only that family's supported grid; this is
    # what lets smaller-context predictors be genuine smaller classification
    # problems instead of receiving permanently invalid output positions.
    cm_model = cm[:, window_indices, :]
    cs_model = cs[:, window_indices, :]
    np.save(os.path.join(model_dir, "curves_mae.npy"), cm_model)
    np.save(os.path.join(model_dir, "curves_mse.npy"), cs_model)
    return cm_model, cs_model, n_done


def _print_data_sanity(
    curves_mae: np.ndarray, n_segments: np.ndarray, family: str,
    window_grid: List[int],
) -> None:
    """Report whether the generated data is actually context-sensitive."""
    valid = ~np.isnan(curves_mae).any(axis=(1, 2))
    if not valid.any():
        print(Fore.RED + "  No completed curves to summarize." + Fore.RESET)
        return
    cm_v = curves_mae[valid]                                # (V, n_win, n_h)
    n_win = cm_v.shape[1]
    win_arr = np.array(window_grid)
    print(Fore.GREEN + f"  Data sanity [{family}] (MAE curves):" + Fore.RESET)
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
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                   help=("Force device. --device cuda fails fast if CUDA is not "
                         "available; without this option the standalone builder "
                         "uses all CUDA devices and otherwise falls back to CPU."))
    p.add_argument("--n-series", type=int, default=N_SERIES,
                   help="Number of synthetic series.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help="Safe TSFM batch size at --batch-reference-context.")
    p.add_argument("--no-dynamic-batching", action="store_true",
                   help="Disable CUDA batch autotuning and use --batch-size exactly.")
    p.add_argument("--batch-reference-context", type=int,
                   default=DYNAMIC_BATCH_REFERENCE_CONTEXT,
                   help="Context width at which --batch-size is the initial batch.")
    p.add_argument("--max-batch-size", type=int, default=DYNAMIC_BATCH_MAX_SIZE,
                   help="Upper bound for dynamically tuned microbatches.")
    p.add_argument("--shard-size", type=int, default=SHARD_SIZE,
                   help="Series per shard (parallelism + resume granularity).")
    p.add_argument("--windows", type=int, nargs="+", default=None,
                   help="Subset of WINDOW_GRID to run (quick smoke tests).")
    p.add_argument("--model-idx", type=int, default=None,
                   help=(
                       "Run only this model index (0-based). "
                       "Default: run all models in sequence. "
                       f"Available: {[f'{i}={m[2]}' for i, m in enumerate(MODELS)]}"
                   ))
    p.add_argument(
        "--regenerate-pool", action="store_true",
        help=("Overwrite an incompatible cached synthetic pool and invalidate "
              "all labeling shards through the versioned pool signature. "
              "master_run_all forwards this when stage 1 is forced."),
    )
    return p.parse_args()


def main() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()

    if args.model_idx is not None and (args.model_idx < 0 or args.model_idx >= len(MODELS)):
        raise ValueError(
            f"--model-idx {args.model_idx} out of range (0-{len(MODELS)-1}).")

    devices = resolve_devices(args.device)
    requested_grid = [w for w in WINDOW_GRID
                      if args.windows is None or w in args.windows]
    if any(w > MAX_WINDOW for w in requested_grid):
        raise ValueError(f"WINDOW_GRID exceeds MAX_WINDOW={MAX_WINDOW}.")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # ---------- Series pool (generated once, shared across all models) ------
    contexts_path = os.path.join(OUTPUT_ROOT, "contexts.npy")
    targets_path  = os.path.join(OUTPUT_ROOT, "targets.npy")
    nseg_path     = os.path.join(OUTPUT_ROOT, "n_segments.npy")
    rlen_path     = os.path.join(OUTPUT_ROOT, "real_lengths.npy")
    pool_meta_path = os.path.join(OUTPUT_ROOT, "meta.json")

    cached_signature = None
    if os.path.isfile(pool_meta_path):
        try:
            with open(pool_meta_path) as handle:
                cached_signature = json.load(handle).get(
                    "synthetic_pool_signature")
        except (OSError, json.JSONDecodeError):
            cached_signature = None
    pool_compatible = cached_signature == synthetic_pool_signature()
    if os.path.isfile(contexts_path) and not pool_compatible:
        if not args.regenerate_pool:
            raise RuntimeError(
                "Existing synthetic pool predates the long-period generator "
                f"{synthetic_pool_signature()}. Re-run stage 1 with "
                "--regenerate-pool (or master_run_all --force 1). Labeling "
                "shards will be recomputed automatically."
            )
        print(Fore.YELLOW
              + "Regenerating synthetic pool: generator signature changed."
              + Fore.RESET)

    if os.path.isfile(contexts_path) and pool_compatible:
        pool_shape = np.load(contexts_path, mmap_mode="r").shape
        if len(pool_shape) != 2 or pool_shape[1] != MAX_WINDOW:
            raise RuntimeError(
                f"Existing synthetic pool has shape {pool_shape}, but this run "
                f"requires (*, {MAX_WINDOW}) for the long-context grid. Start "
                "the recomputation in an empty output root (or archive/remove "
                f"{OUTPUT_ROOT}) so old 8k labels cannot be mixed with the new run."
            )
        n_series = int(pool_shape[0])
        n_segments = np.load(nseg_path)
        print(Fore.CYAN + f"Found existing pool: {n_series} series" + Fore.RESET)
    else:
        n_series = args.n_series
        contexts, targets, n_segments, real_lengths = generate_dataset(n_series, SEED)
        np.save(contexts_path, contexts)
        np.save(targets_path, targets)
        np.save(nseg_path, n_segments)
        np.save(rlen_path, real_lengths)
        n_padded = int((real_lengths < MAX_WINDOW).sum())
        del contexts, targets
        with open(pool_meta_path, "w") as f:
            json.dump({
                "n_series": int(n_series), "seed": SEED,
                "max_window": MAX_WINDOW, "max_horizon": MAX_HORIZON,
                "window_grid": WINDOW_GRID, "horizon_grid": HORIZON_GRID,
                "period_min": PERIOD_MIN,
                "period_core_max": PERIOD_CORE_MAX,
                "period_max": PERIOD_MAX,
                "long_period_probability": LONG_PERIOD_PROB,
                "synthetic_pool_signature": synthetic_pool_signature(),
                "n_segment_choices": N_SEGMENT_CHOICES,
                "min_segment_len": MIN_SEGMENT_LEN,
                "level_shift_std": LEVEL_SHIFT_STD,
                "seasonal_shift_prob": SEASONAL_SHIFT_PROB,
                "transition_max": TRANSITION_MAX,
                "spike_prob": SPIKE_PROB,
                "spike_rate_range": list(SPIKE_RATE_RANGE),
                "spike_std": SPIKE_STD,
                "pad_frac": PAD_FRAC, "min_real_len": MIN_REAL_LEN,
                "n_padded": n_padded,
                "created": datetime.now().isoformat(timespec="seconds"),
            }, f, indent=2)
        print(Fore.GREEN + f"Generated synthetic pool: {n_series} series "
              + f"({n_padded} short/padded)" + Fore.RESET)

    # Defensive: an older pool may predate real_lengths.npy. Treat all series as
    # full-length (no padding) so labeling stays backward-compatible.
    if not os.path.isfile(rlen_path):
        np.save(rlen_path, np.full((n_series,), MAX_WINDOW, dtype=np.int32))

    # ---------- Label with each model (or just the one requested) -----------
    models_to_run = MODELS if args.model_idx is None else [MODELS[args.model_idx]]
    print(Fore.CYAN
          + f"Models to label: {[m[2] for m in models_to_run]}  |  devices={devices}"
          + Fore.RESET)

    for model_id, family, display in models_to_run:
        family_grid = window_grid_for_family(family)
        grid = [w for w in family_grid if w in requested_grid]
        if not grid:
            raise ValueError(
                f"Requested --windows {args.windows} leaves no supported window "
                f"for {display} (family grid: {family_grid}).")
        win_indices = [WINDOW_GRID.index(w) for w in grid]
        model_dir = _model_dir(display)
        os.makedirs(model_dir, exist_ok=True)
        print(Fore.CYAN + f"\n── {display} ({model_id}) ──" + Fore.RESET)
        print(Fore.CYAN + f"   windows={grid}  shard_size={args.shard_size}" + Fore.RESET)

        pending = []
        n_completed = 0
        for shard_id, start in enumerate(range(0, n_series, args.shard_size)):
            end = min(start + args.shard_size, n_series)
            done_path = os.path.join(_shard_dir(model_dir, shard_id), "done.json")
            cached_ok = False
            if os.path.isfile(done_path):
                try:
                    with open(done_path) as f:
                        done_meta = json.load(f)
                    cached_ok = (
                        done_meta.get("window_indices") == win_indices
                        and done_meta.get("horizon_grid") == HORIZON_GRID
                        and done_meta.get("synthetic_pool_signature")
                        == synthetic_pool_signature()
                        and done_meta.get("inference_recipe") == inference_recipe(family)
                    )
                except (OSError, json.JSONDecodeError):
                    cached_ok = False
            if cached_ok:
                n_completed += 1
            else:
                pending.append((shard_id, start, end))
        n_total_shards = n_completed + len(pending)
        pct_done = 100.0 * n_completed / n_total_shards if n_total_shards else 100.0
        print(Fore.CYAN + f"   Shards: {n_completed} cached, {len(pending)} pending "
              + f"({pct_done:.1f}% complete)" + Fore.RESET)

        if pending:
            ctx = mp.get_context("spawn")
            shard_queue = ctx.Queue()
            result_queue = ctx.Queue()
            for spec in pending:
                shard_queue.put(spec)
            for _ in devices:
                shard_queue.put(None)

            workers = []
            for i, dev in enumerate(devices):
                p = ctx.Process(
                    target=gpu_worker,
                    args=(i, dev, shard_queue, result_queue, model_dir,
                          contexts_path, targets_path, rlen_path,
                          model_id, family,
                          MAX_HORIZON, args.batch_size, win_indices,
                          not args.no_dynamic_batching,
                          args.batch_reference_context, args.max_batch_size),
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
                        print(Fore.RED + f"   All workers died with "
                              + f"{n_received}/{len(pending)} shards." + Fore.RESET)
                        break
                    continue
                n_received += 1

            for p in workers:
                p.join(timeout=120)
                if p.is_alive():
                    print(Fore.RED + f"   Worker {p.name} hung — terminating." + Fore.RESET)
                    p.terminate()
                    p.join(timeout=10)
            print(Fore.MAGENTA + f"   Labeling wall-clock: {time.perf_counter() - t0:.1f}s"
                  + Fore.RESET)

        curves_mae, _, n_done = merge_shards(
            model_dir, n_series, win_indices, family)
        total_shards = (n_series + args.shard_size - 1) // args.shard_size

        with open(os.path.join(model_dir, "meta.json"), "w") as f:
            json.dump({
                "model_id": model_id, "model_family": family, "model_display": display,
                "inference_recipe": inference_recipe(family),
                "synthetic_pool_signature": synthetic_pool_signature(),
                "window_indices": win_indices,
                "shards_done": n_done, "shards_total": total_shards,
                "devices": devices, "shard_size": args.shard_size,
                "batching": {
                    "dynamic": not args.no_dynamic_batching,
                    "base_batch_size": args.batch_size,
                    "reference_context": args.batch_reference_context,
                    "max_batch_size": args.max_batch_size,
                },
                "created": datetime.now().isoformat(timespec="seconds"),
                # Pool-level keys repeated here so predict_context_length.py can
                # use this subdir as --dataset-dir without reading the parent meta.
                # Predictor input stays at 8k for <=8k TSFMs and expands to the
                # long grid only when the labeled TSFM can use it.
                "max_window": (15360 if max(grid) > 8192 else 8192),
                "pool_max_window": MAX_WINDOW,
                "model_context_limit": models_config.context_limit(family),
                "window_grid": grid, "horizon_grid": HORIZON_GRID,
            }, f, indent=2)

        _print_data_sanity(curves_mae, n_segments, family, grid)
        idx = MODELS.index((model_id, family, display))
        if n_done == 0:
            # Nothing was labeled at all — almost always a model-load failure in
            # the workers (see the "model load failed" line above). The curves
            # file is entirely NaN, so the predictor would otherwise die later
            # with a misleading "No labeled series in dataset". Fail here, on the
            # build stage, so the error points at the right log.
            raise RuntimeError(
                f"{display}: 0/{total_shards} shards labeled — curves are all "
                f"NaN. Check the worker model-load error above; re-run with "
                f"--model-idx {idx} once fixed.")
        if n_done < total_shards:
            print(Fore.YELLOW + f"   {total_shards - n_done} shard(s) incomplete — "
                  + f"re-run with --model-idx {idx}" + Fore.RESET)

    print(Fore.GREEN + f"\nAll done. Output root: {OUTPUT_ROOT}" + Fore.RESET)


if __name__ == "__main__":
    main()
