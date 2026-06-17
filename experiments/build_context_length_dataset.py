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

# Smoke-test mode (PREDICTCSL_TEST=1, set by experiments/run_all.py --test):
# collapse the ablation grid to just its smallest and largest window so the
# whole pipeline can be exercised end-to-end in minutes. Resolved from the env
# at import time on purpose — `mp spawn` workers re-import this module, so a
# main()-level override would not reach them, but an inherited env var does.
if os.environ.get("PREDICTCSL_TEST") == "1":
    WINDOW_GRID = [WINDOW_GRID[0], WINDOW_GRID[-1]]

# -- Horizon grid (forecast lengths labeled per series) -----------------------
HORIZON_GRID = [16, 32, 64, 128, 512, 1024]
MAX_HORIZON  = max(HORIZON_GRID)

# -- Dataset size -------------------------------------------------------------
N_SERIES   = 50_000
SEED       = 42
BATCH_SIZE = 32              # TSFM inference batch size
SHARD_SIZE = 500             # series per shard (parallelism + resume granularity)

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
MODELS = [
    ("autogluon/chronos-2-small",       "chronos2",     "Chronos2-Small"),
    ("amazon/chronos-bolt-small",       "chronos_bolt", "ChronosBolt-Small"),
    ("Salesforce/moirai-2.0-R-small",   "moirai",       "Moirai2-Small"),
    ("google/timesfm-2.5-200m-pytorch", "timesfm",      "TimesFM2.5-200M"),
    ("ibm-research/patchtst-fm-r1",     "patchtst_fm",  "PatchTST-FM-R1"),
    ("thuml/sundial-base-128m",         "sundial",      "Sundial-Base-128M"),
    ("Maple728/TimeMoE-200M",           "timemoe",      "TimeMoE-200M"),
    # Distinct checkpoint, same chronos2 architecture/loader (load_chronos2) —
    # only the weights differ. Appended (not inserted) so existing --model-idx
    # positions stay stable for direct CLI use.
    ("autogluon/chronos-2-synth",       "chronos2",     "Chronos2-Synth"),
    # Same chronos_bolt loader as ChronosBolt-Small, just the larger checkpoint.
    # Appended (not inserted) to preserve existing --model-idx positions.
    ("amazon/chronos-bolt-base",        "chronos_bolt", "ChronosBolt-Base"),
    # Datadog Toto 2.0 — probabilistic decoder TSFM. 2.0 uses a new module
    # (`toto2.Toto2Model`) with a quantile output head (not the 1.0 sample-based
    # `TotoForecaster`). 313m is the size tier closest to the other Small/200M
    # models here. Needs the `toto2` package importable in the active env.
    ("Datadog/Toto-2.0-313m",           "toto",         "Toto-2.0-313m"),
    # IBM Granite FlowState — SSM-encoder + functional-basis-decoder TSFM (<10M
    # params, quantile output). Loaded via tsfm_public (same package as
    # PatchTST-FM); load_flowstate pins revision r1.1 (the GIFT-Eval checkpoint).
    ("ibm-granite/granite-timeseries-flowstate-r1", "flowstate", "FlowState-R1"),
    # NX-AI TiRex — xLSTM-based zero-shot TSFM (~35M params, quantile output).
    # Recurrent/linear-cost like FlowState, so it's variable-length (genuine-only
    # suffix, no NaN-pad). Loaded via the `tirex` package (pip install tirex-ts).
    ("NX-AI/TiRex",                     "tirex",        "TiRex"),
]

# -- Quantile bookkeeping (per model family) ----------------------------------
MOIRAI2_MEDIAN_IDX              = 4
PATCHTST_FM_MEDIAN_QUANTILE_IDX = 49
SUNDIAL_NUM_SAMPLES             = 20
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
# sampling rate, so 1.0 (treat input cadence as the base) is the neutral choice.
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
# trustworthy (curve flattens past it), like SUNDIAL/FLOWSTATE. 2048 is TiRex's
# pretraining context length (per the paper, arXiv:2505.23719).
TIREX_MAX_CONTEXT               = 2048

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
    if not torch.cuda.is_available():
        return ["cpu"]
    if DEVICES is not None:
        return list(DEVICES)
    n = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(n)] if n > 0 else ["cpu"]


# ==============================================================================
#  SYNTHETIC GENERATOR
# ==============================================================================

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
    log_lo, log_hi = math.log(PERIOD_MIN), math.log(PERIOD_MAX)
    periods = np.exp(rng.uniform(log_lo, log_hi, size=n_periodic))
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
        dominant_period = float(
            np.exp(rng.uniform(math.log(PERIOD_MIN), math.log(PERIOD_MAX))))

        parts: List[np.ndarray] = []
        level = 0.0
        for i in range(n_seg):
            seg_len = bounds[i + 1] - bounds[i]
            seg = _generate_segment(rng, seg_len, force_period=dominant_period)
            seg = seg * float(rng.uniform(*SEG_AMP_RANGE))
            if i > 0:
                level += float(rng.normal(0.0, LEVEL_SHIFT_STD))
                if rng.uniform() < SEASONAL_SHIFT_PROB:
                    dominant_period = float(
                        np.exp(rng.uniform(math.log(PERIOD_MIN), math.log(PERIOD_MAX))))
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
    # PatchTST-FM-R1 has a FIXED context_length (8192) and no observed-mask input;
    # a shorter tensor makes its internal padding NaN out. Left-pad to the native
    # context with NaN, which the model treats as its missing-value indicator and
    # masks out (empirically the least-distorting pad: matches mean-padding, beats
    # zeros). So genuine samples drive the forecast and output stays finite.
    ctx_len = model.config.context_length
    if past_values.shape[1] < ctx_len:
        pad = past_values.new_full(
            (past_values.shape[0], ctx_len - past_values.shape[1]), float("nan"))
        past_values = torch.cat([pad, past_values], dim=1)
    elif past_values.shape[1] > ctx_len:
        past_values = past_values[:, -ctx_len:]
    raw = model(inputs=past_values, prediction_length=horizon)[0]
    if raw.dim() == 4:                                 # (B, Q, H, 1)
        qf = raw[:, :, :horizon, 0]
        return qf[:, PATCHTST_FM_MEDIAN_QUANTILE_IDX, :].to(torch.float32)
    if raw.dim() == 3:                                 # (B, Q, H) or (B, 1, H)
        if raw.shape[1] == 1:
            return raw[:, 0, :horizon].to(torch.float32)
        return torch.median(raw[:, :, :horizon], dim=1).values.to(torch.float32)
    return raw[:, :horizon].to(torch.float32)          # (B, H)


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
    from transformers import AutoModelForCausalLM
    _patch_dynamic_cache_seen_tokens()
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.to(device).eval()
    return model


def predict_sundial(model, x: torch.Tensor, horizon: int, device: str) -> torch.Tensor:
    seqs = x[:, :, 0].to(device, non_blocking=True)
    samples = model.generate(
        seqs, max_new_tokens=horizon, num_samples=SUNDIAL_NUM_SAMPLES,
    )                                                  # (B, S, H)
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
    """Quantile forecast; return the median row (B, horizon).

    ``x`` is (B, W, 1) == (batch, context, channels), which is exactly FlowState's
    ``batch_first=True`` layout, so it's passed through unreshaped. The model emits
    (B, Q, H, 1); we take the middle of its Q=9 quantiles as the point forecast.
    """
    series = x.to(device, non_blocking=True)               # (B, W, 1)
    out = model(
        series,
        scale_factor=FLOWSTATE_SCALE_FACTOR,
        prediction_length=horizon,
        batch_first=True,
    )
    qf = out.prediction_outputs                            # (B, Q, H, 1)
    return qf[:, FLOWSTATE_MEDIAN_QUANTILE_IDX, :horizon, 0].to(torch.float32)


def load_tirex(model_id: str, device: str):
    from tirex import load_model
    model = load_model(model_id)
    # tirex.load_model handles device placement internally; .to is a no-op guard.
    try:
        model.to(device)
    except Exception:
        pass
    return model


def predict_tirex(model, x: torch.Tensor, horizon: int, device: str) -> torch.Tensor:
    """Quantile forecast (TiRex); return the per-step median (B, horizon).

    TiRex takes a (batch, context) tensor and returns (quantiles, mean), where
    quantiles is (B, horizon, 9) over levels [0.1..0.9]; we take the 0.5 row
    (index 4) as the point forecast to stay consistent with the other wrappers.

    NOTE: the quantile axis order/levels below follow TiRex's docs, not its
    package source — verify against the installed `tirex` (quantiles shape and
    median index) before a full run.
    """
    seqs = x[:, :, 0].to(device, non_blocking=True)        # (B, L)
    quantiles, _mean = model.forecast(context=seqs, prediction_length=horizon)
    median = quantiles[:, :horizon, TIREX_MEDIAN_QUANTILE_IDX]
    return median.to(torch.float32)                        # (B, horizon)


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
) -> torch.Tensor:
    """Forecast a uniform-width batch; return median (n, horizon) on `device`.

    moirai/timesfm recompile against `width`, so this is called once per distinct
    context width (see forecast_window's grouping)."""
    n = x_all.shape[0]
    if family == "moirai":
        runner = _build_moirai(base, horizon, width, device)
    elif family == "timesfm":
        runner = load_timesfm(model_id, width, horizon, batch_size)
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
        medians.append(m)

    if family in ("moirai", "timesfm"):
        del runner
        if _is_cuda(device):
            torch.cuda.empty_cache()

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
        med = _forecast_uniform(
            family, base, model_id, x_grp, int(L), horizon, batch_size, device)
        out[torch.as_tensor(idx, device=device, dtype=torch.long)] = med

    return out                                          # (N, H)


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
            real_len = np.ascontiguousarray(
                np.load(real_lengths_path, mmap_mode="r")[start:end])  # (B,)
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
                    batch_size, device)                     # (B, MAX_HORIZON)
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
                           "horizon_grid": HORIZON_GRID}, f)

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
        if _is_cuda(device):
            torch.cuda.empty_cache()

    print(Fore.CYAN + f"  [{dev_label}] worker {worker_id} exited." + Fore.RESET)


# ==============================================================================
#  SHARD MERGE + SANITY
# ==============================================================================

def merge_shards(
    model_dir: str, n_series: int
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
            s, e = d["start"], d["end"]
            cm[s:e] = np.load(os.path.join(sdir, name, "curves_mae.npy"))
            cs[s:e] = np.load(os.path.join(sdir, name, "curves_mse.npy"))
            n_done += 1

    np.save(os.path.join(model_dir, "curves_mae.npy"), cm)
    np.save(os.path.join(model_dir, "curves_mse.npy"), cs)
    return cm, cs, n_done


def _print_data_sanity(
    curves_mae: np.ndarray, n_segments: np.ndarray, family: str
) -> None:
    """Report whether the generated data is actually context-sensitive."""
    valid = ~np.isnan(curves_mae).any(axis=(1, 2))
    if not valid.any():
        print(Fore.RED + "  No completed curves to summarize." + Fore.RESET)
        return
    cm_v = curves_mae[valid]                                # (V, n_win, n_h)
    n_win = cm_v.shape[1]
    win_arr = np.array(WINDOW_GRID)
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
    p.add_argument("--model-idx", type=int, default=None,
                   help=(
                       "Run only this model index (0-based). "
                       "Default: run all models in sequence. "
                       f"Available: {[f'{i}={m[2]}' for i, m in enumerate(MODELS)]}"
                   ))
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
    grid = [w for w in WINDOW_GRID if args.windows is None or w in args.windows]
    if any(w > MAX_WINDOW for w in grid):
        raise ValueError(f"WINDOW_GRID exceeds MAX_WINDOW={MAX_WINDOW}.")
    win_indices = [WINDOW_GRID.index(w) for w in grid]

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # ---------- Series pool (generated once, shared across all models) ------
    contexts_path = os.path.join(OUTPUT_ROOT, "contexts.npy")
    targets_path  = os.path.join(OUTPUT_ROOT, "targets.npy")
    nseg_path     = os.path.join(OUTPUT_ROOT, "n_segments.npy")
    rlen_path     = os.path.join(OUTPUT_ROOT, "real_lengths.npy")

    if os.path.isfile(contexts_path):
        n_series = int(np.load(contexts_path, mmap_mode="r").shape[0])
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
        with open(os.path.join(OUTPUT_ROOT, "meta.json"), "w") as f:
            json.dump({
                "n_series": int(n_series), "seed": SEED,
                "max_window": MAX_WINDOW, "max_horizon": MAX_HORIZON,
                "window_grid": WINDOW_GRID, "horizon_grid": HORIZON_GRID,
                "period_min": PERIOD_MIN, "period_max": PERIOD_MAX,
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
        model_dir = _model_dir(display)
        os.makedirs(model_dir, exist_ok=True)
        print(Fore.CYAN + f"\n── {display} ({model_id}) ──" + Fore.RESET)
        print(Fore.CYAN + f"   windows={grid}  shard_size={args.shard_size}" + Fore.RESET)

        pending = []
        n_completed = 0
        for shard_id, start in enumerate(range(0, n_series, args.shard_size)):
            end = min(start + args.shard_size, n_series)
            if os.path.isfile(os.path.join(_shard_dir(model_dir, shard_id), "done.json")):
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

        curves_mae, _, n_done = merge_shards(model_dir, n_series)
        total_shards = (n_series + args.shard_size - 1) // args.shard_size

        with open(os.path.join(model_dir, "meta.json"), "w") as f:
            json.dump({
                "model_id": model_id, "model_family": family, "model_display": display,
                "window_indices": win_indices,
                "shards_done": n_done, "shards_total": total_shards,
                "devices": devices, "shard_size": args.shard_size,
                "created": datetime.now().isoformat(timespec="seconds"),
                # Pool-level keys repeated here so predict_context_length.py can
                # use this subdir as --dataset-dir without reading the parent meta.
                "max_window": MAX_WINDOW, "max_horizon": MAX_HORIZON,
                "window_grid": WINDOW_GRID, "horizon_grid": HORIZON_GRID,
            }, f, indent=2)

        _print_data_sanity(curves_mae, n_segments, family)
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
