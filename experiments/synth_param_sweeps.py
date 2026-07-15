"""
Synthetic single-factor sweeps: forecast error vs NORMALIZED context, per TSFM.

For each generative factor below, series are generated containing ONLY that
factor plus Gaussian noise (minimal isolated signals — no confounds from the
stage-1 generator's other components), the factor's parameter is swept over a
grid of values ("bins"), and every TSFM forecasts each series at context
lengths sampled on a RATIO grid: context = r × parameter. Plots show
MAE (y) against context/parameter (x), so curves from different parameter
values align on the same normalized axis and expose scale-free relations
(e.g. "error saturates once context covers ~4 periods").

Full design rationale + per-experiment hypotheses: SYNTH_SWEEPS.md (repo root).

Experiments (name → swept parameter → normalizer):
  period      one sinusoid of period T             + noise         → context/T
  seasonality composite repeating pattern (random  harmonic profile
              of length S, per-cycle amplitude jitter)             → context/S
  ar_order    AR(p), p ∈ {1..4}, dominant pole matched to a target
              ACF timescale τ (curves per order)                   → context/τ
  memory      AR(1) with timescale τ = 1/(1−φ) (long-memory
              strength as literal correlation reach)               → context/τ
  delay       trigger→response events: a wide bump (Hann, width 33
              ≈ one patch, NOT a single-step spike) is followed d
              steps later by a deterministic response bump; one
              trigger is planted so its response lands inside the
              horizon — context ≳ d is required to see it          → context/d
  regime      regimes of fixed duration D (level/amp/phase shifts
              at boundaries, shared period); the last boundary sits
              exactly D before forecast time, so ratio > 1 means
              stale regimes enter the window                       → context/D
  horizon     canonical signal (sin T=128 + AR(1) τ=128 + noise),
              forecast horizon h swept; context = r × h            → context/h
  break_age   ONE change point (level/amp/phase shift) at distance
              A before forecast time, after a long homogeneous
              regime — the sharpest "stale history hurts" probe:
              error should be flat for context/A < 1 and RISE past
              1 as pre-break history enters the window. Decouples
              regime AGE from regime DURATION (the `regime`
              experiment confounds them: age = duration = D)       → context/A
  snr         seasonal strength: fixed period T=256, noise σ swept
              (curves per σ). Hypothesis: estimating a periodic
              profile under noise needs ~σ² cycles of averaging,
              so the saturation point in context/T shifts RIGHT as
              σ grows — turns "saturates at k cycles" into "k
              depends on noise like this"                          → context/T
  multiscale  nested periods: inner sinusoid of period T=64 whose
              per-cycle amplitude follows a FIXED random sequence
              of k values repeating every OUTER cycle k·T. The
              next inner cycle's amplitude is only predictable
              from the position-matched cycle one outer period
              back → does the model need to cover the slow cycle
              (daily-within-weekly structure)?                     → context/(k·T)
  period_drift the dominant period WANDERS: log-period follows an
              OU process with correlation time M, so context older
              than ~M carries stale FREQUENCY information (the
              level-staleness analog is regime/break_age)          → context/M
  missing_gap sinusoid with a missing span of length G, mean-filled
              (zeroed in raw space, the stage-1 pad / GiftEval
              NaN→0 model-input convention), occupying
              [end−1.5G, end−0.5G) — scale-free geometry: ratio
              < 0.5 sees only the genuine tail, 0.5–1.5 is inside
              the gap, > 1.5 bridges to pre-gap history. CAVEAT:
              tests "uninformative constant span", not native NaN
              handling (wrappers can't all take NaN) — keep it an
              appendix, not a headline                             → context/G

All series are standardized by their context-pool stats (exactly like stage 1),
so MAE is comparable across bins/models. Non-`horizon` experiments forecast
once at MAX_EVAL_HORIZON and slice the error at every h in EVAL_HORIZONS for
free (the stage-1 trick). Series generation is seeded per (experiment, bin)
independently of the model, so every model sees IDENTICAL series.

Reuses the stage-1 model machinery verbatim (`setup_model`,
`_forecast_uniform` — which already handles Toto's patch-multiple padding,
TimesFM/Moirai per-width rebuilds, PatchTST-FM NaN-padding). Model context
caps (Sundial 2880, TimeMoE 4096−h, Toto/FlowState 4096, TiRex 8192) skip the
offending ratio points, leaving NaN. NOTE: TimesFM recompiles per distinct
context width, so its sweep is slower than the fixed-grid ablation.

Multi-GPU: a queue of (model, experiment) cells, ordered model-major; each
worker caches its currently loaded model and reloads only on change. Each
finished cell writes `results.npz` + `done.json` → re-runs only do what's
missing. Sundial/TimeMoE need the legacy transformers env — run them
separately: `--models Sundial-Base-128M TimeMoE-200M`.

Usage (on the SERVER):
    python -m experiments.synth_param_sweeps                       # run set, all experiments
    python -m experiments.synth_param_sweeps --models Chronos2-Small TiRex2
    python -m experiments.synth_param_sweeps --experiments period delay
    python -m experiments.synth_param_sweeps --plot-only           # replot from cached npz
    python -m experiments.synth_param_sweeps --test                # tiny smoke run

Output tree (server): logs/experiments/synth_param_sweeps/
    <Display>/<experiment>/results.npz + done.json
    plots/<experiment>/<Display>.png          (per-model, lines per bin)
    plots/<experiment>/all_models.png         (cross-model overlay, MAE/min(MAE))
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
import traceback
import zlib
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from colorama import Fore
from scipy.signal import lfilter

from experiments import models_config
from experiments.build_context_length_dataset import (
    FLOWSTATE_MAX_CONTEXT,
    SUNDIAL_MAX_CONTEXT,
    TIMEMOE_MAX_TOTAL,
    TIREX_MAX_CONTEXT,
    TOTO_MAX_CONTEXT,
    _forecast_uniform,
    _is_cuda,
    _physical_gpu_id,
    resolve_devices,
    setup_model,
)

# ==============================================================================
#  CONFIG
# ==============================================================================

SEED         = 42
MAX_WINDOW   = 8192          # context pool length (matches stage 1)
MIN_CONTEXT  = 16            # smallest context ever fed to a model
N_SERIES     = 128           # series per (experiment, bin)
BATCH_SIZE   = 32

# Normalized context grid: context = round(r × parameter), clamped to
# [MIN_CONTEXT, MAX_WINDOW]. The ACTUAL ratio (context/parameter) is what gets
# plotted, so clamped points appear at their true position.
RATIO_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]

# Non-`horizon` experiments: forecast once at MAX_EVAL_HORIZON, slice at each h.
EVAL_HORIZONS    = [16, 64, 256]
MAX_EVAL_HORIZON = max(EVAL_HORIZONS)

NOISE_STD  = 0.2             # observation noise for the wave-based generators
BUMP_WIDTH = 33              # delay-event bump width (Hann); ≈ one patch, so
                             # patch-based models cannot miss it the way they
                             # can a single-step spike

OUTPUT_ROOT = os.environ.get(
    "PREDICTCSL_SWEEP_ROOT", "logs/experiments/synth_param_sweeps")

# Test mode (--test): tiny bins/ratios/series so the whole thing runs in minutes.
TEST_RATIO_GRID = [0.5, 1.0, 2.0, 4.0]
TEST_N_SERIES   = 8


def _stable_seed(experiment: str, label: str) -> int:
    """Model-independent, run-stable seed for one (experiment, bin)."""
    return zlib.crc32(f"{SEED}:{experiment}:{label}".encode()) & 0x7FFFFFFF


# ==============================================================================
#  GENERATORS  (minimal isolated signals; each returns (total_length,) float32)
# ==============================================================================

def _finalize(y: np.ndarray) -> np.ndarray:
    """Sanitize + standardize by context-pool stats (mirrors stage 1: float64
    stats over the pool only, no horizon leakage)."""
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    ctx = y[:MAX_WINDOW].astype(np.float64)
    mu, sigma = float(ctx.mean()), float(ctx.std())
    return ((y - mu) / (sigma + 1e-8)).astype(np.float32, copy=False)


def gen_period(rng: np.random.RandomState, total_length: int, T: int,
               noise_std: float = NOISE_STD) -> np.ndarray:
    """One sinusoid + noise. `noise_std` is only overridden by the `snr`
    experiment (fixed T, σ swept — seasonal strength)."""
    t = np.arange(total_length, dtype=np.float32)
    y = np.sin(2.0 * np.pi * t / float(T) + rng.uniform(0.0, 2.0 * np.pi))
    y += rng.normal(0.0, noise_std, size=total_length)
    return _finalize(y)


def gen_seasonality(rng: np.random.RandomState, total_length: int, S: int) -> np.ndarray:
    """A structured repeating pattern: random harmonic profile of length S,
    tiled with small per-cycle amplitude jitter (a 'weekly profile', not a
    smooth wave — tests complex repetition at the same normalizer)."""
    t = np.arange(S, dtype=np.float32)
    n_harm = int(rng.randint(3, 7))
    harmonics = rng.choice(np.arange(1, 9), size=n_harm, replace=False)
    profile = np.zeros(S, dtype=np.float32)
    for j in harmonics:
        amp = rng.uniform(0.3, 1.0) / math.sqrt(float(j))
        profile += amp * np.sin(
            2.0 * np.pi * float(j) * t / float(S) + rng.uniform(0.0, 2.0 * np.pi))
    profile /= max(float(profile.std()), 1e-6)
    n_cycles = int(math.ceil(total_length / S)) + 1
    jitter = rng.normal(1.0, 0.1, size=n_cycles).astype(np.float32)
    y = (profile[None, :] * jitter[:, None]).ravel()[:total_length]
    y += rng.normal(0.0, NOISE_STD, size=total_length)
    return _finalize(y)


def gen_ar(rng: np.random.RandomState, total_length: int,
           tau: int, order: int) -> np.ndarray:
    """Stable AR(order) whose DOMINANT pole is matched to the target ACF
    timescale τ (pole 1−1/τ); extra poles have |pole| ≤ 0.7, so their memory
    (≲ 3 steps) never competes with τ. lfilter denominator = np.poly(poles)
    (same pole→coefficient identity stage 1 uses)."""
    poles: List[complex] = [1.0 - 1.0 / float(tau)]
    if order >= 3:
        mag = rng.uniform(0.4, 0.7)
        omega = rng.uniform(0.2 * np.pi, 0.8 * np.pi)
        poles += [mag * np.exp(1j * omega), mag * np.exp(-1j * omega)]
    if order in (2, 4):
        poles.append(complex(rng.uniform(-0.6, 0.6)))
    a = np.real(np.poly(poles)).astype(np.float64)
    burn = int(min(6 * tau, 25_000))
    innov = rng.normal(0.0, 1.0, size=total_length + burn)
    y = lfilter([1.0], a, innov)[burn:].astype(np.float32)
    y += rng.normal(0.0, 0.1 * (float(y.std()) + 1e-8), size=total_length)
    return _finalize(y)


def gen_memory(rng: np.random.RandomState, total_length: int, tau: int) -> np.ndarray:
    return gen_ar(rng, total_length, tau=tau, order=1)


def _add_bump(y: np.ndarray, center: int, amp: float, bump: np.ndarray) -> None:
    half = len(bump) // 2
    lo, hi = center - half, center + half + 1
    b_lo, b_hi = max(0, -lo), len(bump) - max(0, hi - len(y))
    lo, hi = max(0, lo), min(len(y), hi)
    if hi > lo:
        y[lo:hi] += amp * bump[b_lo:b_hi]


def gen_delay(rng: np.random.RandomState, total_length: int,
              d: int, horizon: int) -> np.ndarray:
    """Trigger→response pairs at lag d. Each event is a WIDE Hann bump
    (BUMP_WIDTH samples — patch-visible, deliberately not a one-step spike);
    the response is the same bump at ×0.7 amplitude, d steps later. One
    'critical' trigger is planted at MAX_WINDOW − d + k (k inside the horizon)
    so its response falls in the forecast window: only a context reaching back
    ≳ d can see the trigger and predict the response."""
    y = rng.normal(0.0, 0.15, size=total_length).astype(np.float32)
    bump = np.hanning(BUMP_WIDTH).astype(np.float32)
    half = BUMP_WIDTH // 2

    n_events = int(np.clip(MAX_WINDOW // (d + 8 * BUMP_WIDTH), 3, 12))
    centers = list(rng.randint(BUMP_WIDTH, MAX_WINDOW - BUMP_WIDTH, size=n_events))
    k_lo = half + 1
    k_hi = max(k_lo + 1, min(horizon - half - 1, d - 1))
    centers.append(MAX_WINDOW - d + int(rng.randint(k_lo, k_hi + 1)))

    for c in centers:
        amp = float(rng.uniform(1.5, 3.0)) * float(rng.choice([-1.0, 1.0]))
        _add_bump(y, int(c), amp, bump)
        if int(c) + d < total_length:
            _add_bump(y, int(c) + d, 0.7 * amp, bump)
    return _finalize(y)


def gen_regime(rng: np.random.RandomState, total_length: int, D: int) -> np.ndarray:
    """Regimes of duration D: shared period T0=32 sinusoid, but level/amplitude/
    phase shift at every boundary. Boundaries sit at MAX_WINDOW − m·D, so the
    current regime's age is exactly D at forecast time (context/D > 1 ⇒ stale
    regimes enter the window) and it extends through the horizon (no future
    cut — the horizon stays forecastable from the current regime)."""
    T0 = 32
    t = np.arange(total_length, dtype=np.float32)
    edges = sorted(MAX_WINDOW - m * D for m in range(1, MAX_WINDOW // D + 1)
                   if MAX_WINDOW - m * D > 0)
    bounds = [0] + edges + [total_length]

    y = np.empty(total_length, dtype=np.float32)
    level = 0.0
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if i > 0:
            level += float(rng.normal(0.0, 1.5))
        amp = float(rng.uniform(0.6, 1.6))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        y[s:e] = level + amp * np.sin(2.0 * np.pi * t[s:e] / T0 + phase)
    y += rng.normal(0.0, NOISE_STD, size=total_length)
    return _finalize(y)


def gen_break_age(rng: np.random.RandomState, total_length: int, A: int) -> np.ndarray:
    """ONE change point at distance A before forecast time: a long homogeneous
    pre-break regime, then a level/amplitude/phase shift; the post-break regime
    extends through the horizon. Unlike `regime` (where the current regime's
    age EQUALS its duration D), this isolates recency: context/A < 1 stays
    inside the current regime (error flat), context/A > 1 pulls in stale
    pre-break history (error should rise if the model can't discount it).
    This is the single most on-thesis probe of "more context is not always
    better" — it measures the cost of crossing a break, which is what the
    context-length predictor is supposed to exploit."""
    T0 = 32
    t = np.arange(total_length, dtype=np.float32)
    cut = MAX_WINDOW - A
    y = np.empty(total_length, dtype=np.float32)
    amp0 = float(rng.uniform(0.6, 1.6))
    y[:cut] = amp0 * np.sin(
        2.0 * np.pi * t[:cut] / T0 + rng.uniform(0.0, 2.0 * np.pi))
    level1 = float(rng.normal(0.0, 1.5))
    amp1 = float(rng.uniform(0.6, 1.6))
    y[cut:] = level1 + amp1 * np.sin(
        2.0 * np.pi * t[cut:] / T0 + rng.uniform(0.0, 2.0 * np.pi))
    y += rng.normal(0.0, NOISE_STD, size=total_length)
    return _finalize(y)


def gen_multiscale(rng: np.random.RandomState, total_length: int,
                   T: int, k: int) -> np.ndarray:
    """Nested periods: an inner sinusoid of period T whose amplitude in inner
    cycle j is a FIXED random value a[j mod k] — a random envelope sequence
    repeating every OUTER cycle k·T (daily-within-weekly structure). The next
    inner cycle's amplitude is only predictable from the position-matched
    cycle exactly one outer period back, so the hypothesis is a saturation
    knee at context/(k·T) ≈ 1: below it the model can track the inner wave
    but must guess the envelope. T divides MAX_WINDOW and k divides
    MAX_WINDOW/T, so cycle boundaries align with the forecast origin."""
    t = np.arange(total_length, dtype=np.float32)
    amp_seq = rng.uniform(0.3, 1.5, size=k).astype(np.float32)
    cycle_idx = (np.arange(total_length) // T) % k
    y = amp_seq[cycle_idx] * np.sin(
        2.0 * np.pi * t / float(T) + rng.uniform(0.0, 2.0 * np.pi))
    y += rng.normal(0.0, NOISE_STD, size=total_length)
    return _finalize(y)


def gen_period_drift(rng: np.random.RandomState, total_length: int,
                     M: int) -> np.ndarray:
    """The dominant period WANDERS: log-period follows an OU process with
    correlation time M (stationary std 0.25 around T0=64), and the wave's
    phase is the cumulative integral of the instantaneous frequency. Context
    older than ~M carries a stale estimate of the CURRENT period — the
    frequency-staleness analog of regime/break_age's level-staleness. The OU
    recursion x_t = (1−1/M)·x_{t−1} + w_t is run through lfilter with a
    burn-in so x starts in its stationary distribution."""
    T0, sig = 64, 0.25
    burn = int(min(6 * M, 25_000))
    w = rng.normal(0.0, sig * math.sqrt(2.0 / M), size=total_length + burn)
    x = lfilter([1.0], [1.0, -(1.0 - 1.0 / M)], w)[burn:]
    omega = 2.0 * np.pi / (T0 * np.exp(x))
    phase = np.cumsum(omega) + rng.uniform(0.0, 2.0 * np.pi)
    y = np.sin(phase).astype(np.float32)
    y += rng.normal(0.0, NOISE_STD, size=total_length)
    return _finalize(y)


def gen_missing_gap(rng: np.random.RandomState, total_length: int,
                    G: int) -> np.ndarray:
    """Sinusoid (T0=64) with a MISSING span of length G occupying
    [end−1.5G, end−0.5G) of the context pool, zeroed in RAW space before
    standardization — exactly the stage-1 left-pad / GiftEval NaN→0
    model-input convention, so the gap maps to the constant −mu/sigma. The
    geometry is scale-free in context/G: ratio < 0.5 sees only the genuine
    tail, 0.5–1.5 is inside the gap (uninformative constant), > 1.5 bridges
    to pre-gap history. CAVEAT: because wrappers can't uniformly ingest NaN,
    this tests "uninformative constant span", NOT native missing-value
    handling — treat it as an appendix experiment, not a headline."""
    T0 = 64
    t = np.arange(total_length, dtype=np.float32)
    y = np.sin(2.0 * np.pi * t / T0 + rng.uniform(0.0, 2.0 * np.pi))
    y += rng.normal(0.0, NOISE_STD, size=total_length)
    g_hi = MAX_WINDOW - G // 2
    y[g_hi - G:g_hi] = 0.0
    return _finalize(y)


def gen_canonical(rng: np.random.RandomState, total_length: int) -> np.ndarray:
    """Fixed-structure signal for the horizon sweep: sin(T=128) + AR(1, τ=128)
    + noise. Only the horizon varies across bins."""
    t = np.arange(total_length, dtype=np.float32)
    y = np.sin(2.0 * np.pi * t / 128.0 + rng.uniform(0.0, 2.0 * np.pi))
    burn = 768
    innov = rng.normal(0.0, 1.0, size=total_length + burn)
    ar = lfilter([1.0], [1.0, -(1.0 - 1.0 / 128.0)], innov)[burn:].astype(np.float32)
    y += 0.7 * ar / (float(ar.std()) + 1e-8)
    y += rng.normal(0.0, NOISE_STD, size=total_length)
    return _finalize(y)


GENERATORS = {
    "period":       gen_period,
    "seasonality":  gen_seasonality,
    "ar":           gen_ar,
    "memory":       gen_memory,
    "delay":        gen_delay,
    "regime":       gen_regime,
    "canonical":    gen_canonical,
    "break_age":    gen_break_age,
    "multiscale":   gen_multiscale,
    "period_drift": gen_period_drift,
    "missing_gap":  gen_missing_gap,
}


# ==============================================================================
#  EXPERIMENT REGISTRY
# ==============================================================================

class Bin(NamedTuple):
    label: str          # human label; also part of the series seed
    norm: float         # x-axis normalizer (context/norm)
    horizon: int        # forecast horizon for this bin
    gen: str            # GENERATORS key
    kwargs: dict        # generator kwargs


# Per-experiment axis metadata: (param symbol, description)
EXP_META: Dict[str, Tuple[str, str]] = {
    "period":       ("T", "sinusoid period"),
    "seasonality":  ("S", "seasonal pattern length"),
    "ar_order":     ("tau", "AR ACF timescale (curves per order)"),
    "memory":       ("tau", "AR(1) timescale 1/(1-phi)"),
    "delay":        ("d", "trigger-to-response delay"),
    "regime":       ("D", "regime duration"),
    "horizon":      ("h", "forecast horizon"),
    "break_age":    ("A", "age of the single change point"),
    "snr":          ("T", "seasonal strength: noise sigma at fixed T (curves per sigma)"),
    "multiscale":   ("kT", "outer (slow) cycle length k*T"),
    "period_drift": ("M", "period-drift timescale (OU log-period)"),
    "missing_gap":  ("G", "missing-gap length (mean-filled)"),
}

SNR_T        = 256   # fixed period for the `snr` sweep (mid-grid: contexts
                     # 64…4096 across the ratio grid, always on-cap)
MULTISCALE_T = 64    # fixed inner period for the `multiscale` sweep


def build_experiments() -> Dict[str, List[Bin]]:
    exps: Dict[str, List[Bin]] = {}
    h = MAX_EVAL_HORIZON

    exps["period"] = [
        Bin(f"T={T}", float(T), h, "period", {"T": T})
        for T in [32, 64, 128, 256, 512, 1024, 2048]]

    exps["seasonality"] = [
        Bin(f"S={S}", float(S), h, "seasonality", {"S": S})
        for S in [32, 64, 128, 256, 512, 1024, 2048]]

    exps["ar_order"] = [
        Bin(f"p={p},tau={tau}", float(tau), h, "ar", {"tau": tau, "order": p})
        for tau in [64, 256, 1024] for p in [1, 2, 3, 4]]

    exps["memory"] = [
        Bin(f"tau={tau}", float(tau), h, "memory", {"tau": tau})
        for tau in [16, 64, 256, 1024, 4096]]

    exps["delay"] = [
        Bin(f"d={d}", float(d), h, "delay", {"d": d, "horizon": h})
        for d in [64, 128, 256, 512, 1024, 2048, 4096]]

    exps["regime"] = [
        Bin(f"D={D}", float(D), h, "regime", {"D": D})
        for D in [128, 256, 512, 1024, 2048, 4096]]

    exps["horizon"] = [
        Bin(f"h={hh}", float(hh), hh, "canonical", {})
        for hh in [16, 32, 64, 128, 256, 512]]

    exps["break_age"] = [
        Bin(f"A={A}", float(A), h, "break_age", {"A": A})
        for A in [64, 128, 256, 512, 1024, 2048, 4096]]

    exps["snr"] = [
        Bin(f"sigma={s}", float(SNR_T), h, "period",
            {"T": SNR_T, "noise_std": s})
        for s in [0.1, 0.25, 0.5, 1.0, 2.0]]

    exps["multiscale"] = [
        Bin(f"k={k}", float(k * MULTISCALE_T), h, "multiscale",
            {"T": MULTISCALE_T, "k": k})
        for k in [2, 4, 8, 16, 32]]

    exps["period_drift"] = [
        Bin(f"M={M}", float(M), h, "period_drift", {"M": M})
        for M in [256, 512, 1024, 2048, 4096]]

    exps["missing_gap"] = [
        Bin(f"G={G}", float(G), h, "missing_gap", {"G": G})
        for G in [64, 128, 256, 512, 1024, 2048]]

    return exps


EXPERIMENTS = build_experiments()


def resolve_bins(experiment: str, test: bool) -> List[Bin]:
    bins = EXPERIMENTS[experiment]
    return bins[:2] if test else bins


def resolve_ratios(test: bool) -> List[float]:
    return TEST_RATIO_GRID if test else RATIO_GRID


def _context_cap(family: str, horizon: int) -> int:
    """Largest context this family may see (stage-1 caps; beyond → NaN)."""
    cap = MAX_WINDOW
    if family == "sundial":
        cap = min(cap, SUNDIAL_MAX_CONTEXT)
    elif family == "timemoe":
        cap = min(cap, TIMEMOE_MAX_TOTAL - horizon)
    elif family == "toto":
        cap = min(cap, TOTO_MAX_CONTEXT)
    elif family == "flowstate":
        cap = min(cap, FLOWSTATE_MAX_CONTEXT)
    elif family == "tirex":
        cap = min(cap, TIREX_MAX_CONTEXT)
    return cap


# ==============================================================================
#  CELL RUNNER  (one (model, experiment) → results.npz)
# ==============================================================================

def _cell_dir(display: str, experiment: str) -> str:
    return os.path.join(OUTPUT_ROOT, display, experiment)


def _cell_done(display: str, experiment: str) -> bool:
    return os.path.exists(os.path.join(_cell_dir(display, experiment), "done.json"))


def run_cell(
    family: str,
    base,
    model_id: str,
    display: str,
    experiment: str,
    device: str,
    n_series: int,
    batch_size: int,
    test: bool,
) -> None:
    bins = resolve_bins(experiment, test)
    ratios = resolve_ratios(test)
    per_bin_h = experiment == "horizon"
    eval_hs = [-1] if per_bin_h else EVAL_HORIZONS   # -1 → each bin's own horizon
    n_bins, n_ratios, n_h = len(bins), len(ratios), len(eval_hs)

    curves_mae = np.full((n_bins, n_series, n_ratios, n_h), np.nan, dtype=np.float32)
    curves_mse = np.full((n_bins, n_series, n_ratios, n_h), np.nan, dtype=np.float32)
    naive_mae  = np.full((n_bins, n_series, n_h), np.nan, dtype=np.float32)
    contexts   = np.full((n_bins, n_ratios), -1, dtype=np.int64)

    for b_idx, bn in enumerate(bins):
        rng = np.random.RandomState(_stable_seed(experiment, bn.label))
        total = MAX_WINDOW + bn.horizon
        gen = GENERATORS[bn.gen]
        series = np.stack([gen(rng, total, **bn.kwargs) for _ in range(n_series)])
        pool = np.ascontiguousarray(series[:, :MAX_WINDOW])           # (n, 8192)
        tgt = torch.from_numpy(
            np.ascontiguousarray(series[:, MAX_WINDOW:total])).to(device)

        h_slices = [bn.horizon] if per_bin_h else EVAL_HORIZONS
        naive = np.abs(pool[:, -1:] - series[:, MAX_WINDOW:total])    # (n, H)
        for h_idx, hh in enumerate(h_slices):
            naive_mae[b_idx, :, h_idx] = naive[:, :hh].mean(axis=1)

        cap = _context_cap(family, bn.horizon)
        seen: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # L → (mae, mse) rows
        for r_idx, r in enumerate(ratios):
            L = int(round(r * bn.norm))
            L = max(MIN_CONTEXT, min(L, MAX_WINDOW))
            if L > cap:
                continue                                 # stays NaN + context -1
            contexts[b_idx, r_idx] = L
            if L in seen:                                # clamping can repeat L
                curves_mae[b_idx, :, r_idx], curves_mse[b_idx, :, r_idx] = seen[L]
                continue
            x = torch.from_numpy(
                np.ascontiguousarray(pool[:, -L:])).unsqueeze(-1)     # (n, L, 1)
            med = _forecast_uniform(
                family, base, model_id, x, L, bn.horizon, batch_size, device)
            mae_rows = np.empty((n_series, n_h), dtype=np.float32)
            mse_rows = np.empty((n_series, n_h), dtype=np.float32)
            for h_idx, hh in enumerate(h_slices):
                err = med[:, :hh] - tgt[:, :hh]
                mae_rows[:, h_idx] = err.abs().mean(dim=1).cpu().numpy()
                mse_rows[:, h_idx] = err.pow(2).mean(dim=1).cpu().numpy()
            curves_mae[b_idx, :, r_idx] = mae_rows
            curves_mse[b_idx, :, r_idx] = mse_rows
            seen[L] = (mae_rows, mse_rows)

    out_dir = _cell_dir(display, experiment)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "results.npz"),
        curves_mae=curves_mae, curves_mse=curves_mse, naive_mae=naive_mae,
        contexts=contexts, ratios=np.asarray(ratios, dtype=np.float32),
        norms=np.asarray([bn.norm for bn in bins], dtype=np.float32),
        horizons=np.asarray([bn.horizon for bn in bins], dtype=np.int64),
        eval_horizons=np.asarray(eval_hs, dtype=np.int64))
    with open(os.path.join(out_dir, "done.json"), "w") as f:
        json.dump({
            "model": display, "family": family, "experiment": experiment,
            "bins": [{"label": bn.label, "norm": bn.norm, "horizon": bn.horizon,
                      "gen": bn.gen, "kwargs": bn.kwargs} for bn in bins],
            "ratios": ratios, "eval_horizons": eval_hs,
            "n_series": n_series, "seed": SEED, "test": test,
        }, f, indent=2)


# ==============================================================================
#  GPU WORKER  (drains a (model, experiment) cell queue; caches the loaded model)
# ==============================================================================

def gpu_worker(
    worker_id: int,
    device: str,
    task_queue: "mp.Queue",
    result_queue: "mp.Queue",
    n_series: int,
    batch_size: int,
    test: bool,
) -> None:
    dev_label = device
    if _is_cuda(device):
        # Pin before CUDA init (same reason as stage 1: some backends hardcode
        # cuda:0); inside this process the GPU is then always cuda:0.
        os.environ["CUDA_VISIBLE_DEVICES"] = _physical_gpu_id(device)
        device = "cuda:0"
        torch.cuda.set_device(torch.device(device))
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")

    loaded_display: Optional[str] = None
    base = None
    tb_shown = False

    while True:
        task = task_queue.get()
        if task is None:
            break
        model_id, family, display, experiment = task
        t0 = time.perf_counter()
        try:
            if display != loaded_display:
                if loaded_display is not None:
                    del base
                    gc.collect()
                    if _is_cuda(device):
                        torch.cuda.empty_cache()
                base = setup_model(family, model_id, device)
                loaded_display = display
                print(Fore.CYAN + f"  [{dev_label}] worker {worker_id} loaded "
                      + f"{display}." + Fore.RESET)
            run_cell(family, base, model_id, display, experiment, device,
                     n_series, batch_size, test)
            elapsed = time.perf_counter() - t0
            result_queue.put({"cell": (display, experiment), "status": "ok",
                              "elapsed": elapsed})
            print(Fore.YELLOW + f"  [{dev_label}] {display}/{experiment} done "
                  + f"({elapsed:.1f}s)" + Fore.RESET)
        except Exception as exc:
            print(Fore.RED + f"  [{dev_label}] {display}/{experiment} FAILED: "
                  + f"{type(exc).__name__}: {exc}" + Fore.RESET)
            if not tb_shown:
                traceback.print_exc()
                tb_shown = True
            result_queue.put({"cell": (display, experiment),
                              "status": f"error:{type(exc).__name__}"})
            # A load failure would poison every later cell of this model on
            # this worker; drop the cache so the next task retries the load.
            loaded_display, base = None, None
        if _is_cuda(device):
            torch.cuda.empty_cache()

    print(Fore.CYAN + f"  [{dev_label}] worker {worker_id} exited." + Fore.RESET)


# ==============================================================================
#  PLOTS
# ==============================================================================

def _load_cell(display: str, experiment: str):
    path = os.path.join(_cell_dir(display, experiment), "results.npz")
    if not os.path.exists(path):
        return None, None
    meta_path = os.path.join(_cell_dir(display, experiment), "done.json")
    with open(meta_path) as f:
        meta = json.load(f)
    return np.load(path), meta


def _mean_sem(per_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(n_series, n_ratios) → mean and SEM over series, NaN-safe."""
    mean = np.nanmean(per_series, axis=0)
    n = np.sum(np.isfinite(per_series), axis=0)
    sem = np.nanstd(per_series, axis=0) / np.sqrt(np.maximum(n, 1))
    return mean, sem


def plot_model_experiment(display: str, experiment: str, plots_dir: str) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, meta = _load_cell(display, experiment)
    if data is None:
        return False
    curves = data["curves_mae"]                  # (n_bins, n, n_ratios, n_h)
    contexts, norms = data["contexts"], data["norms"]
    eval_hs = list(data["eval_horizons"])
    bins = meta["bins"]
    symbol, desc = EXP_META[experiment]
    per_bin_h = experiment == "horizon"

    if experiment == "ar_order":
        taus = sorted({bn["kwargs"]["tau"] for bn in bins})
        orders = sorted({bn["kwargs"]["order"] for bn in bins})
        n_rows, n_cols = len(taus), len(eval_hs)
    else:
        n_rows, n_cols = 1, len(eval_hs)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.6 * n_cols, 3.6 * n_rows),
        squeeze=False, sharex=True)
    cmap = plt.get_cmap("viridis")

    for h_idx, hh in enumerate(eval_hs):
        for b_idx, bn in enumerate(bins):
            mask = contexts[b_idx] >= 0
            if not mask.any():
                continue
            x = contexts[b_idx][mask] / norms[b_idx]
            mean, sem = _mean_sem(curves[b_idx, :, :, h_idx])
            if experiment == "ar_order":
                row = taus.index(bn["kwargs"]["tau"])
                color = plt.get_cmap("tab10")(orders.index(bn["kwargs"]["order"]))
                label = f"p={bn['kwargs']['order']}"
            else:
                row = 0
                color = cmap(b_idx / max(len(bins) - 1, 1))
                label = bn["label"]
            ax = axes[row][h_idx]
            ax.plot(x, mean[mask], marker="o", ms=3, color=color, label=label)
            ax.fill_between(x, (mean - sem)[mask], (mean + sem)[mask],
                            color=color, alpha=0.15, lw=0)
            if experiment == "ar_order" and h_idx == 0:
                ax.set_ylabel(f"tau={bn['kwargs']['tau']}\nMAE (z-scored)")

    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row][col]
            ax.set_xscale("log", base=2)
            ax.grid(alpha=0.3)
            if row == 0:
                hh = eval_hs[col]
                ax.set_title("per-bin horizon" if hh == -1 else f"horizon {hh}")
            if row == n_rows - 1:
                ax.set_xlabel(f"context / {symbol}")
            if col == 0 and experiment != "ar_order":
                ax.set_ylabel("MAE (z-scored)")
    # De-duplicated legend on the first axis (ar_order repeats labels per tau).
    handles, labels = axes[0][0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[0][0].legend(uniq.values(), uniq.keys(), fontsize=8)
    fig.suptitle(f"{display} — {experiment} ({desc})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = os.path.join(plots_dir, experiment, f"{display}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def plot_all_models_overlay(displays: List[str], experiment: str,
                            plots_dir: str) -> bool:
    """One line per model: each bin's mean-MAE curve is divided by its own
    minimum (relative error), then averaged across bins on the nominal ratio
    grid — a scale-free 'how much does context/parameter matter' summary.
    Uses h=64 for the standard experiments (each bin's own h for `horizon`)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    cmap = plt.get_cmap("tab10")
    symbol, desc = EXP_META[experiment]
    plotted = 0

    for m_idx, display in enumerate(displays):
        data, meta = _load_cell(display, experiment)
        if data is None:
            continue
        curves, contexts = data["curves_mae"], data["contexts"]
        eval_hs = list(data["eval_horizons"])
        h_idx = eval_hs.index(64) if 64 in eval_hs else 0
        ratios = data["ratios"]

        rel = np.full((curves.shape[0], len(ratios)), np.nan, dtype=np.float64)
        for b_idx in range(curves.shape[0]):
            mean, _ = _mean_sem(curves[b_idx, :, :, h_idx])
            mean = np.where(contexts[b_idx] >= 0, mean, np.nan)
            lo = np.nanmin(mean) if np.isfinite(mean).any() else np.nan
            if np.isfinite(lo) and lo > 0:
                rel[b_idx] = mean / lo
        line = np.nanmean(rel, axis=0)
        mask = np.isfinite(line)
        if not mask.any():
            continue
        ax.plot(ratios[mask], line[mask], marker="o", ms=3,
                color=cmap(m_idx % 10), label=display)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return False
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.3)
    ax.set_xlabel(f"context / {symbol} (nominal ratio)")
    ax.set_ylabel("MAE / min MAE  (mean over bins)")
    ax.set_title(f"{experiment} ({desc}) — all models")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = os.path.join(plots_dir, experiment, "all_models.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def make_plots(displays: List[str], experiments: List[str]) -> None:
    plots_dir = os.path.join(OUTPUT_ROOT, "plots")
    for experiment in experiments:
        n = sum(plot_model_experiment(d, experiment, plots_dir) for d in displays)
        overlay = plot_all_models_overlay(displays, experiment, plots_dir)
        print(Fore.GREEN + f"  plots[{experiment}]: {n} per-model"
              + (", overlay" if overlay else "") + Fore.RESET)


# ==============================================================================
#  MAIN
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-factor synthetic sweeps: MAE vs normalized context.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Display names (default: the models_config run set).")
    p.add_argument("--experiments", nargs="+", default=None,
                   choices=sorted(EXPERIMENTS), help="Subset of experiments.")
    p.add_argument("--n-series", type=int, default=None,
                   help=f"Series per bin (default {N_SERIES}; {TEST_N_SERIES} in --test).")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--device", type=str, default=None,
                   help="'cpu' to force CPU (default: all CUDA devices).")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip inference; regenerate plots from cached results.npz.")
    p.add_argument("--force", action="store_true",
                   help="Recompute cells even if their done.json exists.")
    p.add_argument("--test", action="store_true",
                   help="Tiny smoke run (2 bins, 4 ratios, 8 series).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    experiments = args.experiments or sorted(EXPERIMENTS)

    run_set = models_config.models_to_run()
    by_display = {d: (mid, fam, d) for mid, fam, d in models_config.catalog()}
    if args.models:
        unknown = [m for m in args.models if m not in by_display]
        if unknown:
            raise SystemExit(f"Unknown models {unknown}; "
                             f"choose from {sorted(by_display)}")
        models = [by_display[m] for m in args.models]
    else:
        models = run_set
    displays = [d for _, _, d in models]

    if args.plot_only:
        make_plots(displays, experiments)
        return

    n_series = args.n_series or (TEST_N_SERIES if args.test else N_SERIES)

    # Model-major task order so each worker's model cache gets long runs.
    tasks = [(mid, fam, d, e)
             for mid, fam, d in models for e in experiments
             if args.force or not _cell_done(d, e)]
    total_cells = len(models) * len(experiments)
    print(Fore.GREEN + f"synth_param_sweeps: {len(tasks)}/{total_cells} cells to "
          + f"run ({len(models)} models × {len(experiments)} experiments, "
          + f"{n_series} series/bin)" + Fore.RESET)

    if tasks:
        devices = resolve_devices(args.device)
        ctx = mp.get_context("spawn")
        task_queue: "mp.Queue" = ctx.Queue()
        result_queue: "mp.Queue" = ctx.Queue()
        for t in tasks:
            task_queue.put(t)
        for _ in devices:
            task_queue.put(None)

        procs = [ctx.Process(target=gpu_worker,
                             args=(i, dev, task_queue, result_queue,
                                   n_series, args.batch_size, args.test))
                 for i, dev in enumerate(devices)]
        for pr in procs:
            pr.start()
        failures = []
        for _ in range(len(tasks)):
            res = result_queue.get()
            if res["status"] != "ok":
                failures.append(res)
        for pr in procs:
            pr.join()
        if failures:
            cells = ", ".join(f"{d}/{e}" for d, e in (f["cell"] for f in failures))
            print(Fore.RED + f"{len(failures)} cell(s) failed: {cells}" + Fore.RESET)

    make_plots(displays, experiments)


if __name__ == "__main__":
    main()
