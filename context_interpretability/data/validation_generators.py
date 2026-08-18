"""Controlled evaluation pools for attention-masking validation.

The harmonic pools are deliberately transparent: every continuation is an
exact sum of sinusoids, with no regime changes, padding, missing values, or
stochastic innovations.  Three global scales share the same underlying waves,
which makes scale-dependent preprocessing failures easy to detect.

The KernelSynth pool follows the synthetic-data construction from the Chronos
paper and its official ``scripts/kernel-synth.py`` implementation: draw one to
five kernels with replacement from the 33-entry kernel bank, combine them with
random ``+``/``*`` operations, and sample from the resulting zero-mean GP.
This local implementation keeps the method configurable in length and adds a
small diagonal jitter for numerical stability.  Optional per-series standard
deviation normalization is explicit metadata, not part of upstream
KernelSynth.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from context_interpretability.experiments.common import ExperimentData


DEFAULT_PERIODS = (32, 64, 128, 256, 512, 1024, 2048)
DEFAULT_SCALES = (0.5, 1.0, 2.0)


def _scale_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _base_harmonic_series(total_length: int, instance: int, seed: int,
                          periods: Sequence[int], max_tones: int) -> np.ndarray:
    """One deterministic random-phase mixture with balanced tone counts."""
    rng = np.random.default_rng((int(seed), int(instance)))
    n_tones = 1 + (int(instance) % max(1, int(max_tones)))
    chosen = rng.choice(np.asarray(periods, dtype=int), size=n_tones,
                        replace=n_tones > len(periods))
    # Unequal component amplitudes test mixtures without allowing one random
    # series to dominate the aggregate MSE merely because it has more tones.
    relative = rng.choice(np.asarray([0.25, 0.5, 1.0, 2.0]),
                          size=n_tones, replace=True)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_tones)
    t = np.arange(int(total_length), dtype=np.float64)
    x = np.zeros_like(t)
    for period, amplitude, phase in zip(chosen, relative, phases):
        x += amplitude * np.sin(2.0 * np.pi * t / float(period) + phase)
    rms = float(np.sqrt(np.mean(x * x)))
    if rms > 0:
        x /= rms
    return x


def load_harmonic_pools(n_instances: int, series_length: int, horizon: int,
                        seed: int = 0,
                        periods: Sequence[int] = DEFAULT_PERIODS,
                        scales: Sequence[float] = DEFAULT_SCALES,
                        max_tones: int = 4) -> List[ExperimentData]:
    """Return clean multi-tone pools sharing waves across global scales."""
    if not periods or any(int(p) <= 1 for p in periods):
        raise ValueError("harmonic periods must all be greater than one")
    if not scales or any(float(a) <= 0 for a in scales):
        raise ValueError("harmonic scales must all be positive")
    total = int(series_length) + int(horizon)
    base = np.stack([
        _base_harmonic_series(total, i, seed, periods, max_tones)
        for i in range(int(n_instances))
    ])
    pools: List[ExperimentData] = []
    for scale in scales:
        values = (float(scale) * base).astype(np.float32)
        tag = _scale_tag(float(scale))
        pools.append(ExperimentData(
            name=f"harmonic_clean_scale_{tag}",
            contexts=values[:, :series_length],
            targets=values[:, series_length:],
            sample_ids=[f"harmonic_{seed}_{i}" for i in range(len(values))],
            season_length=1,
            metadata={
                "source": "harmonic_clean",
                "seed": int(seed),
                "periods": [int(p) for p in periods],
                "max_tones": int(max_tones),
                "global_scale": float(scale),
                "noise": 0.0,
            },
        ))
    return pools


# Chronos' 33-entry bank, represented without a scikit-learn runtime
# dependency. Periods are specified in timesteps and converted to the [0, 1]
# coordinate system used by the official generator.
_PERIODS = (24, 48, 96, 24 * 7, 48 * 7, 96 * 7, 7, 14, 30, 60,
            365, 365 * 2, 4, 26, 52, 4, 6, 12, 4, 4 * 10, 10)
_KERNEL_BANK = (
    [("periodic", float(p)) for p in _PERIODS]
    + [("dot", 0.0), ("dot", 1.0), ("dot", 10.0)]
    + [("rbf", 0.1), ("rbf", 1.0), ("rbf", 10.0)]
    + [("rq", 0.1), ("rq", 1.0), ("rq", 10.0)]
    + [("white", 0.1), ("white", 1.0), ("constant", 1.0)]
)


def _kernel_covariance(spec: Tuple[str, float], x: np.ndarray,
                       total_length: int) -> np.ndarray:
    kind, value = spec
    delta = x[:, None] - x[None, :]
    if kind == "periodic":
        periodicity = value / float(total_length)
        return np.exp(-2.0 * np.sin(np.pi * delta / periodicity) ** 2)
    if kind == "dot":
        return np.outer(x, x) + value ** 2
    if kind == "rbf":
        return np.exp(-(delta ** 2) / (2.0 * value ** 2))
    if kind == "rq":
        # Official bank varies alpha and leaves length_scale at its default 1.
        return (1.0 + delta ** 2 / (2.0 * value)) ** (-value)
    if kind == "white":
        return value * np.eye(len(x), dtype=np.float64)
    if kind == "constant":
        return np.ones((len(x), len(x)), dtype=np.float64)
    raise ValueError(f"unknown KernelSynth kernel {kind!r}")


def _kernel_synth_series(total_length: int, max_kernels: int,
                         seed: int, instance: int) -> np.ndarray:
    rng = np.random.default_rng((int(seed), int(instance)))
    x = np.linspace(0.0, 1.0, int(total_length), dtype=np.float64)
    n_kernels = int(rng.integers(1, int(max_kernels) + 1))
    chosen = rng.integers(0, len(_KERNEL_BANK), size=n_kernels)
    covariance = _kernel_covariance(
        _KERNEL_BANK[int(chosen[0])], x, total_length)
    for index in chosen[1:]:
        component = _kernel_covariance(
            _KERNEL_BANK[int(index)], x, total_length)
        covariance = (covariance + component if rng.random() < 0.5
                      else covariance * component)
    diag_mean = max(float(np.mean(np.diag(covariance))), 1.0)
    covariance.flat[::total_length + 1] += 1e-8 * diag_mean
    # Sample explicitly from the PSD eigensystem.  NumPy's generic
    # multivariate_normal path emits spurious BLAS overflow warnings for some
    # very low-rank product kernels even though their eigenvalues are finite.
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    coefficients = np.sqrt(eigenvalues) * rng.standard_normal(
        int(total_length))
    # einsum avoids an Accelerate/BLAS warning seen for rank-deficient 32x32
    # kernels on macOS while performing the same matrix-vector product.
    return np.einsum("ij,j->i", eigenvectors, coefficients, optimize=False)


def _stationary_fft_sample(spec: Tuple[str, float], total_length: int,
                           rng: np.random.Generator) -> np.ndarray:
    """O(T log T) circulant-embedding sample for a stationary base kernel."""
    n = int(total_length)
    lags = np.arange(n, dtype=np.float64) / max(1, n - 1)
    kind, value = spec
    if kind == "periodic":
        periodicity = value / float(n)
        covariance = np.exp(
            -2.0 * np.sin(np.pi * lags / periodicity) ** 2)
    elif kind == "rbf":
        covariance = np.exp(-(lags ** 2) / (2.0 * value ** 2))
    elif kind == "rq":
        covariance = (1.0 + lags ** 2 / (2.0 * value)) ** (-value)
    else:
        raise ValueError(f"{kind!r} is not a stationary smooth kernel")
    # Embed the Toeplitz covariance in a symmetric circulant matrix. Tiny
    # negative FFT eigenvalues can arise when the minimal embedding is not
    # positive definite; clipping gives a stable covariance approximation and
    # is recorded in the dataset metadata.
    first_row = np.concatenate([covariance, covariance[-2:0:-1]])
    eigenvalues = np.fft.rfft(first_row).real
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    noise = rng.standard_normal(len(first_row))
    sample = np.fft.irfft(
        np.sqrt(eigenvalues) * np.fft.rfft(noise), n=len(first_row))
    return sample[:n]


def _scalable_base_sample(spec: Tuple[str, float], total_length: int,
                          rng: np.random.Generator) -> np.ndarray:
    kind, value = spec
    if kind in {"periodic", "rbf", "rq"}:
        return _stationary_fft_sample(spec, total_length, rng)
    if kind == "dot":
        x = np.linspace(0.0, 1.0, int(total_length), dtype=np.float64)
        return rng.standard_normal() * x + value * rng.standard_normal()
    if kind == "white":
        return np.sqrt(value) * rng.standard_normal(int(total_length))
    if kind == "constant":
        return np.full(int(total_length), rng.standard_normal(),
                       dtype=np.float64)
    raise ValueError(f"unknown KernelSynth kernel {kind!r}")


def _kernel_synth_series_scalable(total_length: int, max_kernels: int,
                                  seed: int, instance: int) -> np.ndarray:
    """Covariance-matched scalable approximation to composite KernelSynth.

    Independent base-kernel samples are added or multiplied according to the
    upstream random expression. Addition and multiplication preserve the sum
    and product covariance respectively. Product expressions are not Gaussian,
    so this is explicitly a long-context approximation rather than an exact
    sample from the final composite GP.
    """
    rng = np.random.default_rng((int(seed), int(instance)))
    n_kernels = int(rng.integers(1, int(max_kernels) + 1))
    chosen = rng.integers(0, len(_KERNEL_BANK), size=n_kernels)
    sample = _scalable_base_sample(
        _KERNEL_BANK[int(chosen[0])], total_length, rng)
    for index in chosen[1:]:
        component = _scalable_base_sample(
            _KERNEL_BANK[int(index)], total_length, rng)
        sample = sample + component if rng.random() < 0.5 else sample * component
    return sample


def load_kernelsynth_pool(n_instances: int, series_length: int, horizon: int,
                          seed: int = 0, max_kernels: int = 5,
                          normalize_std: bool = True,
                          cache_path: str | None = None,
                          scalable: bool = False) -> ExperimentData:
    """Generate or load a deterministic Chronos-style KernelSynth pool."""
    total = int(series_length) + int(horizon)
    contexts = targets = None
    if cache_path and os.path.isfile(cache_path):
        with np.load(cache_path) as cached:
            if (int(cached["series_length"]) == int(series_length)
                    and int(cached["horizon"]) == int(horizon)
                    and int(cached["n_instances"]) == int(n_instances)
                    and int(cached["seed"]) == int(seed)
                    and int(cached["max_kernels"]) == int(max_kernels)
                    and bool(cached["normalize_std"]) == bool(normalize_std)
                    and bool(cached.get("scalable", False)) == bool(scalable)):
                contexts = cached["contexts"].astype(np.float32)
                targets = cached["targets"].astype(np.float32)
    if contexts is None or targets is None:
        generator = (_kernel_synth_series_scalable if scalable
                     else _kernel_synth_series)
        values = np.stack([
            generator(total, max_kernels, seed, i)
            for i in range(int(n_instances))
        ])
        if normalize_std:
            std = values.std(axis=1, keepdims=True)
            values = values / np.where(std > 1e-8, std, 1.0)
        contexts = values[:, :series_length].astype(np.float32)
        targets = values[:, series_length:].astype(np.float32)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            tmp = cache_path + ".tmp.npz"
            np.savez_compressed(
                tmp, contexts=contexts, targets=targets,
                series_length=int(series_length), horizon=int(horizon),
                n_instances=int(n_instances), seed=int(seed),
                max_kernels=int(max_kernels),
                normalize_std=bool(normalize_std), scalable=bool(scalable))
            os.replace(tmp, cache_path)
    return ExperimentData(
        name=("kernelsynth_long_covariance_approx" if scalable else
              "kernelsynth_chronos"),
        contexts=contexts,
        targets=targets,
        sample_ids=[f"kernelsynth_{seed}_{i}" for i in range(len(contexts))],
        season_length=1,
        metadata={
            "source": "chronos_kernelsynth",
            "seed": int(seed),
            "max_kernels": int(max_kernels),
            "kernel_bank_size": len(_KERNEL_BANK),
            "normalize_std": bool(normalize_std),
            "scalable": bool(scalable),
            "sampling": ("circulant_base_samples_covariance_matched_composition"
                         if scalable else "exact_composite_covariance_eigh"),
            "upstream": "amazon-science/chronos-forecasting/scripts/kernel-synth.py",
        },
    )
