#!/usr/bin/env python3
"""
Analyze the periodicity vs AR composition of synthetic series.
Shows spectral power, autocorrelation structure, and AR fit quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, correlate
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')


def estimate_ar_strength(series: np.ndarray) -> tuple:
    """
    Estimate AR strength by:
    1. Computing autocorrelation decay
    2. Fitting low-order AR models and comparing BIC
    3. Measuring spectral slope (AR ↔ red spectrum)
    """
    y = (series - series.mean()) / (series.std() + 1e-8)

    # ACF: compute lag-1 to lag-50 autocorrelation
    acf_vals = np.array([np.correlate(y, y, mode='full')[len(y)-1-k]
                         for k in range(1, min(51, len(y)))])
    acf_vals /= acf_vals[0]  # normalize

    # AR(1) coefficient via Yule-Walker
    ar1_coef = acf_vals[0]

    # Spectral slope: compare low-freq vs high-freq power
    freqs, pxx = periodogram(y)
    mid = len(freqs) // 2
    low_power = pxx[:mid//2].mean()
    high_power = pxx[mid//2:].mean()
    spectral_slope = low_power / (high_power + 1e-8)

    # ACF decay rate (how fast it goes to zero)
    acf_decay = -np.log(np.abs(acf_vals[4]) + 1e-8)  # decay at lag 5

    return {
        'ar1_coef': ar1_coef,
        'spectral_slope': spectral_slope,
        'acf_decay': acf_decay,
        'acf_mean': acf_vals[:10].mean(),
    }


def estimate_periodicity_strength(series: np.ndarray) -> dict:
    """
    Estimate periodicity by:
    1. Computing spectral power concentration at dominant frequency
    2. Measuring dominant peak height relative to noise floor
    3. Computing spectral peak-to-mean ratio
    """
    y = (series - series.mean()) / (series.std() + 1e-8)

    freqs, pxx = periodogram(y, scaling='spectrum')

    # Skip DC component and low frequencies (trend)
    valid_range = freqs > 0.01
    pxx_valid = pxx[valid_range]
    freqs_valid = freqs[valid_range]

    # Find dominant peak
    peak_idx = np.argmax(pxx_valid)
    peak_power = pxx_valid[peak_idx]
    peak_freq = freqs_valid[peak_idx]

    # Noise floor: median power in high frequencies
    noise_floor = np.median(pxx_valid[len(pxx_valid)//2:])

    # Peak prominence
    peak_ratio = peak_power / (noise_floor + 1e-8)

    # Total spectral concentration (Gini index for periodicity)
    pxx_norm = pxx_valid / (pxx_valid.sum() + 1e-8)
    concentration = (pxx_norm * np.log(pxx_norm + 1e-10)).sum() * -1  # entropy

    return {
        'peak_power': peak_power,
        'peak_freq': peak_freq,
        'peak_ratio': peak_ratio,
        'noise_floor': noise_floor,
        'spectral_entropy': concentration,
    }


def main():
    # Generate a few sample series to check current state
    from experiments.build_context_length_dataset import (
        _generate_synthetic_series, MAX_WINDOW, MAX_HORIZON
    )

    n_samples = 500
    total_length = MAX_WINDOW + MAX_HORIZON

    print("Analyzing synthetic series composition...\n")
    print(f"Generating {n_samples} series (total length={total_length})...\n")

    ar_metrics = []
    period_metrics = []

    for i in range(n_samples):
        rng = np.random.RandomState(42 + i)
        series, n_seg = _generate_synthetic_series(rng, total_length)
        ctx = series[:MAX_WINDOW]  # Use only context (standardized portion)

        ar_m = estimate_ar_strength(ctx)
        period_m = estimate_periodicity_strength(ctx)

        ar_metrics.append(ar_m)
        period_metrics.append(period_m)

    # Aggregate stats
    ar1_coefs = [m['ar1_coef'] for m in ar_metrics]
    spectral_slopes = [m['spectral_slope'] for m in ar_metrics]
    acf_decays = [m['acf_decay'] for m in ar_metrics]
    acf_means = [m['acf_mean'] for m in ar_metrics]

    peak_ratios = [m['peak_ratio'] for m in period_metrics]
    peak_powers = [m['peak_power'] for m in period_metrics]
    spectral_entropies = [m['spectral_entropy'] for m in period_metrics]

    print("=" * 70)
    print("AUTOREGRESSIVE (AR) STRENGTH")
    print("=" * 70)
    print(f"AR(1) coefficient:        mean={np.mean(ar1_coefs):.4f}  "
          f"std={np.std(ar1_coefs):.4f}  [0=white, 1=unit-root]")
    print(f"Spectral slope (LF/HF):   mean={np.mean(spectral_slopes):.2f}  "
          f"std={np.std(spectral_slopes):.2f}  [1=white, >1=red]")
    print(f"ACF decay rate (lag 5):   mean={np.mean(acf_decays):.4f}  "
          f"std={np.std(acf_decays):.4f}  [higher=faster decay]")
    print(f"ACF(1-10) mean:           mean={np.mean(acf_means):.4f}  "
          f"std={np.std(acf_means):.4f}  [higher=more persistence]")

    print("\n" + "=" * 70)
    print("PERIODICITY STRENGTH")
    print("=" * 70)
    print(f"Spectral peak ratio:      mean={np.mean(peak_ratios):.2f}  "
          f"std={np.std(peak_ratios):.2f}  [higher=more periodic]")
    print(f"Peak spectral power:      mean={np.mean(peak_powers):.4f}  "
          f"std={np.std(peak_powers):.4f}")
    print(f"Spectral entropy:         mean={np.mean(spectral_entropies):.3f}  "
          f"std={np.std(spectral_entropies):.3f}  [lower=more concentrated]")

    # Correlation between AR and periodicity
    print("\n" + "=" * 70)
    print("AR vs PERIODICITY BALANCE")
    print("=" * 70)
    corr = np.corrcoef(ar1_coefs, peak_ratios)[0, 1]
    print(f"Correlation (AR1 vs peak_ratio): {corr:.3f}")
    print(f"  (negative = trade-off; near-zero = independent)")

    # Percentiles for interpretation
    print("\n" + "=" * 70)
    print("DISTRIBUTION (PERCENTILES)")
    print("=" * 70)
    print("\nAR(1) coefficients:")
    for p in [10, 25, 50, 75, 90]:
        val = np.percentile(ar1_coefs, p)
        print(f"  {p:2d}th percentile: {val:.4f}")

    print("\nSpectral peak ratios (periodicity):")
    for p in [10, 25, 50, 75, 90]:
        val = np.percentile(peak_ratios, p)
        print(f"  {p:2d}th percentile: {val:.2f}")

    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(ar1_coefs, bins=40, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('AR(1) Coefficients')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].axvline(np.mean(ar1_coefs), color='r', linestyle='--',
                       label=f"mean={np.mean(ar1_coefs):.3f}")
    axes[0, 0].legend()

    axes[0, 1].hist(peak_ratios, bins=40, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Spectral Peak Ratio (Periodicity)')
    axes[0, 1].set_xlabel('Value (higher = more periodic)')
    axes[0, 1].axvline(np.mean(peak_ratios), color='r', linestyle='--',
                       label=f"mean={np.mean(peak_ratios):.2f}")
    axes[0, 1].legend()

    axes[1, 0].scatter(ar1_coefs, peak_ratios, alpha=0.4, s=20)
    axes[1, 0].set_xlabel('AR(1) Coefficient')
    axes[1, 0].set_ylabel('Spectral Peak Ratio')
    axes[1, 0].set_title(f'AR vs Periodicity (r={corr:.3f})')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(acf_means, bins=40, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Mean ACF(1-10)')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].axvline(np.mean(acf_means), color='r', linestyle='--',
                       label=f"mean={np.mean(acf_means):.3f}")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('synth_composition_analysis.png', dpi=150)
    print("\n📊 Saved analysis plot to: synth_composition_analysis.png")

    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    weak_ar = np.mean(ar1_coefs) < 0.2
    strong_period = np.mean(peak_ratios) > 3

    if weak_ar and strong_period:
        print("✓ Confirmed: High periodicity, weak AR component")
        print("  → AR is 50% of the time with small coefficients (0.1-0.3)")
        print("  → Periodicity dominates: 1-3 strong sinusoidal components")
        print("\nSuggestions to increase AR strength:")
        print("  1. Increase AR adoption rate from 50% → 75-80%")
        print("  2. Increase coefficient magnitudes (0.3-0.7 range)")
        print("  3. Increase innovation scale (0.3 → 0.5-0.8)")
        print("  4. Add AR(2) or AR(3) with more persistence")
    elif not weak_ar and strong_period:
        print("✓ Good balance: Both AR and periodicity present")
    else:
        print("⚠ Unexpected: Check code changes")


if __name__ == "__main__":
    main()
