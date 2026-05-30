"""
Demo: Sample AR processes with more variety.
Shows different AR behaviors and how to ensure stability.
"""

import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt


def sample_ar_params_diverse(rng: np.random.RandomState) -> tuple:
    """
    Sample AR(p) parameters with diversity.

    Returns:
        (p, coeffs) where p ∈ {1,2,3} and coeffs are stable AR roots.

    Strategy:
    - AR(1): Sample from full stability range (closer to persistence)
    - AR(2): Mix of persistent, oscillatory, and weak
    - AR(3): Complex behaviors with diverse timescales
    """

    # More series with AR (vs. current 50%)
    if rng.uniform() < 0.5:
        return None, None  # No AR for ~50% (keeps diversity)

    # Choose AR order with different probabilities
    # More emphasis on AR(1) and AR(2) which are more interpretable
    p_choice = rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])

    if p_choice == 1:
        # AR(1): φ ∈ [-1, 1] for stability
        # Weight toward persistence: more mass near 0.5-0.9
        if rng.uniform() < 0.6:
            # Persistent: φ ∈ [0.4, 0.95]
            phi = rng.uniform(0.4, 0.95)
        elif rng.uniform() < 0.5:
            # Weak positive: φ ∈ [0.0, 0.3]
            phi = rng.uniform(0.0, 0.3)
        else:
            # Negative (oscillatory damping): φ ∈ [-0.7, 0.0]
            phi = rng.uniform(-0.7, 0.0)
        coeffs = np.array([phi])

    elif p_choice == 2:
        # AR(2): Sample roots and convert to coefficients
        # Can be: (a) both real, (b) complex conjugate pair (oscillatory)
        if rng.uniform() < 0.5:
            # Oscillatory: complex conjugate pair
            # r ∈ [0.7, 0.98] (magnitude), ω ∈ [0.1π, 0.9π] (frequency)
            r = rng.uniform(0.7, 0.98)
            omega = rng.uniform(0.1 * np.pi, 0.9 * np.pi)
            # Convert: z = r * e^(±iω) → AR coefficients
            c1 = 2 * r * np.cos(omega)
            c2 = -(r ** 2)
            coeffs = np.array([c1, c2])
        else:
            # Two real roots: r1, r2 ∈ [0.2, 0.95]
            r1 = rng.uniform(0.2, 0.95)
            r2 = rng.uniform(0.2, 0.95)
            # AR coefficients from (1 - r1 B)(1 - r2 B) = 1 - (r1+r2)B + r1*r2*B^2
            c1 = -(r1 + r2)
            c2 = r1 * r2
            coeffs = np.array([c1, c2])

    else:  # p_choice == 3
        # AR(3): Complex behavior, sample 3 roots (could be real + complex pair)
        if rng.uniform() < 0.5:
            # One real + one complex conjugate pair
            r_real = rng.uniform(0.3, 0.9)
            r_complex = rng.uniform(0.6, 0.95)
            omega = rng.uniform(0.1 * np.pi, 0.8 * np.pi)

            # Expand (1 - r_real*B) * (1 - 2*r*cos(ω)*B + r^2*B^2)
            c1 = -(r_real + 2 * r_complex * np.cos(omega))
            c2 = 2 * r_real * r_complex * np.cos(omega) + r_complex ** 2
            c3 = -(r_real * r_complex ** 2)
            coeffs = np.array([c1, c2, c3])
        else:
            # Three real roots
            r1 = rng.uniform(0.3, 0.9)
            r2 = rng.uniform(0.3, 0.9)
            r3 = rng.uniform(0.3, 0.9)

            c1 = -(r1 + r2 + r3)
            c2 = r1*r2 + r1*r3 + r2*r3
            c3 = -(r1 * r2 * r3)
            coeffs = np.array([c1, c2, c3])

    return p_choice, coeffs


def test_ar_samples(n_samples: int = 1000, n_per_ar: int = 100):
    """Generate and analyze AR samples to show diversity."""

    all_coeffs_by_order = {1: [], 2: [], 3: []}

    rng = np.random.RandomState(42)
    for _ in range(n_samples):
        p, coeffs = sample_ar_params_diverse(rng)
        if coeffs is not None:
            all_coeffs_by_order[p].append(coeffs)

    print("=" * 70)
    print("AR SAMPLING DIVERSITY")
    print("=" * 70)

    for p in [1, 2, 3]:
        if len(all_coeffs_by_order[p]) == 0:
            continue
        coeffs_arr = np.array(all_coeffs_by_order[p])
        print(f"\nAR({p}): {len(all_coeffs_by_order[p])} samples")
        for i in range(p):
            col = coeffs_arr[:, i]
            print(f"  coeff[{i}]: μ={col.mean():.3f}, σ={col.std():.3f}, "
                  f"range=[{col.min():.3f}, {col.max():.3f}]")

    # Visualize: generate a few AR(1) and AR(2) time series
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    rng = np.random.RandomState(99)
    length = 300

    # AR(1) examples
    for i in range(3):
        p, coeffs = None, None
        while coeffs is None or p != 1:
            p, coeffs = sample_ar_params_diverse(rng)

        innov = rng.normal(0.0, 1.0, size=length)
        a = np.concatenate([[1.0], -coeffs])
        y = lfilter([1.0], a, innov)

        axes[0, i].plot(y, linewidth=1, alpha=0.8)
        axes[0, i].set_title(f"AR(1): φ={coeffs[0]:.2f}")
        axes[0, i].set_ylim(-4, 4)
        axes[0, i].grid(alpha=0.3)

    # AR(2) examples
    for i in range(3):
        p, coeffs = None, None
        while coeffs is None or p != 2:
            p, coeffs = sample_ar_params_diverse(rng)

        innov = rng.normal(0.0, 1.0, size=length)
        a = np.concatenate([[1.0], -coeffs])
        y = lfilter([1.0], a, innov)

        axes[1, i].plot(y, linewidth=1, alpha=0.8)
        axes[1, i].set_title(f"AR(2): φ=[{coeffs[0]:.2f}, {coeffs[1]:.2f}]")
        axes[1, i].set_ylim(-4, 4)
        axes[1, i].grid(alpha=0.3)

    fig.suptitle("Diverse AR Process Samples", fontsize=14)
    plt.tight_layout()
    plt.savefig('ar_diversity_demo.png', dpi=150)
    print("\n📊 Saved to: ar_diversity_demo.png")


if __name__ == "__main__":
    test_ar_samples()
