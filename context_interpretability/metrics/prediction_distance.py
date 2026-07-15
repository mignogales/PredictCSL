"""
Prediction-distance metrics (spec §3.5) — changes in model OUTPUT measured
independently of ground truth. All point-forecast functions take (N, H) arrays
and return (N,) per-sample distances.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

EPS = 1e-8


def _as2d(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a[None, :] if a.ndim == 1 else a


def l1_distance(pred_a, pred_b) -> np.ndarray:
    """Mean absolute prediction difference over the horizon."""
    a, b = _as2d(pred_a), _as2d(pred_b)
    return np.nanmean(np.abs(a - b), axis=-1)


def normalized_distance(pred_a, pred_b) -> np.ndarray:
    """``|y1 - y2|_1 / (|y1|_1 + eps)`` — scale-free prediction change."""
    a, b = _as2d(pred_a), _as2d(pred_b)
    return (np.nansum(np.abs(a - b), axis=-1)
            / (np.nansum(np.abs(a), axis=-1) + EPS))


# -- probabilistic (used only when an adapter exposes quantile forecasts) -----

def quantile_distance(qa, qb) -> np.ndarray:
    """Mean |Δ| across the (N, Q, H) quantile grids."""
    return np.nanmean(np.abs(np.asarray(qa, float) - np.asarray(qb, float)),
                      axis=(1, 2))


def wasserstein1_from_quantiles(qa, qb) -> np.ndarray:
    """W1 between the discretized predictive distributions, averaged over H.

    With both forecasts on the SAME quantile levels, W1 is the mean absolute
    difference of the quantile functions — identical to :func:`quantile_distance`
    on an even level grid; kept separate for uneven grids.
    """
    return quantile_distance(qa, qb)


def crps_difference(crps_a: np.ndarray, crps_b: np.ndarray) -> np.ndarray:
    return np.asarray(crps_a, float) - np.asarray(crps_b, float)


def kl_divergence_gaussian(mean_a, std_a, mean_b, std_b) -> np.ndarray:
    """KL(N_a || N_b), horizon-averaged — ONLY for adapters with parametric
    Gaussian heads (numerically guarded; spec: use only when compatible)."""
    ma, mb = _as2d(mean_a), _as2d(mean_b)
    sa = np.maximum(_as2d(std_a), EPS)
    sb = np.maximum(_as2d(std_b), EPS)
    kl = np.log(sb / sa) + (sa ** 2 + (ma - mb) ** 2) / (2.0 * sb ** 2) - 0.5
    return np.nanmean(kl, axis=-1)
