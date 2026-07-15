"""
Per-sample forecast losses (spec §3.4).

Every function takes ``pred``/``target`` of shape (N, H) and returns a (N,)
per-sample loss so paired statistics (clean vs intervened on the SAME sample)
stay exact. ``loss_delta = intervened - clean`` is computed by the experiments;
positive delta == the intervention worsened forecasting.

MASE follows the project's gluonts-port definition (per-instance ratio
``mean_h(|y - yhat|) / seasonal_error_instance`` — see
experiments/gifteval_mase.py); on synthetic pools ``season_length=1``.

CRPS is available only for probabilistic outputs (quantile forecasts); the
current adapters return median point forecasts, so experiments record CRPS only
when the adapter exposes quantiles (logged otherwise).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

EPS = 1e-8


def _as2d(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a[None, :] if a.ndim == 1 else a


def mae(pred, target) -> np.ndarray:
    p, t = _as2d(pred), _as2d(target)
    return np.nanmean(np.abs(p - t), axis=-1)


def mse(pred, target) -> np.ndarray:
    p, t = _as2d(pred), _as2d(target)
    return np.nanmean((p - t) ** 2, axis=-1)


def smape(pred, target) -> np.ndarray:
    p, t = _as2d(pred), _as2d(target)
    denom = np.abs(p) + np.abs(t)
    return np.nanmean(2.0 * np.abs(p - t) / np.where(denom == 0, np.nan, denom),
                      axis=-1)


def seasonal_error(context, season_length: int = 1) -> np.ndarray:
    """Per-instance in-sample seasonal-naive MAE (the MASE denominator).

    ``context``: (N, T) raw history (NaNs allowed and ignored, matching the
    raw-context seasonal errors used by stage 3).
    """
    c = _as2d(context)
    m = max(1, int(season_length))
    if c.shape[1] <= m:
        return np.full(c.shape[0], np.nan)
    diff = np.abs(c[:, m:] - c[:, :-m])
    return np.nanmean(diff, axis=-1)


def mase(pred, target, context=None, season_length: int = 1,
         seasonal_err: Optional[np.ndarray] = None) -> np.ndarray:
    if seasonal_err is None:
        if context is None:
            raise ValueError("mase needs `context` or a precomputed seasonal_err")
        seasonal_err = seasonal_error(context, season_length)
    se = np.where(np.asarray(seasonal_err) <= 0, np.nan, seasonal_err)
    return mae(pred, target) / se


def crps_from_quantiles(quantile_preds, quantile_levels: Sequence[float],
                        target) -> np.ndarray:
    """Approximate CRPS via the mean weighted quantile (pinball) loss x2.

    ``quantile_preds``: (N, Q, H); standard gluonts-style discretization.
    """
    q = np.asarray(quantile_preds, dtype=np.float64)
    t = _as2d(target)[:, None, :]
    levels = np.asarray(quantile_levels, dtype=np.float64)[None, :, None]
    err = t - q
    pinball = np.maximum(levels * err, (levels - 1.0) * err)
    return 2.0 * np.nanmean(pinball, axis=(1, 2))


_METRICS = {"mae": mae, "mse": mse, "smape": smape}


def compute_loss(name: str, pred, target, context=None,
                 season_length: int = 1) -> np.ndarray:
    """Dispatch a per-sample loss by name (mae|mse|smape|mase)."""
    if name == "mase":
        return mase(pred, target, context=context, season_length=season_length)
    if name not in _METRICS:
        raise ValueError(f"Unknown metric {name!r} (mae|mse|smape|mase)")
    return _METRICS[name](pred, target)
