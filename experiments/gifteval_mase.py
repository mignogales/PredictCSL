"""Leaderboard-faithful (gluonts) MASE primitives.

The project's stage-3 ablation reports MASE as ``global_MAE / pooled_training_
seasonal_naive_MAE`` with a custom seasonality map (``D->7, W->52, ...``). That is
*not* the quantity the HuggingFace GiftEval leaderboard reports, so the numbers
don't line up. The leaderboard uses gluonts' definition:

    * seasonal error is computed **per test instance from that instance's own
      context** (``calculate_seasonal_error``), and
    * MASE is the global mean over valid forecast points of
      ``|y - yhat| / seasonal_error(instance)``, with the seasonality ``m`` taken
      from gluonts' ``DEFAULT_SEASONALITIES`` map.

This module ports those two pieces verbatim (see
https://github.com/awslabs/gluonts ``time_feature/seasonality.py`` and
``evaluation/metrics.py``). The GiftEval ablation
(``test_window_ablation_gifteval_v5.py``) uses them to compute ``mase_gluonts``
as a first-class metric alongside the project's own ``mase`` — the per-instance
seasonal errors are precomputed once per dataset in ``GiftEvalCache``, so existing
cells can even be backfilled from cached per-instance MAE without any TSFM
re-inference. Keeping the definition here — one auditable place — means every
consumer shares exactly the leaderboard semantics.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from pandas.tseries.frequencies import to_offset


# gluonts ``DEFAULT_SEASONALITIES`` (number of steps in one natural cycle):
#   secondly -> an hour, minutely/hourly -> a day, daily/weekly -> lag-1 (!),
#   monthly/quarterly -> a year, business-daily -> a week.
# Note the leaderboard quirk that bites daily/weekly data: D and W map to 1
# (plain lag-1 naive), NOT 7 / 52.
GLUONTS_SEASONALITIES: Dict[str, int] = {
    "S": 3600,
    "T": 1440,
    "H": 24,
    "D": 1,
    "W": 1,
    "M": 12,
    "B": 5,
    "Q": 4,
}

# pandas >= 2.2 renamed several offset aliases; normalise them back to the keys
# above so ``to_offset(freq).name`` lookups keep working across pandas versions.
_ALIAS_TO_KEY: Dict[str, str] = {
    "min": "T", "T": "T",
    "h": "H", "H": "H",
    "s": "S", "S": "S",
    "ME": "M", "MS": "M", "M": "M",
    "QE": "Q", "QS": "Q", "Q": "Q",
    "YE": "Y", "YS": "Y", "Y": "Y", "A": "Y",
    "D": "D", "W": "W", "B": "B",
}


def _norm_freq_str(offset) -> str:
    """gluonts ``norm_freq_str``: base name with any anchoring suffix stripped,
    normalised across pandas alias renames."""
    name = offset.name.split("-")[0]
    return _ALIAS_TO_KEY.get(name, name)


def get_seasonality(freq: str, seasonalities: Dict[str, int] = GLUONTS_SEASONALITIES) -> int:
    """Port of gluonts ``get_seasonality``.

    Looks up the base seasonality for ``freq`` and divides by the offset's
    multiplier (so ``15T`` -> 1440 / 15 = 96). Falls back to 1 when the frequency
    is unknown or the multiplier doesn't divide the base evenly.
    """
    offset = to_offset(freq)
    base = seasonalities.get(_norm_freq_str(offset), 1)
    seasonality, remainder = divmod(base, offset.n)
    if remainder == 0:
        return seasonality
    return 1


def seasonal_error(context: np.ndarray, season: int) -> float:
    """Exact univariate port of ``gluonts.ev.ts_stats.seasonal_error``.

    GluonTS masks invalid context values, switches to lag 1 only when the
    requested seasonality is *longer* than the context, and otherwise returns
    ``mean(abs(x[m:] - x[:-m]))``. It does not replace zero denominators.
    """
    x = np.ma.masked_invalid(np.asarray(context, dtype=np.float64))
    n = x.shape[-1]
    m = int(season)
    if m > n:
        m = 1
    value = np.abs(x[m:] - x[:-m]).mean(axis=-1, keepdims=True)
    # ``evaluate_forecasts`` places the masked-array result into np.array, which
    # intentionally exposes the underlying value (zero for a fully masked mean).
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def per_instance_seasonal_errors(contexts: Sequence[np.ndarray], season: int) -> np.ndarray:
    """Per-instance seasonal error for a list of contexts (one float per series)."""
    return np.array([seasonal_error(c, season) for c in contexts], dtype=np.float64)


# ==============================================================================
#  Vectorized equivalent of the leaderboard's gluonts machinery
# ==============================================================================
# This avoids constructing one Forecast object per test instance and the
# ``evaluate_forecasts`` Python loop (45k instances took about a minute in the
# Bitbrains case). The formulas below were checked against the installed GluonTS
# implementation, including QuantileForecast/SampleForecast 0.5 selection.


def _forecast_point_05(median, quantiles=None, quantile_levels=None, samples=None):
    """Return exactly what GluonTS Forecast objects expose as ``forecast['0.5']``."""
    median = np.asarray(median, dtype=np.float64)
    if quantiles is not None and quantile_levels is not None:
        levels = [float(level) for level in quantile_levels]
        matching = [i for i, level in enumerate(levels) if level == 0.5]
        # _build_gluonts_forecasts previously appended ``median`` as an explicit
        # 0.5 key when a model did not provide one.
        return (np.asarray(quantiles)[:, matching[-1], :].astype(
                    np.float64, copy=False)
                if matching else median)
    if samples is not None:
        values = np.asarray(samples)
        # SampleForecast.quantile: sorted_samples[round((S - 1) * q)]. This is
        # observably different from torch.median for an even number of samples.
        sample_idx = int(np.round((values.shape[1] - 1) * 0.5))
        return np.partition(values, sample_idx, axis=1)[:, sample_idx, :].astype(
            np.float64, copy=False)
    return median


def gluonts_leaderboard_mase(median, starts, contexts, labels, freq,
                             quantiles=None, quantile_levels=None, samples=None,
                             seasonal_errors=None):
    """Exact vectorized equivalent of ``evaluate_forecasts(..., MASE(), axis=None)``.

    All array args are indexed 0..N-1 in the same instance order. ``contexts``
    can be omitted when precomputed ``seasonal_errors`` are supplied. ``starts``
    remains in the signature for compatibility, but MASE itself does not use
    timestamps. Returns the aggregate MASE float.
    """
    del starts
    point = _forecast_point_05(
        median, quantiles=quantiles,
        quantile_levels=quantile_levels, samples=samples)
    labels = np.asarray(labels, dtype=np.float64)
    if point.shape != labels.shape:
        raise ValueError(f"Forecast/label shape mismatch: {point.shape} vs {labels.shape}")
    if np.isnan(point).any():
        raise ValueError("Forecast contains NaN values")
    errors = (per_instance_seasonal_errors(contexts, get_seasonality(freq))
              if seasonal_errors is None else
              np.asarray(seasonal_errors, dtype=np.float64))
    if errors.shape != (labels.shape[0],):
        raise ValueError(
            f"Seasonal-error shape mismatch: {errors.shape} vs {(labels.shape[0],)}")
    masked_labels = np.ma.masked_invalid(labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.ma.abs(masked_labels - point) / errors[:, None]
    # GluonTS Mean(axis=(0, 1)) sums/counts every unmasked forecast point. This
    # is not a mean of per-instance horizon means when validity counts differ.
    return float(np.ma.mean(scaled))
