"""Leaderboard-faithful (gluonts) MASE primitives.

The project's stage-3 ablation reports MASE as ``global_MAE / pooled_training_
seasonal_naive_MAE`` with a custom seasonality map (``D->7, W->52, ...``). That is
*not* the quantity the HuggingFace GiftEval leaderboard reports, so the numbers
don't line up. The leaderboard uses gluonts' definition:

    * seasonal error is computed **per test instance from that instance's own
      context** (``calculate_seasonal_error``), and
    * MASE is ``mean(|y - yhat|) / seasonal_error`` **per instance**, then averaged
      (``mase``), with the seasonality ``m`` taken from gluonts'
      ``DEFAULT_SEASONALITIES`` map.

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
    """Port of gluonts ``calculate_seasonal_error`` over a single instance's
    context: ``mean(|x[m:] - x[:-m]|)``.

    Matches gluonts' fallback to ``m = 1`` when the context is shorter than the
    seasonal lag, ignores NaNs (contexts are already NaN->0 in the ablation
    cache, but be defensive), and clamps the result to a tiny floor so MASE never
    divides by zero.
    """
    x = np.asarray(context, dtype=np.float64)
    n = x.shape[0]
    m = season if 0 < season < n else 1
    if n <= m:
        # Not enough history to form a single seasonal difference.
        level = float(np.nanmean(np.abs(x))) if n else 0.0
        return max(level, 1e-9)
    diff = np.abs(x[m:] - x[:-m])
    diff = diff[~np.isnan(diff)]
    if diff.size == 0:
        return 1e-9
    return max(float(diff.mean()), 1e-9)


def per_instance_seasonal_errors(contexts: Sequence[np.ndarray], season: int) -> np.ndarray:
    """Per-instance seasonal error for a list of contexts (one float per series)."""
    return np.array([seasonal_error(c, season) for c in contexts], dtype=np.float64)
