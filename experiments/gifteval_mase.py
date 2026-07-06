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


# ==============================================================================
#  Leaderboard MASE via the ACTUAL gluonts machinery
# ==============================================================================
# Everything above PORTS the gluonts definition in numpy — cheap and backfillable
# from cached per-instance MAE. The function below instead runs gluonts' own
# ``evaluate_forecasts`` + ``ev.MASE`` on real Forecast objects: the exact path the
# HF GiftEval leaderboard uses, and the ground truth the port reproduces (validated
# to <1%). Stage 3 records its output as the ``mase_gluonts_real`` metric column and
# the standalone ``compare_mase_variants`` script uses it for verification. gluonts
# is imported lazily so the port keeps working without it.


class _ListTestData:
    """Minimal stand-in for gluonts' ``split.TestData``: ``evaluate_forecasts``
    only needs re-iterable ``.input`` / ``.label`` (entry dicts) and to zip them.
    Lists satisfy the re-iteration it does internally (seasonal error over
    ``.input``, then labels over ``.label``)."""

    def __init__(self, inputs, labels):
        self.input = inputs
        self.label = labels

    def __iter__(self):
        return zip(self.input, self.label)

    def __len__(self):
        return len(self.input)


def _build_gluonts_forecasts(median, starts, quantiles=None, quantile_levels=None,
                             samples=None):
    """Per-instance gluonts Forecast objects — QuantileForecast when the model
    emits quantiles, else SampleForecast (from samples, or the median as a lone
    sample). ``median`` is (N, H); ``quantiles`` (N, Q, H); ``samples`` (N, S, H);
    ``starts[i]`` is the pandas Period at which instance i's forecast begins."""
    from gluonts.model.forecast import QuantileForecast, SampleForecast

    median = np.asarray(median, dtype=np.float64)
    n, _h = median.shape
    forecasts = []

    if quantiles is not None and quantile_levels is not None:
        q = np.asarray(quantiles, dtype=np.float64)          # (N, Q, H)
        keys = [str(float(l)) for l in quantile_levels]
        has_median = any(abs(float(l) - 0.5) < 1e-9 for l in quantile_levels)
        for i in range(n):
            arrays, fkeys = q[i], list(keys)
            if not has_median:                                # ensure a 0.5 key
                arrays = np.concatenate([arrays, median[i][None, :]], axis=0)
                fkeys = fkeys + ["0.5"]
            forecasts.append(QuantileForecast(
                forecast_arrays=arrays, start_date=starts[i],
                forecast_keys=fkeys, item_id=str(i)))
    elif samples is not None:
        s = np.asarray(samples, dtype=np.float64)            # (N, S, H)
        for i in range(n):
            forecasts.append(SampleForecast(
                samples=s[i], start_date=starts[i], item_id=str(i)))
    else:
        for i in range(n):                                    # median-only fallback
            forecasts.append(SampleForecast(
                samples=median[i][None, :], start_date=starts[i], item_id=str(i)))

    return forecasts


def gluonts_leaderboard_mase(median, starts, contexts, labels, freq,
                             quantiles=None, quantile_levels=None, samples=None):
    """Aggregate MASE from gluonts' own ``evaluate_forecasts`` + ``ev.MASE``.

    All array args are indexed 0..N-1 in the SAME instance order: row ``i`` of
    ``median`` / ``labels`` (each (N, H)), ``contexts[i]`` (that series' full
    context, from which gluonts derives its per-instance seasonal error), and
    ``starts[i]`` (the forecast-start Period) are the same instance. Lazy-imports
    gluonts (raises ImportError if unavailable). Returns the MASE float.
    """
    from gluonts.model import evaluate_forecasts
    from gluonts.ev.metrics import MASE

    forecasts = _build_gluonts_forecasts(
        median, starts, quantiles=quantiles,
        quantile_levels=quantile_levels, samples=samples)

    labels = np.asarray(labels, dtype=np.float64)
    inputs, label_entries = [], []
    for i in range(len(forecasts)):
        ctx = np.asarray(contexts[i], dtype=np.float64)
        inputs.append({"start": starts[i] - len(ctx), "target": ctx})
        label_entries.append({"start": starts[i], "target": labels[i]})
    test_data = _ListTestData(inputs, label_entries)

    df = evaluate_forecasts(
        forecasts, test_data=test_data, metrics=[MASE()],
        axis=None, seasonality=get_seasonality(freq))
    col = next((c for c in df.columns if str(c).upper().startswith("MASE")), None)
    if col is None:
        raise RuntimeError(
            f"No MASE column in evaluate_forecasts output: {list(df.columns)}")
    return float(np.asarray(df[col]).ravel()[0])
