"""Published GIFT-Eval leaderboard reference values.

The normalized leaderboard score must use the Seasonal Naive MASE published by
GIFT-Eval, not a locally reconstructed forecast.  This module is the single
source of truth for mapping the repo's dataset names to those published rows.
"""

from __future__ import annotations

import csv
import json
import os
from functools import lru_cache
from typing import Dict


REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "leaderboard_reference")
SEASONAL_NAIVE_CSV = os.path.join(
    REFERENCE_DIR, "seasonal_naive_all_results.csv")
MASE_COLUMN = "eval_metrics/MASE[0.5]"
NORMALIZATION_REFERENCE = (
    "leaderboard_reference/seasonal_naive_all_results.csv")

PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


@lru_cache(maxsize=1)
def _dataset_properties() -> dict:
    with open(os.path.join(REFERENCE_DIR, "dataset_properties.json")) as f:
        return json.load(f)


def leaderboard_dataset_key(ds_name: str, term: str) -> str:
    """Return the published CSV key for a GiftEval ``(name, term)`` config."""
    if "/" in ds_name:
        base, freq = ds_name.split("/", 1)
        base = PRETTY_NAMES.get(base.lower(), base.lower())
    else:
        base = PRETTY_NAMES.get(ds_name.lower(), ds_name.lower())
        try:
            freq = _dataset_properties()[base]["frequency"]
        except KeyError as exc:
            raise KeyError(
                f"No published GIFT-Eval dataset metadata for {ds_name!r}"
            ) from exc
    return f"{base}/{freq}/{term}"


@lru_cache(maxsize=1)
def published_seasonal_naive_mase() -> Dict[str, float]:
    """Published Seasonal Naive ``MASE[0.5]``, keyed by leaderboard config."""
    with open(SEASONAL_NAIVE_CSV, newline="") as f:
        rows = {
            row["dataset"]: float(row[MASE_COLUMN])
            for row in csv.DictReader(f)
        }
    if len(rows) != 97:
        raise RuntimeError(
            f"Expected 97 Seasonal Naive rows in {SEASONAL_NAIVE_CSV}, "
            f"found {len(rows)}"
        )
    return rows


def published_naive_record(ds_name: str, term: str) -> dict:
    """Compatibility record for consumers of the two GluonTS MASE columns.

    ``mase_gluonts`` is the repo's NumPy port and ``mase_gluonts_real`` is the
    actual GluonTS machinery, but both normalized views must use the one official
    GIFT-Eval denominator.
    """
    key = leaderboard_dataset_key(ds_name, term)
    try:
        value = published_seasonal_naive_mase()[key]
    except KeyError as exc:
        raise KeyError(
            f"No Seasonal Naive MASE row for {ds_name!r}, term={term!r} "
            f"(expected key {key!r})"
        ) from exc
    return {
        "mase_gluonts": value,
        "mase_gluonts_real": value,
        "_source": "gift_eval_published_csv",
        "_reference_file": os.path.basename(SEASONAL_NAIVE_CSV),
        "_leaderboard_key": key,
    }


@lru_cache(maxsize=1)
def published_naive_by_display() -> Dict[str, dict]:
    """Published denominators keyed as ``<dataset_display>/t<term>``."""
    from experiments import datasets_config

    out = {
        f"{display}/t{term}": published_naive_record(ds_name, term)
        for ds_name, term, display, _to_univariate in datasets_config.catalog()
    }
    if len(out) != 97:
        raise RuntimeError(
            "Dataset display mapping does not cover the 97 official GIFT-Eval "
            f"configs (found {len(out)})"
        )
    return out
