"""
Single source of truth for the GiftEval evaluation dataset set.

Mirrors :mod:`experiments.models_config`: every script that used to carry the
hand-maintained ``DATASETS`` table now reads it from here, so the catalog and the
"what do we actually run" set can't drift apart. ``test_window_ablation_gifteval_v5``
re-exports :func:`datasets_to_run` as its ``DATASETS``, and the whole downstream
pipeline (period eval, robust timing, MASE variants, embedding saturation) imports
``DATASETS`` from there — so flipping a ``run`` flag here reaches every stage.

Two distinct concepts live in one ordered table:

  * **Catalog** — the full official GiftEval config set (97 dataset/freq/term
    combinations). Each row is one ``(ge_name, term)`` cell. Use :func:`catalog`.

  * **Run set** — the subset the pipeline actually evaluates end-to-end, flagged
    per row by ``run``. Use :func:`datasets_to_run`.

Order is the stable, shard-safe iteration order used by stage 3 / the timing
stage (``d_idx`` enumerates the run set). Cache dirs are keyed by
``(display, term, ...)``, NOT by index, so toggling ``run`` flags between runs is
safe for resume — a disabled cell is simply skipped, its cache untouched.

The term column follows GiftEval's freq rule: sub-daily MED_LONG datasets get
short/medium/long; everything else gets short only. Any ``(ge_name, term)`` that
``gift_eval`` rejects is skipped at load time (stage 3 guards
GiftEvalDataset/Cache construction), so a wrong guess self-heals instead of
crashing the run.
"""

from __future__ import annotations

from typing import List, NamedTuple, Tuple


class DatasetSpec(NamedTuple):
    ge_name: str        # gift_eval config name passed to GiftEvalDataset (e.g. "ett1/15T")
    term: str           # "short" | "medium" | "long"
    display: str        # run label / output-dir basename / --datasets selector
    to_univariate: bool # True -> flatten multivariate collection to univariate series
    run: bool           # True -> part of the end-to-end evaluation run set


# Full official GiftEval catalog (97 configs). To drop a cell from a run without
# losing it from the catalog, flip its ``run`` to False.
CATALOG: List[DatasetSpec] = [
    DatasetSpec("jena_weather/10T",            "short",  "JenaWeather-10T", True,  True),
    DatasetSpec("jena_weather/10T",            "medium", "JenaWeather-10T", True,  True),
    DatasetSpec("jena_weather/10T",            "long",   "JenaWeather-10T", True,  True),
    DatasetSpec("jena_weather/H",              "short",  "JenaWeather-H",   True,  True),
    DatasetSpec("jena_weather/H",              "medium", "JenaWeather-H",   True,  True),
    DatasetSpec("jena_weather/H",              "long",   "JenaWeather-H",   True,  True),
    DatasetSpec("jena_weather/D",              "short",  "JenaWeather-D",   True,  True),
    DatasetSpec("bizitobs_application",        "short",  "BizITObsApp",     True,  True),
    DatasetSpec("bizitobs_application",        "medium", "BizITObsApp",     True,  True),
    DatasetSpec("bizitobs_application",        "long",   "BizITObsApp",     True,  True),
    DatasetSpec("bizitobs_service",            "short",  "BizITObsService", True,  True),
    DatasetSpec("bizitobs_service",            "medium", "BizITObsService", True,  True),
    DatasetSpec("bizitobs_service",            "long",   "BizITObsService", True,  True),
    DatasetSpec("bizitobs_l2c/5T",             "short",  "BizITObsL2C-5T",  True,  True),
    DatasetSpec("bizitobs_l2c/5T",             "medium", "BizITObsL2C-5T",  True,  True),
    DatasetSpec("bizitobs_l2c/5T",             "long",   "BizITObsL2C-5T",  True,  True),
    DatasetSpec("bizitobs_l2c/H",              "short",  "BizITObsL2C-H",   True,  True),
    DatasetSpec("bizitobs_l2c/H",              "medium", "BizITObsL2C-H",   True,  True),
    DatasetSpec("bizitobs_l2c/H",              "long",   "BizITObsL2C-H",   True,  True),
    DatasetSpec("bitbrains_fast_storage/5T",   "short",  "BitbrainsFS-5T",  True,  True),
    DatasetSpec("bitbrains_fast_storage/5T",   "medium", "BitbrainsFS-5T",  True,  True),
    DatasetSpec("bitbrains_fast_storage/5T",   "long",   "BitbrainsFS-5T",  True,  True),
    DatasetSpec("bitbrains_fast_storage/H",    "short",  "BitbrainsFS-H",   True,  True),
    DatasetSpec("bitbrains_rnd/5T",            "short",  "BitbrainsRnD-5T", True,  True),
    DatasetSpec("bitbrains_rnd/5T",            "medium", "BitbrainsRnD-5T", True,  True),
    DatasetSpec("bitbrains_rnd/5T",            "long",   "BitbrainsRnD-5T", True,  True),
    DatasetSpec("bitbrains_rnd/H",             "short",  "BitbrainsRnD-H",  True,  True),
    DatasetSpec("restaurant",                  "short",  "Restaurant",      False, True),
    DatasetSpec("ett1/15T",                    "short",  "ETTm1-15T",       True,  True),
    DatasetSpec("ett1/15T",                    "medium", "ETTm1-15T",       True,  True),
    DatasetSpec("ett1/15T",                    "long",   "ETTm1-15T",       True,  True),
    DatasetSpec("ett1/H",                      "short",  "ETTm1-H",         True,  True),
    DatasetSpec("ett1/H",                      "medium", "ETTm1-H",         True,  True),
    DatasetSpec("ett1/H",                      "long",   "ETTm1-H",         True,  True),
    DatasetSpec("ett1/D",                      "short",  "ETTm1-D",         True,  True),
    DatasetSpec("ett1/W",                      "short",  "ETTm1-W",         True,  True),
    DatasetSpec("ett2/15T",                    "short",  "ETTm2-15T",       True,  True),
    DatasetSpec("ett2/15T",                    "medium", "ETTm2-15T",       True,  True),
    DatasetSpec("ett2/15T",                    "long",   "ETTm2-15T",       True,  True),
    DatasetSpec("ett2/H",                      "short",  "ETTm2-H",         True,  True),
    DatasetSpec("ett2/H",                      "medium", "ETTm2-H",         True,  True),
    DatasetSpec("ett2/H",                      "long",   "ETTm2-H",         True,  True),
    DatasetSpec("ett2/D",                      "short",  "ETTm2-D",         True,  True),
    DatasetSpec("ett2/W",                      "short",  "ETTm2-W",         True,  True),
    DatasetSpec("LOOP_SEATTLE/5T",             "short",  "LoopSeattle-5T",  False, True),
    DatasetSpec("LOOP_SEATTLE/5T",             "medium", "LoopSeattle-5T",  False, True),
    DatasetSpec("LOOP_SEATTLE/5T",             "long",   "LoopSeattle-5T",  False, True),
    DatasetSpec("LOOP_SEATTLE/H",              "short",  "LoopSeattle-H",   False, True),
    DatasetSpec("LOOP_SEATTLE/H",              "medium", "LoopSeattle-H",   False, True),
    DatasetSpec("LOOP_SEATTLE/H",              "long",   "LoopSeattle-H",   False, True),
    DatasetSpec("LOOP_SEATTLE/D",              "short",  "LoopSeattle-D",   False, True),
    DatasetSpec("SZ_TAXI/15T",                 "short",  "SZTaxi-15T",      False, True),
    DatasetSpec("SZ_TAXI/15T",                 "medium", "SZTaxi-15T",      False, True),
    DatasetSpec("SZ_TAXI/15T",                 "long",   "SZTaxi-15T",      False, True),
    DatasetSpec("SZ_TAXI/H",                   "short",  "SZTaxi-H",        False, True),
    DatasetSpec("M_DENSE/H",                   "short",  "MDense-H",        False, True),
    DatasetSpec("M_DENSE/H",                   "medium", "MDense-H",        False, True),
    DatasetSpec("M_DENSE/H",                   "long",   "MDense-H",        False, True),
    DatasetSpec("M_DENSE/D",                   "short",  "MDense-D",        False, True),
    DatasetSpec("solar/10T",                   "short",  "Solar-10T",       False, True),
    DatasetSpec("solar/10T",                   "medium", "Solar-10T",       False, True),
    DatasetSpec("solar/10T",                   "long",   "Solar-10T",       False, True),
    DatasetSpec("solar/H",                     "short",  "Solar-H",         False, True),
    DatasetSpec("solar/H",                     "medium", "Solar-H",         False, True),
    DatasetSpec("solar/H",                     "long",   "Solar-H",         False, True),
    DatasetSpec("solar/D",                     "short",  "Solar-D",         False, True),
    # Only W=32 fits, so there is no context-selection decision. It remains in
    # the 97-cell MASE cohort; selector policies fall back to native/full.
    DatasetSpec("solar/W",                     "short",  "Solar-W",         False, True),
    DatasetSpec("hierarchical_sales/D",        "short",  "HierSales-D",     False, True),
    DatasetSpec("hierarchical_sales/W",        "short",  "HierSales-W",     False, True),
    DatasetSpec("m4_yearly",                   "short",  "M4-Yearly",       False, True),
    DatasetSpec("m4_quarterly",                "short",  "M4-Quarterly",    False, True),
    DatasetSpec("m4_monthly",                  "short",  "M4-Monthly",      False, True),
    DatasetSpec("m4_weekly",                   "short",  "M4-Weekly",       False, True),
    DatasetSpec("m4_daily",                    "short",  "M4-Daily",        False, True),
    DatasetSpec("m4_hourly",                   "short",  "M4-Hourly",       False, True),
    DatasetSpec("hospital",                    "short",  "Hospital",        False, True),
    DatasetSpec("covid_deaths",                "short",  "CovidDeaths-D",   False, True),
    DatasetSpec("us_births/D",                 "short",  "USBirths-D",      False, True),
    DatasetSpec("us_births/W",                 "short",  "USBirths-W",      False, True),
    DatasetSpec("us_births/M",                 "short",  "USBirths-M",      False, True),
    DatasetSpec("saugeenday/D",                "short",  "SaugeenDay-D",    False, True),
    DatasetSpec("saugeenday/W",                "short",  "SaugeenDay-W",    False, True),
    DatasetSpec("saugeenday/M",                "short",  "SaugeenDay-M",    False, True),
    DatasetSpec("electricity/15T",             "short",  "Electricity-15T", False, True),
    DatasetSpec("electricity/15T",             "medium", "Electricity-15T", False, True),
    DatasetSpec("electricity/15T",             "long",   "Electricity-15T", False, True),
    DatasetSpec("electricity/H",               "short",  "Electricity-H",   False, True),
    DatasetSpec("electricity/H",               "medium", "Electricity-H",   False, True),
    DatasetSpec("electricity/H",               "long",   "Electricity-H",   False, True),
    DatasetSpec("electricity/D",               "short",  "Electricity-D",   False, True),
    DatasetSpec("electricity/W",               "short",  "Electricity-W",   False, True),
    # ---- Irregular / missing-value datasets (Monash "*_with_missing") --------
    # These close the gap vs the full GiftEval catalog (irregular sampling /
    # injected missing values). Stored as univariate series collections ->
    # to_univariate=False.
    DatasetSpec("kdd_cup_2018_with_missing/H", "short",  "KDDCup2018-H",   False, True),
    DatasetSpec("kdd_cup_2018_with_missing/H", "medium", "KDDCup2018-H",   False, True),
    DatasetSpec("kdd_cup_2018_with_missing/H", "long",   "KDDCup2018-H",   False, True),
    DatasetSpec("kdd_cup_2018_with_missing/D", "short",  "KDDCup2018-D",   False, True),
    # As above, W=32 is the only selectable window; retain it for the 97-cell
    # MASE cohort and use native/full for selector policies.
    DatasetSpec("car_parts_with_missing",      "short",  "CarParts",       False, True),
    DatasetSpec("temperature_rain_with_missing", "short", "TempRain",      False, True),
]


def catalog() -> List[Tuple[str, str, str, bool]]:
    """Full ``(ge_name, term, display, to_univariate)`` catalog, in table order.

    Every official GiftEval config, regardless of ``run`` flag.
    """
    return [(d.ge_name, d.term, d.display, d.to_univariate) for d in CATALOG]


def datasets_to_run() -> List[Tuple[str, str, str, bool]]:
    """``(ge_name, term, display, to_univariate)`` for the run set (``run=True``),
    in table order. Re-exported as ``DATASETS`` by the v5 GiftEval ablation."""
    return [(d.ge_name, d.term, d.display, d.to_univariate) for d in CATALOG if d.run]


def run_displays() -> List[str]:
    """Unique display names of the run set (convenience for CLI help /
    validation)."""
    seen: dict = {}
    for d in CATALOG:
        if d.run:
            seen.setdefault(d.display, None)
    return list(seen)
