"""Clean-room replication of the official GiftEval leaderboard evaluation.

Purpose: sanity-check our numbers against the HF GIFT-Eval leaderboard by
re-running ONE model over ALL 97 official configs with the *exact* recipe of
the official submission, then diffing per config against the leaderboard's
published `all_results.csv`. Until this script reproduces the leaderboard
headline, no pipeline number should be trusted.

The headline is the PUBLISHED aggregation: geomean over the 97 configs of
MASE[0.5] / the leaderboard's published seasonal_naive MASE[0.5] (CSVs shipped
in ``experiments/leaderboard_reference/``). Targets, reproduced exactly from
those CSVs:

    timesfm-2.5   (google/timesfm-2.5-200m-pytorch)   0.7050   <- DEFAULT
    chronos-2     (amazon/chronos-2)                  0.6978
    chronos-2-synth (autogluon/chronos-2-synth)       0.7203

TimesFM-2.5 is the default because its official recipe has no "fancy stuff" —
plain univariate flattening (`to_univariate = target_dim > 1`), independent
series, full context (capped 15360, per-batch compile) — so it is the cleanest
apples-to-apples check against our (univariate, independent) pipeline. The
prediction+evaluation code is a line-for-line port of the OFFICIAL submission
notebooks (linked from each model's results/<model>/config.json on the
leaderboard space):
  https://github.com/SalesforceAIResearch/gift-eval/blob/main/notebooks/timesfm2p5.ipynb
  https://github.com/SalesforceAIResearch/gift-eval/blob/main/notebooks/chronos-2.ipynb

Chronos-2's official recipe DOES have fancy extras our pipeline lacks (native
multivariate + ``predict_batches_jointly`` in-context cross-learning), which is
why pipeline-vs-leaderboard gaps there are expected; knobs below attribute them:

  * ``--univariate``     flatten multivariate sources (chronos-2 only; timesfm
                         official recipe already flattens)
  * ``--independent``    predict_batches_jointly=False (chronos-2 only)
  * ``--max-context N``  truncate every input to its last N steps (both models;
                         mimics the pipeline's window grid, max 8192)

Run on the SERVER (needs GIFT_EVAL data; timesfm and/or chronos-forecasting>=2):

    python -m experiments.sanity_gifteval_leaderboard                     # timesfm-2.5, target 0.7050
    python -m experiments.sanity_gifteval_leaderboard --model chronos-2   # target 0.6978
    python -m experiments.sanity_gifteval_leaderboard --configs m4 ett1   # subset (substring)
    python -m experiments.sanity_gifteval_leaderboard --max-context 8192  # pipeline-style context cap

Per-config results cache under ``logs/experiments/sanity_leaderboard/<tag>/`` and
re-runs resume; the final comparison table prints and lands in ``comparison.csv``
next to them. Knob settings are part of the tag, so attribution runs don't
clobber the faithful one.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
from typing import List

import numpy as np
import pandas as pd
import torch

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sanity_gifteval_leaderboard")

# ------------------------------------------------------------------------------
# Official config universe — verbatim from the submission notebook. 55 short-only
# + 21 med/long datasets x3 terms = 97 configs.
# ------------------------------------------------------------------------------
SHORT_DATASETS = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly "
    "electricity/15T electricity/H electricity/D electricity/W "
    "solar/10T solar/H solar/D solar/W hospital covid_deaths "
    "us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W "
    "temperature_rain_with_missing kdd_cup_2018_with_missing/H "
    "kdd_cup_2018_with_missing/D car_parts_with_missing restaurant "
    "hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H "
    "LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D "
    "ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W "
    "jena_weather/10T jena_weather/H jena_weather/D "
    "bitbrains_fast_storage/5T bitbrains_fast_storage/H "
    "bitbrains_rnd/5T bitbrains_rnd/H "
    "bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
)
MED_LONG_DATASETS = (
    "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H "
    "ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T "
    "bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T "
    "bizitobs_l2c/H"
)

# Leaderboard CSV key naming (verbatim `pretty_names` from the notebook).
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

REF_DIR = os.path.join(os.path.dirname(__file__), "leaderboard_reference")
OUT_ROOT = "logs/experiments/sanity_leaderboard"

# Official reference CSVs shipped in-repo (downloaded 2026-07-17 from
# huggingface.co/spaces/Salesforce/GIFT-Eval results/<model>/all_results.csv).
REFERENCE_CSVS = {
    "amazon/chronos-2": "chronos-2_all_results.csv",
    "autogluon/chronos-2": "chronos-2_all_results.csv",   # same weights, official run under amazon/
    "autogluon/chronos-2-synth": "chronos-2-synth_all_results.csv",
    "google/timesfm-2.5-200m-pytorch": "timesfm-2.5_all_results.csv",
}
SEASONAL_NAIVE_CSV = "seasonal_naive_all_results.csv"

# CLI shorthands -> HF ids of the official submissions.
MODEL_ALIASES = {
    "timesfm": "google/timesfm-2.5-200m-pytorch",
    "timesfm-2.5": "google/timesfm-2.5-200m-pytorch",
    "chronos-2": "amazon/chronos-2",
    "chronos-2-synth": "autogluon/chronos-2-synth",
}


def _model_family(model_name: str) -> str:
    return "timesfm" if "timesfm" in model_name.lower() else "chronos2"


def _dataset_properties() -> dict:
    with open(os.path.join(REF_DIR, "dataset_properties.json")) as f:
        return json.load(f)


def official_configs() -> List[dict]:
    """The 97 (ds_name, term, ds_config_key) rows, notebook naming rules."""
    props = _dataset_properties()
    med_long = set(MED_LONG_DATASETS.split())
    rows = []
    for ds_name in SHORT_DATASETS.split():
        for term in ("short", "medium", "long"):
            if term in ("medium", "long") and ds_name not in med_long:
                continue
            if "/" in ds_name:
                ds_key, ds_freq = ds_name.split("/")
                ds_key = PRETTY_NAMES.get(ds_key.lower(), ds_key.lower())
            else:
                ds_key = PRETTY_NAMES.get(ds_name.lower(), ds_name.lower())
                ds_freq = props[ds_key]["frequency"]
            rows.append({
                "ds_name": ds_name,
                "term": term,
                "key": f"{ds_key}/{ds_freq}/{term}",
                "domain": props[ds_key]["domain"],
                "num_variates": props[ds_key]["num_variates"],
            })
    assert len(rows) == 97, f"expected 97 configs, built {len(rows)}"
    return rows


# ------------------------------------------------------------------------------
# Chronos-2 predictor — verbatim port of the notebook's Chronos2Predictor, with
# two additions: the pipeline is loaded once and reused across datasets (inference
# is stateless), and ``max_context`` optionally truncates each input to its last N
# steps (to mimic our pipeline's window slicing for attribution runs).
# ------------------------------------------------------------------------------
_PIPELINES: dict = {}


def _get_pipeline(model_name: str, device: str):
    if model_name not in _PIPELINES:
        from chronos import BaseChronosPipeline, Chronos2Pipeline

        pipe = BaseChronosPipeline.from_pretrained(
            model_name, device_map=device, torch_dtype="float32")
        assert isinstance(pipe, Chronos2Pipeline), \
            "this sanity script is Chronos-2-only (see the official notebook for others)"
        _PIPELINES[model_name] = pipe
    return _PIPELINES[model_name]


class Chronos2Predictor:
    def __init__(self, model_name: str, prediction_length: int, batch_size: int,
                 device: str = "cuda",
                 quantile_levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                 predict_batches_jointly: bool = False,
                 max_context: int | None = None):
        self.pipeline = _get_pipeline(model_name, device)
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.quantile_levels = list(quantile_levels)
        self.predict_batches_jointly = predict_batches_jointly
        self.max_context = max_context

    def _pack_model_items(self, items):
        for item in items:
            target = item["target"]
            if self.max_context is not None:
                target = target[..., -self.max_context:]
            yield {"target": target}

    def predict(self, test_data_input) -> List:
        from gluonts.model.forecast import QuantileForecast

        pipeline = self.pipeline
        model_batch_size = self.batch_size

        forecast_outputs = []
        input_data = list(self._pack_model_items(test_data_input))
        is_univariate_data = input_data[0]["target"].ndim == 1
        while True:
            try:
                quantiles, _ = pipeline.predict_quantiles(
                    inputs=input_data,
                    prediction_length=self.prediction_length,
                    batch_size=model_batch_size,
                    quantile_levels=self.quantile_levels,
                    predict_batches_jointly=self.predict_batches_jointly,
                )
                quantiles = torch.stack(quantiles)
                # [batch, variates, seq_len, quantiles] -> [batch, quantiles, seq_len, variates]
                quantiles = quantiles.permute(0, 3, 2, 1).cpu().numpy()
                if is_univariate_data:
                    quantiles = quantiles.squeeze(-1)
                assert quantiles.shape[1] == len(self.quantile_levels)
                assert quantiles.shape[2] == self.prediction_length
                forecast_outputs.append(quantiles)
                break
            except torch.cuda.OutOfMemoryError:
                logger.error(f"OOM at batch_size {model_batch_size}, halving")
                model_batch_size //= 2

        forecast_outputs = np.concatenate(forecast_outputs, axis=0)
        assert len(forecast_outputs) == len(input_data)
        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(QuantileForecast(
                forecast_arrays=item,
                forecast_keys=list(map(str, self.quantile_levels)),
                start_date=forecast_start_date,
            ))
        return forecasts


# ------------------------------------------------------------------------------
# TimesFM-2.5 predictor — verbatim port of the official timesfm2p5.ipynb
# notebook's TimesFmPredictor: gluonts batcher over the raw inputs, per-batch
# compile with max_context = batch max length rounded up to the patch size and
# capped at 15360, quantile columns 1..9 of the full prediction as the
# [0.1..0.9] QuantileForecast. ``max_context_knob`` optionally truncates each
# input first (attribution runs mimicking our pipeline's window cap).
# ------------------------------------------------------------------------------
class TimesFmPredictor:
    def __init__(self, tfm, prediction_length: int,
                 max_context_knob: int | None = None):
        self.tfm = tfm
        self.prediction_length = prediction_length
        self.quantiles = list(np.arange(1, 10) / 10.0)
        self.max_context_knob = max_context_knob

    def predict(self, test_data_input, batch_size: int = 1024) -> List:
        import timesfm
        from gluonts.itertools import batcher
        from gluonts.model.forecast import QuantileForecast

        test_data_input = list(test_data_input)
        forecast_outputs = []
        for batch in batcher(test_data_input, batch_size=batch_size):
            context = []
            max_context = 0
            for entry in batch:
                arr = np.array(entry["target"])
                if self.max_context_knob is not None:
                    arr = arr[-self.max_context_knob:]
                max_context = max(max_context, arr.shape[0])
                context.append(arr)
            p = self.tfm.model.p
            max_context = ((max_context + p - 1) // p) * p
            self.tfm.compile(
                timesfm.ForecastConfig(
                    max_context=min(15360, max_context),
                    max_horizon=1024,
                    infer_is_positive=True,
                    use_continuous_quantile_head=True,
                    fix_quantile_crossing=True,
                    force_flip_invariance=True,
                    return_backcast=False,
                    normalize_inputs=True,
                    per_core_batch_size=128,
                ),
            )
            _, full_preds = self.tfm.forecast(
                horizon=self.prediction_length, inputs=context)
            full_preds = full_preds[:, 0:self.prediction_length, 1:]
            forecast_outputs.append(full_preds.transpose((0, 2, 1)))
        forecast_outputs = np.concatenate(forecast_outputs)

        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(QuantileForecast(
                forecast_arrays=item,
                forecast_keys=list(map(str, self.quantiles)),
                start_date=forecast_start_date,
            ))
        return forecasts


def _get_timesfm(model_name: str):
    if model_name not in _PIPELINES:
        import timesfm

        _PIPELINES[model_name] = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_name)
    return _PIPELINES[model_name]


# ------------------------------------------------------------------------------
# Per-dataset evaluation — verbatim port of the notebooks' evaluation cells.
# ------------------------------------------------------------------------------

# pandas >= 2.2 REMOVED the legacy offset aliases the GiftEval HF metadata still
# stores (``A``/``A-DEC`` year, ``Q``/``Q-DEC`` quarter, ``M`` month, ``H`` hour,
# ``T`` minute, ``S`` second, ``U`` microsecond). On a strict build ``to_offset``
# raises on them outright ("'M' is no longer supported ... use 'ME'"), which kills
# gift_eval's own ``prediction_length`` (it does ``to_offset(self.freq)``) and
# gluonts' ``get_seasonality`` before either can normalize. This is the exact
# INVERSE of gift_eval's ``maybe_reconvert_freq`` (new->old): we rewrite the
# stored old alias to the new one so ``to_offset`` parses, and gift_eval's
# downstream ``maybe_reconvert_freq`` maps it right back to its old lookup key.
# Only the leading unit token is touched; multiplier ("15T") and anchor
# ("A-DEC", "W-SUN") are preserved.
_OLD_TO_NEW_OFFSET = {
    "A": "YE", "Y": "YE",
    "Q": "QE",
    "M": "ME",
    "H": "h",
    "T": "min", "MIN": "min",
    "S": "s",
    "U": "us",
    # W, D, B and any already-new alias pass through unchanged.
}


def _normalize_freq(freq: str) -> str:
    """Rewrite a legacy pandas offset alias to its pandas>=2.2 spelling,
    preserving a leading integer multiplier and any ``-ANCHOR`` suffix."""
    freq = str(freq)
    head, dash, anchor = freq.partition("-")
    i = 0
    while i < len(head) and head[i].isdigit():
        i += 1
    mult, unit = head[:i], head[i:]
    new_unit = _OLD_TO_NEW_OFFSET.get(unit.upper(), unit)
    return f"{mult}{new_unit}{dash}{anchor}"


def _force_str_freq() -> None:
    """Make gift_eval's ``Dataset.freq`` return a plain, pandas-current ``str``.

    Two problems with the raw value (``hf_dataset[0]["freq"]``), fixed together
    by wrapping the descriptor once at the source rather than chasing call sites:

      1. It arrives as a ``numpy.str_``; Cython-3-built pandas rejects ``str``
         subclasses in typed signatures ("expected str, got numpy.str_").
      2. It uses LEGACY offset aliases (``A-DEC``, ``Q-DEC``, ``M``, ...) that
         pandas>=2.2 no longer parses (see ``_normalize_freq``).

    The value flows into pandas from OUR calls (get_seasonality) AND gift_eval
    internals (``to_offset`` in ``prediction_length``, test-split Period math), so
    fixing it here covers every consumer. gift_eval defines ``freq`` as a
    ``property`` (``.fget``) or ``functools.cached_property`` (``.func``) across
    versions — handle both, replacing it with a plain recomputing ``property``
    (a data descriptor, so it also overrides any value a cached_property already
    cached on live instances). Idempotent; no-op if ``freq`` isn't a wrappable
    descriptor."""
    from gift_eval import data as ge_data

    Dataset = ge_data.Dataset
    if getattr(Dataset, "_freq_str_patched", False):
        return
    descriptor = next(
        (klass.__dict__["freq"] for klass in Dataset.__mro__
         if "freq" in klass.__dict__), None)
    orig = getattr(descriptor, "fget", None) or getattr(descriptor, "func", None)
    if orig is None:
        return
    Dataset.freq = property(lambda self: _normalize_freq(orig(self)))
    Dataset._freq_str_patched = True


def _metrics():
    from gluonts.ev.metrics import (
        MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
        MeanWeightedSumQuantileLoss,
    )

    return [
        MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(),
        MAPE(), SMAPE(), MSIS(), RMSE(), NRMSE(), ND(),
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    ]


def evaluate_on_dataset(model_name: str, ds_name: str, ds_term: str,
                        batch_size: int, device: str,
                        use_multivariate_data: bool,
                        predict_batches_jointly: bool,
                        max_context: int | None) -> dict:
    from gluonts.model import evaluate_forecasts, evaluate_model
    from gluonts.time_feature import get_seasonality

    from gift_eval.data import Dataset

    _force_str_freq()
    family = _model_family(model_name)

    if family == "timesfm":
        # Official timesfm2p5.ipynb recipe: flatten multivariate sources,
        # independent series, evaluate_model over the whole test split.
        to_univariate = Dataset(
            name=ds_name, term=ds_term, to_univariate=False).target_dim > 1
        dataset = Dataset(name=ds_name, term=ds_term, to_univariate=to_univariate)
        predictor = TimesFmPredictor(
            tfm=_get_timesfm(model_name),
            prediction_length=dataset.prediction_length,
            max_context_knob=max_context,
        )
        res = evaluate_model(
            predictor,
            test_data=dataset.test_data,
            metrics=_metrics(),
            batch_size=1024,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            # str(): gift_eval's Dataset.freq can be a numpy.str_, which the
            # Cython-compiled pandas to_offset inside get_seasonality rejects
            # ("expected str, got numpy.str_") on pandas>=2/Cython 3 builds.
            seasonality=get_seasonality(str(dataset.freq)),
        ).reset_index(drop=True).to_dict(orient="records")
        return res[0]

    # Official chronos-2.ipynb recipe.
    is_multivariate_source = Dataset(
        name=ds_name, term=ds_term, to_univariate=False).target_dim > 1
    dataset = Dataset(
        name=ds_name, term=ds_term,
        to_univariate=is_multivariate_source and not use_multivariate_data)

    predictor = Chronos2Predictor(
        model_name=model_name,
        prediction_length=dataset.prediction_length,
        batch_size=batch_size,
        device=device,
        predict_batches_jointly=predict_batches_jointly,
        max_context=max_context,
    )

    # Predict each rolling window separately: with joint (in-context) prediction,
    # different windows of the same series in one batch would leak.
    forecast_windows = []
    n_windows = dataset.test_data.windows
    for window_idx in range(n_windows):
        entries_window_k = list(itertools.islice(
            dataset.test_data.input, window_idx, None, n_windows))
        forecast_windows.append(list(predictor.predict(entries_window_k)))
    forecasts = [item for items in zip(*forecast_windows) for item in items]

    res = evaluate_forecasts(
        forecasts,
        test_data=dataset.test_data,
        metrics=_metrics(),
        batch_size=1024,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=get_seasonality(str(dataset.freq)),  # str(): see timesfm branch
    ).reset_index(drop=True).to_dict(orient="records")
    return res[0]


# ------------------------------------------------------------------------------
# Comparison against the official leaderboard CSVs
# ------------------------------------------------------------------------------

def _load_reference_mase(csv_name: str) -> dict:
    df = pd.read_csv(os.path.join(REF_DIR, csv_name))
    return dict(zip(df["dataset"], df["eval_metrics/MASE[0.5]"]))


def compare_and_report(ours: dict, model_name: str, out_dir: str) -> None:
    """ours: {config_key: metrics-dict}. Prints the per-config diff table and the
    normalized-geomean headline vs the official one."""
    sn = _load_reference_mase(SEASONAL_NAIVE_CSV)
    ref_csv = REFERENCE_CSVS.get(model_name)
    ref = _load_reference_mase(ref_csv) if ref_csv else {}

    rows = []
    for key, m in sorted(ours.items()):
        mase = m["MASE[0.5]"]
        row = {
            "dataset": key,
            "mase_ours": mase,
            "mase_official": ref.get(key, float("nan")),
            "sn_official": sn.get(key, float("nan")),
        }
        row["rel_diff_pct"] = 100.0 * (mase - row["mase_official"]) / row["mase_official"] \
            if row["mase_official"] == row["mase_official"] else float("nan")
        row["norm_ours"] = mase / row["sn_official"]
        row["norm_official"] = row["mase_official"] / row["sn_official"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "comparison.csv"), index=False)

    def geomean(s):
        s = s.dropna()
        return float(np.exp(np.log(s).mean())) if len(s) else float("nan")

    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df.round(4).to_string(index=False))

    n = len(df)
    print(f"\n==== {model_name} on {n}/97 configs ====")
    print("PUBLISHED aggregation: geomean of MASE[0.5] / published seasonal_naive MASE[0.5]")
    print(f"geomean normalized MASE (ours)     : {geomean(df['norm_ours']):.4f}")
    print(f"geomean normalized MASE (official) : {geomean(df['norm_official']):.4f}"
          f"   <- leaderboard headline on the same {n} configs")
    print(f"raw geomean MASE (ours)            : {geomean(df['mase_ours']):.4f}")
    print(f"raw geomean MASE (official)        : {geomean(df['mase_official']):.4f}")
    if n < 97:
        print(f"NOTE: only {n}/97 configs — the headline is comparable ONLY via "
              "the 'official' column above, not to the full-board number.")
    worst = df.reindex(df["rel_diff_pct"].abs().sort_values(ascending=False).index).head(10)
    print("\nlargest per-config deviations (ours vs official):")
    print(worst[["dataset", "mase_ours", "mase_official", "rel_diff_pct"]]
          .round(4).to_string(index=False))


# ------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="timesfm-2.5",
                    help="shorthand (timesfm-2.5, chronos-2, chronos-2-synth) "
                         "or a full HF id")
    ap.add_argument("--batch-size", type=int, default=100,
                    help="chronos-2 official notebook value: 100 (timesfm "
                         "batches internally at 1024)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="substring filter on ds_name (e.g. m4 ett1)")
    # Attribution knobs — DEFAULTS are the official recipe; each flag moves one
    # step toward our ablation pipeline's setup.
    ap.add_argument("--univariate", action="store_true",
                    help="chronos-2 only: flatten multivariate sources (our "
                         "pipeline does this; official chronos-2 uses native "
                         "multivariate; timesfm's official recipe already flattens)")
    ap.add_argument("--independent", action="store_true",
                    help="chronos-2 only: predict_batches_jointly=False "
                         "(official uses True: in-context cross-learning)")
    ap.add_argument("--max-context", type=int, default=None,
                    help="truncate inputs to last N steps (our grid tops at 8192; "
                         "official feeds everything)")
    args = ap.parse_args()

    args.model = MODEL_ALIASES.get(args.model, args.model)
    if _model_family(args.model) == "timesfm" and (args.univariate or args.independent):
        logger.warning("--univariate/--independent are no-ops for timesfm: its "
                       "official recipe is already univariate and independent")

    tag = os.path.basename(args.model)
    if args.univariate:
        tag += "_univariate"
    if args.independent:
        tag += "_independent"
    if args.max_context:
        tag += f"_ctx{args.max_context}"
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)

    configs = official_configs()
    if args.configs:
        configs = [c for c in configs
                   if any(f.lower() in c["ds_name"].lower() for f in args.configs)]
        logger.info(f"filtered to {len(configs)} configs")

    ours: dict = {}
    for i, cfg in enumerate(configs):
        cell_path = os.path.join(out_dir, cfg["key"].replace("/", "_") + ".json")
        if os.path.exists(cell_path):
            with open(cell_path) as f:
                ours[cfg["key"]] = json.load(f)
            logger.info(f"[{i + 1}/{len(configs)}] {cfg['key']}  (cached)")
            continue
        logger.info(f"[{i + 1}/{len(configs)}] {cfg['key']}  running...")
        try:
            m = evaluate_on_dataset(
                model_name=args.model,
                ds_name=cfg["ds_name"],
                ds_term=cfg["term"],
                batch_size=args.batch_size,
                device=args.device,
                use_multivariate_data=not args.univariate,
                predict_batches_jointly=not args.independent,
                max_context=args.max_context,
            )
        except Exception as exc:  # keep going; a hole in the table is loud enough
            logger.error(f"{cfg['key']} FAILED: {exc}", exc_info=True)
            continue
        m = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
             for k, v in m.items()}
        with open(cell_path, "w") as f:
            json.dump(m, f, indent=1)
        ours[cfg["key"]] = m
        logger.info(f"    MASE[0.5] = {m['MASE[0.5]']:.4f}")

    if not ours:
        raise SystemExit("no results produced")
    compare_and_report(ours, args.model, out_dir)


if __name__ == "__main__":
    main()
