"""Clean-room replication of the official GIFT-Eval leaderboard evaluation.

Goal: verify our numbers against the HF GIFT-Eval leaderboard by re-running ONE
model over ALL 97 official configs with the *exact* recipe of the official
submission, then diffing per config against the leaderboard's published
``all_results.csv``. Until this reproduces the leaderboard headline for a model,
no pipeline number for that model should be trusted.

This is a deliberately faithful port of the OFFICIAL submission notebooks
(linked from each model's ``results/<model>/config.json`` on the leaderboard
space). PatchTST-FM uses the same v5 pipeline wrapper as the ablation so its
sanity result is also an integration gate for short-context inference:

  timesfm-2.5   https://github.com/SalesforceAIResearch/gift-eval/blob/main/notebooks/timesfm2p5.ipynb
  chronos-2     https://github.com/SalesforceAIResearch/gift-eval/blob/main/notebooks/chronos-2.ipynb
  patchtst-fm   https://github.com/SalesforceAIResearch/gift-eval/blob/main/notebooks/patchtst_fm.ipynb

The headline is the PUBLISHED aggregation: geomean over the 97 configs of
MASE[0.5] / the leaderboard's published seasonal_naive MASE[0.5] (reference CSVs
shipped in ``experiments/leaderboard_reference/``). Targets, reproduced exactly
from those CSVs:

    timesfm-2.5     (google/timesfm-2.5-200m-pytorch)   0.7050   <- DEFAULT
    chronos-2       (amazon/chronos-2)                  0.6978
    chronos-2-synth (autogluon/chronos-2-synth)         0.7203
    patchtst-fm-r1  (ibm-research/patchtst-fm-r1)        0.7069

TimesFM-2.5 is the default because its official recipe has no "fancy stuff" —
plain univariate flattening (``to_univariate = target_dim > 1``), independent
series, full context (capped 15360, per-batch compile) — so it is the cleanest
apples-to-apples check against our (univariate, independent) pipeline.

Chronos-2's official recipe DOES have extras our pipeline lacks (native
multivariate + ``predict_batches_jointly`` in-context cross-learning), so
pipeline-vs-leaderboard gaps there are expected; the attribution knobs below
move the recipe stepwise toward our pipeline's setup to localize the gap:

  * ``--univariate``     flatten multivariate sources (chronos-2 only; the
                         timesfm official recipe already flattens)
  * ``--independent``    predict_batches_jointly=False (chronos-2 only)
  * ``--max-context N``  truncate every input to its last N steps (both models;
                         mimics the pipeline's window grid, which tops at 8192)

ENVIRONMENT — READ THIS IF CONFIGS DIE WITH A pandas "Invalid frequency" ERROR:
gift-eval pins ``gluonts~=0.15.1``, which uses the LEGACY pandas offset aliases
(``M``, ``Q-DEC``, ``A-DEC``, ``H``, ``T``) the GIFT-Eval data is stored with.
That stack runs only on **pandas < 3**. pandas 3.0 removed the legacy offset
alias but still requires it for ``pd.Period`` — so ``to_offset`` wants ``ME``
while ``Period`` wants ``M``, a contradiction no freq string can satisfy, and
yearly/quarterly/monthly configs die no matter how the freq is spelled. Fix is
the ENV, not the code (the leaderboard itself ran on pandas 2.x): install the
eval env with ``pip install 'pandas<3'`` (2.2.x / 2.3.x keep the old aliases
working in both paths and stay compatible with the TSFM stack). This script does
NOT monkeypatch the freq — it relies on the correct env, exactly like the
notebooks.

Run on the SERVER (needs GIFT_EVAL data + pandas<3; timesfm and/or
chronos-forecasting>=2 depending on --model):

    python -m experiments.sanity_gifteval_leaderboard                     # timesfm-2.5, target 0.7050
    python -m experiments.sanity_gifteval_leaderboard --model chronos-2   # target 0.6978
    python -m experiments.sanity_gifteval_leaderboard --configs m4 ett1   # substring subset
    python -m experiments.sanity_gifteval_leaderboard --max-context 8192  # pipeline-style context cap

Per-config results cache under ``logs/experiments/sanity_leaderboard/<tag>/`` and
re-runs resume; the comparison table prints and lands in ``comparison.csv`` next
to them. Knob settings are folded into the tag, so attribution runs never clobber
the faithful one.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import logging
import math
import numbers
import os
from typing import Callable, List, Optional, Tuple, TypeVar

from experiments.gifteval_reference import published_seasonal_naive_mase
from experiments.gifteval_inference_recipes import inference_recipe

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sanity_gifteval_leaderboard")

T = TypeVar("T")


def _require_numpy(purpose: str):
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            f"numpy is required to {purpose}. Activate the GiftEval server env "
            "(predictcsl-main / TSFM_moirai) before running model evaluation."
        ) from exc
    return np


def _require_pandas(purpose: str):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            f"pandas is required to {purpose}. Activate the GiftEval server env "
            "(predictcsl-main / TSFM_moirai) before running this step."
        ) from exc
    return pd


def _require_torch(purpose: str):
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            f"torch is required to {purpose}. Activate the GiftEval server env "
            "(predictcsl-main / TSFM_moirai) before running Chronos-2 evaluation."
        ) from exc
    return torch


def _jsonable_metric_value(v):
    """Convert numpy/scalar metric values to JSON-safe Python values."""
    if isinstance(v, bool):
        return v
    if isinstance(v, numbers.Real):
        return float(v)
    item = getattr(v, "item", None)
    if callable(item):
        try:
            scalar = item()
        except Exception:
            return v
        if isinstance(scalar, bool):
            return scalar
        if isinstance(scalar, numbers.Real):
            return float(scalar)
    return v


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    oom_fragments = (
        "out of memory",
        "cuda oom",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
        "mps backend out of memory",
        "unable to allocate",
        "resource exhausted",
        "allocation failed",
    )
    return any(fragment in msg for fragment in oom_fragments)


def _clear_accelerator_cache() -> None:
    gc.collect()
    try:
        torch = _require_torch("clear accelerator cache after OOM")
    except ImportError:
        return
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "empty_cache"):
        try:
            mps.empty_cache()
        except Exception:
            pass


def _run_with_dynamic_batch(label: str, initial_batch_size: int,
                            run: Callable[[int], T]) -> Tuple[T, int]:
    """Run ``run(batch_size)`` and halve the batch size on accelerator OOM.

    Some model stacks raise non-``RuntimeError`` exceptions for accelerator
    allocation failures, so inspect the message instead of keying only on the
    exception class.
    """
    batch_size = max(1, int(initial_batch_size))
    while True:
        try:
            return run(batch_size), batch_size
        except Exception as exc:
            if not _is_oom_error(exc) or batch_size <= 1:
                raise
            next_batch_size = max(1, batch_size // 2)
            logger.warning("%s OOM at batch_size=%d; retrying with %d",
                           label, batch_size, next_batch_size)
            batch_size = next_batch_size
        # Clear only after leaving the ``except`` suite: the exception and its
        # traceback can retain the failed attempt's CUDA tensors until then.
        _clear_accelerator_cache()


# ==============================================================================
# Official config universe — verbatim from the submission notebooks.
# 55 short-only + 21 med/long datasets, x {short, medium, long} where allowed
# = 97 configs.
# ==============================================================================
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

# Leaderboard CSV key naming (verbatim ``pretty_names`` from the notebooks).
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

REF_DIR = os.path.join(os.path.dirname(__file__), "leaderboard_reference")
OUT_ROOT = "logs/experiments/sanity_leaderboard"

# Official reference CSVs shipped in-repo (from
# huggingface.co/spaces/Salesforce/GIFT-Eval  results/<model>/all_results.csv).
REFERENCE_CSVS = {
    "google/timesfm-2.5-200m-pytorch": "timesfm-2.5_all_results.csv",
    "amazon/chronos-2": "chronos-2_all_results.csv",
    "autogluon/chronos-2": "chronos-2_all_results.csv",  # same weights
    "autogluon/chronos-2-synth": "chronos-2-synth_all_results.csv",
    "ibm-research/patchtst-fm-r1": "patchtst-fm-r1_all_results.csv",
}

# CLI shorthand -> HF id of the official submission.
MODEL_ALIASES = {
    "timesfm": "google/timesfm-2.5-200m-pytorch",
    "timesfm-2.5": "google/timesfm-2.5-200m-pytorch",
    "chronos-2": "amazon/chronos-2",
    "chronos-2-synth": "autogluon/chronos-2-synth",
    "patchtst": "ibm-research/patchtst-fm-r1",
    "patchtst-fm": "ibm-research/patchtst-fm-r1",
}

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Model ids that should stay on the official notebook recipe by default. Use the
# display name from experiments.models_config (e.g. TimesFM2.5-200M,
# Chronos2-Base, Moirai2-Small) to run the repo/pipeline wrapper instead.
OFFICIAL_RECIPE_IDS = {
    "google/timesfm-2.5-200m-pytorch",
    "amazon/chronos-2",
    "autogluon/chronos-2",
    "autogluon/chronos-2-synth",
}


def _model_family(model_name: str) -> str:
    """Which official notebook governs this model."""
    return "timesfm" if "timesfm" in model_name.lower() else "chronos2"


def _pipeline_model_index() -> dict:
    from experiments.models_config import CATALOG

    idx = {}
    for spec in CATALOG:
        for key in (spec.display, spec.model_id):
            idx[key.lower()] = (spec.model_id, spec.family, spec.display)
    return idx


def resolve_model(model_arg: str) -> Tuple[str, Optional[str], str, str]:
    """Return (model_id, pipeline_family, display, recipe).

    recipe is "official" for the two published notebook ports and "pipeline" for
    models evaluated through this repo's v5 ablation wrappers.
    """
    aliased = MODEL_ALIASES.get(model_arg, model_arg)
    if aliased in OFFICIAL_RECIPE_IDS:
        return aliased, None, os.path.basename(aliased), "official"

    spec = _pipeline_model_index().get(model_arg.lower()) \
        or _pipeline_model_index().get(aliased.lower())
    if spec is not None:
        model_id, family, display = spec
        return model_id, family, display, "pipeline"

    # Unknown ids keep the previous behavior: assume an official-style recipe
    # family based on the model name, but warn that no pipeline family was found.
    logger.warning("model %r is not in experiments.models_config; using the "
                   "official-style recipe dispatch", model_arg)
    return aliased, None, os.path.basename(aliased), "official"


# ==============================================================================
# Config enumeration
# ==============================================================================
def _dataset_properties() -> dict:
    with open(os.path.join(REF_DIR, "dataset_properties.json")) as f:
        return json.load(f)


def official_configs() -> List[dict]:
    """The 97 (ds_name, term) rows with the notebook's config-key naming."""
    props = _dataset_properties()
    med_long = set(MED_LONG_DATASETS.split())
    rows = []
    for ds_name in SHORT_DATASETS.split():
        for term in ("short", "medium", "long"):
            if term in ("medium", "long") and ds_name not in med_long:
                continue
            if "/" in ds_name:
                base, ds_freq = ds_name.split("/")
                ds_key = PRETTY_NAMES.get(base.lower(), base.lower())
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


# ==============================================================================
# Shared metric list — identical across both notebooks.
# ==============================================================================
def build_metrics():
    from gluonts.ev.metrics import (
        MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
        MeanWeightedSumQuantileLoss,
    )

    return [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS),
    ]


# ==============================================================================
# TimesFM-2.5 — verbatim port of timesfm2p5.ipynb's TimesFmPredictor.
# Only addition: ``max_context_knob`` optionally truncates each input to its last
# N steps first (attribution runs mimicking our pipeline's window cap).
# ==============================================================================
class TimesFmPredictor:
    def __init__(self, tfm, prediction_length: int,
                 max_context_knob: Optional[int] = None,
                 default_batch_size: int = 1024):
        np = _require_numpy("construct the TimesFM predictor")

        self.tfm = tfm
        self.prediction_length = prediction_length
        self.quantiles = list(np.arange(1, 10) / 10.0)
        self.max_context_knob = max_context_knob
        self.default_batch_size = int(default_batch_size)

    @staticmethod
    def _model_context(entry_target):
        from experiments.timesfm_gifteval import model_context
        return model_context(entry_target)

    def predict(self, test_data_input, batch_size: Optional[int] = None) -> List:
        np = _require_numpy("run TimesFM forecasts")

        from gluonts.model.forecast import QuantileForecast
        from experiments.timesfm_gifteval import forecast_quantiles

        if batch_size is None:
            batch_size = self.default_batch_size
        test_data_input = list(test_data_input)
        forecast_outputs = forecast_quantiles(
            self.tfm,
            [entry["target"] for entry in test_data_input],
            self.prediction_length,
            batch_size=batch_size,
            max_context_knob=self.max_context_knob,
        )

        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(QuantileForecast(
                forecast_arrays=item,
                forecast_keys=list(map(str, self.quantiles)),
                start_date=forecast_start_date,
            ))
        return forecasts


# ==============================================================================
# Chronos-2 — verbatim port of chronos-2.ipynb's Chronos2Predictor.
# Only additions: the pipeline is loaded once and reused (inference is
# stateless), and ``max_context`` optionally truncates each input first.
# ==============================================================================
class Chronos2Predictor:
    def __init__(self, pipeline, prediction_length: int, batch_size: int,
                 quantile_levels=QUANTILE_LEVELS,
                 predict_batches_jointly: bool = False,
                 max_context: Optional[int] = None):
        self.pipeline = pipeline
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
        np = _require_numpy("run Chronos-2 forecasts")
        torch = _require_torch("run Chronos-2 forecasts")

        from gluonts.model.forecast import QuantileForecast

        pipeline = self.pipeline
        input_data = list(self._pack_model_items(test_data_input))
        is_univariate_data = input_data[0]["target"].ndim == 1

        def _run(model_batch_size: int):
            quantiles, _ = pipeline.predict_quantiles(
                inputs=input_data,
                prediction_length=self.prediction_length,
                batch_size=model_batch_size,
                quantile_levels=self.quantile_levels,
                predict_batches_jointly=self.predict_batches_jointly,
            )
            quantiles = torch.stack(quantiles)
            # [batch, variates, seq, quantiles] -> [batch, quantiles, seq, variates]
            quantiles = quantiles.permute(0, 3, 2, 1).cpu().numpy()
            if is_univariate_data:
                quantiles = quantiles.squeeze(-1)
            assert quantiles.shape[1] == len(self.quantile_levels)
            assert quantiles.shape[2] == self.prediction_length
            return quantiles

        forecast_outputs, self.batch_size = _run_with_dynamic_batch(
            "Chronos-2", self.batch_size, _run)
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


# ==============================================================================
# Model loading — cached; each family uses its OFFICIAL load path.
# ==============================================================================
_MODEL_HANDLE: dict = {}


def _load_timesfm(model_name: str):
    """Official timesfm2p5.ipynb load path:
        tfm = timesfm_2p5_torch.TimesFM_2p5_200M_torch()
        tfm.load_checkpoint()
    Some installed TimesFM builds changed that method to require an explicit
    checkpoint path; those fall back to the same HF id used everywhere else in
    this repo, or to TIMESFM_2P5_CHECKPOINT / TIMESFM_CHECKPOINT if set."""
    from experiments.timesfm_gifteval import load_model

    if model_name not in _MODEL_HANDLE:
        _MODEL_HANDLE[model_name] = load_model(model_name)
    return _MODEL_HANDLE[model_name]


def _load_chronos2(model_name: str, device: str):
    """Official chronos-2.ipynb load path: BaseChronosPipeline.from_pretrained,
    asserting it resolves to a Chronos2Pipeline."""
    if model_name not in _MODEL_HANDLE:
        from chronos import BaseChronosPipeline, Chronos2Pipeline

        pipe = BaseChronosPipeline.from_pretrained(
            model_name, device_map=device, torch_dtype="float32")
        assert isinstance(pipe, Chronos2Pipeline), (
            "this sanity script's chronos path is Chronos-2-only "
            "(see the official chronos-bolt notebook for other chronos models)")
        _MODEL_HANDLE[model_name] = pipe
    return _MODEL_HANDLE[model_name]


def _pipeline_context_cap(family: str, horizon: int, requested: Optional[int],
                          max_available: int) -> int:
    import experiments.test_window_ablation_gifteval_v5 as wab

    hard_caps = {
        "chronos2": 8192,
        "chronos_bolt": 2048,
        "moirai": wab._moirai_max_context(horizon),
        "moirai_1_1": max(1, 8192 - int(horizon)),
        "timesfm": 15360,
        "patchtst_fm": 8192,
        "sundial": getattr(wab, "SUNDIAL_MAX_CONTEXT", max_available),
        "toto": getattr(wab, "TOTO_MAX_CONTEXT", max_available),
        "flowstate": getattr(wab, "FLOWSTATE_MAX_CONTEXT", max_available),
        "tirex": getattr(wab, "TIREX_MAX_CONTEXT", max_available),
    }
    if family == "timemoe":
        hard_caps[family] = max(
            1, getattr(wab, "TIMEMOE_MAX_TOTAL", max_available) - int(horizon))

    cap = int(requested) if requested is not None else int(max_available)
    if family in hard_caps:
        cap = min(cap, int(hard_caps[family]))
    return max(1, min(cap, int(max_available)))


def _pipeline_full_context_batches(cache, cap: int, batch_size: int,
                                   device: str, pin_memory: bool,
                                   preserve_missing: bool = False):
    """Exact full-native batches: every instance uses its last min(len, cap)
    samples, grouped by that effective length so wrappers see rectangular tensors
    without synthetic left padding."""
    np = _require_numpy("build pipeline full-context batches")
    import torch

    lengths = np.maximum(1, np.minimum(cache.context_lengths, int(cap)))
    groups = []
    source_contexts = cache.contexts_raw if preserve_missing else cache.contexts
    for width in np.unique(lengths):
        width = int(width)
        idx = np.flatnonzero(lengths == width)
        x_np = np.empty((idx.size, width), dtype=np.float32)
        for j, i in enumerate(idx):
            ctx = np.asarray(source_contexts[i], dtype=np.float32)
            if ctx.size >= width:
                x_np[j] = ctx[-width:]
            else:
                x_np[j] = np.concatenate([
                    np.zeros(width - ctx.size, dtype=np.float32),
                    ctx,
                ])
        y_np = cache.labels_np[idx]

        all_x = torch.from_numpy(x_np).unsqueeze(-1)
        all_y = torch.from_numpy(y_np).unsqueeze(-1)
        if pin_memory and device == "cuda":
            ax, ay = all_x.pin_memory(), all_y.pin_memory()
        else:
            ax, ay = all_x, all_y

        batches = []
        for start in range(0, idx.size, batch_size):
            stop = min(start + batch_size, idx.size)
            batches.append({"x": ax[start:stop], "y": ay[start:stop]})
        groups.append((idx, width, batches))
    return groups


def evaluate_pipeline_on_dataset(model_id: str, model_family: str,
                                 ds_name: str, ds_term: str, batch_size: int,
                                 device: str, max_context: Optional[int]) -> dict:
    import numpy as np
    import torch

    import experiments.test_window_ablation_gifteval_v5 as wab
    from gift_eval.data import Dataset

    to_univariate = Dataset(
        name=ds_name, term=ds_term, to_univariate=False).target_dim > 1
    dataset = Dataset(name=ds_name, term=ds_term, to_univariate=to_univariate)
    cache = wab.GiftEvalCache(dataset, ds_name)
    horizon = cache.horizon
    cap = _pipeline_context_cap(
        model_family, horizon, max_context, cache.max_context)

    logger.info("pipeline full-context %s/%s: cap=%d ctx_range=[%d,%d] n=%d",
                ds_name, ds_term, cap, cache.min_context, cache.max_context,
                cache.n_total)

    handle = (None if model_family == "context_parroting"
              else wab.load_handle(model_family, model_id, device))

    try:
        def _run(eval_batch_size: int) -> dict:
            groups = None
            results = []
            try:
                if model_family in {
                        "timesfm", "chronos_bolt", "patchtst_fm", "sundial"}:
                    contexts = [
                        np.asarray(ctx[-cap:], dtype=np.float32)
                        for ctx in cache.contexts_raw
                    ]
                    fr, _tgts = wab.predict_official_full_contexts(
                        model_family, handle, contexts, cache.labels_np,
                        horizon, device, eval_batch_size)
                else:
                    groups = _pipeline_full_context_batches(
                        cache, cap, eval_batch_size, device,
                        pin_memory=(device == "cuda"),
                        preserve_missing=wab.preserves_missing(model_family))
                    for idx, width, batches in groups:
                        fr, _tgts = wab._forecast_cell(
                            model_family, handle, model_id, batches, width, horizon,
                            device, eval_batch_size,
                            flowstate_scale=cache.flowstate_scale)
                        results.append((idx, fr, _tgts))
                    fr, _tgts = wab._merge_grouped(
                        results, cache.n_total, horizon, device)
                mase = wab.cell_mase_gluonts_real(
                    fr, cache, np.arange(cache.n_total))
                return {
                    "MASE[0.5]": float(mase),
                    "_recipe": "pipeline_full_context",
                    "_inference_recipe": inference_recipe(model_family),
                    "_context_cap": int(cap),
                    "_model_family": model_family,
                    "_batch_size": int(eval_batch_size),
                }
            except Exception:
                del results
                if groups is not None:
                    del groups
                _clear_accelerator_cache()
                raise

        metrics, _ = _run_with_dynamic_batch(
            f"{model_family} {ds_name}/{ds_term}", batch_size, _run)
        return metrics
    finally:
        del handle
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==============================================================================
# Per-config evaluation — dispatch on family, each branch a notebook port.
# ==============================================================================
def evaluate_on_dataset(model_name: str, ds_name: str, ds_term: str,
                        batch_size: int, device: str,
                        use_multivariate_data: bool,
                        predict_batches_jointly: bool,
                        max_context: Optional[int],
                        pipeline_family: Optional[str] = None) -> dict:
    if pipeline_family is not None:
        return evaluate_pipeline_on_dataset(
            model_name, pipeline_family, ds_name, ds_term, batch_size,
            device, max_context)

    from gluonts.model import evaluate_forecasts, evaluate_model
    from gluonts.time_feature import get_seasonality

    from gift_eval.data import Dataset

    metrics = build_metrics()

    if _model_family(model_name) == "timesfm":
        # ---- timesfm2p5.ipynb recipe ----
        # Flatten only genuinely multivariate sources; evaluate the whole test
        # split through evaluate_model (which calls predictor.predict).
        to_univariate = Dataset(
            name=ds_name, term=ds_term, to_univariate=False).target_dim > 1
        dataset = Dataset(name=ds_name, term=ds_term, to_univariate=to_univariate)
        predictor = TimesFmPredictor(
            tfm=_load_timesfm(model_name),
            prediction_length=dataset.prediction_length,
            max_context_knob=max_context,
            default_batch_size=1024,
        )
        def _run(eval_batch_size: int):
            predictor.default_batch_size = eval_batch_size
            return evaluate_model(
                predictor,
                test_data=dataset.test_data,
                metrics=metrics,
                batch_size=eval_batch_size,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                # str(): gift_eval's Dataset.freq can arrive as numpy.str_,
                # which the Cython-compiled pandas inside get_seasonality
                # rejects. Harmless when it is already a plain str.
                seasonality=get_seasonality(str(dataset.freq)),
            ).reset_index(drop=True).to_dict(orient="records")

        res, effective_batch_size = _run_with_dynamic_batch(
            f"TimesFM {ds_name}/{ds_term}", 1024, _run)
        out = res[0]
        out["_batch_size"] = int(effective_batch_size)
        return out

    # ---- chronos-2.ipynb recipe ----
    # Native multivariate unless --univariate flattens it.
    is_multivariate_source = Dataset(
        name=ds_name, term=ds_term, to_univariate=False).target_dim > 1
    dataset = Dataset(
        name=ds_name, term=ds_term,
        to_univariate=is_multivariate_source and not use_multivariate_data)
    predictor = Chronos2Predictor(
        pipeline=_load_chronos2(model_name, device),
        prediction_length=dataset.prediction_length,
        batch_size=batch_size,
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
        metrics=metrics,
        batch_size=1024,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=get_seasonality(str(dataset.freq)),
    ).reset_index(drop=True).to_dict(orient="records")
    out = res[0]
    out["_batch_size"] = int(predictor.batch_size)
    return out


# ==============================================================================
# Comparison against the official leaderboard CSVs
# ==============================================================================
def _load_reference_mase(csv_name: str) -> dict:
    pd = _require_pandas("load leaderboard reference CSVs")

    df = pd.read_csv(os.path.join(REF_DIR, csv_name))
    return dict(zip(df["dataset"], df["eval_metrics/MASE[0.5]"]))


def _geomean(s) -> float:
    s = s.dropna()
    if not len(s):
        return float("nan")
    vals = [float(v) for v in s]
    if any(v < 0.0 for v in vals):
        return float("nan")
    if any(v == 0.0 for v in vals):
        return 0.0
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


def compare_and_report(ours: dict, model_name: str, out_dir: str) -> None:
    """``ours``: {config_key: metrics-dict}. Prints the per-config diff table and
    the normalized-geomean headline vs the official one, and writes
    ``comparison.csv``."""
    pd = _require_pandas("compare against leaderboard reference CSVs")

    sn = published_seasonal_naive_mase()
    ref_csv = REFERENCE_CSVS.get(model_name)
    ref = _load_reference_mase(ref_csv) if ref_csv else {}
    if not ref:
        logger.warning(f"no reference CSV for {model_name}; only 'ours' columns "
                       "will be meaningful")

    rows = []
    for key, m in sorted(ours.items()):
        mase = m.get("MASE[0.5]", float("nan"))
        mase_off = ref.get(key, float("nan"))
        sn_off = sn.get(key, float("nan"))
        rows.append({
            "dataset": key,
            "mase_ours": mase,
            "mase_official": mase_off,
            "sn_official": sn_off,
            "rel_diff_pct": (100.0 * (mase - mase_off) / mase_off
                             if mase_off == mase_off else float("nan")),
            "norm_ours": mase / sn_off,
            "norm_official": mase_off / sn_off,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "comparison.csv"), index=False)

    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df.round(4).to_string(index=False))

    n = len(df)
    print(f"\n==== {model_name} on {n}/97 configs ====")
    print("PUBLISHED aggregation: geomean of MASE[0.5] / published seasonal_naive MASE[0.5]")
    print(f"geomean normalized MASE (ours)     : {_geomean(df['norm_ours']):.4f}")
    print(f"geomean normalized MASE (official) : {_geomean(df['norm_official']):.4f}"
          f"   <- leaderboard headline on these {n} configs")
    print(f"raw geomean MASE (ours)            : {_geomean(df['mase_ours']):.4f}")
    print(f"raw geomean MASE (official)        : {_geomean(df['mase_official']):.4f}")
    if n < 97:
        print(f"NOTE: only {n}/97 configs — compare the headline ONLY to the "
              "'official' column above, not to the full-board number.")
    worst = df.reindex(
        df["rel_diff_pct"].abs().sort_values(ascending=False).index).head(10)
    print("\nlargest per-config deviations (ours vs official):")
    print(worst[["dataset", "mase_ours", "mase_official", "rel_diff_pct"]]
          .round(4).to_string(index=False))


# ==============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="timesfm-2.5",
                    help="official shorthand (timesfm-2.5, chronos-2, "
                         "chronos-2-synth, patchtst-fm), a models_config display name "
                         "(Moirai2-Small, PatchTST-FM-R1, ...), or a full HF id")
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
                         "pipeline does this; timesfm already flattens)")
    ap.add_argument("--independent", action="store_true",
                    help="chronos-2 only: predict_batches_jointly=False "
                         "(official uses True: in-context cross-learning)")
    ap.add_argument("--max-context", type=int, default=None,
                    help="truncate inputs to last N steps (our grid tops at "
                         "8192; the official recipe feeds everything)")
    args = ap.parse_args()

    model_id, pipeline_family, model_display, recipe = resolve_model(args.model)
    args.model = model_id
    if recipe == "official" and _model_family(args.model) == "timesfm" \
            and (args.univariate or args.independent):
        logger.warning("--univariate/--independent are no-ops for timesfm: its "
                       "official recipe is already univariate and independent")
    if recipe == "pipeline" and (args.univariate or args.independent):
        logger.warning("--univariate/--independent only apply to the official "
                       "Chronos-2 recipe; pipeline models already use the repo's "
                       "univariate wrappers")

    tag = os.path.basename(args.model) if recipe == "official" else model_display
    if recipe == "pipeline":
        tag += "_pipeline_full"
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
                cached = json.load(f)
            expected_recipe = (None if recipe == "official"
                               else inference_recipe(pipeline_family))
            cached_mase = cached.get("MASE[0.5]", float("nan"))
            if ((expected_recipe is None or
                 cached.get("_inference_recipe") == expected_recipe)
                    and math.isfinite(float(cached_mase))):
                ours[cfg["key"]] = cached
                logger.info(f"[{i + 1}/{len(configs)}] {cfg['key']}  (cached)")
                continue
            logger.warning(
                "[%d/%d] %s stale/non-finite cache; recomputing",
                i + 1, len(configs), cfg["key"],
            )
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
                pipeline_family=pipeline_family,
            )
        except ImportError as exc:
            raise SystemExit(
                f"{cfg['key']} cannot run because a dependency is missing: {exc}"
            ) from exc
        except Exception as exc:  # keep going; a hole in the table is loud enough
            logger.error(f"{cfg['key']} FAILED: {exc}", exc_info=True)
            continue
        m = {k: _jsonable_metric_value(v) for k, v in m.items()}
        with open(cell_path, "w") as f:
            json.dump(m, f, indent=1)
        ours[cfg["key"]] = m
        logger.info(f"    MASE[0.5] = {m.get('MASE[0.5]', float('nan')):.4f}")

    if not ours:
        raise SystemExit("no results produced")
    compare_and_report(ours, args.model, out_dir)


if __name__ == "__main__":
    main()
