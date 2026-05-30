"""
Window-size ablation on GiftEval + zero-shot predictor curve overlay.

Same ablation pipeline as test_window_ablation_gifteval_v4.py, with one added
deliverable: for every (dataset, term) we also run the trained context-length
predictor (from predict_context_length.py) on every test instance and overlay
the predicted error-vs-context curve against the real one.

Differences vs v4
-----------------
- Predictor checkpoint auto-discovered from the latest run under
  CACHE_ROOT_PREDICTOR (best_model.pt + best_config.json).
- Ablation window grid is overridden by the predictor's training window_grid,
  so both curves share the same x-axis exactly.
- For each (dataset, term):
    * real curve : aggregate metric per window from the v4 ablation loop
                   (using the predictor's curve_metric, default "mae").
    * predicted  : per-instance predictor output (mean +/- std across
                   instances). Predictor input is each test instance's
                   suffix, left-padded with zeros to context_length and
                   standardized to zero mean / unit std (matching
                   build_context_length_dataset.py).
    * both curves are z-scored along the windows axis (the predictor only
      learns curve shape; argmin is invariant to shift/scale) and overlaid.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from colorama import Fore
from dotenv import load_dotenv
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()

from gift_eval.data import Dataset as GiftEvalDataset

try:
    from tsfm_public import PatchTSTFMForPrediction
except ImportError:
    PatchTSTFMForPrediction = None

from experiments.predict_context_length import PatchTSTContextLength


# ==============================================================================
#  STATIC CONFIG
# ==============================================================================

MODELS = [
    ("autogluon/chronos-2-small",       "chronos2",    "Chronos2-Small"),
    ("amazon/chronos-bolt-small",       "chronos_bolt","ChronosBolt-Small"),
    ("Salesforce/moirai-2.0-R-small",   "moirai",      "Moirai2-Small"),
    ("google/timesfm-2.5-200m-pytorch", "timesfm",     "TimesFM2.5-200M"),
    ("ibm-research/patchtst-fm-r1",     "patchtst_fm", "PatchTST-FM-R1"),
    ("thuml/sundial-base-128m",         "sundial",     "Sundial-Base-128M"),
    ("Maple728/TimeMoE-200M",           "timemoe",     "TimeMoE-200M"),
]

DATASETS = [
    ("jena_weather/10T",            "short",  "JenaWeather-10T", True),
    ("jena_weather/10T",            "medium", "JenaWeather-10T", True),
    ("jena_weather/10T",            "long",   "JenaWeather-10T", True),
    ("jena_weather/H",              "short",  "JenaWeather-H",   True),
    ("jena_weather/H",              "medium", "JenaWeather-H",   True),
    ("jena_weather/H",              "long",   "JenaWeather-H",   True),
    ("jena_weather/D",              "short",  "JenaWeather-D",   True),
    ("bizitobs_application",        "short",  "BizITObsApp",     True),
    ("bizitobs_application",        "medium", "BizITObsApp",     True),
    ("bizitobs_application",        "long",   "BizITObsApp",     True),
    ("bizitobs_service",            "short",  "BizITObsService", True),
    ("bizitobs_service",            "medium", "BizITObsService", True),
    ("bizitobs_service",            "long",   "BizITObsService", True),
    ("bizitobs_l2c/5T",             "short",  "BizITObsL2C-5T",  True),
    ("bizitobs_l2c/5T",             "medium", "BizITObsL2C-5T",  True),
    ("bizitobs_l2c/5T",             "long",   "BizITObsL2C-5T",  True),
    ("bizitobs_l2c/H",              "short",  "BizITObsL2C-H",   True),
    ("bizitobs_l2c/H",              "medium", "BizITObsL2C-H",   True),
    ("bizitobs_l2c/H",              "long",   "BizITObsL2C-H",   True),
    ("bitbrains_fast_storage/5T",   "short",  "BitbrainsFS-5T",  True),
    ("bitbrains_fast_storage/5T",   "medium", "BitbrainsFS-5T",  True),
    ("bitbrains_fast_storage/5T",   "long",   "BitbrainsFS-5T",  True),
    ("bitbrains_fast_storage/H",    "short",  "BitbrainsFS-H",   True),
    ("bitbrains_rnd/5T",            "short",  "BitbrainsRnD-5T", True),
    ("bitbrains_rnd/5T",            "medium", "BitbrainsRnD-5T", True),
    ("bitbrains_rnd/5T",            "long",   "BitbrainsRnD-5T", True),
    ("bitbrains_rnd/H",             "short",  "BitbrainsRnD-H",  True),
    ("restaurant",                  "short",  "Restaurant",      False),
    ("ett1/15T",                    "short",  "ETTm1-15T",       True),
    ("ett1/15T",                    "medium", "ETTm1-15T",       True),
    ("ett1/15T",                    "long",   "ETTm1-15T",       True),
    ("ett1/H",                      "short",  "ETTm1-H",         True),
    ("ett1/H",                      "medium", "ETTm1-H",         True),
    ("ett1/H",                      "long",   "ETTm1-H",         True),
    ("ett1/D",                      "short",  "ETTm1-D",         True),
    ("ett1/W",                      "short",  "ETTm1-W",         True),
    ("ett2/15T",                    "short",  "ETTm2-15T",       True),
    ("ett2/15T",                    "medium", "ETTm2-15T",       True),
    ("ett2/15T",                    "long",   "ETTm2-15T",       True),
    ("ett2/H",                      "short",  "ETTm2-H",         True),
    ("ett2/H",                      "medium", "ETTm2-H",         True),
    ("ett2/H",                      "long",   "ETTm2-H",         True),
    ("ett2/D",                      "short",  "ETTm2-D",         True),
    ("ett2/W",                      "short",  "ETTm2-W",         True),
    ("LOOP_SEATTLE/5T",             "short",  "LoopSeattle-5T",  False),
    ("LOOP_SEATTLE/5T",             "medium", "LoopSeattle-5T",  False),
    ("LOOP_SEATTLE/5T",             "long",   "LoopSeattle-5T",  False),
    ("LOOP_SEATTLE/H",              "short",  "LoopSeattle-H",   False),
    ("LOOP_SEATTLE/H",              "medium", "LoopSeattle-H",   False),
    ("LOOP_SEATTLE/H",              "long",   "LoopSeattle-H",   False),
    ("LOOP_SEATTLE/D",              "short",  "LoopSeattle-D",   False),
    ("SZ_TAXI/15T",                 "short",  "SZTaxi-15T",      False),
    ("SZ_TAXI/15T",                 "medium", "SZTaxi-15T",      False),
    ("SZ_TAXI/15T",                 "long",   "SZTaxi-15T",      False),
    ("SZ_TAXI/H",                   "short",  "SZTaxi-H",        False),
    ("M_DENSE/H",                   "short",  "MDense-H",        False),
    ("M_DENSE/H",                   "medium", "MDense-H",        False),
    ("M_DENSE/H",                   "long",   "MDense-H",        False),
    ("M_DENSE/D",                   "short",  "MDense-D",        False),
    ("solar/10T",                   "short",  "Solar-10T",       False),
    ("solar/10T",                   "medium", "Solar-10T",       False),
    ("solar/10T",                   "long",   "Solar-10T",       False),
    ("solar/H",                     "short",  "Solar-H",         False),
    ("solar/H",                     "medium", "Solar-H",         False),
    ("solar/H",                     "long",   "Solar-H",         False),
    ("solar/D",                     "short",  "Solar-D",         False),
    ("solar/W",                     "short",  "Solar-W",         False),
    ("hierarchical_sales/D",        "short",  "HierSales-D",     False),
    ("hierarchical_sales/W",        "short",  "HierSales-W",     False),
    ("m4_yearly",                   "short",  "M4-Yearly",       False),
    ("m4_quarterly",                "short",  "M4-Quarterly",    False),
    ("m4_monthly",                  "short",  "M4-Monthly",      False),
    ("m4_weekly",                   "short",  "M4-Weekly",       False),
    ("m4_daily",                    "short",  "M4-Daily",        False),
    ("m4_hourly",                   "short",  "M4-Hourly",       False),
    ("hospital",                    "short",  "Hospital",        False),
    ("covid_deaths",                "short",  "CovidDeaths-D",   False),
    ("us_births/D",                 "short",  "USBirths-D",      False),
    ("us_births/W",                 "short",  "USBirths-W",      False),
    ("us_births/M",                 "short",  "USBirths-M",      False),
    ("saugeenday/D",                "short",  "SaugeenDay-D",    False),
    ("saugeenday/W",                "short",  "SaugeenDay-W",    False),
    ("saugeenday/M",                "short",  "SaugeenDay-M",    False),
    ("electricity/15T",             "short",  "Electricity-15T", False),
    ("electricity/15T",             "medium", "Electricity-15T", False),
    ("electricity/15T",             "long",   "Electricity-15T", False),
    ("electricity/H",               "short",  "Electricity-H",   False),
    ("electricity/H",               "medium", "Electricity-H",   False),
    ("electricity/H",               "long",   "Electricity-H",   False),
    ("electricity/D",               "short",  "Electricity-D",   False),
    ("electricity/W",               "short",  "Electricity-W",   False),
]

MOIRAI2_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MOIRAI2_MEDIAN_IDX = 4
MOIRAI_1_1_NUM_SAMPLES = 100
MOIRAI_1_1_PATCH_SIZE = 32
TIMESFM_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PATCHTST_FM_QUANTILE_LEVELS = [i / 100.0 for i in range(1, 100)]
PATCHTST_FM_MEDIAN_QUANTILE_IDX = 49
SUNDIAL_NUM_SAMPLES = 20
SUNDIAL_MAX_CONTEXT = 2880
TIMEMOE_MAX_TOTAL   = 4096   # context + horizon must not exceed this

N_BEST_WORST = 10
PLOT_METRICS = ["mae", "mse", "rmse", "mase", "smape", "crps"]
CACHE_ROOT = "logs/experiments/window_ablation_gifteval"
CACHE_ROOT_PREDICTOR = "logs/experiments/context_length_predictor"


# ==============================================================================
#  FORECAST CONTAINER
# ==============================================================================

@dataclass
class ForecastResult:
    median: torch.Tensor
    samples: Optional[torch.Tensor] = None
    quantiles: Optional[torch.Tensor] = None
    quantile_levels: Optional[List[float]] = None


# ==============================================================================
#  SEASONALITY (frequency-based only)
# ==============================================================================

def _get_seasonality(freq_str: str) -> int:
    freq_map = {
        "S": 86400, "T": 1440, "5T": 288, "10T": 144, "15T": 96,
        "30T": 48, "H": 24, "D": 7, "W": 52, "M": 12, "Q": 4, "Y": 1,
        "1H": 24, "1D": 7, "1W": 52, "1M": 12,
    }
    f = freq_str.upper().replace("MIN", "T").replace("HOURLY", "H")
    if f in freq_map:
        return freq_map[f]
    for k, v in freq_map.items():
        if f.endswith(k):
            return v
    return 1


# ==============================================================================
#  DATASET PRECOMPUTATION (single pass per (name, term))
# ==============================================================================

class GiftEvalCache:
    def __init__(self, ge_dataset: GiftEvalDataset, dataset_display: str):
        self.ge_dataset = ge_dataset
        self.dataset_display = dataset_display
        self.freq: str = ge_dataset.freq
        self.horizon: int = ge_dataset.prediction_length
        self.season: int = _get_seasonality(self.freq)

        contexts: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        for test_input, test_label in ge_dataset.test_data:
            target = test_input["target"]
            if target.ndim > 1:
                raise ValueError(
                    f"Expected univariate context; got shape {target.shape}. "
                    "Set to_univariate=True."
                )
            label = test_label["target"]
            if label.ndim > 1:
                raise ValueError(
                    f"Expected univariate label; got shape {label.shape}."
                )
            if len(label) < self.horizon:
                continue
            ctx = np.nan_to_num(np.asarray(target, dtype=np.float32), nan=0.0)
            lbl = np.asarray(label[: self.horizon], dtype=np.float32)
            contexts.append(ctx)
            labels.append(lbl)

        if not contexts:
            raise RuntimeError(
                f"No valid test instances for {dataset_display}/{ge_dataset.freq}."
            )

        self.contexts: List[np.ndarray] = contexts
        self.context_lengths: np.ndarray = np.array(
            [len(c) for c in contexts], dtype=np.int64
        )
        self.labels_np: np.ndarray = np.stack(labels, axis=0)
        self.max_context: int = int(self.context_lengths.max())
        self.min_context: int = int(self.context_lengths.min())
        self.n_total: int = len(contexts)
        self.naive_seasonal_mae_train: float = self._compute_naive_mae_train()

    def _compute_naive_mae_train(self) -> float:
        season = self.season
        abs_diffs = []
        for entry in self.ge_dataset.training_dataset:
            target = entry["target"]
            if target.ndim > 1:
                target = target[0]
            target = np.asarray(target, dtype=np.float64)
            if len(target) <= season:
                continue
            d = np.abs(target[season:] - target[:-season])
            d = d[~np.isnan(d)]
            if d.size:
                abs_diffs.append(d)
        if not abs_diffs:
            return 1.0
        return max(float(np.mean(np.concatenate(abs_diffs))), 1e-9)

    def can_serve(self, window_size: int) -> bool:
        return window_size <= self.max_context

    def build_batches(
        self,
        window_size: int,
        batch_size: int,
        device: str,
        pin_memory: bool,
    ) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, np.ndarray]:
        valid_mask = self.context_lengths >= window_size
        valid_indices = np.flatnonzero(valid_mask)
        if valid_indices.size == 0:
            raise RuntimeError(
                f"No instance has enough context for window_size={window_size} "
                f"(max_context={self.max_context})."
            )

        n = valid_indices.size
        x_np = np.empty((n, window_size), dtype=np.float32)
        for j, i in enumerate(valid_indices):
            x_np[j] = self.contexts[i][-window_size:]

        y_np = self.labels_np[valid_indices]

        all_x = torch.from_numpy(x_np).unsqueeze(-1)
        all_y = torch.from_numpy(y_np).unsqueeze(-1)

        if pin_memory and device == "cuda":
            all_x_pinned = all_x.pin_memory()
            all_y_pinned = all_y.pin_memory()
        else:
            all_x_pinned, all_y_pinned = all_x, all_y

        batches = []
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batches.append({
                "x": all_x_pinned[start:stop],
                "y": all_y_pinned[start:stop],
            })

        return batches, all_x, all_y, valid_indices


# ==============================================================================
#  NAIVE BASELINE
# ==============================================================================

def _naive_cache_path(dataset_display: str, term: str) -> str:
    return os.path.join(
        CACHE_ROOT, dataset_display, "_naive_seasonal", f"t{term}", "metrics.json"
    )


def compute_naive_seasonal_test_metrics(cache: GiftEvalCache) -> Dict[str, float]:
    season = cache.season
    horizon = cache.horizon
    n = cache.n_total

    preds_np = np.empty((n, horizon), dtype=np.float32)
    for i, ctx in enumerate(cache.contexts):
        if len(ctx) >= season:
            tail = ctx[-season:]
            repeats = (horizon // season) + 1
            preds_np[i] = np.tile(tail, repeats)[:horizon]
        else:
            preds_np[i] = ctx[-1] if len(ctx) else 0.0
    preds_np = np.nan_to_num(preds_np, nan=0.0)

    preds = torch.from_numpy(preds_np)
    tgts = torch.from_numpy(cache.labels_np)
    return compute_all_metrics(
        ForecastResult(median=preds), tgts, cache.naive_seasonal_mae_train
    )


def load_or_compute_naive_baseline(
    cache: GiftEvalCache, term: str
) -> Dict[str, float]:
    path = _naive_cache_path(cache.dataset_display, term)
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    metrics = compute_naive_seasonal_test_metrics(cache)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ==============================================================================
#  RESULT CACHE
# ==============================================================================

def _cache_dir(dataset_display: str, model_short: str, term: str, window_size: int) -> str:
    return os.path.join(
        CACHE_ROOT, dataset_display, model_short, f"t{term}", f"w{window_size}"
    )


def _result_cached(dataset_display, model_short, term, window_size) -> bool:
    path = os.path.join(_cache_dir(dataset_display, model_short, term, window_size), "metrics.json")
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r") as f:
            metrics = json.load(f)
        for key in ("mae", "mse", "rmse"):
            v = metrics.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return False
    except (json.JSONDecodeError, OSError):
        return False
    return True


def _load_cached_result(dataset_display, model_short, term, window_size) -> dict:
    path = os.path.join(_cache_dir(dataset_display, model_short, term, window_size), "metrics.json")
    with open(path, "r") as f:
        return json.load(f)


def _save_result(dataset_display, model_short, term, window_size, metrics: dict) -> None:
    d = _cache_dir(dataset_display, model_short, term, window_size)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def _save_per_sample_metrics(dataset_display, model_short, term, window_size, per_sample: dict) -> None:
    d = _cache_dir(dataset_display, model_short, term, window_size)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "per_sample_metrics.npz"), **per_sample)


# ==============================================================================
#  METRICS
# ==============================================================================

def crps_energy_score(samples: torch.Tensor, targets: torch.Tensor) -> float:
    N, S, H = samples.shape
    term1 = (samples - targets.unsqueeze(1)).abs().mean(dim=1)
    sorted_samples, _ = samples.sort(dim=1)
    indices = torch.arange(1, S + 1, device=samples.device, dtype=samples.dtype)
    weights = (2 * indices - S - 1).reshape(1, S, 1)
    term2 = (weights * sorted_samples).sum(dim=1) / (S * S)
    return float((term1 - term2).mean().item())


def crps_quantile_loss(quantiles: torch.Tensor, quantile_levels: List[float], targets: torch.Tensor) -> float:
    Q = len(quantile_levels)
    tau = torch.tensor(quantile_levels, dtype=quantiles.dtype, device=quantiles.device).reshape(1, Q, 1)
    errors = targets.unsqueeze(1) - quantiles
    pinball = torch.where(errors >= 0, tau * errors, (tau - 1) * errors)
    return float((2.0 / Q) * pinball.mean().item())


def compute_all_metrics(
    forecast_result: ForecastResult,
    targets: torch.Tensor,
    naive_seasonal_mae: float = 1.0,
) -> dict:
    pred = forecast_result.median
    y = targets

    valid = ~torch.isnan(y)
    if not valid.all():
        y_safe = y.clone(); y_safe[~valid] = 0.0
        pred_safe = pred.clone(); pred_safe[~valid] = 0.0
    else:
        y_safe, pred_safe = y, pred

    abs_err = (pred_safe - y_safe).abs()
    sq_err = (pred_safe - y_safe) ** 2
    n_valid = valid.sum()

    if n_valid == 0:
        return {k: float("nan") for k in
                ["mae", "mse", "rmse", "mase", "smape", "mape", "nd", "nrmse", "crps"]}

    mae = float(abs_err[valid].mean().item())
    mse = float(sq_err[valid].mean().item())
    rmse = float(np.sqrt(mse))
    mase = mae / naive_seasonal_mae

    denom_smape = (pred_safe.abs() + y_safe.abs()).clamp(min=1e-13)
    smape = float((2.0 * abs_err / denom_smape)[valid].mean().item())
    mape = float((abs_err / y_safe.abs().clamp(min=1e-13))[valid].mean().item())
    nd = float(abs_err[valid].sum().item() / y_safe.abs()[valid].sum().clamp(min=1e-13).item())
    nrmse = rmse / float(y_safe.abs()[valid].mean().clamp(min=1e-13).item())

    crps = float("nan")
    if forecast_result.samples is not None:
        crps = crps_energy_score(forecast_result.samples, y_safe)
    elif forecast_result.quantiles is not None and forecast_result.quantile_levels is not None:
        crps = crps_quantile_loss(forecast_result.quantiles, forecast_result.quantile_levels, y_safe)

    return {
        "mae": mae, "mse": mse, "rmse": rmse, "mase": mase,
        "smape": smape, "mape": mape, "nd": nd, "nrmse": nrmse, "crps": crps,
    }


def compute_per_sample_metrics(
    forecast_result: ForecastResult,
    targets: torch.Tensor,
    naive_seasonal_mae: float = 1.0,
) -> dict:
    pred = forecast_result.median
    y = targets
    abs_err = (pred - y).abs()
    sq_err = (pred - y) ** 2

    per_mae = abs_err.mean(dim=1).cpu().numpy()
    per_mse = sq_err.mean(dim=1).cpu().numpy()
    per_rmse = np.sqrt(per_mse)
    per_mase = per_mae / naive_seasonal_mae
    out = {"mae": per_mae, "mse": per_mse, "rmse": per_rmse, "mase": per_mase}

    if forecast_result.samples is not None:
        S = forecast_result.samples
        num_s = S.shape[1]
        term1 = (S - y.unsqueeze(1)).abs().mean(dim=(1, 2))
        sorted_s, _ = S.sort(dim=1)
        idx = torch.arange(1, num_s + 1, dtype=S.dtype, device=S.device).reshape(1, num_s, 1)
        w = (2 * idx - num_s - 1)
        term2 = ((w * sorted_s).sum(dim=1) / (num_s * num_s)).mean(dim=1)
        out["crps"] = (term1 - term2).cpu().numpy()
    return out


# ==============================================================================
#  MODEL LOADERS + INFERENCE
# ==============================================================================

def load_chronos_bolt(model_id, device):
    from chronos import ChronosBoltPipeline
    return ChronosBoltPipeline.from_pretrained(
        model_id, device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )


def predict_chronos_bolt(pipeline, batches, horizon, device):
    all_samples, all_tgts = [], []
    for batch in tqdm(batches, desc="  Chronos-Bolt", leave=False):
        x, y = batch["x"], batch["y"]
        context = x[:, :, 0]
        samples = pipeline.predict(inputs=context, prediction_length=horizon)
        all_samples.append(samples.to(device=device, dtype=torch.float32, non_blocking=True))
        all_tgts.append(y[:, :, 0].to(device, non_blocking=True))
    all_samples = torch.cat(all_samples, 0)
    all_tgts = torch.cat(all_tgts, 0)
    median = torch.median(all_samples, dim=1).values
    return ForecastResult(median=median, samples=all_samples), all_tgts


def load_chronos2(model_id, device):
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(
        model_id, device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )


def predict_chronos2(pipeline, batches, horizon, device):
    all_quantiles, all_tgts = [], []
    for batch in tqdm(batches, desc="  Chronos2", leave=False):
        x, y = batch["x"], batch["y"]
        context = x.permute(0, 2, 1)
        samples = pipeline.predict(inputs=context, prediction_length=horizon)
        samples = (torch.stack(samples, dim=0).squeeze(1)
                   if isinstance(samples, list) else samples)
        if samples.dim() == 4:
            samples = samples.squeeze(2)
        all_quantiles.append(samples.to(device=device, dtype=torch.float32, non_blocking=True))
        all_tgts.append(y[:, :, 0].to(device, non_blocking=True))
    all_quantiles = torch.cat(all_quantiles, 0)
    all_tgts = torch.cat(all_tgts, 0)
    median = torch.median(all_quantiles, dim=1).values
    return ForecastResult(median=median,
                          quantiles=all_quantiles,
                          quantile_levels=pipeline.quantiles), all_tgts


def load_moirai_module(model_id):
    from uni2ts.model.moirai2 import Moirai2Module
    return Moirai2Module.from_pretrained(model_id)


def build_moirai_forecast(module, horizon, window_size, device):
    from uni2ts.model.moirai2 import Moirai2Forecast
    return Moirai2Forecast(
        module=module, prediction_length=horizon, context_length=window_size,
        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
    ).to(device)


def predict_moirai(model, batches, horizon, device):
    all_quantiles, all_tgts = [], []
    for batch in tqdm(batches, desc="  Moirai", leave=False):
        x, y = batch["x"], batch["y"]
        x_cpu = x[:, :, 0].numpy()
        context_list = [x_cpu[i] for i in range(x_cpu.shape[0])]
        forecast = model.predict(past_target=context_list)
        forecast_t = torch.as_tensor(forecast[:, :, :horizon], dtype=torch.float32, device=device)
        all_quantiles.append(forecast_t)
        all_tgts.append(y[:, :, 0].to(device, non_blocking=True))
    all_quantiles = torch.cat(all_quantiles, 0)
    all_tgts = torch.cat(all_tgts, 0)
    median = all_quantiles[:, MOIRAI2_MEDIAN_IDX, :]
    return ForecastResult(
        median=median, quantiles=all_quantiles,
        quantile_levels=MOIRAI2_QUANTILE_LEVELS,
    ), all_tgts


def load_moirai_1_1_module(model_id):
    from uni2ts.model.moirai import MoiraiModule
    return MoiraiModule.from_pretrained(model_id)


def build_moirai_1_1_forecast(module, horizon, window_size, device):
    from uni2ts.model.moirai import MoiraiForecast
    return MoiraiForecast(
        module=module, prediction_length=horizon, context_length=window_size,
        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        num_samples=MOIRAI_1_1_NUM_SAMPLES, patch_size=MOIRAI_1_1_PATCH_SIZE,
    ).to(device)


def predict_moirai_1_1(model, batches, horizon, device):
    all_samples, all_tgts = [], []
    for batch in tqdm(batches, desc="  Moirai1.1", leave=False):
        x_cpu, y_cpu = batch["x"], batch["y"]
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        bs, ctx_len, _ = x.shape
        past_observed = ~torch.isnan(x)
        past_is_pad = torch.zeros(bs, ctx_len, dtype=torch.bool, device=device)
        samples = model(
            past_target=x,
            past_observed_target=past_observed,
            past_is_pad=past_is_pad,
        )
        if samples.dim() == 4:
            samples = samples.squeeze(-1)
        samples = samples[:, :, :horizon].to(torch.float32)
        all_samples.append(samples)
        all_tgts.append(y[:, :, 0])
    all_samples = torch.cat(all_samples, 0)
    all_tgts = torch.cat(all_tgts, 0)
    median = torch.median(all_samples, dim=1).values
    return ForecastResult(median=median, samples=all_samples), all_tgts


def load_timesfm(model_id, window_size, horizon, batch_size):
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
    model.compile(
        timesfm.ForecastConfig(
            max_context=window_size, max_horizon=horizon,
            normalize_inputs=True, use_continuous_quantile_head=True,
            force_flip_invariance=True, per_core_batch_size=batch_size,
            infer_is_positive=True, fix_quantile_crossing=True,
        )
    )
    return model


def predict_timesfm(model, batches, horizon, device):
    PATCH_SIZE = 32
    all_medians, all_quantiles, all_tgts = [], [], []
    for batch in tqdm(batches, desc="  TimesFM", leave=False):
        x, y = batch["x"], batch["y"]
        bs, ws, _ = x.shape
        x_np = x[:, :, 0].numpy()
        remainder = ws % PATCH_SIZE
        pad_len = (PATCH_SIZE - remainder) % PATCH_SIZE
        if pad_len > 0:
            padding = np.zeros((bs, pad_len), dtype=np.float32)
            padded = np.concatenate([padding, x_np], axis=1)
            mask_pad = np.ones((bs, pad_len), dtype=bool)
            mask_valid = np.zeros((bs, ws), dtype=bool)
            masks_np = np.concatenate([mask_pad, mask_valid], axis=1)
        else:
            padded = x_np
            masks_np = np.zeros((bs, ws), dtype=bool)

        values = [padded[i] for i in range(bs)]
        masks = [masks_np[i] for i in range(bs)]
        point_forecast, quantile_forecast = model.compiled_decode(horizon, values, masks)
        pf = torch.as_tensor(point_forecast[:, :horizon], dtype=torch.float32, device=device)
        qf = torch.as_tensor(
            quantile_forecast[:, :horizon, 1:], dtype=torch.float32, device=device
        ).permute(0, 2, 1)
        all_medians.append(pf)
        all_quantiles.append(qf)
        all_tgts.append(y[:, :, 0].to(device, non_blocking=True))
    all_medians = torch.cat(all_medians, 0)
    all_quantiles = torch.cat(all_quantiles, 0)
    all_tgts = torch.cat(all_tgts, 0)
    return ForecastResult(
        median=all_medians, quantiles=all_quantiles,
        quantile_levels=TIMESFM_QUANTILE_LEVELS,
    ), all_tgts


def load_patchtst_fm(model_id, device):
    if PatchTSTFMForPrediction is None:
        raise RuntimeError("tsfm_public is not installed.")
    model = PatchTSTFMForPrediction.from_pretrained(model_id, device_map=device)
    model.eval()
    return model


def predict_patchtst_fm(model, batches, horizon, device):
    all_quantiles, all_tgts = [], []
    for batch in tqdm(batches, desc="  PatchTST-FM", leave=False):
        x_cpu, y_cpu = batch["x"], batch["y"]
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        B = x.shape[0]
        past_values = x.squeeze(-1)
        output = model(inputs=past_values, prediction_length=horizon)
        raw = output[0]
        if raw.dim() == 4:
            qf = raw[:, :, :horizon, 0]
            pred_len = qf.shape[2]
            if pred_len < horizon:
                pad = qf[:, :, -1:].expand(B, qf.shape[1], horizon - pred_len)
                qf = torch.cat([qf, pad], dim=2)
            all_quantiles.append(qf.to(torch.float32))
        elif raw.dim() == 3:
            preds = raw[:, :, :horizon].unsqueeze(1) if raw.shape[1] == 1 else raw[:, :, :horizon]
            all_quantiles.append(preds.to(torch.float32))
        elif raw.dim() == 2:
            preds = raw[:, :horizon].unsqueeze(1)
            all_quantiles.append(preds.to(torch.float32))
        else:
            raise ValueError(f"Unexpected output shape {raw.shape}")
        all_tgts.append(y[:, :, 0])

    all_quantiles = torch.cat(all_quantiles, 0)
    all_tgts = torch.cat(all_tgts, 0)
    if all_quantiles.shape[1] == len(PATCHTST_FM_QUANTILE_LEVELS):
        median = all_quantiles[:, PATCHTST_FM_MEDIAN_QUANTILE_IDX, :]
        return ForecastResult(
            median=median, quantiles=all_quantiles,
            quantile_levels=PATCHTST_FM_QUANTILE_LEVELS,
        ), all_tgts
    return ForecastResult(median=all_quantiles.squeeze(1)), all_tgts


def load_sundial(model_id, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.to(device).eval()
    return model


def predict_sundial(model, batches, horizon, device):
    all_samples, all_tgts = [], []
    for batch in tqdm(batches, desc="  Sundial", leave=False):
        x, y = batch["x"], batch["y"]
        seqs = x[:, :, 0].to(device, non_blocking=True)
        samples = model.generate(
            seqs, max_new_tokens=horizon, num_samples=SUNDIAL_NUM_SAMPLES,
        )
        samples = samples[:, :, :horizon].to(torch.float32)
        all_samples.append(samples)
        all_tgts.append(y[:, :, 0].to(device, non_blocking=True))
    all_samples = torch.cat(all_samples, 0)
    all_tgts = torch.cat(all_tgts, 0)
    median = torch.median(all_samples, dim=1).values
    return ForecastResult(median=median, samples=all_samples), all_tgts


def load_timemoe(model_id, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.to(device).eval()
    return model


def predict_timemoe(model, batches, horizon, device):
    all_preds, all_tgts = [], []
    for batch in tqdm(batches, desc="  TimeMoE", leave=False):
        x, y = batch["x"], batch["y"]
        seqs = x[:, :, 0].to(device, non_blocking=True)
        mean = seqs.mean(dim=-1, keepdim=True)
        std = seqs.std(dim=-1, keepdim=True)
        normed = (seqs - mean) / (std + 1e-8)
        out = model.generate(normed, max_new_tokens=horizon)
        preds = out[:, -horizon:].to(torch.float32) * std + mean
        all_preds.append(preds)
        all_tgts.append(y[:, :, 0].to(device, non_blocking=True))
    all_preds = torch.cat(all_preds, 0)
    all_tgts = torch.cat(all_tgts, 0)
    return ForecastResult(median=all_preds), all_tgts


def predict_context_parroting(batches, horizon, device):
    all_preds, all_tgts = [], []
    for batch in tqdm(batches, desc="  Parroting", leave=False):
        x_cpu, y_cpu = batch["x"], batch["y"]
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        ws = x.shape[1]
        if ws >= horizon:
            pred = x[:, -horizon:, 0]
        else:
            repeats = (horizon // ws) + 1
            pred = x[:, :, 0].repeat(1, repeats)[:, -horizon:]
        all_preds.append(pred)
        all_tgts.append(y[:, :, 0])
    return ForecastResult(median=torch.cat(all_preds, 0)), torch.cat(all_tgts, 0)


# ==============================================================================
#  PLOTTING (per-(model, window) cells, unchanged from v4)
# ==============================================================================

def plot_sample_predictions(all_inputs, all_predictions, all_targets,
                            model_short, window_size, horizon, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    num_plots = min(4, all_predictions.shape[0])
    if num_plots == 0:
        return
    indices = np.linspace(0, all_predictions.shape[0] - 1, num_plots, dtype=int)
    h = all_predictions.shape[1]
    ws = all_inputs.shape[1]

    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 3 * num_plots))
    if num_plots == 1:
        axes = [axes]
    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.plot(np.arange(ws), all_inputs[idx].squeeze().numpy(), label="Input", color="blue", alpha=0.7)
        ax.plot(np.arange(ws, ws + h), all_targets[idx].squeeze().numpy(), label="Target", marker="o", markersize=3, color="green")
        ax.plot(np.arange(ws, ws + h), all_predictions[idx].squeeze().numpy(), label="Prediction", marker="x", markersize=3, color="red")
        ax.axvline(x=ws - 0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Sample {idx}")
        ax.legend(fontsize=8)
    plt.suptitle(f"{model_short} -- w={window_size}, h={horizon}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"samples_{model_short}_w{window_size}_h{horizon}.png"), dpi=120)
    plt.close()


def find_best_worst_samples(predictions, targets, n=N_BEST_WORST):
    per_sample_mae = (predictions - targets).abs().mean(dim=(1, 2)).cpu().numpy()
    n = min(n, len(per_sample_mae))
    sorted_idx = np.argsort(per_sample_mae)
    return sorted_idx[:n], sorted_idx[-n:][::-1].copy(), per_sample_mae


def plot_best_worst_samples(all_inputs, predictions, targets,
                            best_indices, worst_indices, per_sample_mae,
                            model_short, window_size, horizon, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    h = predictions.shape[1]
    ws = all_inputs.shape[1]
    for kind, indices in [("best", best_indices), ("worst", worst_indices)]:
        n = len(indices)
        if n == 0:
            continue
        fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
        if n == 1:
            axes = [axes]
        for i, idx in enumerate(indices):
            ax = axes[i]
            ax.plot(np.arange(ws), all_inputs[idx].squeeze().numpy(), label="Input", color="blue", alpha=0.7)
            ax.plot(np.arange(ws, ws + h), targets[idx].squeeze().numpy(), label="Target", marker="o", markersize=3, color="green")
            ax.plot(np.arange(ws, ws + h), predictions[idx].squeeze().numpy(), label="Prediction", marker="x", markersize=3, color="red")
            ax.axvline(x=ws - 0.5, color="gray", linestyle="--", alpha=0.5)
            ax.set_title(f"{kind.capitalize()} #{i+1}  (sample {idx}, MAE={per_sample_mae[idx]:.6f})")
            ax.legend(fontsize=8)
        plt.suptitle(f"{model_short} -- w={window_size}, h={horizon} -- {kind.upper()} {n}", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{kind}_samples_{model_short}_w{window_size}_h{horizon}.png"), dpi=120)
        plt.close()

    manifest = {
        "best": [{"index": int(i), "mae": float(per_sample_mae[i])} for i in best_indices],
        "worst": [{"index": int(i), "mae": float(per_sample_mae[i])} for i in worst_indices],
    }
    with open(os.path.join(save_dir, "best_worst_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def plot_error_distributions(per_sample_metrics, model_short, window_size, horizon, save_dir,
                             naive_baseline: Optional[Dict[str, float]] = None,
                             percentile_clip=99.0):
    os.makedirs(save_dir, exist_ok=True)
    metric_names = ["mae", "mse", "rmse"]
    colors = {"mae": "#2196F3", "mse": "#FF9800", "rmse": "#4CAF50"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, name in zip(axes, metric_names):
        vals = per_sample_metrics.get(name)
        if vals is None:
            ax.set_title(f"{name.upper()} (no data)"); continue
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            ax.set_title(f"{name.upper()} (no data)"); continue
        clip_val = np.percentile(vals, percentile_clip)
        clipped = vals[vals <= clip_val]
        n_bins = min(80, max(20, len(clipped) // 10))
        ax.hist(clipped, bins=n_bins, density=True, alpha=0.55,
                color=colors[name], edgecolor="white", linewidth=0.5)
        if len(clipped) > 1 and np.std(clipped) > 0:
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(clipped, bw_method="silverman")
                xg = np.linspace(clipped.min(), clipped.max(), 300)
                ax.plot(xg, kde(xg), color=colors[name], linewidth=2, label="KDE")
            except Exception:
                pass
        ax.axvline(vals.mean(), color="red", linestyle="--", linewidth=1.2, label=f"Mean={vals.mean():.4f}")
        ax.axvline(np.median(vals), color="purple", linestyle=":", linewidth=1.2, label=f"Median={np.median(vals):.4f}")
        if naive_baseline is not None and name in naive_baseline:
            bv = naive_baseline[name]
            if bv is not None and not np.isnan(bv) and bv <= clip_val:
                ax.axvline(bv, color="black", linestyle="-.", linewidth=1.5, alpha=0.8, label=f"S.Naive={bv:.4f}")
        ax.set_xlim(left=0, right=clip_val)
        ax.set_title(f"{name.upper()}  (n={len(vals)})", fontsize=11)
        ax.set_xlabel(name.upper()); ax.set_ylabel("Density")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    fig.suptitle(f"{model_short} -- w={window_size}, h={horizon}  |  Error Distributions",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, f"error_dist_{model_short}_w{window_size}_h{horizon}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_ablation_summary(results_df, model_names_in_run, window_sizes,
                          dataset_display, term, run_dir,
                          naive_baseline: Optional[Dict[str, float]] = None):
    n_metrics = len(PLOT_METRICS)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 6 * n_rows))
    axes = np.array(axes).flatten() if n_metrics > 1 else np.array([axes])
    colors = plt.cm.tab10.colors

    for col, metric_name in enumerate(PLOT_METRICS):
        ax = axes[col]
        for i, (model_id, model_short) in enumerate(model_names_in_run):
            mdf = results_df[
                (results_df["model"] == model_id)
                & (results_df["dataset_display"] == dataset_display)
                & (results_df["term"] == term)
            ].sort_values("window_size")
            if mdf.empty or metric_name not in mdf.columns:
                continue
            vals = mdf[metric_name].values
            if np.all(np.isnan(vals)):
                continue
            ax.plot(mdf["window_size"], vals, marker="o", linewidth=2,
                    label=model_short, color=colors[i % len(colors)])
        if naive_baseline is not None and metric_name in naive_baseline:
            bv = naive_baseline[metric_name]
            if bv is not None and not np.isnan(bv):
                ax.axhline(y=bv, color="black", linestyle="--", linewidth=1.8,
                           alpha=0.7, label=f"Seasonal Naive ({bv:.4f})", zorder=0)
        ax.set_xlabel("Input Window Size"); ax.set_ylabel(metric_name.upper())
        ax.set_title(metric_name.upper(), fontsize=14, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks(window_sizes)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.3)

    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    horizons = results_df.loc[
        (results_df["dataset_display"] == dataset_display)
        & (results_df["term"] == term), "horizon"
    ].unique()
    horizon_str = str(int(horizons[0])) if len(horizons) == 1 else "/".join(str(int(h)) for h in sorted(horizons))
    fig.suptitle(f"Window Size Ablation -- {dataset_display}  (term={term}, horizon={horizon_str})",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(run_dir, f"summary_{dataset_display}_t{term}_h{horizon_str}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ==============================================================================
#  PREDICTOR LOADING + INFERENCE
# ==============================================================================

def find_latest_predictor_run(root: str) -> str:
    """Most recent run directory under `root` that has best_model.pt+best_config."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Predictor cache root not found: {root}")
    candidates = []
    for name in os.listdir(root):
        rd = os.path.join(root, name)
        if not os.path.isdir(rd):
            continue
        if (os.path.isfile(os.path.join(rd, "best_model.pt"))
                and os.path.isfile(os.path.join(rd, "best_config.json"))):
            candidates.append((os.path.getmtime(rd), rd))
    if not candidates:
        raise FileNotFoundError(
            f"No run under {root} with best_model.pt + best_config.json.")
    candidates.sort(reverse=True)
    return candidates[0][1]


def load_predictor(predictor_dir: str, device: str
                   ) -> Tuple[PatchTSTContextLength, dict]:
    with open(os.path.join(predictor_dir, "best_config.json")) as f:
        cfg = json.load(f)
    model = PatchTSTContextLength(
        context_length      = cfg["context_length"],
        patch_length        = cfg["patch_length"],
        d_model             = cfg["d_model"],
        num_hidden_layers   = cfg["num_hidden_layers"],
        num_attention_heads = cfg["num_attention_heads"],
        dropout             = cfg["dropout"],
        mask_ratio          = cfg["mask_ratio"],
        n_windows           = cfg["n_windows"],
        n_horizons          = cfg["n_horizons"],
    )
    state = torch.load(
        os.path.join(predictor_dir, "best_model.pt"),
        map_location="cpu",
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg


def _closest_horizon_idx(horizon: int, horizon_grid: List[int]) -> int:
    arr = np.asarray(horizon_grid)
    return int(np.argmin(np.abs(arr - int(horizon))))


def _prepare_predictor_inputs(
    contexts: List[np.ndarray], context_length: int
) -> np.ndarray:
    """Left-pad-with-zero (matching build_context_length_dataset.py) + standardize.

    Each gifteval test instance keeps its suffix (the most recent samples)
    truncated/padded to exactly `context_length`. Standardization is
    per-instance to (mean=0, std=1), matching what the synthetic builder
    does (ctx = (series - mu)/sigma over the observed pool).
    """
    n = len(contexts)
    out = np.zeros((n, context_length), dtype=np.float32)
    for i, ctx in enumerate(contexts):
        ctx = np.nan_to_num(np.asarray(ctx, dtype=np.float32), nan=0.0)
        if len(ctx) >= context_length:
            x = ctx[-context_length:]
        else:
            x = np.concatenate([
                np.zeros(context_length - len(ctx), dtype=np.float32),
                ctx,
            ])
        mu = float(x.mean())
        sigma = float(x.std())
        out[i] = (x - mu) / (sigma + 1e-8)
    return out


def predict_curves_for_dataset(
    predictor: PatchTSTContextLength,
    cache: GiftEvalCache,
    context_length: int,
    horizon_idx: int,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Per-instance predicted z-scored error curve.

    Returns
    -------
    np.ndarray of shape (n_instances, n_windows). Values are already z-scored
    (the predictor is trained to output z-scored curves directly).
    """
    x_np = _prepare_predictor_inputs(cache.contexts, context_length)
    x = torch.from_numpy(x_np).unsqueeze(-1)  # (N, L, 1)

    preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            xb = x[start:stop].to(device, non_blocking=True)
            h_idx = torch.full(
                (xb.shape[0],), horizon_idx, dtype=torch.long, device=device,
            )
            curve_pred, _, _, _ = predictor(xb, horizon_idx=h_idx)
            preds.append(curve_pred.float().cpu().numpy())
    return np.concatenate(preds, axis=0)


# ==============================================================================
#  COMPARISON PLOT (real vs predicted, z-scored)
# ==============================================================================

def _zscore_curve(curve: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score along the windows axis, ignoring NaNs.

    Returns the z-scored array (with NaNs preserved) and a boolean valid-mask.
    """
    valid = ~np.isnan(curve)
    if valid.sum() < 2:
        return np.full_like(curve, np.nan, dtype=np.float64), valid
    vals = curve[valid].astype(np.float64)
    mu = vals.mean()
    sd = vals.std()
    if sd < 1e-12:
        return np.full_like(curve, 0.0, dtype=np.float64), valid
    out = np.full_like(curve, np.nan, dtype=np.float64)
    out[valid] = (vals - mu) / sd
    return out, valid


def plot_real_vs_predicted_curve(
    window_grid: List[int],
    real_curve: np.ndarray,                  # (n_windows,) raw metric values
    predicted_curves: np.ndarray,            # (n_inst, n_windows) z-scored
    model_short: str,
    dataset_display: str,
    term: str,
    horizon_real: int,
    horizon_pred: int,
    curve_metric: str,
    save_dir: str,
) -> str:
    """Overlay z-scored real curve and z-scored predicted curve (mean +/- std).

    The predictor is trained to output z-scored curves (shape only), so we
    z-score the real curve the same way before overlaying. Argmin is invariant
    to shift/scale so this is the natural shape comparison.
    """
    os.makedirs(save_dir, exist_ok=True)
    ws = np.asarray(window_grid)

    real_z, real_valid = _zscore_curve(np.asarray(real_curve, dtype=np.float64))
    pred_mean = predicted_curves.mean(axis=0)
    pred_std = predicted_curves.std(axis=0)
    n_inst = predicted_curves.shape[0]

    fig, ax = plt.subplots(figsize=(11, 6.5))

    if real_valid.any():
        ax.plot(ws[real_valid], real_z[real_valid],
                marker="o", linewidth=2.0, color="#2ca02c",
                label=f"Real (z-scored {curve_metric.upper()})")
        argmin_real = int(np.nanargmin(real_z))
        ax.axvline(ws[argmin_real], color="#2ca02c",
                   linestyle=":", alpha=0.6,
                   label=f"Real argmin = {ws[argmin_real]}")
    else:
        ax.text(0.5, 0.6, "no valid real-curve points",
                ha="center", va="center", transform=ax.transAxes, color="#999")

    ax.plot(ws, pred_mean,
            marker="x", linewidth=2.0, color="#d62728",
            label=f"Predicted (mean over {n_inst} inst.)")
    ax.fill_between(ws, pred_mean - pred_std, pred_mean + pred_std,
                    color="#d62728", alpha=0.18,
                    label="Predicted $\\pm$1$\\sigma$")
    argmin_pred = int(np.argmin(pred_mean))
    ax.axvline(ws[argmin_pred], color="#d62728",
               linestyle=":", alpha=0.6,
               label=f"Predicted argmin = {ws[argmin_pred]}")

    ax.set_xscale("log", base=2)
    ax.set_xticks(ws)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Input Window Size")
    ax.set_ylabel(f"Z-scored {curve_metric.upper()} along windows")
    ax.set_title(
        f"{dataset_display}  (term={term}, h_real={horizon_real}, "
        f"h_pred={horizon_pred})  --  {model_short}",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3); ax.axhline(0.0, color="gray", alpha=0.4, lw=0.8)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    path = os.path.join(
        save_dir,
        f"compare_{dataset_display}_t{term}_{model_short}.png",
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ==============================================================================
#  RUN
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Window-size ablation on GiftEval with predictor overlay.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--models", type=str, nargs="+", default=None)
    p.add_argument("--datasets", type=str, nargs="+", default=None)
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    p.add_argument("--no-plots", action="store_true",
                   help="Skip per-(model,window) plots (summary + comparison still produced).")
    p.add_argument("--cache-root", type=str, default=CACHE_ROOT)
    p.add_argument("--predictor-cache-root", type=str, default=CACHE_ROOT_PREDICTOR,
                   help="Root holding context-length-predictor runs. Latest run is auto-picked.")
    p.add_argument("--predictor-dir", type=str, default=None,
                   help="Override: use this specific predictor run directory.")
    p.add_argument("--predictor-batch-size", type=int, default=64,
                   help="Batch size used when running the predictor on test-instance contexts.")
    return p.parse_args()


def main():
    args = parse_args()

    global CACHE_ROOT
    CACHE_ROOT = args.cache_root

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    # ---- Load predictor + derive window grid + horizon grid ------------------
    predictor_dir = args.predictor_dir or find_latest_predictor_run(
        args.predictor_cache_root)
    print(Fore.CYAN + f"Predictor run: {predictor_dir}" + Fore.RESET)
    predictor, pred_cfg = load_predictor(predictor_dir, device)
    window_grid: List[int] = list(pred_cfg["window_grid"])
    horizon_grid: List[int] = list(pred_cfg["horizon_grid"])
    pred_ctx_len: int = int(pred_cfg["context_length"])
    pred_curve_metric: str = pred_cfg.get("curve_metric", "mae")
    print(Fore.CYAN
          + f"  window_grid ({len(window_grid)}): {window_grid}\n"
          + f"  horizon_grid: {horizon_grid}\n"
          + f"  context_length={pred_ctx_len}  curve_metric={pred_curve_metric}\n"
          + f"  label_model={pred_cfg.get('label_model')}"
          + Fore.RESET)

    window_sizes = sorted(set(window_grid))
    print(Fore.CYAN + f"Device: {device}  |  ablation windows: {window_sizes}"
          + Fore.RESET)

    models = [m for m in MODELS if (args.models is None or m[2] in args.models)]
    datasets = [d for d in DATASETS if (args.datasets is None or d[2] in args.datasets)]

    # Deterministic per-family layout: run dir is the predictor's family
    # (= basename of predictor_dir), so re-running overwrites the same place.
    family_tag = os.path.basename(os.path.normpath(predictor_dir))
    run_dir = os.path.join(CACHE_ROOT, family_tag)
    os.makedirs(run_dir, exist_ok=True)

    ge_cache: Dict[Tuple[str, str], GiftEvalCache] = {}
    naive_baseline_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
    all_results: List[dict] = []

    # ----- Model -> Dataset -> Window ----------------------------------------
    for model_id, model_family, model_short in models:
        print(Fore.CYAN + "\n" + "=" * 78 + Fore.RESET)
        print(Fore.CYAN + f"  MODEL: {model_id}  ({model_family})" + Fore.RESET)
        print(Fore.CYAN + "=" * 78 + Fore.RESET)

        pipeline = None
        moirai_module = None
        moirai_1_1_module = None
        patchtst_fm_model = None
        sundial_model = None
        timemoe_model = None

        if model_family == "chronos_bolt":
            pipeline = load_chronos_bolt(model_id, device)
        elif model_family == "chronos2":
            pipeline = load_chronos2(model_id, device)
        elif model_family == "moirai":
            moirai_module = load_moirai_module(model_id)
        elif model_family == "moirai_1_1":
            moirai_1_1_module = load_moirai_1_1_module(model_id)
        elif model_family == "patchtst_fm":
            patchtst_fm_model = load_patchtst_fm(model_id, device)
        elif model_family == "sundial":
            sundial_model = load_sundial(model_id, device)
        elif model_family == "timemoe":
            timemoe_model = load_timemoe(model_id, device)

        for ge_name, term, dataset_display, to_univariate in datasets:
            ds_key = (ge_name, term)
            if ds_key not in ge_cache:
                print(Fore.CYAN + f"\n  Loading GiftEval: {ge_name}  term={term}" + Fore.RESET)
                ge_dataset = GiftEvalDataset(name=ge_name, term=term, to_univariate=to_univariate)
                cache = GiftEvalCache(ge_dataset, dataset_display)
                ge_cache[ds_key] = cache
                print(Fore.CYAN
                      + f"    freq={cache.freq}  horizon={cache.horizon}  "
                      + f"n_test={cache.n_total}  "
                      + f"ctx_range=[{cache.min_context},{cache.max_context}]  "
                      + f"naive_mae_train={cache.naive_seasonal_mae_train:.6f}"
                      + Fore.RESET)
                naive_baseline_cache[(dataset_display, term)] = load_or_compute_naive_baseline(cache, term)
            cache = ge_cache[ds_key]
            naive_bl = naive_baseline_cache[(dataset_display, term)]
            horizon = cache.horizon

            for window_size in window_sizes:
                tag = f"{model_short} | {dataset_display} | t={term} | h={horizon} | w={window_size}"

                if _result_cached(dataset_display, model_short, term, window_size):
                    cached = _load_cached_result(dataset_display, model_short, term, window_size)
                    print(Fore.WHITE + f"  CACHED  {tag}  -> MAE={cached['mae']:.6f}" + Fore.RESET)
                    all_results.append({
                        "model": model_id, "model_short": model_short, "model_family": model_family,
                        "dataset": ge_name, "dataset_display": dataset_display, "term": term,
                        "horizon": horizon, "window_size": window_size, **cached,
                    })
                    continue

                if not cache.can_serve(window_size):
                    print(Fore.RED + f"  SKIP    {tag}  (max_context={cache.max_context} < ws)"
                          + Fore.RESET)
                    continue

                if model_family == "sundial" and window_size > SUNDIAL_MAX_CONTEXT:
                    print(Fore.RED + f"  SKIP    {tag}  (Sundial max context={SUNDIAL_MAX_CONTEXT} < ws)"
                          + Fore.RESET)
                    continue

                if model_family == "timemoe" and window_size + horizon > TIMEMOE_MAX_TOTAL:
                    print(Fore.RED + f"  SKIP    {tag}  (TimeMoE ws+h={window_size + horizon} > {TIMEMOE_MAX_TOTAL})"
                          + Fore.RESET)
                    continue

                print(Fore.YELLOW + f"\n  > {tag}" + Fore.RESET)
                t_start = time.perf_counter()

                try:
                    batches, all_x_cpu, all_y_cpu, _ = cache.build_batches(
                        window_size, args.batch_size, device, pin_memory=True
                    )
                except RuntimeError as exc:
                    print(Fore.RED + f"    SKIP: {exc}" + Fore.RESET); continue

                n_valid = all_x_cpu.shape[0]
                print(f"    Samples: {n_valid}  Batches: {len(batches)}  W={window_size}  H={horizon}")

                if model_family == "chronos_bolt":
                    fr, tgts = predict_chronos_bolt(pipeline, batches, horizon, device)
                elif model_family == "chronos2":
                    fr, tgts = predict_chronos2(pipeline, batches, horizon, device)
                elif model_family == "moirai":
                    moirai_model = build_moirai_forecast(moirai_module, horizon, window_size, device)
                    fr, tgts = predict_moirai(moirai_model, batches, horizon, device)
                    del moirai_model
                    if device == "cuda": torch.cuda.empty_cache()
                elif model_family == "moirai_1_1":
                    moirai_1_1_model = build_moirai_1_1_forecast(moirai_1_1_module, horizon, window_size, device)
                    fr, tgts = predict_moirai_1_1(moirai_1_1_model, batches, horizon, device)
                    del moirai_1_1_model
                    if device == "cuda": torch.cuda.empty_cache()
                elif model_family == "timesfm":
                    tfm = load_timesfm(model_id, window_size, horizon, args.batch_size)
                    fr, tgts = predict_timesfm(tfm, batches, horizon, device)
                    del tfm
                    if device == "cuda": torch.cuda.empty_cache()
                elif model_family == "context_parroting":
                    fr, tgts = predict_context_parroting(batches, horizon, device)
                elif model_family == "patchtst_fm":
                    fr, tgts = predict_patchtst_fm(patchtst_fm_model, batches, horizon, device)
                elif model_family == "sundial":
                    fr, tgts = predict_sundial(sundial_model, batches, horizon, device)
                elif model_family == "timemoe":
                    fr, tgts = predict_timemoe(timemoe_model, batches, horizon, device)
                else:
                    raise ValueError(f"Unknown model family: {model_family}")

                elapsed = time.perf_counter() - t_start

                metrics = compute_all_metrics(fr, tgts, cache.naive_seasonal_mae_train)
                metrics["elapsed_seconds"] = round(elapsed, 3)
                metrics["horizon"] = horizon

                for k, v in metrics.items():
                    if isinstance(v, float):
                        print(Fore.YELLOW + f"    {k}: {v:.6f}" + Fore.RESET)
                print(Fore.MAGENTA + f"    TIME  {elapsed:.1f}s" + Fore.RESET)

                per_sample = compute_per_sample_metrics(fr, tgts, cache.naive_seasonal_mae_train)
                _save_per_sample_metrics(dataset_display, model_short, term, window_size, per_sample)

                if not args.no_plots:
                    pred_3d = fr.median.detach().cpu().unsqueeze(-1)
                    tgt_3d = tgts.detach().cpu().unsqueeze(-1)
                    cell_dir = _cache_dir(dataset_display, model_short, term, window_size)
                    plot_sample_predictions(all_x_cpu, pred_3d, tgt_3d,
                                            model_short, window_size, horizon, cell_dir)
                    best_idx, worst_idx, psm = find_best_worst_samples(pred_3d, tgt_3d, n=N_BEST_WORST)
                    plot_best_worst_samples(all_x_cpu, pred_3d, tgt_3d,
                                            best_idx, worst_idx, psm,
                                            model_short, window_size, horizon, cell_dir)
                    plot_error_distributions(per_sample, model_short, window_size, horizon,
                                             cell_dir, naive_baseline=naive_bl)

                _save_result(dataset_display, model_short, term, window_size, metrics)

                all_results.append({
                    "model": model_id, "model_short": model_short, "model_family": model_family,
                    "dataset": ge_name, "dataset_display": dataset_display, "term": term,
                    "horizon": horizon, "window_size": window_size, **metrics,
                })

                del fr, tgts, batches, all_x_cpu, all_y_cpu, per_sample
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        del pipeline, moirai_module, moirai_1_1_module, patchtst_fm_model, sundial_model, timemoe_model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # ==========================================================================
    #  AGGREGATE + SAVE + COMPARISON PLOTS
    # ==========================================================================
    if not all_results:
        print(Fore.RED + "No results produced." + Fore.RESET); return

    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(run_dir, "results.csv")
    results_df.to_csv(csv_path, index=False)
    print(Fore.GREEN + f"\n  Results CSV: {csv_path}" + Fore.RESET)

    model_names_in_run = [(m[0], m[2]) for m in models]

    # ---- Comparison plot per (model, dataset, term) -------------------------
    compare_dir = os.path.join(run_dir, "compare_real_vs_predicted")
    os.makedirs(compare_dir, exist_ok=True)
    compare_records: List[dict] = []

    for model_id, model_family, model_short in models:
        seen_ds_term = set()
        for ge_name, term, dataset_display, _ in datasets:
            key = (dataset_display, term)
            if key in seen_ds_term:
                continue
            seen_ds_term.add(key)

            mdf = results_df[
                (results_df["model"] == model_id)
                & (results_df["dataset_display"] == dataset_display)
                & (results_df["term"] == term)
            ].sort_values("window_size")
            if mdf.empty or "mase" not in mdf.columns:
                continue

            # Build real curve aligned to window_grid (NaN where window absent).
            ws_to_val = dict(zip(mdf["window_size"].values,
                                 mdf["mase"].values))
            real_curve = np.array(
                [ws_to_val.get(w, np.nan) for w in window_grid],
                dtype=np.float64,
            )
            if np.sum(~np.isnan(real_curve)) < 2:
                print(Fore.YELLOW
                      + f"  Compare SKIP  {dataset_display} t={term} "
                      + f"({model_short}): <2 real-curve points." + Fore.RESET)
                continue

            cache = ge_cache.get((ge_name, term))
            if cache is None:
                continue
            horizon_real = cache.horizon
            h_idx = _closest_horizon_idx(horizon_real, horizon_grid)
            horizon_pred = horizon_grid[h_idx]

            print(Fore.CYAN
                  + f"  Predictor inference: {dataset_display} t={term} "
                  + f"({model_short})  h_real={horizon_real} -> h_pred={horizon_pred}  "
                  + f"n_inst={cache.n_total}"
                  + Fore.RESET)

            predicted_curves = predict_curves_for_dataset(
                predictor, cache, pred_ctx_len, h_idx, device,
                batch_size=args.predictor_batch_size,
            )

            plot_path = plot_real_vs_predicted_curve(
                window_grid, real_curve, predicted_curves,
                model_short, dataset_display, term,
                horizon_real, horizon_pred, "mase", compare_dir,
            )

            # Per-(model,dataset,term) numerical artifact for downstream eval.
            real_z, _ = _zscore_curve(real_curve)
            pred_mean = predicted_curves.mean(axis=0)
            pred_std = predicted_curves.std(axis=0)
            argmin_real = (
                int(np.nanargmin(real_curve)) if np.any(~np.isnan(real_curve))
                else None
            )
            argmin_pred = int(np.argmin(pred_mean))
            np.savez_compressed(
                os.path.join(
                    compare_dir,
                    f"compare_{dataset_display}_t{term}_{model_short}.npz",
                ),
                window_grid=np.asarray(window_grid),
                real_curve=real_curve,
                real_curve_zscored=real_z,
                predicted_curves=predicted_curves,
                predicted_mean=pred_mean,
                predicted_std=pred_std,
            )
            compare_records.append({
                "model": model_id, "model_short": model_short,
                "dataset_display": dataset_display, "term": term,
                "horizon_real": int(horizon_real),
                "horizon_pred_idx": int(h_idx),
                "horizon_pred": int(horizon_pred),
                "n_instances": int(cache.n_total),
                "best_window_real": (int(window_grid[argmin_real])
                                     if argmin_real is not None else None),
                "best_window_pred": int(window_grid[argmin_pred]),
                "plot_path": plot_path,
            })

    if compare_records:
        cdf = pd.DataFrame(compare_records)
        cdf.to_csv(os.path.join(compare_dir, "compare_summary.csv"), index=False)
        print(Fore.GREEN + f"  Comparison summary: "
              + f"{os.path.join(compare_dir, 'compare_summary.csv')}"
              + Fore.RESET)

    # ---- Standard summary plot per (dataset, term) --------------------------
    seen = set()
    for ge_name, term, dataset_display, _ in datasets:
        key = (dataset_display, term)
        if key in seen: continue
        seen.add(key)
        plot_ablation_summary(
            results_df, model_names_in_run, window_sizes,
            dataset_display, term, run_dir,
            naive_baseline=naive_baseline_cache.get(key),
        )

    naive_path = os.path.join(run_dir, "naive_baselines.json")
    with open(naive_path, "w") as f:
        json.dump({f"{k[0]}/t{k[1]}": v for k, v in naive_baseline_cache.items()}, f, indent=2)
    print(Fore.GREEN + f"  Naive baselines: {naive_path}" + Fore.RESET)

    # ---- Persist a small marker tying this run to the predictor checkpoint --
    with open(os.path.join(run_dir, "predictor_meta.json"), "w") as f:
        json.dump({
            "predictor_dir":  predictor_dir,
            "window_grid":    window_grid,
            "horizon_grid":   horizon_grid,
            "context_length": pred_ctx_len,
            "curve_metric":   pred_curve_metric,
            "label_model":    pred_cfg.get("label_model"),
        }, f, indent=2)

    print(Fore.GREEN + "Done." + Fore.RESET)


if __name__ == "__main__":
    main()
