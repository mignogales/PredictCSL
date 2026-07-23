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
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

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

# Leaderboard-faithful (gluonts) MASE primitives: per-instance seasonal error +
# gluonts seasonality map. Used to compute the `mase_gluonts` metric alongside the
# project's own `mase` (see compute_all_metrics). `gluonts_leaderboard_mase` runs
# gluonts' OWN evaluate_forecasts to produce the `mase_gluonts_real` column.
from experiments.gifteval_mase import (
    get_seasonality, per_instance_seasonal_errors, gluonts_leaderboard_mase,
)

try:
    from tsfm_public import PatchTSTFMForPrediction
except ImportError:
    PatchTSTFMForPrediction = None

from experiments.predict_context_length import (
    PatchTSTContextLength, build_predictor)
from experiments import models_config
from experiments import datasets_config


# ==============================================================================
#  STATIC CONFIG
# ==============================================================================

# Run set (model_id, family, display) from the single config in
# experiments.models_config. To add/drop a model here, flip its `run` flag there.
MODELS = models_config.models_to_run()

# Run set (ge_name, term, display, to_univariate) from the single config in
# experiments.datasets_config. To add/drop a dataset cell here, flip its `run`
# flag there. Every downstream stage (period eval, robust timing, MASE variants,
# embedding saturation) imports DATASETS from this module, so the config reaches
# the whole pipeline.
DATASETS = datasets_config.datasets_to_run()

MOIRAI2_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MOIRAI2_MEDIAN_IDX = 4
MOIRAI_1_1_NUM_SAMPLES = 100
MOIRAI_1_1_PATCH_SIZE = 32
TIMESFM_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TIMESFM_FULL_CONTEXT = 15360
FULL_NATIVE_WINDOW = "full_native"
PATCHTST_FM_QUANTILE_LEVELS = [i / 100.0 for i in range(1, 100)]
PATCHTST_FM_MEDIAN_QUANTILE_IDX = 49
SUNDIAL_NUM_SAMPLES = 20
SUNDIAL_MAX_CONTEXT = 2880
TIMEMOE_MAX_TOTAL   = 4096   # context + horizon must not exceed this
TOTO_NUM_QUANTILES       = 9     # Toto 2.0 quantile head: [0.1..0.9]
TOTO_MEDIAN_QUANTILE_IDX = 4     # middle of the 9 quantiles (0.5)
TOTO_MAX_CONTEXT         = 4096  # skip wider windows (mirrors stage-1 label cap)
TOTO_PATCH_SIZE          = 32    # forecast() context length must be a multiple of this
FLOWSTATE_REVISION            = "r1.1"  # GIFT-Eval checkpoint (mirrors stage-1)
FLOWSTATE_NUM_QUANTILES       = 9
FLOWSTATE_MEDIAN_QUANTILE_IDX = 4       # middle of the 9 output quantiles (0.5)
FLOWSTATE_SCALE_FACTOR        = 1.0     # fallback only (unknown cadence)
FLOWSTATE_MAX_CONTEXT         = 4096    # skip wider windows (mirrors stage-1 label cap)
TIREX_NUM_QUANTILES           = 9       # TiRex quantile head: [0.1..0.9]
TIREX_MEDIAN_QUANTILE_IDX     = 4       # middle of the 9 quantiles (0.5)
TIREX_MAX_CONTEXT             = 8192    # TiRex2 pretraining context (mirrors stage-1 cap)

# ---- FlowState per-dataset scale factor (leaderboard parity) -----------------
# FlowState is cadence-conditioned: its S5 discretization step and decoder basis
# are stretched by `scale_factor` = pretraining base seasonality (24) / samples
# per dominant cycle. IBM's #1 GIFT-Eval submission sets it per dataset
# (granite-tsfm notebooks/hfdemo/flowstate/gift_wrapper.py + get_fixed_factor in
# tsfm_public/models/flowstate/utils/utils.py); running everything at 1.0
# measures a different, handicapped model. Exact port of their recipe:
#   * frequency -> samples/cycle (sub-daily -> daily cycle, D -> weekly cycle in
#     human-rhythm domains else yearly, W -> yearly, M -> 12, Q/A -> 4),
#   * domain decides daily data's cycle: Transport/Healthcare/Sales have weekly
#     rhythm (their `has_weekly`); the base-name set below is exactly the
#     gift-eval dataset_properties.json entries in those three domains,
#   * bizitobs_l2c has no daily cycle -> extra /7 (their `no_daily` special case).
_FLOWSTATE_BASE_SEASON = 24.0
_FLOWSTATE_WEEKLY_BASES = {
    "hospital", "covid_deaths", "us_births",            # Healthcare
    "car_parts", "restaurant", "hierarchical_sales",    # Sales
    "loop_seattle", "sz_taxi", "m_dense",               # Transport
}


def flowstate_scale_factor(freq: str, ge_name: str = "") -> float:
    """IBM's GIFT-Eval `scale_factor` recipe for one dataset (see block comment).

    `freq` is the gluonts/pandas offset string (`"15T"`, `"H"`, `"W-SUN"`, ...);
    `ge_name` the gift_eval config name (`"LOOP_SEATTLE/D"`), whose base decides
    the daily-data domain rule. Unknown frequencies warn loudly and fall back to
    the neutral 1.0 rather than killing a multi-model run."""
    base_name = ge_name.split("/")[0].lower()
    if base_name.endswith("_with_missing"):
        base_name = base_name[: -len("_with_missing")]
    has_weekly = base_name in _FLOWSTATE_WEEKLY_BASES

    head = freq.split("-")[0].strip()                 # "W-SUN" -> "W"
    digits = "".join(ch for ch in head if ch.isdigit())
    n = int(digits) if digits else 1
    unit = head[len(digits):].upper()
    # new-style pandas aliases -> the old-style ones the port matches on
    unit = {"MIN": "T", "ME": "M", "QE": "Q", "YE": "A", "Y": "A"}.get(unit, unit)

    if unit == "S":
        factor = _FLOWSTATE_BASE_SEASON / (3600.0 / n)          # daily... hourly cycle
    elif unit == "T":
        factor = _FLOWSTATE_BASE_SEASON / (24.0 * 60.0 / n)     # daily cycle
    elif unit == "H":
        factor = _FLOWSTATE_BASE_SEASON / (24.0 / n)            # daily cycle
    elif unit == "D":
        factor = _FLOWSTATE_BASE_SEASON / (7.0 if has_weekly else 365.0)
        factor *= n
    elif unit == "W":
        factor = _FLOWSTATE_BASE_SEASON / (365.0 / 7.0)         # yearly cycle
    elif unit == "M":
        factor = _FLOWSTATE_BASE_SEASON / 12.0
    elif unit in ("Q", "A"):
        factor = _FLOWSTATE_BASE_SEASON / 4.0   # upstream uses 4 for A too (sic)
    else:
        print(Fore.RED + f"  [flowstate] unknown freq {freq!r} -> scale_factor=1.0"
              + Fore.RESET)
        return FLOWSTATE_SCALE_FACTOR
    if "l2c" in base_name:
        factor /= 7.0                           # weekly-only season (no daily cycle)
    return factor

N_BEST_WORST = 10
PLOT_METRICS = ["mae", "mse", "rmse", "mase", "mase_gluonts", "smape", "crps"]
CACHE_ROOT = "logs/experiments/window_ablation_gifteval"
CACHE_ROOT_PREDICTOR = "logs/experiments/context_length_predictor"

# Version stamp for the gluonts-MASE computation (model cells + naive baseline).
# Bump whenever the `mase_gluonts` definition changes so cached cells re-derive it
# cheaply (backfill from per-instance MAE, NO TSFM re-inference) on the next
# `--force 3` pass instead of silently keeping stale numbers.
#   v1: initial port      v2: seasonal_error zero-fallback fixed to 1.0 (gluonts),
#                             was 1e-9 -> exploded MASE on constant/intermittent cells
#   v3: seasonal errors computed from RAW contexts (NaNs preserved, like the
#       leaderboard) instead of the NaN->0-filled model-input copy; naive
#       baseline's mase_gluonts_real now runs the actual gluonts machinery.
MASE_GLUONTS_VER = 3

# Inference-only dynamic batching. ``--batch-size`` is interpreted as the safe
# size at this reference width; shorter contexts start proportionally larger.
# OOM-derived safe sizes are cached per model/width/horizon; accelerator OOMs
# halve the candidate and retry the complete cell.
DYNAMIC_BATCH_REFERENCE_CONTEXT = 1024
DYNAMIC_BATCH_MAX_SIZE = 4096
_DYNAMIC_BATCH_CACHE: Dict[Tuple[str, str, int, int], int] = {}


# ==============================================================================
#  FORECAST CONTAINER
# ==============================================================================

@dataclass
class ForecastResult:
    median: torch.Tensor
    samples: Optional[torch.Tensor] = None
    quantiles: Optional[torch.Tensor] = None
    quantile_levels: Optional[List[float]] = None


def _as_horizon_matrix(t: torch.Tensor, horizon: int, name: str) -> torch.Tensor:
    """Return univariate forecast/target tensors as (N, horizon)."""
    if t.dim() == 3 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    elif t.dim() == 3 and t.shape[1] == 1 and t.shape[2] == horizon:
        t = t.squeeze(1)
    if t.dim() != 2 or t.shape[1] != horizon:
        raise ValueError(
            f"{name} must have shape (N, {horizon}) after univariate squeeze; "
            f"got {tuple(t.shape)}"
        )
    return t


def _as_horizon_cube(t: Optional[torch.Tensor], horizon: int, name: str) -> Optional[torch.Tensor]:
    """Return sample/quantile tensors as (N, S_or_Q, horizon)."""
    if t is None:
        return None
    if t.dim() == 4 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    if t.dim() != 3 or t.shape[2] != horizon:
        raise ValueError(
            f"{name} must have shape (N, K, {horizon}) after univariate squeeze; "
            f"got {tuple(t.shape)}"
        )
    return t


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
        # Cadence conditioning for FlowState (every other family ignores it).
        self.flowstate_scale: float = flowstate_scale_factor(
            self.freq, getattr(ge_dataset, "name", ""))

        contexts: List[np.ndarray] = []
        contexts_raw: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        # Per-instance forecast-start Periods, captured in the SAME (kept) order as
        # contexts/labels so they align 1:1. Feeds the gluonts-machinery
        # `mase_gluonts_real` metric (each forecast needs its start_date).
        starts: List = []
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
            # Two copies of the context, and the split matters for leaderboard
            # exactness: `contexts` is NaN->0 filled and feeds the MODEL wrappers
            # (unchanged, so every cached forecast stays valid); `contexts_raw`
            # keeps the NaNs and feeds the METRIC paths (gluonts seasonal errors +
            # the evaluate_forecasts machinery), which is what the leaderboard
            # sees. Zero-filling before the seasonal error skewed the MASE
            # denominator on every config with missing values.
            raw = np.asarray(target, dtype=np.float32)
            ctx = np.nan_to_num(raw, nan=0.0)
            lbl = np.asarray(label[: self.horizon], dtype=np.float32)
            contexts.append(ctx)
            contexts_raw.append(raw)
            labels.append(lbl)
            starts.append(test_label["start"])

        if not contexts:
            raise RuntimeError(
                f"No valid test instances for {dataset_display}/{ge_dataset.freq}."
            )

        self.starts: List = starts
        self.contexts: List[np.ndarray] = contexts
        self.contexts_raw: List[np.ndarray] = contexts_raw
        self.context_lengths: np.ndarray = np.array(
            [len(c) for c in contexts], dtype=np.int64
        )
        self.labels_np: np.ndarray = np.stack(labels, axis=0)
        self.max_context: int = int(self.context_lengths.max())
        self.min_context: int = int(self.context_lengths.min())
        self.n_total: int = len(contexts)
        self.naive_seasonal_mae_train: float = self._compute_naive_mae_train()

        # Leaderboard (gluonts) MASE ingredients: the seasonality lag from
        # gluonts' map, and the per-instance seasonal error computed from each
        # series' OWN context (mean(|x[m:]-x[:-m]|)). These feed the `mase_gluonts`
        # metric, which — unlike `mase` (global MAE / one pooled training-set
        # seasonal-naive MAE) — averages per-instance ratios, matching the HF
        # GiftEval leaderboard. Cheap: one O(context) pass per series.
        # Computed from the RAW contexts (NaNs preserved, excluded from the mean)
        # exactly like the leaderboard — the zero-filled copy would understate the
        # error wherever values are missing.
        self.season_gluonts: int = get_seasonality(self.freq)
        self.seasonal_errors_gluonts: np.ndarray = per_instance_seasonal_errors(
            self.contexts_raw, self.season_gluonts)

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

    def build_batches_padded(
        self,
        window_size: int,
        batch_size: int,
        device: str,
        pin_memory: bool,
        window_grid: List[int],
    ) -> List[Tuple[int, List[Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, np.ndarray]]:
        """Pad-mode batching: NO instance is skipped.

        Every instance contributes ``min(window_size, context_length)`` of its
        genuine context (the most recent samples). Instances are grouped by that
        effective width — bucketed *down* to ``window_grid`` so models recompiling
        per context length build at most one runner per distinct grid width. The
        per-window error curve is then averaged over the *same* full instance set
        at every window (it flattens once an instance runs out of context), rather
        than over only the instances long enough to serve that window.

        Returns a list of ``(L, batches, all_x, all_y, indices)`` groups, where
        ``indices`` are positions into the full instance list (0..n_total-1).
        """
        ctx_lens = self.context_lengths
        eff = np.minimum(int(window_size), ctx_lens)
        grid = np.asarray(sorted(set(window_grid)))
        # Largest grid width <= eff; per-instance min() guards the rare eff < grid[0].
        eff_buck = np.minimum(
            eff, grid[np.clip(np.searchsorted(grid, eff, side="right") - 1, 0, None)]
        )

        groups: List[Tuple[int, List[Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, np.ndarray]] = []
        for L in np.unique(eff_buck):
            L = int(L)
            idx = np.flatnonzero(eff_buck == L)
            g = idx.size
            x_np = np.empty((g, L), dtype=np.float32)
            for j, i in enumerate(idx):
                x_np[j] = self.contexts[i][-L:]
            y_np = self.labels_np[idx]

            all_x = torch.from_numpy(x_np).unsqueeze(-1)
            all_y = torch.from_numpy(y_np).unsqueeze(-1)
            if pin_memory and device == "cuda":
                ax, ay = all_x.pin_memory(), all_y.pin_memory()
            else:
                ax, ay = all_x, all_y

            batches = []
            for start in range(0, g, batch_size):
                stop = min(start + batch_size, g)
                batches.append({"x": ax[start:stop], "y": ay[start:stop]})
            groups.append((L, batches, all_x, all_y, idx))

        return groups

    def build_batches_full_native(
        self,
        context_cap: int,
        batch_size: int,
        device: str,
        pin_memory: bool,
    ) -> Tuple[
        List[Tuple[int, List[Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, np.ndarray]],
        np.ndarray,
    ]:
        """Full-native batching: serve every instance with its available history.

        Each instance contributes ``min(context_cap, context_length)`` genuine
        samples. Groups are exact effective widths so wrappers that require a
        rectangular tensor still see only real context, matching the sanity
        leaderboard path.
        """
        lengths = np.maximum(1, np.minimum(self.context_lengths, int(context_cap)))

        groups: List[Tuple[int, List[Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, np.ndarray]] = []
        for L in np.unique(lengths):
            L = int(L)
            idx = np.flatnonzero(lengths == L)
            g = idx.size
            x_np = np.empty((g, L), dtype=np.float32)
            for j, i in enumerate(idx):
                ctx = np.asarray(self.contexts[i], dtype=np.float32)
                if ctx.size >= L:
                    x_np[j] = ctx[-L:]
                else:
                    x_np[j] = np.concatenate([
                        np.zeros(L - ctx.size, dtype=np.float32),
                        ctx,
                    ])
            y_np = self.labels_np[idx]

            all_x = torch.from_numpy(x_np).unsqueeze(-1)
            all_y = torch.from_numpy(y_np).unsqueeze(-1)
            if pin_memory and device == "cuda":
                ax, ay = all_x.pin_memory(), all_y.pin_memory()
            else:
                ax, ay = all_x, all_y

            batches = []
            for start in range(0, g, batch_size):
                stop = min(start + batch_size, g)
                batches.append({"x": ax[start:stop], "y": ay[start:stop]})
            groups.append((L, batches, all_x, all_y, idx))

        return groups, lengths.astype(np.int32, copy=False)


# ==============================================================================
#  NAIVE BASELINE
# ==============================================================================

def _naive_cache_path(dataset_display: str, term: str) -> str:
    return os.path.join(
        CACHE_ROOT, "datasets", dataset_display, "_naive_seasonal", f"t{term}", "metrics.json"
    )


def _seasonal_naive_preds(cache: GiftEvalCache, season: int) -> np.ndarray:
    """Seasonal-naive forecast: repeat each series' last ``season`` values across
    the horizon (plain persistence when ``season == 1``). Series shorter than one
    season fall back to the last observed value. Returns an (n_total, horizon)
    float32 array with NaNs zeroed."""
    horizon = cache.horizon
    preds = np.empty((cache.n_total, horizon), dtype=np.float32)
    for i, ctx in enumerate(cache.contexts):
        if len(ctx) >= season:
            tail = ctx[-season:]
            repeats = (horizon // season) + 1
            preds[i] = np.tile(tail, repeats)[:horizon]
        else:
            preds[i] = ctx[-1] if len(ctx) else 0.0
    return np.nan_to_num(preds, nan=0.0)


def compute_naive_seasonal_test_metrics(cache: GiftEvalCache) -> Dict[str, float]:
    """Seasonal-naive baseline metrics for one (dataset, term).

    Two seasonalities are in play and MUST NOT be conflated:
      * project metrics (mae/mse/`mase`) use the project season map (D->7, W->52),
        matching the rest of the ablation and the naive reference lines in plots;
      * the leaderboard `mase_gluonts` normaliser must use gluonts' Seasonal-Naive,
        i.e. the forecast tiled with ``get_seasonality(freq)`` (D->1, W->1). So we
        compute `mase_gluonts` from a SEPARATE gluonts-season forecast and splice it
        in — otherwise D/W/S datasets would divide by the wrong baseline and the
        normalised aggregate wouldn't line up with the HF GiftEval leaderboard.
    """
    tgts = torch.from_numpy(cache.labels_np)

    # Project-season forecast -> mae/mse/`mase` + the plot reference lines. Naive
    # preds cover ALL instances in order, so the gluonts seasonal errors align 1:1.
    proj_preds = torch.from_numpy(_seasonal_naive_preds(cache, cache.season))
    metrics = compute_all_metrics(
        ForecastResult(median=proj_preds), tgts, cache.naive_seasonal_mae_train,
        seasonal_errors=cache.seasonal_errors_gluonts,
    )

    # gluonts-season forecast -> leaderboard-faithful `mase_gluonts` normaliser.
    # Mask NaN targets exactly like the model path (compute_all_metrics): the
    # *_with_missing datasets carry NaN horizon labels, and an unmasked mean would
    # turn the whole naive baseline into NaN — leaving those cells with no
    # denominator (they'd silently drop out of the normalised aggregate).
    gl_preds = torch.from_numpy(_seasonal_naive_preds(cache, cache.season_gluonts))
    gl_valid = ~torch.isnan(tgts)
    gl_abs = (gl_preds - torch.nan_to_num(tgts, nan=0.0)).abs()
    metrics["mase_gluonts"] = compute_mase_gluonts(
        gl_abs, gl_valid, cache.seasonal_errors_gluonts)
    # `mase_gluonts_real` for the naive: run the ACTUAL gluonts machinery on the
    # naive forecast (we hold preds + starts + raw contexts, so nothing stops us).
    # This is the normalisation denominator for the leaderboard-style aggregate —
    # a port stand-in here would make every normalised number slightly off. Falls
    # back to the port only if gluonts is unavailable or the eval fails.
    try:
        metrics["mase_gluonts_real"] = gluonts_leaderboard_mase(
            gl_preds.numpy().astype(np.float64), cache.starts,
            cache.contexts_raw, cache.labels_np, cache.freq)
    except Exception as exc:  # noqa: BLE001 - never sink the baseline on this
        print(Fore.YELLOW + f"  [naive mase_gluonts_real] gluonts eval failed "
              f"({exc}) -> standing in with the port value." + Fore.RESET)
        metrics["mase_gluonts_real"] = metrics["mase_gluonts"]

    # Cache-version sentinels: `_gluonts_naive_faithful` marks the gluonts-season
    # forecast (vs the old project-season one); `_gluonts_naive_ver` tracks the
    # gluonts-MASE definition so a seasonal-error fix invalidates stale naives too.
    metrics["_gluonts_naive_faithful"] = True
    metrics["_gluonts_naive_ver"] = MASE_GLUONTS_VER
    return metrics


def load_or_compute_naive_baseline(
    cache: GiftEvalCache, term: str
) -> Dict[str, float]:
    path = _naive_cache_path(cache.dataset_display, term)
    if os.path.isfile(path):
        with open(path, "r") as f:
            cached = json.load(f)
        # Reuse only a faithful naive at the CURRENT gluonts-MASE version. Older
        # caches (project-season forecast, or a superseded seasonal_error) are
        # recomputed here (the loaded cache is on hand anyway, so it's cheap).
        if cached.get("_gluonts_naive_faithful") \
                and cached.get("_gluonts_naive_ver", 0) >= MASE_GLUONTS_VER:
            return cached
    metrics = compute_naive_seasonal_test_metrics(cache)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ==============================================================================
#  RESULT CACHE
# ==============================================================================

WindowKey = Union[int, str]


def _cache_dir(dataset_display: str, model_short: str, term: str, window_size: WindowKey) -> str:
    return os.path.join(
        CACHE_ROOT, "datasets", dataset_display, model_short, f"t{term}", f"w{window_size}"
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


def _backfill_mase_gluonts(dataset_display, model_short, term, window_size,
                           cache: "GiftEvalCache", short_mode: str):
    """Cheaply derive the cell's gluonts MASE from the cached per-instance MAE +
    the data-derived seasonal errors (NO TSFM re-inference). Returns the float, or
    None when the per-sample cache is missing / its instance set doesn't line up
    (so the caller leaves the cell untouched)."""
    path = os.path.join(_cache_dir(dataset_display, model_short, term, window_size),
                        "per_sample_metrics.npz")
    if not os.path.isfile(path):
        return None
    with np.load(path) as d:
        if "mae" not in d:
            return None
        per_mae = np.asarray(d["mae"], dtype=np.float64)
    # Reproduce the instance set/order stage 3 used for this cell: in skip mode
    # only instances whose context >= window are served (ascending index order);
    # in pad mode every instance is served (full 0..n-1 order).
    if window_size == FULL_NATIVE_WINDOW or short_mode == "pad":
        vi = np.arange(cache.n_total)
    else:
        vi = np.flatnonzero(cache.context_lengths >= window_size)
    if vi.shape[0] != per_mae.shape[0]:
        return None
    se = cache.seasonal_errors_gluonts[vi]
    # Mirror compute_mase_gluonts' instance masking: an instance with no valid
    # horizon points (all-NaN labels on *_with_missing-style cells) has NaN
    # per-instance MAE — drop it instead of letting one NaN poison the cell mean
    # (which left the whole gluonts curve NaN and the cell out of stage 4).
    ok = np.isfinite(per_mae) & np.isfinite(se) & (se > 0)
    if not ok.any():
        return None
    return float(np.mean(per_mae[ok] / se[ok]))


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
    seasonal_errors: Optional[np.ndarray] = None,
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
                ["mae", "mse", "rmse", "mase", "mase_gluonts", "mase_gluonts_real",
                 "smape", "mape", "nd", "nrmse", "crps"]}

    mae = float(abs_err[valid].mean().item())
    mse = float(sq_err[valid].mean().item())
    rmse = float(np.sqrt(mse))
    mase = mae / naive_seasonal_mae

    # Leaderboard (gluonts) MASE: average of PER-INSTANCE ratios
    # mean(|y-yhat|)_horizon / seasonal_error_instance. `seasonal_errors` must be
    # aligned to the instances in this cell (one float per row of `pred`).
    mase_gluonts = compute_mase_gluonts(abs_err, valid, seasonal_errors)
    # `mase_gluonts_real` (the actual gluonts machinery value) is filled by the
    # caller, which holds the forecast objects + start Periods needed to run
    # evaluate_forecasts. Placeholder NaN here keeps the dict shape stable.
    mase_gluonts_real = float("nan")

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
        "mase_gluonts": mase_gluonts, "mase_gluonts_real": mase_gluonts_real,
        "smape": smape, "mape": mape, "nd": nd, "nrmse": nrmse, "crps": crps,
    }


def compute_mase_gluonts(
    abs_err: torch.Tensor,
    valid: torch.Tensor,
    seasonal_errors: Optional[np.ndarray],
) -> float:
    """Leaderboard (gluonts) MASE aggregate from per-(instance, horizon) absolute
    error and a per-instance seasonal error.

    ``abs_err`` / ``valid`` are (N, H); ``seasonal_errors`` is length-N (one
    seasonal error per instance, in the SAME row order as ``abs_err``). Returns
    the mean over instances of ``mean_h(|y-yhat|) / seasonal_error`` (NaN when no
    seasonal errors are supplied or no instance is usable)."""
    if seasonal_errors is None:
        return float("nan")
    se = torch.as_tensor(
        np.asarray(seasonal_errors, dtype=np.float64),
        dtype=abs_err.dtype, device=abs_err.device,
    )
    if se.shape[0] != abs_err.shape[0]:
        return float("nan")
    vmask = valid.to(abs_err.dtype)
    n_per = vmask.sum(dim=1)
    inst_ok = n_per > 0
    if not bool(inst_ok.any()):
        return float("nan")
    per_inst_mae = (abs_err * vmask).sum(dim=1) / n_per.clamp(min=1)
    ratios = per_inst_mae[inst_ok] / se[inst_ok]
    return float(ratios.mean().item())


# Warn once (per process) if gluonts is unavailable, so the log isn't spammed.
_GLUONTS_REAL_WARNED = [False]


def cell_mase_gluonts_real(fr, cache: "GiftEvalCache", served_indices) -> float:
    """Leaderboard MASE for one cell via the ACTUAL gluonts machinery (the
    `mase_gluonts_real` metric). ``served_indices`` are the instance positions the
    forecast ``fr`` covers, in row order: skip mode -> ``valid_indices``; pad mode
    -> ``arange(n_total)``. Returns NaN (warning once) if gluonts is unavailable or
    evaluation fails, so the ablation never crashes on it."""
    try:
        idx = np.asarray(served_indices)
        median = fr.median.detach().cpu().float().numpy()
        quantiles = (fr.quantiles.detach().cpu().float().numpy()
                     if fr.quantiles is not None else None)
        samples = (fr.samples.detach().cpu().float().numpy()
                   if fr.samples is not None else None)
        starts = [cache.starts[i] for i in idx]
        # RAW contexts (NaNs preserved): gluonts derives each instance's seasonal
        # error from these, and the leaderboard sees the missing values as NaN.
        contexts = [cache.contexts_raw[i] for i in idx]
        labels = cache.labels_np[idx]
        return gluonts_leaderboard_mase(
            median, starts, contexts, labels, cache.freq,
            quantiles=quantiles, quantile_levels=fr.quantile_levels, samples=samples)
    except ImportError:
        if not _GLUONTS_REAL_WARNED[0]:
            print(Fore.YELLOW + "  [mase_gluonts_real] gluonts unavailable -> "
                  "leaving NaN (the mase_gluonts port still populates)." + Fore.RESET)
            _GLUONTS_REAL_WARNED[0] = True
        return float("nan")
    except Exception as exc:  # noqa: BLE001 - never let the real metric sink a cell
        print(Fore.YELLOW + f"  [mase_gluonts_real] eval failed: {exc}" + Fore.RESET)
        return float("nan")


def compute_per_sample_metrics(
    forecast_result: ForecastResult,
    targets: torch.Tensor,
    naive_seasonal_mae: float = 1.0,
    seasonal_errors: Optional[np.ndarray] = None,
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
    if seasonal_errors is not None:
        se = np.asarray(seasonal_errors, dtype=np.float64)
        if se.shape[0] == per_mae.shape[0]:
            out["mase_gluonts"] = per_mae / se

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

def _run_batches(batches, desc, step):
    """Drive the standard per-batch forecast loop shared by every predictor.

    ``step(x, y)`` runs one batch and returns ``(outs, tgt)`` where ``outs`` is a
    dict of named per-batch tensors to accumulate and ``tgt`` the per-batch
    target. Each named tensor and the targets are concatenated along dim 0;
    returns ``(dict[str, Tensor], targets)``.
    """
    acc, tgts = {}, []
    with torch.inference_mode():
        for batch in tqdm(batches, desc=desc, leave=False):
            outs, tgt = step(batch["x"], batch["y"])
            for key, val in outs.items():
                acc.setdefault(key, []).append(val.detach())
            tgts.append(tgt.detach())
    cat = {key: torch.cat(vals, 0) for key, vals in acc.items()}
    return cat, torch.cat(tgts, 0)


def _is_accelerator_oom(exc: BaseException) -> bool:
    """Recognise allocation failures across CUDA and wrapped model stacks."""
    message = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or any(fragment in message for fragment in (
        "out of memory", "cuda oom", "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed", "mps backend out of memory",
        "unable to allocate", "resource exhausted", "allocation failed",
    ))


def _clear_accelerator_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "empty_cache"):
        try:
            mps.empty_cache()
        except Exception:
            pass
    gc.collect()


def _rebatch(batches, batch_size: int):
    """Re-slice a uniform-width CPU batch list without rebuilding the dataset."""
    if not batches:
        return []
    all_x = torch.cat([batch["x"] for batch in batches], dim=0)
    all_y = torch.cat([batch["y"] for batch in batches], dim=0)
    return [
        {"x": all_x[start:start + batch_size],
         "y": all_y[start:start + batch_size]}
        for start in range(0, all_x.shape[0], batch_size)
    ]


def _forecast_cell_dynamic(model_family, handle, model_id, batches, width,
                           horizon, device, base_batch_size,
                           flowstate_scale=FLOWSTATE_SCALE_FACTOR):
    """Forecast one width using context-scaled batching with OOM backoff.

    The returned batch size is persisted in cell metadata. Fixed-native-context
    PatchTST-FM does not scale with the requested window, so it starts at the
    configured base instead of receiving the short-window multiplier.
    """
    n_samples = sum(int(batch["x"].shape[0]) for batch in batches)
    key = (str(model_id), model_family, int(width), int(horizon))
    if key in _DYNAMIC_BATCH_CACHE:
        candidate = min(_DYNAMIC_BATCH_CACHE[key], n_samples)
    else:
        scale = (1 if model_family == "patchtst_fm" else
                 max(1, DYNAMIC_BATCH_REFERENCE_CONTEXT // max(1, int(width))))
        candidate = min(n_samples, DYNAMIC_BATCH_MAX_SIZE,
                        max(1, int(base_batch_size)) * scale)

    reduced_after_oom = False
    while True:
        trial_batches = _rebatch(batches, candidate)
        try:
            result = _forecast_cell(
                model_family, handle, model_id, trial_batches, width, horizon,
                device, candidate, flowstate_scale=flowstate_scale)
            # Do not cache a size merely capped by a tiny dataset: that would
            # unnecessarily throttle a later, larger dataset at the same width.
            if reduced_after_oom or key in _DYNAMIC_BATCH_CACHE:
                _DYNAMIC_BATCH_CACHE[key] = candidate
            return result[0], result[1], candidate
        except Exception as exc:
            if not _is_accelerator_oom(exc) or candidate <= 1:
                raise
            next_candidate = max(1, candidate // 2)
            print(Fore.YELLOW
                  + f"    dynamic batch {model_family} W={width}: OOM at "
                    f"{candidate}; retrying with {next_candidate}"
                  + Fore.RESET)
            del trial_batches
            _clear_accelerator_cache()
            candidate = next_candidate
            reduced_after_oom = True


def load_chronos_bolt(model_id, device):
    from chronos import ChronosBoltPipeline
    return ChronosBoltPipeline.from_pretrained(
        model_id, device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )


def predict_chronos_bolt(pipeline, batches, horizon, device):
    def step(x, y):
        samples = pipeline.predict(inputs=x[:, :, 0], prediction_length=horizon)
        return ({"samples": samples.to(device=device, dtype=torch.float32, non_blocking=True)},
                y[:, :, 0].to(device, non_blocking=True))
    out, all_tgts = _run_batches(batches, "  Chronos-Bolt", step)
    all_samples = out["samples"]
    median = torch.median(all_samples, dim=1).values
    return ForecastResult(median=median, samples=all_samples), all_tgts


def load_chronos2(model_id, device):
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(
        model_id, device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )


def predict_chronos2(pipeline, batches, horizon, device):
    def step(x, y):
        context = x.permute(0, 2, 1)
        samples = pipeline.predict(inputs=context, prediction_length=horizon)
        samples = (torch.stack(samples, dim=0).squeeze(1)
                   if isinstance(samples, list) else samples)
        if samples.dim() == 4:
            samples = samples.squeeze(2)
        return ({"quantiles": samples.to(device=device, dtype=torch.float32, non_blocking=True)},
                y[:, :, 0].to(device, non_blocking=True))
    out, all_tgts = _run_batches(batches, "  Chronos2", step)
    all_quantiles = out["quantiles"]
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
    def step(x, y):
        x_cpu = x[:, :, 0].numpy()
        context_list = [x_cpu[i] for i in range(x_cpu.shape[0])]
        forecast = model.predict(past_target=context_list)
        forecast_t = torch.as_tensor(forecast[:, :, :horizon], dtype=torch.float32, device=device)
        return {"quantiles": forecast_t}, y[:, :, 0].to(device, non_blocking=True)
    out, all_tgts = _run_batches(batches, "  Moirai", step)
    all_quantiles = out["quantiles"]
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
    def step(x_cpu, y_cpu):
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
        return {"samples": samples}, y[:, :, 0]
    out, all_tgts = _run_batches(batches, "  Moirai1.1", step)
    all_samples = out["samples"]
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
    """Median + quantile forecast.

    NOTE: TimesFM-2.5's ``point_forecast`` IS the 0.5-quantile, not the mean —
    ``compiled_decode`` returns ``full_forecast[..., 5]`` (``decode_index=5``;
    columns are ``[mean-head, q0.1..q0.9]``), which is exactly what the GiftEval
    leaderboard scores as MASE[0.5]. Do NOT "fix" this to the mean head (col 0).
    The ``[:, :, 1:]`` slice below keeps the 9 quantiles, mirroring the official
    notebook's ``full_preds[:, :, 1:]``.
    """
    PATCH_SIZE = 32

    def step(x, y):
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
        return ({"median": pf, "quantiles": qf},
                y[:, :, 0].to(device, non_blocking=True))
    out, all_tgts = _run_batches(batches, "  TimesFM", step)
    return ForecastResult(
        median=out["median"], quantiles=out["quantiles"],
        quantile_levels=TIMESFM_QUANTILE_LEVELS,
    ), all_tgts


def load_patchtst_fm(model_id, device):
    if PatchTSTFMForPrediction is None:
        raise RuntimeError("tsfm_public is not installed.")
    model = PatchTSTFMForPrediction.from_pretrained(model_id, device_map=device)
    model.eval()
    return model


def predict_patchtst_fm(model, batches, horizon, device):
    def step(x_cpu, y_cpu):
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        B = x.shape[0]
        # PatchTST-FM-R1 has a FIXED context_length (8192) and no observed-mask
        # input; a shorter tensor makes its internal padding NaN out. Left-pad to
        # the native context with NaN, which the model treats as its missing-value
        # indicator and masks (empirically the least-distorting pad: matches
        # mean-padding, beats zeros). Genuine samples drive the forecast; output
        # stays finite. Any incoming NaN pad region (pad-mode short inputs) is
        # likewise masked.
        ctx_len = model.config.context_length
        past_values = x.squeeze(-1)                          # (B, W)
        if past_values.shape[1] < ctx_len:
            pad = past_values.new_full((B, ctx_len - past_values.shape[1]), float("nan"))
            past_values = torch.cat([pad, past_values], dim=1)
        elif past_values.shape[1] > ctx_len:
            past_values = past_values[:, -ctx_len:]
        # tsfm_public's multi-step rollout (forecast_single_step) builds its mask
        # with a bare `torch.ones(f_i)` (no device arg) -> CPU, tripping a device
        # mismatch against the CUDA hidden state when horizon exceeds the model's
        # single-shot length (long-term GiftEval). Run under a default-device
        # context so those factory tensors land on `device`.
        with torch.device(device):
            # Granite-TSFM renamed the first argument from ``inputs`` to
            # ``past_values``. Positional dispatch supports both APIs.
            output = model(past_values, prediction_length=horizon)
        raw = output[0]
        if raw.dim() == 4:
            qf = raw[:, :, :horizon, 0]
            pred_len = qf.shape[2]
            if pred_len < horizon:
                pad = qf[:, :, -1:].expand(B, qf.shape[1], horizon - pred_len)
                qf = torch.cat([qf, pad], dim=2)
            preds = qf.to(torch.float32)
        elif raw.dim() == 3:
            if raw.shape[1] == 1:
                preds = raw[:, :, :horizon]
            elif raw.shape[2] == 1:
                preds = raw[:, :horizon, 0].unsqueeze(1)
            elif raw.shape[1] == len(PATCHTST_FM_QUANTILE_LEVELS):
                preds = raw[:, :, :horizon]
            elif raw.shape[2] == len(PATCHTST_FM_QUANTILE_LEVELS):
                preds = raw[:, :horizon, :].permute(0, 2, 1)
            else:
                preds = raw[:, :, :horizon]
            preds = preds.to(torch.float32)
        elif raw.dim() == 2:
            preds = raw[:, :horizon].unsqueeze(1).to(torch.float32)
        else:
            raise ValueError(f"Unexpected output shape {raw.shape}")
        return {"quantiles": preds}, y[:, :, 0]
    out, all_tgts = _run_batches(batches, "  PatchTST-FM", step)
    all_quantiles = out["quantiles"]
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
    def step(x, y):
        seqs = x[:, :, 0].to(device, non_blocking=True)
        samples = model.generate(
            seqs, max_new_tokens=horizon, num_samples=SUNDIAL_NUM_SAMPLES,
        )
        samples = samples[:, :, :horizon].to(torch.float32)
        return {"samples": samples}, y[:, :, 0].to(device, non_blocking=True)
    out, all_tgts = _run_batches(batches, "  Sundial", step)
    all_samples = out["samples"]
    median = torch.median(all_samples, dim=1).values
    return ForecastResult(median=median, samples=all_samples), all_tgts


def load_timemoe(model_id, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.to(device).eval()
    return model


def predict_timemoe(model, batches, horizon, device):
    def step(x, y):
        seqs = x[:, :, 0].to(device, non_blocking=True)
        mean = seqs.mean(dim=-1, keepdim=True)
        std = seqs.std(dim=-1, keepdim=True)
        normed = (seqs - mean) / (std + 1e-8)
        out = model.generate(normed, max_new_tokens=horizon)
        preds = out[:, -horizon:].to(torch.float32) * std + mean
        return {"preds": preds}, y[:, :, 0].to(device, non_blocking=True)
    out, all_tgts = _run_batches(batches, "  TimeMoE", step)
    return ForecastResult(median=out["preds"]), all_tgts


def load_toto(model_id, device):
    from toto2 import Toto2Model
    model = Toto2Model.from_pretrained(model_id)
    model.to(device).eval()
    return model


def predict_toto(model, batches, horizon, device):
    """Quantile forecast (Toto 2.0); return the per-step median (B, horizon).

    Mirrors stage-1's predict_toto: each univariate series is one batch element
    with a single variate (no cross-series attention). forecast() returns
    quantiles of shape (9, B, n_variates, horizon); take the 0.5 row (index 4).
    The patcher requires the context length to be a multiple of TOTO_PATCH_SIZE,
    so left-pad the oldest steps (forecast origin at the right is untouched) and
    mask the pad as missing.
    """
    def step(x, y):
        seqs = x[:, :, 0].to(device, non_blocking=True)        # (B, L)
        b, length = seqs.shape
        pad = (-length) % TOTO_PATCH_SIZE
        if pad:
            seqs = torch.cat([seqs.new_zeros((b, pad)), seqs], dim=1)  # (B, L+pad)
        target = seqs.unsqueeze(1)                             # (B, 1, L')
        target_mask = torch.ones_like(target, dtype=torch.bool)
        if pad:
            target_mask[:, :, :pad] = False
        series_ids = torch.arange(b, device=device, dtype=torch.long).unsqueeze(1)
        quantiles = model.forecast(
            {"target": target, "target_mask": target_mask, "series_ids": series_ids},
            horizon=horizon,
            has_missing_values=bool(pad),
        )                                                      # (9, B, 1, horizon)
        median = quantiles[TOTO_MEDIAN_QUANTILE_IDX, :, 0, :horizon].to(torch.float32)
        return {"median": median}, y[:, :, 0].to(device, non_blocking=True)
    out, all_tgts = _run_batches(batches, "  Toto", step)
    return ForecastResult(median=out["median"]), all_tgts


def load_flowstate(model_id, device):
    from tsfm_public import FlowStateForPrediction
    model = FlowStateForPrediction.from_pretrained(
        model_id, revision=FLOWSTATE_REVISION).to(device)
    model.eval()
    return model


def predict_flowstate(model, batches, horizon, device,
                      scale_factor=FLOWSTATE_SCALE_FACTOR):
    """Quantile forecast (FlowState r1.1); return the per-step median (B, horizon).

    ``x`` is (B, W, 1), exactly FlowState's ``batch_first=True`` layout, so it's
    passed through unreshaped. ``scale_factor`` is the per-dataset cadence factor
    (see ``flowstate_scale_factor``); callers with a GiftEvalCache pass
    ``cache.flowstate_scale``. The median comes from ``quantile_outputs``
    (B, Q=9, H, C), index 4 — reading ``prediction_outputs`` instead is a trap on
    current tsfm_public: the r1.1 config's ``prediction_type='quantile'`` is
    deprecated there and silently coerced to ``'mean'``, so ``prediction_outputs``
    is the quantile-weighted MEAN (3-D), not the median every other wrapper
    contributes. Older tsfm_public (no ``quantile_outputs``) put the quantiles in
    ``prediction_outputs`` as 4-D (B, Q, H, C) — kept as fallback; a 3-D output
    there is a true point forecast.
    """
    def step(x, y):
        series = x.to(device, non_blocking=True)               # (B, W, 1)
        out = model(
            series,
            scale_factor=scale_factor,
            prediction_length=horizon,
            batch_first=True,
        )
        qf = getattr(out, "quantile_outputs", None)
        if qf is not None:                                     # (B, Q, H, C)
            med = qf[:, FLOWSTATE_MEDIAN_QUANTILE_IDX, :horizon, 0]
        else:
            pf = out.prediction_outputs
            if pf.dim() == 4:                                  # (B, Q, H, C)
                med = pf[:, FLOWSTATE_MEDIAN_QUANTILE_IDX, :horizon, 0]
            else:                                              # (B, H, C)
                med = pf[:, :horizon, 0]
        return {"median": med.to(torch.float32)}, y[:, :, 0].to(device, non_blocking=True)
    out, all_tgts = _run_batches(batches, "  FlowState", step)
    return ForecastResult(median=out["median"]), all_tgts


def load_tirex(model_id, device):
    # TiRex2 lives in the `tirex2` package (TiRex-1 was `tirex`); load_model takes
    # the device up front and handles placement internally. Its API accepts only
    # bare device types ("cuda"/"cpu"); multi-GPU workers mask themselves to one
    # visible card, so "cuda" still selects the intended GPU.
    from experiments.tirex_compat import configure_tirex_backend
    from tirex2 import load_model
    tirex_device = "cuda" if str(device).startswith("cuda") else "cpu"
    backend = configure_tirex_backend(tirex_device)
    print(f"  TiRex FlashRNN backend: {backend}", flush=True)
    return load_model(model_id, device=tirex_device)


def predict_tirex(model, batches, horizon, device):
    """Quantile forecast (TiRex2); return the per-step median (B, horizon).

    Mirrors stage-1's predict_tirex. TiRex2's API differs from TiRex-1: it takes a
    LIST of ``TimeseriesType`` objects (one per series, ``target`` shaped
    (n_variates, length)) rather than a batched ``context=`` tensor, and each
    forecast is shaped (n_targets, 9 quantiles, prediction_length) — the quantile
    axis is the MIDDLE one now (TiRex-1 was (B, horizon, 9)). The 9 levels are
    still [0.1..0.9], so index 4 (0.5) is the point forecast. ``output_type=
    "numpy"`` is the documented return format; convert back to `device` for the
    downstream stitching/metrics. Horizons above ``model.future_len`` roll
    forward in chunks using each preceding median as context.
    """
    def step(x, y):
        seqs = x[:, :, 0].to(device, non_blocking=True)        # (B, L)
        from experiments.tirex_compat import forecast_tirex_medians
        medians = forecast_tirex_medians(
            model, seqs, horizon, TIREX_MEDIAN_QUANTILE_IDX
        )                                                       # (B, horizon)
        return ({"median": torch.as_tensor(medians, dtype=torch.float32,
                                           device=device)},
                y[:, :, 0].to(device, non_blocking=True))
    out, all_tgts = _run_batches(batches, "  TiRex", step)
    return ForecastResult(median=out["median"]), all_tgts


def predict_context_parroting(batches, horizon, device):
    def step(x_cpu, y_cpu):
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        ws = x.shape[1]
        if ws >= horizon:
            pred = x[:, -horizon:, 0]
        else:
            repeats = (horizon // ws) + 1
            pred = x[:, :, 0].repeat(1, repeats)[:, -horizon:]
        return {"preds": pred}, y[:, :, 0]
    out, all_tgts = _run_batches(batches, "  Parroting", step)
    return ForecastResult(median=out["preds"]), all_tgts


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
        # NaNs and infinities cannot produce meaningful histogram limits.
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            ax.set_title(f"{name.upper()} (no data)"); continue
        clip_val = np.percentile(vals, percentile_clip)
        clipped = vals[vals <= clip_val]
        # Numerical roundoff in percentile() should not normally make this
        # empty, but retaining the finite values is safer than passing an empty
        # array to histogram().
        if len(clipped) == 0:
            clipped = vals
        n_bins = min(80, max(20, len(clipped) // 10))
        try:
            bin_edges = np.histogram_bin_edges(clipped, bins=n_bins)
        except ValueError:
            # At a large offset, a very narrow data range may contain fewer
            # representable floats than requested bin edges (NumPy raises
            # "Too many bins for data range"). One bin remains informative for
            # this effectively degenerate distribution and is always enough.
            bin_edges = np.histogram_bin_edges(clipped, bins=1)
        ax.hist(clipped, bins=bin_edges, density=True, alpha=0.55,
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
        if clip_val > 0:
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
    datasets_dir = os.path.join(run_dir, "datasets", dataset_display)
    os.makedirs(datasets_dir, exist_ok=True)
    path = os.path.join(datasets_dir, f"summary_{dataset_display}_t{term}_h{horizon_str}.png")
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
    # build_predictor branches on cfg["arch"] (defaulting to "patchtst" for
    # pre-v4 checkpoints that predate the arch key), so a Mamba predictor loads
    # exactly like a Transformer one here.
    model = build_predictor(cfg, cfg["n_windows"], cfg["n_horizons"])
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
    training_objective: str = "curve",
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
            if training_objective == "classification":
                # Downstream strategy code historically chooses argmin of a
                # predicted cost curve. Convert class logits to negative
                # sigmoid scores so argmin remains the single decision contract
                # while preserving the classifier's argmax exactly.
                curve_pred = -torch.sigmoid(curve_pred)
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
    training_objective: str = "curve",
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

    pred_label = (f"Classifier decision score (mean over {n_inst} inst.)"
                  if training_objective == "classification"
                  else f"Predicted curve (mean over {n_inst} inst.)")
    ax.plot(ws, pred_mean,
            marker="x", linewidth=2.0, color="#d62728",
            label=pred_label)
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
    ax.set_ylabel(
        "Real z-score / classifier score (-probability)"
        if training_objective == "classification"
        else f"Z-scored {curve_metric.upper()} along windows")
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

def load_handle(model_family: str, model_id: str, device: str):
    """Load the base model for a family, or None for families that need no
    persistent handle (timesfm recompiles per-cell; context_parroting is
    parameter-free). Single source of truth for the per-family load dispatch,
    shared by the ablation's lazy `ensure_handle` and the robust-timing stage."""
    if model_family == "chronos_bolt":
        return load_chronos_bolt(model_id, device)
    if model_family == "chronos2":
        return load_chronos2(model_id, device)
    if model_family == "moirai":
        return load_moirai_module(model_id)
    if model_family == "moirai_1_1":
        return load_moirai_1_1_module(model_id)
    if model_family == "patchtst_fm":
        return load_patchtst_fm(model_id, device)
    if model_family == "sundial":
        return load_sundial(model_id, device)
    if model_family == "timemoe":
        return load_timemoe(model_id, device)
    if model_family == "toto":
        return load_toto(model_id, device)
    if model_family == "flowstate":
        return load_flowstate(model_id, device)
    if model_family == "tirex":
        return load_tirex(model_id, device)
    # timesfm (recompiled per-cell in _forecast_cell) + context_parroting need none.
    return None


def _forecast_cell(model_family, handle, model_id, batches, width, horizon,
                   device, batch_size, flowstate_scale=FLOWSTATE_SCALE_FACTOR):
    """Run one model family on a uniform-width set of batches.

    `width` is the context length these batches share; moirai/timesfm recompile
    against it. `handle` is the loaded base model (None for timesfm — recompiled
    here — and context_parroting). `flowstate_scale` is FlowState's per-dataset
    cadence factor (`cache.flowstate_scale`); ignored by every other family.
    Returns (ForecastResult, tgts).
    """
    if model_family == "chronos_bolt":
        return predict_chronos_bolt(handle, batches, horizon, device)
    if model_family == "chronos2":
        return predict_chronos2(handle, batches, horizon, device)
    if model_family == "moirai":
        m = build_moirai_forecast(handle, horizon, width, device)
        fr, tgts = predict_moirai(m, batches, horizon, device)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
        return fr, tgts
    if model_family == "moirai_1_1":
        m = build_moirai_1_1_forecast(handle, horizon, width, device)
        fr, tgts = predict_moirai_1_1(m, batches, horizon, device)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
        return fr, tgts
    if model_family == "timesfm":
        tfm = load_timesfm(model_id, width, horizon, batch_size)
        fr, tgts = predict_timesfm(tfm, batches, horizon, device)
        del tfm
        if device == "cuda":
            torch.cuda.empty_cache()
        return fr, tgts
    if model_family == "context_parroting":
        return predict_context_parroting(batches, horizon, device)
    if model_family == "patchtst_fm":
        return predict_patchtst_fm(handle, batches, horizon, device)
    if model_family == "sundial":
        return predict_sundial(handle, batches, horizon, device)
    if model_family == "timemoe":
        return predict_timemoe(handle, batches, horizon, device)
    if model_family == "toto":
        return predict_toto(handle, batches, horizon, device)
    if model_family == "flowstate":
        return predict_flowstate(handle, batches, horizon, device,
                                 scale_factor=flowstate_scale)
    if model_family == "tirex":
        return predict_tirex(handle, batches, horizon, device)
    raise ValueError(f"Unknown model family: {model_family}")


def build_forward(model_family, handle, model_id, batches, width, horizon,
                  device, batch_size, flowstate_scale=FLOWSTATE_SCALE_FACTOR):
    """Do all per-window SETUP now (weight load / compile / forecaster build) and
    return a zero-arg ``forward()`` that runs ONLY the forward pass over ``batches``.

    Separating setup from the forward lets a benchmark time the inference itself
    rather than re-paying timesfm's reload+compile or moirai's wrapper build on
    every repeat. For families whose ``_forecast_cell`` is already a pure forward
    over a persistent handle, ``forward()`` is just that call. Returns
    ``(forward, teardown)``; call ``teardown()`` once after the repeats to free any
    model built here (no-op for persistent-handle families)."""
    def _noop():
        pass

    if model_family == "moirai":
        m = build_moirai_forecast(handle, horizon, width, device)

        def _td():
            nonlocal m
            del m
            if device == "cuda":
                torch.cuda.empty_cache()
        return (lambda: predict_moirai(m, batches, horizon, device)), _td

    if model_family == "moirai_1_1":
        m = build_moirai_1_1_forecast(handle, horizon, width, device)

        def _td():
            nonlocal m
            del m
            if device == "cuda":
                torch.cuda.empty_cache()
        return (lambda: predict_moirai_1_1(m, batches, horizon, device)), _td

    if model_family == "timesfm":
        tfm = load_timesfm(model_id, width, horizon, batch_size)

        def _td():
            nonlocal tfm
            del tfm
            if device == "cuda":
                torch.cuda.empty_cache()
        return (lambda: predict_timesfm(tfm, batches, horizon, device)), _td

    # Persistent-handle families (+ context_parroting): _forecast_cell is already a
    # pure forward, so no separate setup is needed.
    return (lambda: _forecast_cell(model_family, handle, model_id, batches,
                                   width, horizon, device, batch_size,
                                   flowstate_scale=flowstate_scale)), _noop


def _merge_grouped(results, n_total, horizon, device):
    """Reassemble per-width-group forecasts into a single instance-ordered result.

    `results` is a list of (indices, ForecastResult, tgts). All groups come from
    the same model, so they share sample count / quantile levels.
    """
    first, first_tgts = results[0][1], results[0][2]
    first_median = _as_horizon_matrix(first.median, horizon, "median")
    first_tgts = _as_horizon_matrix(first_tgts, horizon, "targets")
    first_samples = _as_horizon_cube(first.samples, horizon, "samples")
    first_quantiles = _as_horizon_cube(first.quantiles, horizon, "quantiles")

    median = torch.empty((n_total, horizon), device=device, dtype=first_median.dtype)
    tgts = torch.empty((n_total, horizon), device=device, dtype=first_tgts.dtype)

    samples = quantiles = None
    qlevels = first.quantile_levels
    if first_samples is not None:
        S = first_samples.shape[1]
        samples = torch.empty((n_total, S, horizon), device=device, dtype=first_samples.dtype)
    if first_quantiles is not None:
        Q = first_quantiles.shape[1]
        quantiles = torch.empty((n_total, Q, horizon), device=device, dtype=first_quantiles.dtype)

    for idx, fr, t in results:
        ii = torch.as_tensor(idx, device=device, dtype=torch.long)
        median[ii] = _as_horizon_matrix(fr.median, horizon, "median")
        tgts[ii] = _as_horizon_matrix(t, horizon, "targets")
        if samples is not None:
            samples[ii] = _as_horizon_cube(fr.samples, horizon, "samples")
        if quantiles is not None:
            quantiles[ii] = _as_horizon_cube(fr.quantiles, horizon, "quantiles")

    return ForecastResult(median=median, samples=samples,
                          quantiles=quantiles, quantile_levels=qlevels), tgts


def _full_native_context_cap(model_family: str, horizon: int, max_available: int) -> int:
    caps = {
        "timesfm": TIMESFM_FULL_CONTEXT,
        "patchtst_fm": 8192,
        "sundial": SUNDIAL_MAX_CONTEXT,
        "toto": TOTO_MAX_CONTEXT,
        "flowstate": FLOWSTATE_MAX_CONTEXT,
        "tirex": TIREX_MAX_CONTEXT,
    }
    if model_family == "timemoe":
        caps[model_family] = max(1, TIMEMOE_MAX_TOTAL - int(horizon))
    cap = int(caps.get(model_family, max_available))
    return max(1, min(cap, int(max_available)))


def _missing_metric(metrics: dict, key: str) -> bool:
    v = metrics.get(key)
    return v is None or (isinstance(v, float) and np.isnan(v))


def _run_full_native_baseline(
    cache: GiftEvalCache,
    dataset_display: str,
    ge_name: str,
    term: str,
    model_id: str,
    model_family: str,
    model_short: str,
    batch_size: int,
    device: str,
    ensure_handle,
) -> None:
    horizon = cache.horizon
    cap = _full_native_context_cap(model_family, horizon, cache.max_context)
    tag = (f"{model_short} | {dataset_display} | t={term} | h={horizon} | "
           f"w={FULL_NATIVE_WINDOW} cap={cap}")

    if _result_cached(dataset_display, model_short, term, FULL_NATIVE_WINDOW):
        cached = _load_cached_result(dataset_display, model_short, term, FULL_NATIVE_WINDOW)
        changed = False
        stale_ver = cached.get("_mase_gluonts_ver", 0) < MASE_GLUONTS_VER
        if _missing_metric(cached, "mase_gluonts") or stale_ver:
            mg = _backfill_mase_gluonts(
                dataset_display, model_short, term, FULL_NATIVE_WINDOW,
                cache, "pad")
            if mg is not None:
                cached["mase_gluonts"] = mg
                cached["_mase_gluonts_ver"] = MASE_GLUONTS_VER
                changed = True
        if (_missing_metric(cached, "mase_gluonts_real") or stale_ver) \
                and not _missing_metric(cached, "mase_gluonts"):
            cached["mase_gluonts_real"] = cached["mase_gluonts"]
            cached["_mase_gluonts_real_standin"] = True
            changed = True
        if changed:
            _save_result(dataset_display, model_short, term,
                         FULL_NATIVE_WINDOW, cached)
        print(Fore.WHITE + f"  CACHED  {tag}  -> MAE={cached['mae']:.6f}" + Fore.RESET)
        return

    print(Fore.YELLOW + f"\n  > {tag}" + Fore.RESET)
    groups, effective_lengths = cache.build_batches_full_native(
        cap, batch_size, device, pin_memory=True)

    def _sync():
        if device == "cuda":
            torch.cuda.synchronize()

    results = []
    effective_batch_sizes = []
    _sync()
    t_start = time.perf_counter()
    for L, batches_L, _ax, _ay, idx_L in groups:
        fr_L, tgts_L, effective_bs = _forecast_cell_dynamic(
            model_family, ensure_handle(), model_id, batches_L,
            L, horizon, device, batch_size,
            flowstate_scale=cache.flowstate_scale)
        results.append((idx_L, fr_L, tgts_L))
        effective_batch_sizes.append(effective_bs)
    fr, tgts = _merge_grouped(results, cache.n_total, horizon, device)
    _sync()
    elapsed = time.perf_counter() - t_start

    served_idx = np.arange(cache.n_total)
    se_cell = cache.seasonal_errors_gluonts
    metrics = compute_all_metrics(fr, tgts, cache.naive_seasonal_mae_train,
                                  seasonal_errors=se_cell)
    metrics["mase_gluonts_real"] = cell_mase_gluonts_real(fr, cache, served_idx)
    if np.isnan(metrics["mase_gluonts_real"]) and not np.isnan(metrics["mase_gluonts"]):
        metrics["mase_gluonts_real"] = metrics["mase_gluonts"]
        metrics["_mase_gluonts_real_standin"] = True
    else:
        metrics["_mase_gluonts_real_standin"] = False
    metrics["_mase_gluonts_ver"] = MASE_GLUONTS_VER
    metrics["elapsed_seconds"] = round(elapsed, 3)
    metrics["horizon"] = horizon
    metrics["_full_native_baseline"] = True
    metrics["_context_cap"] = int(cap)
    metrics["_min_effective_context"] = int(np.min(effective_lengths))
    metrics["_mean_effective_context"] = float(np.mean(effective_lengths))
    metrics["_max_effective_context"] = int(np.max(effective_lengths))
    metrics["_n_width_groups"] = int(len(groups))
    metrics["_dynamic_batch_sizes"] = effective_batch_sizes

    for k, v in metrics.items():
        if isinstance(v, float):
            print(Fore.YELLOW + f"    {k}: {v:.6f}" + Fore.RESET)
    print(Fore.MAGENTA + f"    TIME  {elapsed:.1f}s" + Fore.RESET)
    print(f"    Samples: {cache.n_total} ({len(groups)} exact width-groups, full-native)  "
          f"cap={cap}  H={horizon}")

    per_sample = compute_per_sample_metrics(
        fr, tgts, cache.naive_seasonal_mae_train, seasonal_errors=se_cell)
    per_sample["effective_context"] = effective_lengths
    per_sample["served_index"] = served_idx.astype(np.int32, copy=False)
    _save_per_sample_metrics(dataset_display, model_short, term,
                             FULL_NATIVE_WINDOW, per_sample)
    _save_result(dataset_display, model_short, term, FULL_NATIVE_WINDOW, metrics)

    del fr, tgts, per_sample, results
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Window-size ablation on GiftEval with predictor overlay.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--models", type=str, nargs="+", default=None)
    p.add_argument("--datasets", type=str, nargs="+", default=None)
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    p.add_argument("--no-plots", action="store_true",
                   help="Skip per-(model,window) plots (summary + comparison still produced).")
    p.add_argument("--no-cell-cache", action="store_true",
                   help="Do not persist per-cell metrics.json / per-sample .npz. Results "
                        "are still aggregated in memory into results.csv and the comparison "
                        "plots, so downstream stages work — but nothing is cached for resume. "
                        "Used by run_all.py --test smoke runs.")
    p.add_argument("--test-datasets", type=int, default=None,
                   help="Smoke test: randomly sample this many (dataset, term) entries to "
                        "run, instead of the full grid. Applied after --datasets. The sample "
                        "is seeded (--test-datasets-seed) so every sharded worker agrees on "
                        "the same subset. Used by run_all.py --test.")
    p.add_argument("--test-datasets-seed", type=int, default=0,
                   help="Seed for --test-datasets sampling (kept identical across workers).")
    p.add_argument("--cache-root", type=str, default=CACHE_ROOT)
    p.add_argument("--predictor-cache-root", type=str, default=CACHE_ROOT_PREDICTOR,
                   help="Root holding context-length-predictor runs. Latest run is auto-picked.")
    p.add_argument("--predictor-dir", type=str, default=None,
                   help="Override: use this specific predictor run directory.")
    p.add_argument("--predictor-batch-size", type=int, default=64,
                   help="Batch size used when running the predictor on test-instance contexts.")
    p.add_argument("--short-context-mode", choices=["skip", "pad"], default="skip",
                   help="How to handle instances whose context is shorter than the "
                        "ablation window. 'skip' (default): exclude them at that window "
                        "(original behaviour; the curve at window w averages only "
                        "instances with >=w history). 'pad': never skip — feed each "
                        "instance its min(w, context) genuine samples (PatchTST-FM is "
                        "NaN-padded to its native context), so every window averages the "
                        "same full instance set and the curve flattens past available context.")
    p.add_argument("--no-full-native-baseline", action="store_true",
                   help="Do not write the extra wfull_native baseline used by stage 4 "
                        "as the stricter full-context/FLOPs reference.")
    p.add_argument("--num-gpus", type=int, default=0,
                   help="GPUs to shard the ablation across. 0 = auto (all visible, "
                        "respecting CUDA_VISIBLE_DEVICES); >0 caps to min(n, device_count). "
                        "1 (or CPU) keeps the original single-process behavior.")
    # Internal: set by the coordinator when it spawns one worker per GPU.
    p.add_argument("--shard-id", type=int, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--num-shards", type=int, default=1,
                   help=argparse.SUPPRESS)
    return p.parse_args()


def _run_coordinator(args, device: str, n_gpus: int, n_visible: int) -> None:
    """Spawn one worker subprocess per GPU (each pinned + given a dataset shard),
    wait for all, then run a single aggregation pass over the now-filled cache.

    Workers are fresh `python -m` processes (no CUDA-in-fork hazard) writing to
    the shared per-cell cache; the coordinator produces the CSV/plots/marker."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    phys = visible.split(",") if visible else [str(j) for j in range(n_visible)]

    print(Fore.CYAN
          + f"Coordinator: sharding ablation across {n_gpus} GPU(s) "
          + f"(physical {phys[:n_gpus]}), by dataset." + Fore.RESET)

    procs = []
    for i in range(n_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = phys[i]
        cmd = [sys.executable, "-m",
               "experiments.test_window_ablation_gifteval_v5",
               *sys.argv[1:],
               "--shard-id", str(i), "--num-shards", str(n_gpus)]
        procs.append(subprocess.Popen(cmd, env=env))

    rcs = [p.wait() for p in procs]
    failed = [i for i, rc in enumerate(rcs) if rc != 0]
    if failed:
        raise SystemExit(
            Fore.RED
            + f"stage-3 worker(s) on GPU index {failed} failed "
            + "(see output above); aggregation skipped, cache may be incomplete."
            + Fore.RESET)

    # Every ablation cell is cached now → this pass hits the CACHED branch
    # (no GPU ablation, no foundation-model load) and only runs the aggregation
    # tail + the small predictor inference on a single GPU.
    print(Fore.CYAN + "Coordinator: all shards done — aggregating." + Fore.RESET)
    run_ablation(args, device, shard_id=None, num_shards=1)


def main():
    args = parse_args()

    global CACHE_ROOT
    CACHE_ROOT = args.cache_root

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    is_worker = args.shard_id is not None
    n_visible = torch.cuda.device_count() if device == "cuda" else 0
    n_gpus = n_visible if args.num_gpus == 0 else min(args.num_gpus, n_visible)

    if device == "cuda" and n_gpus > 1 and not is_worker:
        _run_coordinator(args, device, n_gpus, n_visible)
    else:
        run_ablation(args, device,
                     shard_id=args.shard_id,
                     num_shards=(args.num_shards if is_worker else 1))


def run_ablation(args, device: str, shard_id: Optional[int] = None,
                 num_shards: int = 1) -> None:
    global CACHE_ROOT
    CACHE_ROOT = args.cache_root

    # ---- Load predictor + derive window grid + horizon grid ------------------
    predictor_dir = args.predictor_dir or find_latest_predictor_run(
        args.predictor_cache_root)
    print(Fore.CYAN + f"Predictor run: {predictor_dir}" + Fore.RESET)
    predictor, pred_cfg = load_predictor(predictor_dir, device)
    window_grid: List[int] = list(pred_cfg["window_grid"])
    horizon_grid: List[int] = list(pred_cfg["horizon_grid"])
    pred_ctx_len: int = int(pred_cfg["context_length"])
    pred_curve_metric: str = pred_cfg.get("curve_metric", "mae")
    training_objective: str = pred_cfg.get("training_objective", "curve")
    print(Fore.CYAN
          + f"  window_grid ({len(window_grid)}): {window_grid}\n"
          + f"  horizon_grid: {horizon_grid}\n"
          + f"  context_length={pred_ctx_len}  curve_metric={pred_curve_metric}\n"
          + f"  training_objective={training_objective}\n"
          + f"  label_model={pred_cfg.get('label_model')}"
          + Fore.RESET)

    window_sizes = sorted(set(window_grid))
    short_mode = args.short_context_mode
    print(Fore.CYAN + f"Device: {device}  |  ablation windows: {window_sizes}"
          + f"  |  short-context mode: {short_mode}" + Fore.RESET)

    models = [m for m in MODELS if (args.models is None or m[2] in args.models)]
    # Fail loud on an empty model filter. Otherwise the ablation runs over zero
    # models, writes nothing, and exits 0 — which downstream surfaces as an
    # opaque "No records for models [...]" in stage 4 (compare). A requested
    # display name that isn't in this MODELS catalog is the usual cause.
    if args.models is not None and not models:
        known = [m[2] for m in MODELS]
        raise SystemExit(
            Fore.RED
            + f"--models {args.models} matched none of this script's known models.\n"
            + f"  Known: {known}\n"
            + "  Add the missing entry to MODELS (and ensure its family has "
            + "load_*/predict_* wrappers + dispatch)." + Fore.RESET)
    datasets = [d for d in DATASETS if (args.datasets is None or d[2] in args.datasets)]

    # Smoke test: keep only a random subset of (dataset, term) entries. Seeded so
    # every sharded worker samples the identical subset (d_idx must stay stable
    # across processes). Sorted back to original order for a sane shard split.
    if args.test_datasets is not None and args.test_datasets < len(datasets):
        rng = random.Random(args.test_datasets_seed)
        datasets = sorted(rng.sample(datasets, args.test_datasets),
                          key=lambda d: DATASETS.index(d))
        print(Fore.YELLOW
              + f"SMOKE TEST: sampled {len(datasets)} datasets "
              + f"(seed={args.test_datasets_seed}): {[d[2] + '/' + d[1] for d in datasets]}"
              + Fore.RESET)

    # Deterministic per-family layout: run dir is the predictor's family
    # (= basename of predictor_dir), so re-running overwrites the same place.
    run_dir = CACHE_ROOT
    os.makedirs(run_dir, exist_ok=True)

    # ---- Resume estimate: how many (model, dataset, window) cells are already
    # on disk. Cheap file-stat scan, no dataset loading. The denominator is the
    # full grid, so cells that will be skipped (can't serve / model context
    # limits) are still counted — treat the percentage as a lower bound.
    include_full_native = (not args.no_cell_cache
                           and not args.no_full_native_baseline)
    n_planned = len(models) * len(datasets) * (
        len(window_sizes) + (1 if include_full_native else 0))
    n_cached = sum(
        1
        for _, _, model_short in models
        for _, term, dataset_display, _ in datasets
        for window_size in window_sizes
        if _result_cached(dataset_display, model_short, term, window_size)
    )
    if include_full_native:
        n_cached += sum(
            1
            for _, _, model_short in models
            for _, term, dataset_display, _ in datasets
            if _result_cached(dataset_display, model_short, term, FULL_NATIVE_WINDOW)
        )
    pct_cached = 100.0 * n_cached / n_planned if n_planned else 100.0
    print(Fore.CYAN
          + f"Resume: {n_cached}/{n_planned} cells cached ({pct_cached:.1f}% complete)"
          + Fore.RESET)

    ge_cache: Dict[Tuple[str, str], GiftEvalCache] = {}
    naive_baseline_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
    all_results: List[dict] = []

    # ----- Model -> Dataset -> Window ----------------------------------------
    for model_id, model_family, model_short in models:
        print(Fore.CYAN + "\n" + "=" * 78 + Fore.RESET)
        print(Fore.CYAN + f"  MODEL: {model_id}  ({model_family})" + Fore.RESET)
        print(Fore.CYAN + "=" * 78 + Fore.RESET)

        # Foundation models are loaded lazily + memoized: the first cell that
        # actually needs prediction triggers the load. This keeps a sharded
        # worker from loading models it owns no work for, and lets the
        # coordinator's all-cached aggregation pass skip GPU model loads
        # entirely. (timesfm loads per-cell; context_parroting needs nothing.)
        _handle = [None]

        def ensure_handle():
            if _handle[0] is None:
                _handle[0] = load_handle(model_family, model_id, device)
            return _handle[0]

        for d_idx, (ge_name, term, dataset_display, to_univariate) in enumerate(datasets):
            # Dataset-level sharding: each worker owns a round-robin slice of
            # datasets (all their windows), so each dataset is loaded by exactly
            # one process. Index is stable across workers (same filtered list).
            if num_shards > 1 and d_idx % num_shards != shard_id:
                continue
            ds_key = (ge_name, term)
            if ds_key not in ge_cache:
                print(Fore.CYAN + f"\n  Loading GiftEval: {ge_name}  term={term}" + Fore.RESET)
                # Guard construction so an unsupported (name, term) or a
                # to_univariate mismatch (raises in GiftEvalCache) skips this
                # dataset with a warning instead of sinking the whole run. Lets
                # newly-added configs (e.g. the *_with_missing datasets) self-heal
                # if a term/flag guess is off.
                try:
                    ge_dataset = GiftEvalDataset(name=ge_name, term=term,
                                                 to_univariate=to_univariate)
                    cache = GiftEvalCache(ge_dataset, dataset_display)
                except Exception as exc:  # noqa: BLE001
                    print(Fore.RED + f"    SKIP dataset {ge_name} term={term} "
                          f"({dataset_display}): {exc}" + Fore.RESET)
                    continue
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

            if not args.no_cell_cache and not args.no_full_native_baseline:
                _run_full_native_baseline(
                    cache, dataset_display, ge_name, term,
                    model_id, model_family, model_short,
                    args.batch_size, device, ensure_handle)

            for window_size in window_sizes:
                tag = f"{model_short} | {dataset_display} | t={term} | h={horizon} | w={window_size}"

                if _result_cached(dataset_display, model_short, term, window_size):
                    cached = _load_cached_result(dataset_display, model_short, term, window_size)
                    # Backfill the gluonts MASE into pre-existing cells WITHOUT
                    # re-inference: per-instance MAE is in the per_sample cache and
                    # the seasonal error comes from the data. Lets a plain re-run of
                    # the ablation populate `mase_gluonts` everywhere cheaply.
                    def _missing(key):
                        v = cached.get(key)
                        return v is None or (isinstance(v, float) and np.isnan(v))

                    changed = False
                    # Backfill when absent OR when the cached value predates the
                    # current gluonts-MASE version (e.g. the seasonal_error fix).
                    # Both paths are cheap: derive from the per-instance MAE cache,
                    # no TSFM re-inference.
                    _stale_ver = cached.get("_mase_gluonts_ver", 0) < MASE_GLUONTS_VER
                    if _missing("mase_gluonts") or _stale_ver:
                        mg = _backfill_mase_gluonts(
                            dataset_display, model_short, term, window_size,
                            cache, short_mode)
                        if mg is not None:
                            cached["mase_gluonts"] = mg
                            cached["_mase_gluonts_ver"] = MASE_GLUONTS_VER
                            changed = True
                    # A cached cell has no stored forecast objects, so the real
                    # gluonts machinery can't be re-run here. Stand in with the port
                    # `mase_gluonts` (they match to <1%; a forced --force 3 re-run
                    # recomputes the true value from fresh forecasts). Refresh the
                    # stand-in whenever the port was just re-derived.
                    if (_missing("mase_gluonts_real") or _stale_ver) \
                            and not _missing("mase_gluonts"):
                        cached["mase_gluonts_real"] = cached["mase_gluonts"]
                        # Audit marker: this cell's `_real` is the port stand-in,
                        # not the machinery — `--force 3` recomputes it for real.
                        cached["_mase_gluonts_real_standin"] = True
                        changed = True
                    if changed:
                        _save_result(dataset_display, model_short, term,
                                     window_size, cached)
                    print(Fore.WHITE + f"  CACHED  {tag}  -> MAE={cached['mae']:.6f}" + Fore.RESET)
                    all_results.append({
                        "model": model_id, "model_short": model_short, "model_family": model_family,
                        "dataset": ge_name, "dataset_display": dataset_display, "term": term,
                        "horizon": horizon, "window_size": window_size, **cached,
                    })
                    continue

                # In skip mode a window wider than every instance's context has no
                # servable instances -> skip. In pad mode we still evaluate it: each
                # instance contributes its available context (the curve flattens).
                if short_mode == "skip" and not cache.can_serve(window_size):
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

                if model_family == "toto" and window_size > TOTO_MAX_CONTEXT:
                    print(Fore.RED + f"  SKIP    {tag}  (Toto max context={TOTO_MAX_CONTEXT} < ws)"
                          + Fore.RESET)
                    continue

                if model_family == "flowstate" and window_size > FLOWSTATE_MAX_CONTEXT:
                    print(Fore.RED + f"  SKIP    {tag}  (FlowState max context={FLOWSTATE_MAX_CONTEXT} < ws)"
                          + Fore.RESET)
                    continue

                if model_family == "tirex" and window_size > TIREX_MAX_CONTEXT:
                    print(Fore.RED + f"  SKIP    {tag}  (TiRex max context={TIREX_MAX_CONTEXT} < ws)"
                          + Fore.RESET)
                    continue

                print(Fore.YELLOW + f"\n  > {tag}" + Fore.RESET)

                # Batch building (CPU->GPU copy, pinning, width-grouping) happens
                # OUTSIDE the timer: elapsed_seconds must capture pure forward-pass
                # cost so it lines up with the forward-only theoretical-FLOPs proxy
                # in the strategy comparison.
                try:
                    if short_mode == "pad":
                        groups = cache.build_batches_padded(
                            window_size, args.batch_size, device,
                            pin_memory=True, window_grid=window_sizes)
                    else:
                        batches, all_x_cpu, all_y_cpu, valid_indices = cache.build_batches(
                            window_size, args.batch_size, device, pin_memory=True)
                except RuntimeError as exc:
                    print(Fore.RED + f"    SKIP: {exc}" + Fore.RESET); continue

                # cuda.synchronize() brackets the timer so it measures completed GPU
                # work, not just enqueued kernel launches (forecast calls launch
                # async). No-op on CPU. Syncing right before t_start also drains any
                # outstanding work from the batch build so it can't leak in.
                def _sync():
                    if device == "cuda":
                        torch.cuda.synchronize()

                if short_mode == "pad":
                    # Run each width-group (its own moirai/timesfm runner), then
                    # stitch back into one instance-ordered result over ALL n_total.
                    results = []
                    effective_batch_sizes = []
                    _sync()
                    t_start = time.perf_counter()
                    for L, batches_L, _ax, _ay, idx_L in groups:
                        fr_L, tgts_L, effective_bs = _forecast_cell_dynamic(
                            model_family, ensure_handle(), model_id, batches_L,
                            L, horizon, device, args.batch_size,
                            flowstate_scale=cache.flowstate_scale)
                        results.append((idx_L, fr_L, tgts_L))
                        effective_batch_sizes.append(effective_bs)
                    n_valid = cache.n_total
                    fr, tgts = _merge_grouped(results, n_valid, horizon, device)
                    # Pad mode serves every instance in 0..n-1 order (see
                    # _merge_grouped), so the per-instance seasonal errors map 1:1.
                    se_cell = cache.seasonal_errors_gluonts
                    _sync()
                    elapsed = time.perf_counter() - t_start
                    # Uniform-width (NaN-left-padded) inputs, for plotting only.
                    # Built AFTER the timer stops — pure CPU plotting prep, not
                    # inference.
                    all_x_np = np.full((n_valid, window_size, 1), np.nan, dtype=np.float32)
                    for L, _b, ax_L, _ay, idx_L in groups:
                        all_x_np[idx_L, -int(L):, :] = ax_L.numpy()
                    all_x_cpu = torch.from_numpy(all_x_np)
                    print(f"    Samples: {n_valid} ({len(groups)} width-groups, pad)  "
                          f"W={window_size}  H={horizon}")
                else:
                    n_valid = all_x_cpu.shape[0]
                    print(f"    Samples: {n_valid}  Batches: {len(batches)}  W={window_size}  H={horizon}")
                    _sync()
                    t_start = time.perf_counter()
                    fr, tgts, effective_bs = _forecast_cell_dynamic(
                        model_family, ensure_handle(), model_id, batches,
                        window_size, horizon, device, args.batch_size,
                        flowstate_scale=cache.flowstate_scale)
                    effective_batch_sizes = [effective_bs]
                    _sync()
                    elapsed = time.perf_counter() - t_start
                    # Skip mode serves only instances with context >= window, in
                    # ascending index order (build_batches' valid_indices) — pick the
                    # matching per-instance seasonal errors for mase_gluonts.
                    se_cell = cache.seasonal_errors_gluonts[valid_indices]

                metrics = compute_all_metrics(fr, tgts, cache.naive_seasonal_mae_train,
                                              seasonal_errors=se_cell)
                # Fill the leaderboard-machinery MASE (`mase_gluonts_real`) now that
                # we hold the forecast objects. Served-instance order matches se_cell:
                # pad mode serves every instance (0..n-1); skip mode serves
                # valid_indices (context >= window, ascending).
                served_idx = (np.arange(cache.n_total) if short_mode == "pad"
                              else valid_indices)
                metrics["mase_gluonts_real"] = cell_mase_gluonts_real(
                    fr, cache, served_idx)
                metrics["_mase_gluonts_real_standin"] = bool(
                    np.isnan(metrics["mase_gluonts_real"]))
                metrics["_mase_gluonts_ver"] = MASE_GLUONTS_VER
                metrics["elapsed_seconds"] = round(elapsed, 3)
                metrics["horizon"] = horizon
                metrics["_dynamic_batch_sizes"] = effective_batch_sizes

                for k, v in metrics.items():
                    if isinstance(v, float):
                        print(Fore.YELLOW + f"    {k}: {v:.6f}" + Fore.RESET)
                print(Fore.MAGENTA + f"    TIME  {elapsed:.1f}s" + Fore.RESET)

                per_sample = compute_per_sample_metrics(fr, tgts, cache.naive_seasonal_mae_train,
                                                        seasonal_errors=se_cell)
                if not args.no_cell_cache:
                    _save_per_sample_metrics(dataset_display, model_short, term, window_size, per_sample)
                    # Persist the expensive inference result before generating
                    # optional diagnostic plots. A plotting failure must never
                    # force this cell to be inferred again on resume.
                    _save_result(dataset_display, model_short, term, window_size, metrics)

                if not args.no_plots:
                    try:
                        pred_3d = fr.median.detach().cpu().unsqueeze(-1)
                        tgt_3d = tgts.detach().cpu().unsqueeze(-1)
                        cell_dir = _cache_dir(dataset_display, model_short, term, window_size)
                        plot_sample_predictions(all_x_cpu, pred_3d, tgt_3d,
                                                model_short, window_size, horizon, cell_dir)
                        best_idx, worst_idx, psm = find_best_worst_samples(
                            pred_3d, tgt_3d, n=N_BEST_WORST)
                        plot_best_worst_samples(all_x_cpu, pred_3d, tgt_3d,
                                                best_idx, worst_idx, psm,
                                                model_short, window_size, horizon, cell_dir)
                        plot_error_distributions(per_sample, model_short, window_size, horizon,
                                                 cell_dir, naive_baseline=naive_bl)
                    except Exception as exc:  # noqa: BLE001
                        print(Fore.YELLOW + f"    WARNING: plots failed for {tag}: {exc}"
                              + Fore.RESET)

                all_results.append({
                    "model": model_id, "model_short": model_short, "model_family": model_family,
                    "dataset": ge_name, "dataset_display": dataset_display, "term": term,
                    "horizon": horizon, "window_size": window_size, **metrics,
                })

                del fr, tgts, all_x_cpu, per_sample
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        _handle[0] = None
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # A sharded worker only fills the per-cell cache; the coordinator runs the
    # aggregation tail once over the combined cache (see _run_coordinator).
    if shard_id is not None:
        print(Fore.GREEN
              + f"Worker shard {shard_id}/{num_shards} done." + Fore.RESET)
        return

    # ==========================================================================
    #  AGGREGATE + SAVE + COMPARISON PLOTS
    # ==========================================================================
    if not all_results:
        print(Fore.RED + "No results produced." + Fore.RESET); return

    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(run_dir, "results.csv")

    # run_dir is the SHARED general folder: run_all invokes v5 once per model,
    # so merge this run's rows into any results.csv left by earlier models
    # (dropping stale rows for the models we just recomputed) to keep a single
    # combined "general results" table across all models.
    merged_df = results_df
    if os.path.isfile(csv_path):
        try:
            prev = pd.read_csv(csv_path)
            cur_models = set(results_df["model_short"].unique())
            prev = prev[~prev["model_short"].isin(cur_models)]
            merged_df = pd.concat([prev, results_df], ignore_index=True)
        except Exception as exc:
            print(Fore.YELLOW + f"  Could not merge existing results.csv: {exc}"
                  + Fore.RESET)
    merged_df.to_csv(csv_path, index=False)
    print(Fore.GREEN + f"\n  Results CSV: {csv_path}" + Fore.RESET)

    # Summary plots span every model present in the combined table, not just
    # this run's, so the shared folder shows a true cross-model comparison.
    model_names_in_run = list(
        merged_df[["model", "model_short"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    # ---- Comparison plot per (model, dataset, term) -------------------------
    for model_id, model_family, model_short in models:
        compare_dir = os.path.join(run_dir, "models", model_short, "compare_real_vs_predicted")
        os.makedirs(compare_dir, exist_ok=True)
        compare_records: List[dict] = []
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
            # Parallel curves for the leaderboard (gluonts) MASE, so stage 4 can
            # drive the strategy comparison off any metric (--mase-metric):
            #   real_curve_gluonts       -> the cheap numpy port (`mase_gluonts`)
            #   real_curve_gluonts_real  -> the actual gluonts machinery
            #                               (`mase_gluonts_real`)
            def _curve_for(col):
                if col not in mdf.columns:
                    return np.full(len(window_grid), np.nan, dtype=np.float64)
                m = dict(zip(mdf["window_size"].values, mdf[col].values))
                return np.array([m.get(w, np.nan) for w in window_grid],
                                dtype=np.float64)

            real_curve_gluonts = _curve_for("mase_gluonts")
            real_curve_gluonts_real = _curve_for("mase_gluonts_real")
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
                training_objective=training_objective,
                batch_size=args.predictor_batch_size,
            )

            plot_path = ""
            if not args.no_plots:
                plot_path = plot_real_vs_predicted_curve(
                    window_grid, real_curve, predicted_curves,
                    model_short, dataset_display, term,
                    horizon_real, horizon_pred, "mase", compare_dir,
                    training_objective=training_objective,
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
                real_curve_gluonts=real_curve_gluonts,
                real_curve_gluonts_real=real_curve_gluonts_real,
                real_curve_zscored=real_z,
                predicted_curves=predicted_curves,
                predicted_mean=pred_mean,
                predicted_std=pred_std,
                training_objective=np.asarray(training_objective),
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
            merged_df, model_names_in_run, window_sizes,
            dataset_display, term, run_dir,
            naive_baseline=naive_baseline_cache.get(key),
        )

    # Merge naive baselines into the shared file (keyed by dataset/term, so
    # baselines from earlier model runs are preserved).
    naive_path = os.path.join(run_dir, "naive_baselines.json")
    naive_all: Dict[str, dict] = {}
    if os.path.isfile(naive_path):
        try:
            with open(naive_path) as f:
                naive_all = json.load(f)
        except (OSError, json.JSONDecodeError):
            naive_all = {}
    naive_all.update({f"{k[0]}/t{k[1]}": v for k, v in naive_baseline_cache.items()})
    with open(naive_path, "w") as f:
        json.dump(naive_all, f, indent=2)
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
