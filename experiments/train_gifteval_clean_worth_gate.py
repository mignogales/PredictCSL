#!/usr/bin/env python3
"""Train a shortening gate from GIFT-Eval train/validation data only.

The context selector remains frozen.  For rolling origins contained entirely in
the official training split, this script runs Chronos-2 at only two context
lengths: the selector proposal and the native context.  The resulting relative
MASE gain supervises a lightweight gate.  The official validation extension is
used only to calibrate gate thresholds, and the official test labels are never
used for fitting or threshold selection.

The final evaluation reuses the previously computed per-instance test oracle
cache.  This keeps the expensive test forecasts identical to the main window
ablation and, importantly, does not expose them to the training procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import torch
from gift_eval.data import Dataset as GiftEvalDataset
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import datasets_config, models_config
from experiments.compare_window_strategies_gifteval import (
    DEFAULT_PATCH_SIZES,
    theoretical_flops,
)
from experiments.gifteval_mase import get_seasonality, seasonal_error
from experiments.predict_context_length import MambaContextLength
from experiments.test_window_ablation_gifteval_v5 import (
    _as_horizon_matrix,
    _closest_horizon_idx,
    _forecast_cell_dynamic,
    _full_native_context_cap,
    _prepare_predictor_inputs,
    flowstate_scale_factor,
)
from experiments.gifteval_inference_recipes import preserves_missing
from experiments.train_shortening_worth_gate import features, load_selector, predict


SELECTOR_NO_DECISION_CELLS = {"Solar-W/short", "CarParts/short"}


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _one_dimensional_target(entry: dict) -> np.ndarray:
    target = np.asarray(entry["target"], dtype=np.float32)
    if target.ndim != 1:
        raise ValueError(f"Expected a univariate target, got {target.shape}")
    return target


def _entry_map(entries: Iterable[dict]) -> dict[str, dict]:
    return {str(entry.get("item_id", f"item_{i}")): entry for i, entry in enumerate(entries)}


def _load_frozen_selector(checkpoint: str, legacy_config: str, device: str):
    if not legacy_config:
        return load_selector(checkpoint, device)
    config = json.loads(Path(legacy_config).read_text())
    model = MambaContextLength(
        config["context_length"],
        config["patch_length"],
        config["d_model"],
        config["num_hidden_layers"],
        config["d_state"],
        config["d_conv"],
        config["expand"],
        config["dropout"],
        config["mask_ratio"],
        config["n_windows"],
        config["n_horizons"],
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    bundle = {
        "windows": config["window_grid"],
        "horizons": config["horizon_grid"],
        "model_config": config,
    }
    return model.to(device).eval(), bundle


def _load_forecaster(model_id: str, family: str, device: str):
    from experiments.test_window_ablation_gifteval_v5 import load_handle

    return load_handle(family, model_id, device)


def _median_forecasts(
    handle,
    model_id: str,
    family: str,
    contexts: list[np.ndarray],
    labels: list[np.ndarray],
    horizon: int,
    batch_size: int,
    device: str,
    flowstate_scale: float,
) -> np.ndarray:
    if not contexts:
        return np.empty((0, horizon), dtype=np.float32)
    predictions = np.empty((len(contexts), horizon), dtype=np.float32)
    source_contexts = (
        contexts if preserves_missing(family)
        else [np.nan_to_num(context, nan=0.0) for context in contexts]
    )
    widths = np.asarray([len(context) for context in source_contexts])
    for width in np.unique(widths):
        indices = np.flatnonzero(widths == width)
        x = torch.from_numpy(np.stack([source_contexts[i] for i in indices])).unsqueeze(-1)
        y = torch.from_numpy(np.stack([labels[i] for i in indices])).unsqueeze(-1)
        if device == "cuda":
            x = x.pin_memory()
            y = y.pin_memory()
        batches = [
            {"x": x[start : start + batch_size], "y": y[start : start + batch_size]}
            for start in range(0, len(indices), batch_size)
        ]
        forecast, _, _ = _forecast_cell_dynamic(
            family,
            handle,
            model_id,
            batches,
            int(width),
            int(horizon),
            device,
            int(batch_size),
            flowstate_scale=flowstate_scale,
        )
        median = _as_horizon_matrix(forecast.median, horizon, "median")
        predictions[indices] = median.detach().float().cpu().numpy()
    return predictions


def _per_instance_mase(
    predictions: np.ndarray,
    labels: list[np.ndarray],
    contexts: list[np.ndarray],
    freq: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full(len(labels), np.nan, dtype=np.float64)
    counts = np.zeros(len(labels), dtype=np.int64)
    season = get_seasonality(freq)
    for i, (forecast, label, context) in enumerate(zip(predictions, labels, contexts)):
        label = np.asarray(label, dtype=np.float64)
        forecast = np.asarray(forecast, dtype=np.float64)
        valid = np.isfinite(label) & np.isfinite(forecast)
        scale = seasonal_error(context, season)
        if valid.any() and np.isfinite(scale) and scale > 0:
            values[i] = np.mean(np.abs(label[valid] - forecast[valid]) / scale)
            counts[i] = int(valid.sum())
    return values, counts


def _rolling_origins(
    train_entries: list[dict],
    validation_entries: list[dict],
    horizon: int,
    origins_per_series: int,
    max_items: int,
    min_context: int,
    cell: str,
    seed: int,
):
    train_map = _entry_map(train_entries)
    validation_map = _entry_map(validation_entries)
    item_ids = sorted(set(train_map) & set(validation_map))
    if max_items > 0 and len(item_ids) > max_items:
        rng = np.random.RandomState(_stable_seed(cell, seed))
        item_ids = sorted(rng.choice(item_ids, max_items, replace=False).tolist())

    contexts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    splits: list[str] = []
    items: list[str] = []
    orders: list[int] = []
    for item_id in item_ids:
        train_target = _one_dimensional_target(train_map[item_id])
        validation_target = _one_dimensional_target(validation_map[item_id])
        cutoffs = []
        for step in range(origins_per_series, 0, -1):
            cutoff = len(train_target) - step * horizon
            if cutoff >= min_context and cutoff + horizon <= len(train_target):
                cutoffs.append(cutoff)
        for order, cutoff in enumerate(sorted(set(cutoffs))):
            contexts.append(train_target[:cutoff].copy())
            labels.append(train_target[cutoff : cutoff + horizon].copy())
            splits.append("train")
            items.append(item_id)
            orders.append(order)

        # GiftEval's validation target extends the training target by one
        # prediction horizon.  That extension is the calibration label.
        val_label = validation_target[len(train_target) : len(train_target) + horizon]
        if len(train_target) >= min_context and len(val_label) == horizon:
            contexts.append(train_target.copy())
            labels.append(val_label.copy())
            splits.append("val")
            items.append(item_id)
            orders.append(len(cutoffs))
    return contexts, labels, splits, items, orders


def _cell_path(root: Path, display: str, term: str) -> Path:
    safe = display.replace("/", "_")
    return root / "cells" / f"{safe}__{term}.npz"


def _prepared_cell_path(root: Path, display: str, term: str) -> Path:
    safe = display.replace("/", "_")
    return root / "prepared_cells" / f"{safe}__{term}.npz"


def prepare_cell_oracle(
    spec,
    selector,
    selector_bundle: dict,
    family: str,
    output_root: Path,
    device: str,
    origins_per_series: int,
    max_items: int,
    min_context: int,
    seed: int,
    force: bool,
) -> dict:
    """Build rolling origins and frozen-selector features without a TSFM.

    This stage intentionally runs in the Mamba-capable main environment.  The
    serialized contexts are consumed by ``forecast_prepared_cell`` in a model's
    native environment, which is necessary for Toto and TiRex.
    """
    ge_name, term, display, to_univariate = spec
    final_path = _cell_path(output_root, display, term)
    path = _prepared_cell_path(output_root, display, term)
    if final_path.exists() and not force:
        with np.load(final_path) as cached:
            return {
                "cell": f"{display}/{term}",
                "status": "oracle_cached",
                "n": int(len(cached["split"])),
            }
    if path.exists() and not force:
        with np.load(path, allow_pickle=True) as cached:
            return {
                "cell": f"{display}/{term}",
                "status": "prepared_cached",
                "n": int(len(cached["split"])),
            }

    dataset = GiftEvalDataset(name=ge_name, term=term, to_univariate=to_univariate)
    horizon = int(dataset.prediction_length)
    context_cap = _full_native_context_cap(
        family, horizon, models_config.context_limit(family)
    )
    cell = f"{display}/{term}"
    contexts, labels, splits, items, orders = _rolling_origins(
        list(dataset.training_dataset),
        list(dataset.validation_dataset),
        horizon,
        origins_per_series,
        max_items,
        min_context,
        cell,
        seed,
    )
    if not contexts:
        return {"cell": cell, "status": "empty", "n": 0}

    windows = np.asarray(selector_bundle["windows"], dtype=np.int64)
    horizons = list(selector_bundle["horizons"])
    horizon_idx = _closest_horizon_idx(horizon, horizons)
    predictor_context = int(selector_bundle["model_config"]["context_length"])
    prepared = _prepare_predictor_inputs(contexts, predictor_context)
    predicted_curves = predict(
        selector,
        prepared,
        np.full(len(contexts), horizon_idx, dtype=np.int64),
        device,
    )
    x, action = features(
        prepared,
        predicted_curves,
        np.full(len(contexts), horizons[horizon_idx], dtype=np.int64),
        windows,
    )
    native_context = np.asarray(
        [min(len(context), context_cap) for context in contexts], dtype=np.int64
    )
    proposed_context = np.minimum(windows[action], native_context)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        contexts=np.asarray(contexts, dtype=object),
        labels=np.asarray(labels, dtype=object),
        x=x,
        predicted_curves=predicted_curves.astype(np.float32),
        action=action,
        proposed_context=proposed_context,
        native_context=native_context,
        split=np.asarray(splits),
        item=np.asarray(items),
        origin_order=np.asarray(orders, dtype=np.int64),
        cell=np.asarray([cell] * len(contexts)),
        horizon=np.asarray([horizon] * len(contexts), dtype=np.int64),
    )
    return {
        "cell": cell,
        "status": "prepared",
        "n": len(contexts),
        "n_train": int(np.sum(np.asarray(splits) == "train")),
        "n_val": int(np.sum(np.asarray(splits) == "val")),
    }


def forecast_prepared_cell(
    spec,
    forecaster,
    model_id: str,
    family: str,
    output_root: Path,
    device: str,
    batch_size: int,
    force: bool,
) -> dict:
    """Forecast a prepared selector cell inside the TSFM-native environment."""
    ge_name, term, display, to_univariate = spec
    path = _cell_path(output_root, display, term)
    prepared_path = _prepared_cell_path(output_root, display, term)
    cell = f"{display}/{term}"
    if path.exists() and not force:
        with np.load(path) as cached:
            return {"cell": cell, "status": "cached", "n": int(len(cached["split"]))}
    if not prepared_path.exists():
        return {"cell": cell, "status": "missing_prepared", "n": 0}

    with np.load(prepared_path, allow_pickle=True) as data:
        contexts = [np.asarray(value, dtype=np.float32) for value in data["contexts"]]
        labels = [np.asarray(value, dtype=np.float32) for value in data["labels"]]
        payload = {
            key: data[key].copy()
            for key in (
                "x", "predicted_curves", "action", "proposed_context",
                "native_context", "split", "item", "origin_order", "cell", "horizon",
            )
        }
    horizon = int(payload["horizon"][0])
    proposed_context = payload["proposed_context"]
    native_context = payload["native_context"]
    proposed_inputs = [
        context[-int(width):] for context, width in zip(contexts, proposed_context)
    ]
    native_inputs = [
        context[-int(width):] for context, width in zip(contexts, native_context)
    ]
    dataset = GiftEvalDataset(name=ge_name, term=term, to_univariate=to_univariate)
    started = time.time()
    forecasts = _median_forecasts(
        forecaster,
        model_id,
        family,
        proposed_inputs + native_inputs,
        labels + labels,
        horizon,
        batch_size,
        device,
        flowstate_scale_factor(dataset.freq, ge_name),
    )
    proposed_forecast = forecasts[:len(contexts)]
    native_forecast = forecasts[len(contexts):]
    proposed_mase, proposed_count = _per_instance_mase(
        proposed_forecast, labels, contexts, dataset.freq
    )
    native_mase, native_count = _per_instance_mase(
        native_forecast, labels, contexts, dataset.freq
    )
    native_flops = np.asarray(
        [theoretical_flops(model_id, int(width), horizon, DEFAULT_PATCH_SIZES)
         for width in native_context],
        dtype=np.float64,
    )
    proposed_flops = np.asarray(
        [theoretical_flops(model_id, int(width), horizon, DEFAULT_PATCH_SIZES)
         for width in proposed_context],
        dtype=np.float64,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **payload,
        proposed_mase=proposed_mase,
        native_mase=native_mase,
        proposed_count=proposed_count,
        native_count=native_count,
        proposed_flops=proposed_flops,
        native_flops=native_flops,
    )
    return {
        "cell": cell,
        "status": "built",
        "n": len(contexts),
        "n_train": int(np.sum(payload["split"] == "train")),
        "n_val": int(np.sum(payload["split"] == "val")),
        "seconds": time.time() - started,
    }


def build_cell_oracle(
    spec,
    selector,
    selector_bundle: dict,
    forecaster,
    model_id: str,
    family: str,
    output_root: Path,
    device: str,
    origins_per_series: int,
    max_items: int,
    min_context: int,
    batch_size: int,
    seed: int,
    force: bool,
) -> dict:
    prepared = prepare_cell_oracle(
        spec, selector, selector_bundle, family, output_root, device,
        origins_per_series, max_items, min_context, seed, force,
    )
    if prepared["status"] in {"empty", "oracle_cached"}:
        return prepared
    return forecast_prepared_cell(
        spec, forecaster, model_id, family, output_root, device, batch_size, force,
    )


def _concatenate_oracle(root: Path) -> dict[str, np.ndarray]:
    keys = [
        "x", "action", "proposed_context", "native_context", "proposed_mase",
        "native_mase", "proposed_count", "native_count", "proposed_flops",
        "native_flops", "split", "item", "origin_order", "cell", "horizon",
    ]
    chunks = {key: [] for key in keys}
    for path in sorted((root / "cells").glob("*.npz")):
        with np.load(path) as data:
            for key in keys:
                chunks[key].append(data[key])
    if not chunks["x"]:
        raise RuntimeError(f"No oracle cells found under {root / 'cells'}")
    return {key: np.concatenate(values) for key, values in chunks.items()}


def _policy_metrics(
    native: np.ndarray,
    proposed: np.ndarray,
    native_flops: np.ndarray,
    proposed_flops: np.ndarray,
    eligible: np.ndarray,
    score: np.ndarray,
    threshold: float,
    cells=None,
) -> dict:
    use = eligible & (score >= threshold)
    final = np.where(use, proposed, native)
    ratio = final / np.maximum(native, 1e-12)
    saved = np.where(use, native_flops - proposed_flops, 0.0)
    result = {
        "threshold": float(threshold),
        "coverage": float(np.mean(use)),
        "eligible_coverage": float(np.mean(use[eligible])) if eligible.any() else 0.0,
        "mean_mase_ratio": float(np.mean(ratio)),
        "relative_mase_change_pct": float(100.0 * (np.mean(ratio) - 1.0)),
        "harm_rate": float(np.mean(ratio > 1.0)),
        "harm5_rate": float(np.mean(ratio > 1.05)),
        "improvement_rate": float(np.mean(ratio < 1.0)),
        "flops_saved_pct": float(100.0 * saved.sum() / np.maximum(native_flops.sum(), 1e-12)),
    }
    if cells is not None:
        cell_ratios = []
        cell_harm5 = []
        for cell in np.unique(cells):
            mask = cells == cell
            cell_ratios.append(float(np.mean(final[mask]) / np.maximum(np.mean(native[mask]), 1e-12)))
            cell_harm5.append(float(np.mean(ratio[mask] > 1.05)))
        cell_ratios = np.asarray(cell_ratios)
        result.update({
            "n_cells": int(len(cell_ratios)),
            "macro_cell_mase_ratio": float(np.mean(cell_ratios)),
            "macro_cell_mase_change_pct": float(100.0 * np.mean(cell_ratios - 1.0)),
            "geomean_cell_mase_change_pct": float(
                100.0 * np.expm1(np.mean(np.log(cell_ratios)))
            ),
            "cell_win_rate": float(np.mean(cell_ratios < 1.0)),
            "macro_cell_harm5_rate": float(np.mean(cell_harm5)),
        })
    return result


def _calibrate_profiles(
    native: np.ndarray,
    proposed: np.ndarray,
    native_flops: np.ndarray,
    proposed_flops: np.ndarray,
    eligible: np.ndarray,
    score: np.ndarray,
    cells=None,
) -> dict:
    finite = score[np.isfinite(score)]
    candidates = np.unique(np.r_[np.inf, np.quantile(finite, np.linspace(0, 1, 501))])
    specs = {
        "conservative": (0.005, 1.000),
        "balanced": (0.010, 1.001),
        "aggressive": (0.030, 1.005),
    }
    profiles = {}
    for name, (harm_limit, mean_limit) in specs.items():
        rows = [
            _policy_metrics(
                native, proposed, native_flops, proposed_flops,
                eligible, score, threshold, cells,
            )
            for threshold in candidates
        ]
        feasible = [
            row for row in rows
            if (
                row.get("macro_cell_harm5_rate", row["harm5_rate"]) <= harm_limit
                and row.get("macro_cell_mase_ratio", row["mean_mase_ratio"]) <= mean_limit
            )
        ]
        profiles[name] = (
            max(feasible, key=lambda row: (row["flops_saved_pct"], -row["mean_mase_ratio"]))
            if feasible
            else _policy_metrics(
                native, proposed, native_flops, proposed_flops,
                eligible, score, np.inf, cells,
            )
        )
    return profiles


def _tree_mean_std(model: ExtraTreesRegressor, x: np.ndarray, chunk: int = 8192):
    mean = np.empty(len(x), dtype=np.float64)
    std = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), chunk):
        stop = min(len(x), start + chunk)
        values = np.stack([tree.predict(x[start:stop]) for tree in model.estimators_])
        mean[start:stop] = values.mean(0)
        std[start:stop] = values.std(0)
    return mean, std


def _test_arrays(path: str, exclude_cells=SELECTOR_NO_DECISION_CELLS):
    with np.load(path) as data:
        x = data["x"].copy()
        action = data["action"].copy()
        error = data["error"].copy()
        save = data["save"].copy()
        cells = data["cells"].copy()
    row = np.arange(len(x))
    native = error[:, -1]
    proposed = error[row, action]
    eligible = action < error.shape[1] - 1
    native_flops = save[:, 1]
    proposed_flops = native_flops - save[:, 0]
    valid = (
        np.isfinite(native) & (native > 0) & np.isfinite(proposed)
        & np.isfinite(native_flops) & (native_flops > 0)
    )
    if exclude_cells:
        valid &= ~np.isin(cells, list(exclude_cells))
    return {
        "x": x[valid], "native": native[valid], "proposed": proposed[valid],
        "eligible": eligible[valid], "native_flops": native_flops[valid],
        "proposed_flops": proposed_flops[valid], "cells": cells[valid],
        "action": action[valid],
    }


def _target_encoding_features(data, gain, train, test):
    """Leakage-safe cell/action performance priors from training origins.

    Validation and test rows receive statistics computed from all clean training
    origins.  A training row receives leave-one-out statistics, preventing its
    own label from appearing in its features.  Bayesian smoothing keeps tiny
    cells and rare selector actions close to the global/cell prior.
    """
    global_values = gain[train]
    global_mean = float(np.mean(global_values))
    global_second = float(np.mean(global_values ** 2))
    global_harm = float(np.mean(global_values < -0.05))
    global_positive = float(np.mean(global_values > 0.0))
    cell_alpha = 20.0
    action_alpha = 10.0

    def sufficient(mask):
        values = gain[mask]
        return {
            "n": int(len(values)),
            "sum": float(values.sum()),
            "sum2": float(np.sum(values ** 2)),
            "harm": float(np.sum(values < -0.05)),
            "positive": float(np.sum(values > 0.0)),
        }

    cells = data["cell"]
    actions = data["action"]
    cell_stats = {
        cell: sufficient(train & (cells == cell)) for cell in np.unique(cells)
    }
    cell_action_stats = {
        (cell, int(action)): sufficient(train & (cells == cell) & (actions == action))
        for cell in np.unique(cells)
        for action in np.unique(actions[cells == cell])
    }

    def smoothed(stats, prior, alpha):
        denom = stats["n"] + alpha
        mean = (stats["sum"] + alpha * prior[0]) / denom
        second = (stats["sum2"] + alpha * prior[1]) / denom
        return (
            mean,
            math.sqrt(max(0.0, second - mean * mean)),
            (stats["harm"] + alpha * prior[2]) / denom,
            (stats["positive"] + alpha * prior[3]) / denom,
            math.log1p(stats["n"]),
        )

    def encode(row_cells, row_actions, row_indices=None):
        result = np.empty((len(row_cells), 10), dtype=np.float32)
        for j, (cell, action) in enumerate(zip(row_cells, row_actions)):
            cell_raw = dict(cell_stats.get(cell, {"n": 0, "sum": 0.0, "sum2": 0.0, "harm": 0.0, "positive": 0.0}))
            action_raw = dict(cell_action_stats.get((cell, int(action)), {"n": 0, "sum": 0.0, "sum2": 0.0, "harm": 0.0, "positive": 0.0}))
            if row_indices is not None:
                i = int(row_indices[j])
                if train[i]:
                    value = float(gain[i])
                    for stats in (cell_raw, action_raw):
                        stats["n"] -= 1
                        stats["sum"] -= value
                        stats["sum2"] -= value * value
                        stats["harm"] -= float(value < -0.05)
                        stats["positive"] -= float(value > 0.0)
            cell_encoded = smoothed(
                cell_raw,
                (global_mean, global_second, global_harm, global_positive),
                cell_alpha,
            )
            action_encoded = smoothed(
                action_raw,
                (cell_encoded[0], cell_encoded[0] ** 2 + cell_encoded[1] ** 2,
                 cell_encoded[2], cell_encoded[3]),
                action_alpha,
            )
            result[j] = np.asarray(cell_encoded + action_encoded, dtype=np.float32)
        return result

    clean_encoding = encode(cells, actions, np.arange(len(cells)))
    test_encoding = encode(test["cells"], test["action"])
    return clean_encoding, test_encoding


def train_and_evaluate(
    root: Path,
    test_cache: str,
    seed: int,
    n_estimators: int,
    use_target_encoding: bool = False,
    model_short: str = "Chronos2-Small",
):
    data = _concatenate_oracle(root)
    gain = (data["native_mase"] - data["proposed_mase"]) / np.maximum(
        data["native_mase"], 1e-12
    )
    valid = (
        np.isfinite(data["native_mase"]) & (data["native_mase"] > 0)
        & np.isfinite(data["proposed_mase"])
        & (data["native_count"] > 0) & (data["proposed_count"] > 0)
    )
    eligible = data["proposed_context"] < data["native_context"]
    train = valid & eligible & (data["split"] == "train")
    val = valid & (data["split"] == "val")
    val_eligible = val & eligible
    test = _test_arrays(test_cache)
    if data["x"].shape[1] != test["x"].shape[1]:
        raise ValueError(
            f"Feature mismatch: clean={data['x'].shape[1]}, test={test['x'].shape[1]}"
        )
    if use_target_encoding:
        clean_encoding, test_encoding = _target_encoding_features(data, gain, train, test)
        clean_x = np.column_stack([data["x"], clean_encoding]).astype(np.float32)
        test_x = np.column_stack([test["x"], test_encoding]).astype(np.float32)
        feature_description = f"{data['x'].shape[1]} selector/series features plus 10 leave-one-out-smoothed cell/action performance priors"
    else:
        clean_x = data["x"]
        test_x = test["x"]
        feature_description = f"{data['x'].shape[1]} selector/series features"

    candidates = {}
    fitted = {}
    train_cells = data["cell"][train]
    cell_counts = {cell: int(np.sum(train_cells == cell)) for cell in np.unique(train_cells)}
    train_weight = np.asarray([1.0 / cell_counts[cell] for cell in train_cells])
    train_weight *= len(train_weight) / train_weight.sum()
    for leaf in (2, 8, 24):
        name = f"extra_trees_leaf{leaf}"
        regressor = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=leaf,
            max_features=0.7,
            n_jobs=-1,
            random_state=seed + leaf,
        )
        regressor.fit(clean_x[train], gain[train], sample_weight=train_weight)
        val_mean, val_std = _tree_mean_std(regressor, clean_x[val])
        test_mean, test_std = _tree_mean_std(regressor, test_x)
        fitted[name] = regressor
        for uncertainty_penalty in (0.0, 0.25, 0.5, 1.0):
            score_name = f"{name}_lcb{uncertainty_penalty:g}"
            candidates[score_name] = {
                "model": name,
                "kind": "gain_lcb",
                "uncertainty_penalty": uncertainty_penalty,
                "val_score": val_mean - uncertainty_penalty * val_std,
                "test_score": test_mean - uncertainty_penalty * test_std,
            }

    harm = gain < -0.05
    classifier = ExtraTreesClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=8,
        max_features=0.7,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed + 1000,
    )
    classifier.fit(clean_x[train], harm[train], sample_weight=train_weight)
    fitted["harm_classifier"] = classifier
    val_harm = classifier.predict_proba(clean_x[val])[:, 1]
    test_harm = classifier.predict_proba(test_x)[:, 1]
    base = candidates["extra_trees_leaf8_lcb0"]
    for penalty in (0.05, 0.10, 0.20, 0.40):
        name = f"gain_minus_{penalty:g}_harm"
        candidates[name] = {
            "model": "extra_trees_leaf8+harm_classifier",
            "kind": "risk_adjusted_gain",
            "harm_penalty": penalty,
            "val_score": base["val_score"] - penalty * val_harm,
            "test_score": base["test_score"] - penalty * test_harm,
        }

    report_candidates = {}
    for name, candidate in candidates.items():
        profiles = _calibrate_profiles(
            data["native_mase"][val], data["proposed_mase"][val],
            data["native_flops"][val], data["proposed_flops"][val],
            eligible[val], candidate["val_score"], data["cell"][val],
        )
        result = {"profiles": {}}
        for profile, validation_metrics in profiles.items():
            result["profiles"][profile] = {
                "validation": validation_metrics,
                "test": _policy_metrics(
                    test["native"], test["proposed"], test["native_flops"],
                    test["proposed_flops"], test["eligible"],
                    candidate["test_score"], validation_metrics["threshold"],
                    test["cells"],
                ),
            }
        result["raw_validation"] = _policy_metrics(
            data["native_mase"][val], data["proposed_mase"][val],
            data["native_flops"][val], data["proposed_flops"][val],
            eligible[val], candidate["val_score"], -np.inf, data["cell"][val],
        )
        result["raw_test"] = _policy_metrics(
            test["native"], test["proposed"], test["native_flops"],
            test["proposed_flops"], test["eligible"], candidate["test_score"], -np.inf,
            test["cells"],
        )
        report_candidates[name] = result

    # Choose exclusively by the balanced validation profile.  The test metrics
    # above are diagnostics and cannot affect this selection.
    selected = max(
        report_candidates,
        key=lambda name: (
            report_candidates[name]["profiles"]["balanced"]["validation"]["flops_saved_pct"],
            -report_candidates[name]["profiles"]["balanced"]["validation"]["macro_cell_mase_ratio"],
        ),
    )
    metadata = {
        key: value for key, value in candidates[selected].items()
        if key not in {"val_score", "test_score"}
    }
    selected_models = {
        name: model for name, model in fitted.items()
        if name in metadata["model"]
    }
    joblib.dump(
        {
            "models": selected_models,
            "score": metadata,
            "profiles": {
                profile: values["validation"]
                for profile, values in report_candidates[selected]["profiles"].items()
            },
        },
        root / "gate.joblib",
        compress=3,
    )
    report = {
        "method": "frozen selector + clean GIFT train rolling-origin worth gate",
        "model": model_short,
        "training_and_calibration_weighting": "each GIFT dataset cell has equal total training weight; thresholds constrain cell-macro validation risk",
        "features": feature_description,
        "test_labels_used_for_training_or_calibration": False,
        "benchmark_classification": "fine-tuned, no test-data leakage (not zero-shot)",
        "selector": "frozen",
        "n_cells_oracle": int(len(np.unique(data["cell"]))),
        "n_train": int(train.sum()),
        "n_validation": int(val.sum()),
        "n_validation_eligible": int(val_eligible.sum()),
        "n_test": int(len(test["x"])),
        "n_test_cells": int(len(np.unique(test["cells"]))),
        "excluded_no_decision_test_cells": sorted(SELECTOR_NO_DECISION_CELLS),
        "selected_from_validation": selected,
        "selected_result": report_candidates[selected],
        "candidates": report_candidates,
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--legacy-config", default="")
    parser.add_argument("--model-short", default="Chronos2-Small")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stage",
        choices=["prepare", "forecast", "oracle", "train", "all"],
        default="all",
    )
    parser.add_argument("--origins-per-series", type=int, default=3)
    parser.add_argument("--max-items-per-cell", type=int, default=64)
    parser.add_argument("--min-context", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--target-encoding", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=[])
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_specs = {
        display: (model_id, family)
        for model_id, family, display in models_config.models_to_run()
    }
    if args.model_short not in model_specs:
        raise ValueError(
            f"Unknown run model {args.model_short!r}; choose from {sorted(model_specs)}"
        )
    model_id, family = model_specs[args.model_short]
    specs = datasets_config.datasets_to_run()
    if args.datasets:
        wanted = set(args.datasets)
        specs = [spec for spec in specs if spec[2] in wanted or f"{spec[2]}/{spec[1]}" in wanted]

    if args.stage == "prepare":
        selector, bundle = _load_frozen_selector(
            args.checkpoint, args.legacy_config, args.device
        )
        manifest = []
        for number, spec in enumerate(specs, 1):
            try:
                result = prepare_cell_oracle(
                    spec, selector, bundle, family, output, args.device,
                    args.origins_per_series, args.max_items_per_cell,
                    args.min_context, args.seed, args.force,
                )
            except Exception as exc:
                result = {
                    "cell": f"{spec[2]}/{spec[1]}",
                    "status": "error",
                    "error": repr(exc),
                }
            manifest.append(result)
            (output / "prepare_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            print(f"[{number}/{len(specs)}] {json.dumps(result)}", flush=True)
        del selector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.stage == "forecast":
        forecaster = _load_forecaster(model_id, family, args.device)
        manifest = []
        for number, spec in enumerate(specs, 1):
            try:
                result = forecast_prepared_cell(
                    spec, forecaster, model_id, family, output,
                    args.device, args.batch_size, args.force,
                )
            except Exception as exc:
                result = {
                    "cell": f"{spec[2]}/{spec[1]}",
                    "status": "error",
                    "error": repr(exc),
                }
            manifest.append(result)
            (output / "oracle_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            print(f"[{number}/{len(specs)}] {json.dumps(result)}", flush=True)
        del forecaster
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.stage in {"oracle", "all"}:
        selector, bundle = _load_frozen_selector(
            args.checkpoint, args.legacy_config, args.device
        )
        forecaster = _load_forecaster(model_id, family, args.device)
        manifest = []
        for number, spec in enumerate(specs, 1):
            try:
                result = build_cell_oracle(
                    spec, selector, bundle, forecaster, model_id, family,
                    output, args.device,
                    args.origins_per_series, args.max_items_per_cell,
                    args.min_context, args.batch_size, args.seed, args.force,
                )
            except Exception as exc:
                result = {
                    "cell": f"{spec[2]}/{spec[1]}",
                    "status": "error",
                    "error": repr(exc),
                }
            manifest.append(result)
            (output / "oracle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"[{number}/{len(specs)}] {json.dumps(result)}", flush=True)
        del forecaster, selector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.stage in {"train", "all"}:
        report = train_and_evaluate(
            output,
            args.test_cache,
            args.seed,
            args.n_estimators,
            use_target_encoding=args.target_encoding,
            model_short=args.model_short,
        )
        print(json.dumps(report["selected_result"], indent=2), flush=True)


if __name__ == "__main__":
    main()
