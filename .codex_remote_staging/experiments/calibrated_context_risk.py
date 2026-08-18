#!/usr/bin/env python3
"""Zero-shot context selection by calibrated expected forecast risk.

The original PredictCSL target is the realized argmin of one noisy forecast
error curve.  This experiment instead learns the model-specific log error
ratio of every candidate window relative to native/full context.  A policy is
then calibrated *only on held-out synthetic series* to choose the shortest
window whose predicted risk is acceptable.  GIFT-Eval labels are used once,
after fitting and calibration, for zero-shot evaluation.

The long-form risk model also lets a predictor trained on the coarse synthetic
grid interpolate to the complementary midpoint grid used by the real ablation.
No TSFM is loaded by this module; it consumes Stage-1 curves and Stage-3 caches.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import joblib
import numpy as np
from gift_eval.data import Dataset as GiftEvalDataset
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from experiments import datasets_config
from experiments.compare_window_strategies_gifteval import (
    DEFAULT_PATCH_SIZES,
    theoretical_flops,
)
from experiments.test_window_ablation_gifteval_v5 import (
    GiftEvalCache,
    _prepare_predictor_inputs,
)


VERSION = 2
HARM_THRESHOLD = math.log1p(0.05)
LOG_RATIO_CLIP = 3.0
DEFAULT_SCALES = (32, 128, 512, 2048, 8192, 15360)
DEFAULT_LAGS = (1, 2, 4, 8, 16, 24, 48, 96, 168, 336, 720)
PROFILE_LIMITS = {
    "conservative": {"harm5_rate": 0.005, "mean_ratio": 1.000},
    "balanced": {"harm5_rate": 0.010, "mean_ratio": 1.001},
    "aggressive": {"harm5_rate": 0.030, "mean_ratio": 1.005},
    "very_aggressive": {"harm5_rate": 0.050, "mean_ratio": 1.010},
    "efficiency": {"harm5_rate": 0.150, "mean_ratio": 1.040},
    "max_efficiency": {"harm5_rate": 0.200, "mean_ratio": 1.070},
}


@dataclass(frozen=True)
class PolicyConfig:
    uncertainty_weight: float
    harm_weight: float
    threshold: float


@dataclass
class RealCell:
    dataset: str
    term: str
    windows: np.ndarray
    errors: np.ndarray
    counts: np.ndarray
    native_error: np.ndarray
    native_count: np.ndarray
    native_context: np.ndarray
    contexts: list[np.ndarray]
    horizon: int

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.term}"


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _safe_signal(row: np.ndarray, valid_length: int) -> np.ndarray:
    length = max(1, min(int(valid_length), int(row.shape[0])))
    return np.nan_to_num(
        np.asarray(row[-length:], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _autocorrelation(signal: np.ndarray, lag: int) -> tuple[float, float]:
    if lag <= 0 or signal.size <= lag + 3:
        return 0.0, 0.0
    left = signal[:-lag]
    right = signal[lag:]
    left = left - left.mean()
    right = right - right.mean()
    denom = float(left.std() * right.std())
    value = float(np.mean(left * right) / denom) if denom > 1e-12 else 0.0
    return float(np.clip(value, -1.0, 1.0)), 1.0


def _spectral_features(signal: np.ndarray) -> list[float]:
    tail = signal[-min(4096, signal.size):]
    if tail.size < 8:
        return [0.0] * 5
    tail = tail - tail.mean()
    power = np.abs(np.fft.rfft(tail)) ** 2
    power = power[1:]
    total = float(power.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return [0.0] * 5
    probability = power / total
    entropy = -float(np.sum(probability * np.log(probability + 1e-12)))
    entropy /= max(math.log(probability.size), 1e-12)
    peak = int(np.argmax(power)) + 1
    peak_share = float(power[peak - 1] / total)
    peak_period = float(tail.size / peak)
    low_share = float(power[:max(1, power.size // 8)].sum() / total)
    frequencies = np.arange(1, power.size + 1, dtype=np.float64)
    centroid = float(np.sum(frequencies * probability) / power.size)
    return [entropy, peak_share, math.log2(max(peak_period, 1.0)), low_share, centroid]


def extract_series_features(
    contexts: np.ndarray,
    valid_lengths: np.ndarray,
    max_context: int,
    scales: Sequence[int] = DEFAULT_SCALES,
    lags: Sequence[int] = DEFAULT_LAGS,
) -> np.ndarray:
    """Scale-invariant recent/remote summaries for each input series.

    ``valid_lengths`` excludes artificial left padding.  The features deliberately
    expose both conditional-forecasting information (recent differences and
    autocorrelation) and generative-process information (multi-scale, spectral,
    and regime-shift summaries).
    """
    contexts = np.asarray(contexts)
    valid_lengths = np.asarray(valid_lengths, dtype=np.int64)
    active_scales = tuple(int(v) for v in scales if int(v) <= int(max_context))
    rows: list[list[float]] = []
    for row, length in zip(contexts, valid_lengths):
        signal = _safe_signal(row, int(length))
        center = float(signal.mean())
        scale = max(float(signal.std()), 1e-6)
        values: list[float] = [
            math.log2(max(signal.size, 1)) / math.log2(max(max_context, 2)),
            signal.size / max(max_context, 1),
        ]
        for desired in active_scales:
            width = min(int(desired), signal.size)
            suffix = signal[-width:]
            normalized = (suffix - center) / scale
            difference = np.diff(suffix)
            if difference.size:
                mad_diff = float(np.mean(np.abs(difference)) / scale)
                std_diff = float(difference.std() / scale)
            else:
                mad_diff = std_diff = 0.0
            q05, q95 = np.quantile(normalized, [0.05, 0.95])
            previous_available = float(signal.size >= 2 * desired)
            if previous_available:
                previous = signal[-2 * desired:-desired]
                mean_shift = float((suffix.mean() - previous.mean()) / scale)
                log_scale_shift = float(math.log(
                    (suffix.std() + 1e-6) / (previous.std() + 1e-6)))
            else:
                mean_shift = log_scale_shift = 0.0
            values.extend([
                width / desired,
                float((suffix.mean() - center) / scale),
                float(suffix.std() / scale),
                float((suffix[-1] - center) / scale),
                float((suffix[-1] - suffix[0]) / scale),
                mad_diff,
                std_diff,
                float(q95 - q05),
                previous_available,
                mean_shift,
                log_scale_shift,
            ])
        for lag in lags:
            values.extend(_autocorrelation(signal[-min(4096, signal.size):], int(lag)))
        values.extend(_spectral_features(signal))
        rows.append(values)
    result = np.asarray(rows, dtype=np.float32)
    return np.nan_to_num(result, nan=0.0, posinf=20.0, neginf=-20.0).clip(-20, 20)


def make_pair_features(
    base: np.ndarray,
    valid_lengths: np.ndarray,
    windows: np.ndarray,
    horizons: np.ndarray,
    max_context: int,
) -> np.ndarray:
    """Append candidate-window and horizon geometry to series features."""
    base = np.asarray(base, dtype=np.float32)
    valid = np.asarray(valid_lengths, dtype=np.float64)
    window = np.asarray(windows, dtype=np.float64)
    horizon = np.asarray(horizons, dtype=np.float64)
    if not (len(base) == len(valid) == len(window) == len(horizon)):
        raise ValueError("base/length/window/horizon rows must align")
    effective = np.minimum(window, valid)
    denominator = math.log2(max(int(max_context), 2))
    extra = np.column_stack([
        np.log2(np.maximum(window, 1.0)) / denominator,
        window / max(int(max_context), 1),
        np.log2(np.maximum(effective, 1.0)) / denominator,
        effective / np.maximum(valid, 1.0),
        np.log2(np.maximum(horizon, 1.0)) / denominator,
        np.log1p(horizon / np.maximum(window, 1.0)),
        np.log1p(window / np.maximum(horizon, 1.0)),
        np.log1p(horizon / np.maximum(effective, 1.0)),
        (window >= int(max_context)).astype(np.float64),
        (window < valid).astype(np.float64),
    ]).astype(np.float32)
    return np.column_stack([base, extra]).astype(np.float32, copy=False)


def _positive_probability(classifier, features: np.ndarray) -> np.ndarray:
    classes = np.asarray(classifier.classes_)
    if not np.any(classes == 1):
        return np.zeros(len(features), dtype=np.float32)
    index = int(np.flatnonzero(classes == 1)[0])
    return classifier.predict_proba(features)[:, index].astype(np.float32)


def predict_risk(
    regressor,
    classifier,
    features: np.ndarray,
    chunk_size: int = 32768,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tree-ensemble mean/std plus probability of >5% native harm."""
    mean = np.empty(len(features), dtype=np.float32)
    std = np.empty(len(features), dtype=np.float32)
    harm = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), int(chunk_size)):
        stop = min(len(features), start + int(chunk_size))
        block = features[start:stop]
        predictions = np.stack(
            [tree.predict(block) for tree in regressor.estimators_], axis=0,
        ).astype(np.float32)
        mean[start:stop] = predictions.mean(axis=0)
        std[start:stop] = predictions.std(axis=0)
        harm[start:stop] = _positive_probability(classifier, block)
    return mean, std, harm


def select_shortest_safe(
    score: np.ndarray,
    threshold: float,
    windows: np.ndarray,
    native_context: np.ndarray,
    available: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return numeric action indices, or -1 for native/full abstention."""
    score = np.asarray(score, dtype=np.float64)
    windows = np.asarray(windows, dtype=np.int64)
    native_context = np.asarray(native_context, dtype=np.int64)
    if score.shape != (native_context.size, windows.size):
        raise ValueError("score shape must be (rows, windows)")
    feasible = np.isfinite(score) & (windows[None, :] < native_context[:, None])
    if available is not None:
        feasible &= np.asarray(available, dtype=bool)
    accepted = feasible & (score <= float(threshold))
    has = accepted.any(axis=1)
    choice = np.argmax(accepted, axis=1).astype(np.int64)
    return np.where(has, choice, -1)


def _chosen_arrays(
    errors: np.ndarray,
    counts: np.ndarray,
    windows: np.ndarray,
    native_error: np.ndarray,
    native_count: np.ndarray,
    native_context: np.ndarray,
    action: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.arange(len(action))
    numeric = action >= 0
    safe_action = np.maximum(action, 0)
    chosen_error = np.where(numeric, errors[rows, safe_action], native_error)
    chosen_count = np.where(numeric, counts[rows, safe_action], native_count)
    chosen_context = np.where(
        numeric,
        np.minimum(windows[safe_action], native_context),
        native_context,
    )
    available = np.isfinite(errors) & (counts > 0)
    candidates = np.column_stack([np.where(available, errors, np.inf), native_error])
    oracle_error = np.min(candidates, axis=1)
    oracle_action = np.argmin(candidates, axis=1)
    return {
        "chosen_error": chosen_error,
        "chosen_count": chosen_count,
        "chosen_context": chosen_context,
        "numeric": numeric,
        "oracle_error": oracle_error,
        "oracle_action": oracle_action,
    }


def policy_metrics(
    errors: np.ndarray,
    counts: np.ndarray,
    windows: np.ndarray,
    native_error: np.ndarray,
    native_count: np.ndarray,
    native_context: np.ndarray,
    action: np.ndarray,
) -> dict[str, float]:
    chosen = _chosen_arrays(
        errors, counts, windows, native_error, native_count, native_context, action)
    valid = (
        np.isfinite(chosen["chosen_error"])
        & np.isfinite(native_error)
        & (native_error > 0)
        & (chosen["chosen_count"] > 0)
        & (native_count > 0)
    )
    ratio = chosen["chosen_error"][valid] / native_error[valid]
    # Some benchmark rows have a numerically zero oracle MASE. Dividing by that
    # value makes an otherwise ordinary miss look like 1e10 regret and renders
    # the aggregate useless. The native-relative floor preserves ordering while
    # bounding the influence of those degenerate rows.
    regret_floor = np.maximum(native_error[valid] * 1e-3, 1e-12)
    regret = (
        chosen["chosen_error"][valid] - chosen["oracle_error"][valid]
    ) / np.maximum(chosen["oracle_error"][valid], regret_floor)
    log_regret = np.log(
        (chosen["chosen_error"][valid] + regret_floor)
        / (chosen["oracle_error"][valid] + regret_floor))
    context_ratio = chosen["chosen_context"][valid] / np.maximum(native_context[valid], 1)
    predicted_action = np.where(
        action[valid] >= 0, action[valid], errors.shape[1])
    return {
        "n": int(valid.sum()),
        "coverage": float(chosen["numeric"][valid].mean()),
        "mean_ratio": float(ratio.mean()),
        "median_ratio": float(np.median(ratio)),
        "harm_rate": float(np.mean(ratio > 1.0 + 1e-12)),
        "harm5_rate": float(np.mean(ratio > 1.05)),
        "improvement_rate": float(np.mean(ratio < 1.0 - 1e-12)),
        "mean_regret": float(regret.mean()),
        "p90_regret": float(np.quantile(regret, 0.90)),
        "p95_regret": float(np.quantile(regret, 0.95)),
        "mean_log_regret": float(log_regret.mean()),
        "p90_log_regret": float(np.quantile(log_regret, 0.90)),
        "exact_argmin_accuracy": float(np.mean(
            predicted_action == chosen["oracle_action"][valid])),
        "context_saved_pct": float(100.0 * (1.0 - context_ratio.mean())),
        "mean_selected_context": float(chosen["chosen_context"][valid].mean()),
    }


def _calibrate_profiles(
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    predicted_harm: np.ndarray,
    errors: np.ndarray,
    windows: np.ndarray,
    native_context: np.ndarray,
    uncertainty_weights: Sequence[float],
    harm_weights: Sequence[float],
    n_quantiles: int,
) -> tuple[dict[str, dict], list[dict]]:
    native_error = errors[:, -1]
    numeric_errors = errors[:, :-1]
    numeric_windows = windows[:-1]
    ones = np.ones_like(numeric_errors, dtype=np.float64)
    native_count = np.ones(len(errors), dtype=np.float64)
    candidates: list[dict] = []
    score_families: list[dict] = []
    for uncertainty_weight in uncertainty_weights:
        for harm_weight in harm_weights:
            score = (
                predicted_mean
                + float(uncertainty_weight) * predicted_std
                + float(harm_weight) * predicted_harm
            )
            eligible_score = score[
                numeric_windows[None, :] < native_context[:, None]]
            finite = eligible_score[np.isfinite(eligible_score)]
            thresholds = np.unique(np.r_[
                -np.inf,
                np.quantile(finite, np.linspace(0.0, 1.0, int(n_quantiles))),
            ])
            family_rows: list[dict] = []
            for threshold in thresholds:
                action = select_shortest_safe(
                    score, float(threshold), numeric_windows, native_context)
                metrics = policy_metrics(
                    numeric_errors, ones, numeric_windows,
                    native_error, native_count, native_context, action)
                row = {
                    "config": asdict(PolicyConfig(
                        float(uncertainty_weight), float(harm_weight),
                        float(threshold))),
                    "validation": metrics,
                }
                family_rows.append(row)
                candidates.append(row)
            family_profiles: dict[str, dict] = {}
            for name, limits in PROFILE_LIMITS.items():
                feasible = [
                    row for row in family_rows
                    if row["validation"]["harm5_rate"] <= limits["harm5_rate"]
                    and row["validation"]["mean_ratio"] <= limits["mean_ratio"]
                ]
                # The -inf threshold is all-native, so every score family must
                # have at least one feasible row for every profile.
                if not feasible:
                    raise RuntimeError(
                        f"No feasible synthetic calibration for {name}")
                family_profiles[name] = max(
                    feasible,
                    key=lambda row: (
                        row["validation"]["context_saved_pct"],
                        -row["validation"]["mean_ratio"],
                        -row["validation"]["harm5_rate"],
                    ),
                )
            score_families.append({
                "uncertainty_weight": float(uncertainty_weight),
                "harm_weight": float(harm_weight),
                "profiles": family_profiles,
            })

    # Choose the risk-score definition once, using only the balanced synthetic
    # profile, then calibrate every threshold on that frozen score. This
    # avoids one model-selection search per profile and makes the profiles
    # nested: conservative <= balanced <= aggressive in admitted context.
    selected_family = max(
        score_families,
        key=lambda family: (
            family["profiles"]["balanced"]["validation"]["context_saved_pct"],
            -family["profiles"]["balanced"]["validation"]["mean_ratio"],
            -family["profiles"]["balanced"]["validation"]["harm5_rate"],
        ),
    )
    profiles = selected_family["profiles"]
    return profiles, candidates


def _synthetic_paths(model_dir: Path) -> tuple[Path, Path, Path]:
    pool = model_dir.parent
    return pool / "contexts.npy", pool / "real_lengths.npy", model_dir / "curves_mae.npy"


def _synthetic_split_predictions(
    split_index: np.ndarray,
    split_base: np.ndarray,
    split_lengths: np.ndarray,
    curves: np.ndarray,
    windows: np.ndarray,
    horizons: np.ndarray,
    max_context: int,
    regressor,
    classifier,
    prediction_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict the complete window grid for one held-out synthetic split."""
    task_series, task_horizon = np.meshgrid(
        np.arange(len(split_index)), np.arange(len(horizons)), indexing="ij")
    task_series = task_series.ravel()
    task_horizon = task_horizon.ravel()
    n_tasks = len(task_series)
    repeated_series = np.repeat(task_series, len(windows))
    repeated_horizon = np.repeat(task_horizon, len(windows))
    repeated_window = np.tile(np.arange(len(windows)), n_tasks)
    features = make_pair_features(
        split_base[repeated_series], split_lengths[repeated_series],
        windows[repeated_window], horizons[repeated_horizon], max_context)
    predicted_mean, predicted_std, predicted_harm = predict_risk(
        regressor, classifier, features, prediction_chunk)
    shape = (n_tasks, len(windows))
    errors = np.asarray(
        curves[split_index[task_series], :, task_horizon], dtype=np.float64)
    return (
        predicted_mean.reshape(shape)[:, :-1],
        predicted_std.reshape(shape)[:, :-1],
        predicted_harm.reshape(shape)[:, :-1],
        errors,
        split_lengths[task_series],
    )


def train_policy(args: argparse.Namespace) -> dict:
    model_dir = Path(args.synthetic_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    if meta["model_display"] != args.model_short:
        raise ValueError(
            f"Synthetic labels are {meta['model_display']}, not {args.model_short}")
    context_path, length_path, curve_path = _synthetic_paths(model_dir)
    contexts = np.load(context_path, mmap_mode="r")
    lengths = np.load(length_path, mmap_mode="r")
    curves = np.load(curve_path, mmap_mode="r")
    windows = np.asarray(meta["window_grid"], dtype=np.int64)
    horizons = np.asarray(meta["horizon_grid"], dtype=np.int64)
    max_context = int(meta["max_window"])
    if curves.shape != (len(contexts), len(windows), len(horizons)):
        raise ValueError(f"Unexpected synthetic curve shape {curves.shape}")

    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(contexts))
    n_calibration = min(int(args.max_val_series), max(1, len(order) // 5))
    remaining = len(order) - n_calibration
    n_selection = min(int(args.max_selection_series), max(1, remaining // 5))
    n_train = min(int(args.max_train_series), remaining - n_selection)
    calibration_index = np.sort(order[:n_calibration])
    selection_index = np.sort(order[n_calibration:n_calibration + n_selection])
    train_index = np.sort(order[
        n_calibration + n_selection:n_calibration + n_selection + n_train])
    selected = np.concatenate([train_index, selection_index, calibration_index])
    prepared = np.asarray(contexts[selected, -max_context:], dtype=np.float32)
    selected_lengths = np.minimum(lengths[selected], max_context).astype(np.int64)
    print(f"Extracting synthetic features for {len(selected):,} series...", flush=True)
    base = extract_series_features(prepared, selected_lengths, max_context)
    train_base = base[:len(train_index)]
    selection_stop = len(train_index) + len(selection_index)
    selection_base = base[len(train_index):selection_stop]
    calibration_base = base[selection_stop:]
    train_lengths = selected_lengths[:len(train_index)]
    selection_lengths = selected_lengths[len(train_index):selection_stop]
    calibration_lengths = selected_lengths[selection_stop:]

    pair_count = int(args.train_pairs)
    series_pos = rng.randint(0, len(train_index), size=pair_count)
    horizon_pos = rng.randint(0, len(horizons), size=pair_count)
    window_pos = rng.randint(0, len(windows), size=pair_count)
    raw_error = np.asarray(
        curves[train_index[series_pos], window_pos, horizon_pos], dtype=np.float64)
    native_error = np.asarray(
        curves[train_index[series_pos], -1, horizon_pos], dtype=np.float64)
    valid = (
        np.isfinite(raw_error) & np.isfinite(native_error)
        & (raw_error > 0) & (native_error > 0)
    )
    series_pos = series_pos[valid]
    horizon_pos = horizon_pos[valid]
    window_pos = window_pos[valid]
    target = np.log(raw_error[valid] / native_error[valid])
    target_clipped = np.clip(target, -LOG_RATIO_CLIP, LOG_RATIO_CLIP)
    train_features = make_pair_features(
        train_base[series_pos], train_lengths[series_pos],
        windows[window_pos], horizons[horizon_pos], max_context)
    sample_weight = 1.0 + 2.0 * (target > HARM_THRESHOLD) + 0.25 * np.abs(target_clipped)

    regressor = ExtraTreesRegressor(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    classifier = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        class_weight="balanced",
        n_jobs=args.n_jobs,
        random_state=args.seed + 1,
    )
    print(f"Fitting risk trees on {len(train_features):,} sampled pairs...", flush=True)
    regressor.fit(train_features, target_clipped, sample_weight=sample_weight)
    classifier.fit(
        train_features, (target > HARM_THRESHOLD).astype(np.int8),
        sample_weight=sample_weight)

    selection = _synthetic_split_predictions(
        selection_index, selection_base, selection_lengths, curves, windows,
        horizons, max_context, regressor, classifier, args.prediction_chunk)
    selected_profiles, _selection_candidates = _calibrate_profiles(
        selection[0], selection[1], selection[2], selection[3], windows,
        selection[4], args.uncertainty_weights, args.harm_weights,
        args.calibration_quantiles,
    )
    selected_config = selected_profiles["balanced"]["config"]

    calibration = _synthetic_split_predictions(
        calibration_index, calibration_base, calibration_lengths, curves,
        windows, horizons, max_context, regressor, classifier,
        args.prediction_chunk)
    profiles, candidates = _calibrate_profiles(
        calibration[0], calibration[1], calibration[2], calibration[3],
        windows, calibration[4],
        [float(selected_config["uncertainty_weight"])],
        [float(selected_config["harm_weight"])],
        args.calibration_quantiles,
    )
    n_tasks = len(calibration[3])

    report = {
        "version": VERSION,
        "method": "long-form ExtraTrees expected log-risk + synthetic-only shortest-safe calibration",
        "model": args.model_short,
        "model_id": meta["model_id"],
        "synthetic_dir": str(model_dir),
        "real_labels_used_for_training_or_calibration": False,
        "target": "log(MAE_window / MAE_native), clipped for regression; >5% harm classified separately",
        "n_train_series": int(len(train_index)),
        "n_selection_series": int(len(selection_index)),
        "n_calibration_series": int(len(calibration_index)),
        "n_validation_series": int(len(calibration_index)),
        "n_train_pairs": int(len(train_features)),
        "n_validation_tasks": int(n_tasks),
        "n_features": int(train_features.shape[1]),
        "windows": windows.tolist(),
        "horizons": horizons.tolist(),
        "max_context": max_context,
        "profiles": profiles,
        "score_selection_balanced_profile": selected_profiles["balanced"],
        "selected_risk_score": {
            "uncertainty_weight": profiles["balanced"]["config"]["uncertainty_weight"],
            "harm_weight": profiles["balanced"]["config"]["harm_weight"],
        },
        "profile_limits": PROFILE_LIMITS,
        "search_candidates": int(len(candidates)),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "version": VERSION,
        "regressor": regressor,
        "classifier": classifier,
        "meta": meta,
        "profiles": profiles,
        "feature_spec": {
            "scales": [v for v in DEFAULT_SCALES if v <= max_context],
            "lags": list(DEFAULT_LAGS),
            "max_context": max_context,
        },
        "training_report": report,
    }, output / "policy.joblib", compress=3)
    (output / "synthetic_calibration.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v["validation"] for k, v in profiles.items()}, indent=2))
    return report


def _metric_vector(path: str, n: int, field: str) -> tuple[np.ndarray, np.ndarray, dict]:
    values_out = np.full(n, np.nan, dtype=np.float64)
    counts_out = np.zeros(n, dtype=np.float64)
    with np.load(path) as data:
        source = field
        if source not in data.files:
            if field == "mase_gluonts_real" and "mase_gluonts" in data.files:
                source = "mase_gluonts"
            else:
                return values_out, counts_out, {"source": "missing"}
        values = np.asarray(data[source], dtype=np.float64)
        index = (
            np.asarray(data["served_index"], dtype=np.int64)
            if "served_index" in data.files
            else np.arange(len(values), dtype=np.int64)
        )
        counts = (
            np.asarray(data["valid_count"], dtype=np.float64)
            if "valid_count" in data.files
            else np.ones(len(values), dtype=np.float64)
        )
        ok = (index >= 0) & (index < n)
        values_out[index[ok]] = values[ok]
        counts_out[index[ok]] = counts[ok]
        extra = {"source": source}
        if "effective_context" in data.files:
            effective = np.full(n, -1, dtype=np.int64)
            effective[index[ok]] = np.asarray(data["effective_context"], dtype=np.int64)[ok]
            extra["effective_context"] = effective
    return values_out, counts_out, extra


def _cell_specs() -> dict[tuple[str, str], tuple[str, bool]]:
    return {
        (display, str(term)): (name, bool(to_univariate))
        for name, term, display, to_univariate in datasets_config.datasets_to_run()
    }


def _discover_cell_paths(
    roots: Sequence[str], model: str,
) -> dict[tuple[str, str], list[str]]:
    result: dict[tuple[str, str], list[str]] = {}
    for root in roots:
        pattern = os.path.join(root, "datasets", "*", model, "t*", "w*", "per_sample_metrics.npz")
        for path in glob.glob(pattern):
            parts = Path(path).parts
            model_pos = parts.index(model)
            dataset = parts[model_pos - 1]
            term = parts[model_pos + 1][1:]
            result.setdefault((dataset, term), []).append(path)
    return result


def load_real_cell(
    dataset: str,
    term: str,
    paths: Sequence[str],
    metric: str,
) -> RealCell:
    spec = _cell_specs().get((dataset, str(term)))
    if spec is None:
        raise KeyError(f"No active GIFT-Eval spec for {dataset}/{term}")
    ge_dataset = GiftEvalDataset(
        name=spec[0], term=term, to_univariate=spec[1])
    cache = GiftEvalCache(ge_dataset, dataset)
    numeric: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    native: Optional[tuple[np.ndarray, np.ndarray, dict]] = None
    for path in sorted(paths):
        folder = Path(path).parent.name
        if folder == "wfull_native":
            native = _metric_vector(path, cache.n_total, metric)
            continue
        match = re.fullmatch(r"w(\d+)", folder)
        if match is None:
            continue
        window = int(match.group(1))
        values, counts, _ = _metric_vector(path, cache.n_total, metric)
        if window in numeric:
            old_values, _old_counts = numeric[window]
            overlap = np.isfinite(old_values) & np.isfinite(values)
            if overlap.any() and not np.allclose(
                    old_values[overlap], values[overlap], rtol=1e-5, atol=1e-8):
                raise ValueError(f"Conflicting duplicate cache for {dataset}/{term}/w{window}")
            values = np.where(np.isfinite(old_values), old_values, values)
            counts = np.where(np.isfinite(old_values), numeric[window][1], counts)
        numeric[window] = values, counts
    if not numeric or native is None:
        raise RuntimeError(f"Missing numeric/native caches for {dataset}/{term}")
    windows = np.asarray(sorted(numeric), dtype=np.int64)
    errors = np.column_stack([numeric[int(w)][0] for w in windows])
    counts = np.column_stack([numeric[int(w)][1] for w in windows])
    native_error, native_count, native_meta = native
    native_context = np.asarray(
        native_meta.get("effective_context", cache.context_lengths), dtype=np.int64)
    native_context = np.where(native_context > 0, native_context, cache.context_lengths)
    return RealCell(
        dataset=dataset,
        term=str(term),
        windows=windows,
        errors=errors,
        counts=counts,
        native_error=native_error,
        native_count=native_count,
        native_context=native_context,
        contexts=cache.contexts_raw,
        horizon=int(cache.horizon),
    )


def _profile_actions(
    mean: np.ndarray,
    std: np.ndarray,
    harm: np.ndarray,
    profile: dict,
    windows: np.ndarray,
    native_context: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    config = profile["config"]
    score = (
        mean
        + float(config["uncertainty_weight"]) * std
        + float(config["harm_weight"]) * harm
    )
    return select_shortest_safe(
        score, float(config["threshold"]), windows, native_context, available)


def _requested_window_action(
    requested: np.ndarray | int,
    windows: np.ndarray,
    native_context: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    """Map a label-free requested length to the nearest cached feasible action.

    Native/full is selected whenever the capped request reaches the row's native
    context.  Otherwise the closest effective cached window is used; ties prefer
    the shorter action so the baseline does not receive an accidental compute
    advantage from a longer tie-break.
    """
    windows = np.asarray(windows, dtype=np.int64)
    native_context = np.asarray(native_context, dtype=np.int64)
    requested_array = np.broadcast_to(
        np.asarray(requested, dtype=np.int64), native_context.shape)
    target = np.minimum(np.maximum(requested_array, 1), native_context)
    effective = np.minimum(windows[None, :], native_context[:, None])
    feasible = np.asarray(available, dtype=bool) & (
        effective < native_context[:, None])
    distance = np.abs(effective - target[:, None]).astype(np.float64)
    # Stable column order is ascending by window, so argmin resolves equal
    # distances toward the shorter cached action.
    choice = np.argmin(np.where(feasible, distance, np.inf), axis=1)
    has_numeric = feasible.any(axis=1) & (target < native_context)
    return np.where(has_numeric, choice, -1).astype(np.int64)


def _oracle_action(cell: RealCell, tolerance: float = 0.0) -> np.ndarray:
    available = np.isfinite(cell.errors) & (cell.counts > 0)
    candidate = np.column_stack([
        np.where(available, cell.errors, np.inf), cell.native_error])
    best = np.min(candidate, axis=1)
    allowed = candidate <= best[:, None] * (1.0 + float(tolerance)) + 1e-12
    context = np.column_stack([
        np.broadcast_to(cell.windows, cell.errors.shape), cell.native_context])
    choice = np.argmin(np.where(allowed, context, np.inf), axis=1)
    return np.where(choice < len(cell.windows), choice, -1).astype(np.int64)


def _cell_result(cell: RealCell, method: str, action: np.ndarray) -> tuple[dict, dict]:
    arrays = _chosen_arrays(
        cell.errors, cell.counts, cell.windows,
        cell.native_error, cell.native_count, cell.native_context, action)
    valid = (
        np.isfinite(arrays["chosen_error"]) & np.isfinite(cell.native_error)
        & (cell.native_error > 0) & (arrays["chosen_count"] > 0)
        & (cell.native_count > 0)
    )
    selected_mase = float(np.average(
        arrays["chosen_error"][valid], weights=arrays["chosen_count"][valid]))
    native_mase = float(np.average(
        cell.native_error[valid], weights=cell.native_count[valid]))
    metrics = policy_metrics(
        cell.errors, cell.counts, cell.windows,
        cell.native_error, cell.native_count, cell.native_context, action)
    row = {
        "model": "",
        "dataset": cell.dataset,
        "term": cell.term,
        "method": method,
        "horizon": cell.horizon,
        "n_windows": int(len(cell.windows)),
        "n_instances": int(valid.sum()),
        "selected_mase": selected_mase,
        "native_mase": native_mase,
        "cell_mase_ratio": selected_mase / max(native_mase, 1e-12),
        **metrics,
    }
    regret_floor = np.maximum(cell.native_error[valid] * 1e-3, 1e-12)
    audit = {
        "ratio": arrays["chosen_error"][valid] / cell.native_error[valid],
        "regret": (
            arrays["chosen_error"][valid] - arrays["oracle_error"][valid]
        ) / np.maximum(arrays["oracle_error"][valid], regret_floor),
        "log_regret": np.log(
            (arrays["chosen_error"][valid] + regret_floor)
            / (arrays["oracle_error"][valid] + regret_floor)),
        "selected_context": arrays["chosen_context"][valid],
        "native_context": cell.native_context[valid],
        "numeric": arrays["numeric"][valid],
        "horizon": int(cell.horizon),
    }
    return row, audit


def _context_flops(
    model: str,
    contexts: np.ndarray,
    horizon: int,
    cache: dict[tuple[int, int], float],
) -> np.ndarray:
    """Return the repository's theoretical transformer-cost proxy per row."""
    values = np.asarray(contexts, dtype=np.int64)
    result = np.empty(values.shape, dtype=np.float64)
    for context in np.unique(values):
        key = (int(context), int(horizon))
        if key not in cache:
            cache[key] = theoretical_flops(
                model, key[0], key[1], DEFAULT_PATCH_SIZES)
        result[values == context] = cache[key]
    return result


def _aggregate_real(rows: list[dict], audits: dict[str, list[dict]]) -> dict:
    output: dict[str, dict] = {}
    flops_cache: dict[tuple[int, int], float] = {}
    for method in sorted(audits):
        method_rows = [row for row in rows if row["method"] == method]
        model = str(method_rows[0]["model"])
        cell_ratio = np.asarray([row["cell_mase_ratio"] for row in method_rows])
        ratios = np.concatenate([row["ratio"] for row in audits[method]])
        regret = np.concatenate([row["regret"] for row in audits[method]])
        log_regret = np.concatenate(
            [row["log_regret"] for row in audits[method]])
        selected_context = np.concatenate(
            [row["selected_context"] for row in audits[method]])
        native_context = np.concatenate(
            [row["native_context"] for row in audits[method]])
        numeric = np.concatenate([row["numeric"] for row in audits[method]])
        selected_flops = sum(
            float(_context_flops(
                model, row["selected_context"], row["horizon"], flops_cache).sum())
            for row in audits[method]
        )
        native_flops = sum(
            float(_context_flops(
                model, row["native_context"], row["horizon"], flops_cache).sum())
            for row in audits[method]
        )
        output[method] = {
            "n_cells": int(len(method_rows)),
            "n_instances": int(len(ratios)),
            "geomean_cell_mase_ratio": float(np.exp(np.mean(np.log(cell_ratio)))),
            "macro_cell_mase_ratio": float(cell_ratio.mean()),
            "cell_win_rate": float(np.mean(cell_ratio < 1.0)),
            "instance_mean_ratio": float(ratios.mean()),
            "instance_harm_rate": float(np.mean(ratios > 1.0 + 1e-12)),
            "instance_harm5_rate": float(np.mean(ratios > 1.05)),
            "instance_improvement_rate": float(np.mean(ratios < 1.0 - 1e-12)),
            "mean_regret": float(regret.mean()),
            "p90_regret": float(np.quantile(regret, 0.90)),
            "p95_regret": float(np.quantile(regret, 0.95)),
            "mean_log_regret": float(log_regret.mean()),
            "p90_log_regret": float(np.quantile(log_regret, 0.90)),
            "coverage": float(numeric.mean()),
            "context_saved_pct": float(100.0 * (
                1.0 - selected_context.sum() / np.maximum(native_context.sum(), 1))),
            "theoretical_flops_saved_pct": float(100.0 * (
                1.0 - selected_flops / max(native_flops, 1e-12))),
            "selected_theoretical_macs": float(selected_flops),
            "native_theoretical_macs": float(native_flops),
            "mean_selected_context": float(selected_context.mean()),
        }
    return output


def evaluate_real(args: argparse.Namespace) -> dict:
    output = Path(args.output_dir)
    bundle = joblib.load(output / "policy.joblib")
    meta = bundle["meta"]
    if meta["model_display"] != args.model_short:
        raise ValueError("Policy/model mismatch")
    max_context = int(bundle["feature_spec"]["max_context"])
    cell_paths = _discover_cell_paths(args.cache_roots, args.model_short)
    keys = sorted(set(cell_paths) & set(_cell_specs()))
    if args.max_real_cells:
        keys = keys[:int(args.max_real_cells)]
    rows: list[dict] = []
    audits: dict[str, list[dict]] = {}
    histogram_rows: list[dict] = []
    cell_flops_cache: dict[tuple[int, int], float] = {}
    for number, (dataset, term) in enumerate(keys, 1):
        try:
            cell = load_real_cell(dataset, term, cell_paths[(dataset, term)], args.metric)
            valid_row = (
                np.isfinite(cell.native_error) & (cell.native_error > 0)
                & (cell.native_count > 0)
                & (np.isfinite(cell.errors) & (cell.counts > 0)).any(axis=1)
            )
            index = np.flatnonzero(valid_row)
            if args.max_real_instances_per_cell and len(index) > args.max_real_instances_per_cell:
                rng = np.random.RandomState(_stable_seed(cell.key, args.seed))
                index = np.sort(rng.choice(
                    index, int(args.max_real_instances_per_cell), replace=False))
            cell.errors = cell.errors[index]
            cell.counts = cell.counts[index]
            cell.native_error = cell.native_error[index]
            cell.native_count = cell.native_count[index]
            cell.native_context = np.minimum(cell.native_context[index], max_context)
            contexts = [cell.contexts[i] for i in index]
            prepared = _prepare_predictor_inputs(contexts, max_context)
            base = extract_series_features(
                prepared, cell.native_context, max_context,
                bundle["feature_spec"]["scales"], bundle["feature_spec"]["lags"])
            n, n_windows = cell.errors.shape
            mean = np.empty((n, n_windows), dtype=np.float32)
            std = np.empty_like(mean)
            harm = np.empty_like(mean)
            for start in range(0, n, args.prediction_row_batch):
                stop = min(n, start + args.prediction_row_batch)
                count = stop - start
                pair_features = make_pair_features(
                    np.repeat(base[start:stop], n_windows, axis=0),
                    np.repeat(cell.native_context[start:stop], n_windows),
                    np.tile(cell.windows, count),
                    np.full(count * n_windows, cell.horizon),
                    max_context,
                )
                values = predict_risk(
                    bundle["regressor"], bundle["classifier"],
                    pair_features, args.prediction_chunk)
                mean[start:stop] = values[0].reshape(count, n_windows)
                std[start:stop] = values[1].reshape(count, n_windows)
                harm[start:stop] = values[2].reshape(count, n_windows)
            available = np.isfinite(cell.errors) & (cell.counts > 0)
            actions = {
                name: _profile_actions(
                    mean, std, harm, profile, cell.windows,
                    cell.native_context, available)
                for name, profile in bundle["profiles"].items()
            }
            expected_score = mean + harm
            expected_score = np.where(available, expected_score, np.inf)
            best = np.argmin(expected_score, axis=1)
            actions["expected_risk_argmin"] = np.where(
                np.isfinite(expected_score[np.arange(n), best])
                & (mean[np.arange(n), best] < 0.0)
                & (cell.windows[best] < cell.native_context), best, -1)
            balanced_config = bundle["profiles"].get(
                "balanced", next(iter(bundle["profiles"].values())))['config']
            full_score = (
                mean
                + float(balanced_config["uncertainty_weight"]) * std
                + float(balanced_config["harm_weight"]) * harm
            )
            full_score = np.where(
                available & (cell.windows[None, :] < cell.native_context[:, None]),
                full_score, np.inf)
            full_best = np.argmin(full_score, axis=1)
            actions["risk_argmin_no_abstain"] = np.where(
                np.isfinite(full_score[np.arange(n), full_best]), full_best, -1)
            actions["full_native"] = np.full(n, -1, dtype=np.int64)
            actions["oracle_argmin"] = _oracle_action(cell, 0.0)
            actions["oracle_shortest_within_5pct"] = _oracle_action(cell, 0.05)
            if args.fixed_baselines:
                for requested in args.fixed_windows:
                    actions[f"fixed_w{int(requested)}"] = _requested_window_action(
                        int(requested), cell.windows, cell.native_context, available)
                for multiple in args.horizon_multiples:
                    requested = np.rint(float(multiple) * cell.horizon).astype(np.int64)
                    label = str(float(multiple)).replace(".", "p")
                    actions[f"horizon_x{label}"] = _requested_window_action(
                        requested, cell.windows, cell.native_context, available)
            for method, action in actions.items():
                row, audit = _cell_result(cell, method, action)
                row["model"] = args.model_short
                selected_flops = float(_context_flops(
                    args.model_short, audit["selected_context"], cell.horizon,
                    cell_flops_cache).sum())
                native_flops = float(_context_flops(
                    args.model_short, audit["native_context"], cell.horizon,
                    cell_flops_cache).sum())
                row["selected_theoretical_macs"] = selected_flops
                row["native_theoretical_macs"] = native_flops
                row["theoretical_flops_saved_pct"] = float(
                    100.0 * (1.0 - selected_flops / max(native_flops, 1e-12)))
                rows.append(row)
                audits.setdefault(method, []).append(audit)
                requested_window = np.where(
                    action >= 0,
                    cell.windows[np.maximum(action, 0)],
                    int(np.max(cell.native_context)),
                ).astype(np.int64)
                for window_size, count in zip(
                    *np.unique(requested_window, return_counts=True)
                ):
                    histogram_rows.append({
                        "model": args.model_short,
                        "dataset": cell.dataset,
                        "term": cell.term,
                        "horizon": int(cell.horizon),
                        "method": method,
                        "window_size": int(window_size),
                        "n_instances": int(count),
                        "cell_instances": int(n),
                    })
            print(
                f"[{number}/{len(keys)}] {cell.key}: n={n}, windows={n_windows}",
                flush=True)
        except Exception as exc:
            print(f"[{number}/{len(keys)}] SKIP {dataset}/{term}: {exc!r}", flush=True)

    if not rows:
        raise RuntimeError("No real cells evaluated")
    aggregate = _aggregate_real(rows, audits)
    report = {
        "version": VERSION,
        "method": "synthetic-only calibrated expected-risk policy; zero-shot GIFT-Eval evaluation",
        "model": args.model_short,
        "metric": args.metric,
        "cache_roots": list(args.cache_roots),
        "real_labels_used_for_training_or_calibration": False,
        "real_labels_used_for_final_evaluation": True,
        "synthetic_profiles": bundle["profiles"],
        "aggregate": aggregate,
    }
    (output / "real_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "real_cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if histogram_rows:
        with (output / "selected_window_histograms.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(histogram_rows[0]))
            writer.writeheader()
            writer.writerows(histogram_rows)
    print(json.dumps(aggregate, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["train", "evaluate", "all"], default="all")
    parser.add_argument("--model-short", required=True)
    parser.add_argument("--synthetic-dir", required=True)
    parser.add_argument("--cache-roots", nargs="*", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", default="mase_gluonts_real")
    parser.add_argument("--max-train-series", type=int, default=12000)
    parser.add_argument(
        "--max-selection-series", type=int, default=1000,
        help="Held-out synthetic series used only to select risk-score weights.")
    parser.add_argument("--max-val-series", type=int, default=2000)
    parser.add_argument("--train-pairs", type=int, default=600000)
    parser.add_argument("--n-estimators", type=int, default=192)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--max-features", type=float, default=0.7)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--uncertainty-weights", nargs="+", type=float,
                        default=[0.0, 0.5, 1.0, 1.645])
    parser.add_argument("--harm-weights", nargs="+", type=float,
                        default=[0.0, 0.1, 0.25, 0.5])
    parser.add_argument("--calibration-quantiles", type=int, default=81)
    parser.add_argument("--prediction-chunk", type=int, default=32768)
    parser.add_argument("--prediction-row-batch", type=int, default=256)
    parser.add_argument("--max-real-cells", type=int, default=0)
    parser.add_argument("--max-real-instances-per-cell", type=int, default=0)
    parser.add_argument(
        "--fixed-baselines", action="store_true",
        help="Also evaluate label-free global fixed-window and horizon-multiple baselines.")
    parser.add_argument(
        "--fixed-windows", nargs="+", type=int,
        default=[32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192])
    parser.add_argument(
        "--horizon-multiples", nargs="+", type=float,
        default=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.stage in {"evaluate", "all"} and not args.cache_roots:
        parser.error("--cache-roots is required for real evaluation")
    if args.calibration_quantiles < 2:
        parser.error("--calibration-quantiles must be >= 2")
    if args.max_selection_series < 1 or args.max_val_series < 1:
        parser.error("selection and calibration splits must each contain at least one series")
    return args


def main() -> None:
    args = parse_args()
    if args.stage in {"train", "all"}:
        train_policy(args)
    if args.stage in {"evaluate", "all"}:
        evaluate_real(args)


if __name__ == "__main__":
    main()
