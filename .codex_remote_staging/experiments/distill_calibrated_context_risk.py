#!/usr/bin/env python3
"""Distill the frozen ExtraTrees context-risk score into compact students.

The teacher is the already calibrated synthetic-only ExtraTrees policy.  The
students see the same engineered pair features and learn the teacher's deployed
composite risk score.  Thresholds are then recalibrated from true held-out
synthetic curves, never from GIFT-Eval outcomes.

Candidate families:
  * Ridge regression: a global glass-box linear score.
  * One shallow decision tree: a directly inspectable rule list.
  * One tiny MLP: a low-memory, low-latency nonlinear student.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments import calibrated_context_risk as base


VERSION = 1
PROFILE_LIMITS = {
    **base.PROFILE_LIMITS,
    "extreme": {"harm5_rate": 0.150, "mean_ratio": 1.040},
    "very_extreme": {"harm5_rate": 0.200, "mean_ratio": 1.070},
}
base.PROFILE_LIMITS = PROFILE_LIMITS


class OneEstimatorEnsemble:
    """Expose a single student through the risk scorer's ensemble protocol."""

    def __init__(self, estimator):
        self.estimators_ = [estimator]


class MeanScalarEstimator:
    """Average several tiny students while exposing one deterministic score."""

    def __init__(self, members: Sequence):
        self.members = list(members)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.mean(
            [member.predict(features) for member in self.members], axis=0)


class ZeroHarmClassifier:
    """The student regresses the complete teacher score, including harm risk."""

    classes_ = np.asarray([0, 1], dtype=np.int64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = np.zeros(len(features), dtype=np.float32)
        return np.column_stack([1.0 - positive, positive])


class ScalarMLP(nn.Module):
    def __init__(self, n_features: int, hidden: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        width = int(n_features)
        for next_width in hidden:
            layers.extend([nn.Linear(width, int(next_width)), nn.SiLU()])
            width = int(next_width)
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class FrozenScalarMLP:
    """Pickle-friendly NumPy prediction wrapper for one tiny MLP."""

    def __init__(
        self,
        n_features: int,
        hidden: Sequence[int],
        state_dict: dict[str, torch.Tensor],
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        prediction_batch_size: int,
    ):
        self.n_features = int(n_features)
        self.hidden = tuple(int(value) for value in hidden)
        self.state_dict = {key: value.detach().cpu() for key, value in state_dict.items()}
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float32)
        self.prediction_batch_size = int(prediction_batch_size)
        self._network: ScalarMLP | None = None

    def _model(self) -> ScalarMLP:
        if self._network is None:
            network = ScalarMLP(self.n_features, self.hidden)
            network.load_state_dict(self.state_dict)
            network.eval()
            self._network = network
        return self._network

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        output = np.empty(len(features), dtype=np.float32)
        network = self._model()
        with torch.inference_mode():
            for start in range(0, len(features), self.prediction_batch_size):
                stop = min(len(features), start + self.prediction_batch_size)
                normalized = (
                    features[start:stop] - self.feature_mean
                ) / self.feature_scale
                output[start:stop] = network(torch.from_numpy(normalized)).numpy()
        return output


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def feature_names(max_context: int) -> list[str]:
    names = ["valid_length_log_fraction", "valid_length_fraction"]
    scale_fields = (
        "coverage", "mean_z", "std_z", "last_z", "change_z",
        "mean_abs_diff_z", "std_diff_z", "q90_range_z",
        "previous_available", "previous_mean_shift_z", "previous_log_scale_shift",
    )
    for scale in base.DEFAULT_SCALES:
        if scale <= int(max_context):
            names.extend(f"scale{scale}_{field}" for field in scale_fields)
    for lag in base.DEFAULT_LAGS:
        names.extend([f"autocorrelation_lag{lag}", f"lag{lag}_available"])
    names.extend([
        "spectral_entropy", "spectral_peak_share", "spectral_peak_period_log2",
        "spectral_low_frequency_share", "spectral_centroid",
    ])
    names.extend([
        "window_log_fraction", "window_fraction", "effective_window_log_fraction",
        "effective_window_valid_fraction", "horizon_log_fraction",
        "log1p_horizon_over_window", "log1p_window_over_horizon",
        "log1p_horizon_over_effective_window", "window_is_native_cap",
        "window_shorter_than_valid_history",
    ])
    return names


def _prepare_data(args: argparse.Namespace) -> dict:
    model_dir = Path(args.synthetic_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    if meta["model_display"] != args.model_short:
        raise ValueError(
            f"Synthetic labels are {meta['model_display']}, not {args.model_short}")
    context_path, length_path, curve_path = base._synthetic_paths(model_dir)
    contexts = np.load(context_path, mmap_mode="r")
    lengths = np.load(length_path, mmap_mode="r")
    curves = np.load(curve_path, mmap_mode="r")
    windows = np.asarray(meta["window_grid"], dtype=np.int64)
    horizons = np.asarray(meta["horizon_grid"], dtype=np.int64)
    max_context = int(meta["max_window"])

    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(contexts))
    n_val = min(int(args.max_val_series), max(1, len(order) // 5))
    n_train = min(int(args.max_train_series), len(order) - n_val)
    val_index = np.sort(order[:n_val])
    train_index = np.sort(order[n_val:n_val + n_train])
    selected = np.concatenate([train_index, val_index])
    prepared = np.asarray(contexts[selected, -max_context:], dtype=np.float32)
    selected_lengths = np.minimum(lengths[selected], max_context).astype(np.int64)
    print(f"Extracting features for {len(selected):,} series...", flush=True)
    series_features = base.extract_series_features(
        prepared, selected_lengths, max_context)
    train_base = series_features[:len(train_index)]
    val_base = series_features[len(train_index):]
    train_lengths = selected_lengths[:len(train_index)]
    val_lengths = selected_lengths[len(train_index):]

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
    raw_target = np.log(raw_error[valid] / native_error[valid])
    train_features = base.make_pair_features(
        train_base[series_pos], train_lengths[series_pos],
        windows[window_pos], horizons[horizon_pos], max_context)
    sample_weight = (
        1.0 + 2.0 * (raw_target > base.HARM_THRESHOLD)
        + 0.25 * np.abs(np.clip(raw_target, -base.LOG_RATIO_CLIP, base.LOG_RATIO_CLIP))
    ).astype(np.float32)

    task_series, task_horizon = np.meshgrid(
        np.arange(len(val_index)), np.arange(len(horizons)), indexing="ij")
    task_series = task_series.ravel()
    task_horizon = task_horizon.ravel()
    n_tasks = len(task_series)
    repeated_series = np.repeat(task_series, len(windows))
    repeated_horizon = np.repeat(task_horizon, len(windows))
    repeated_window = np.tile(np.arange(len(windows)), n_tasks)
    val_features = base.make_pair_features(
        val_base[repeated_series], val_lengths[repeated_series],
        windows[repeated_window], horizons[repeated_horizon], max_context)
    validation_errors = np.asarray(
        curves[val_index[task_series], :, task_horizon], dtype=np.float64)
    return {
        "meta": meta,
        "windows": windows,
        "horizons": horizons,
        "max_context": max_context,
        "train_index": train_index,
        "val_index": val_index,
        "train_features": train_features,
        "sample_weight": sample_weight,
        "raw_log_risk": np.clip(
            raw_target, -base.LOG_RATIO_CLIP, base.LOG_RATIO_CLIP).astype(np.float32),
        "val_features": val_features,
        "val_lengths": val_lengths,
        "task_series": task_series,
        "validation_errors": validation_errors,
    }


def _teacher_scores(
    bundle: dict,
    features: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    profile = bundle["profiles"]["balanced"]["config"]
    mean, std, harm = base.predict_risk(
        bundle["regressor"], bundle["classifier"], features, chunk_size)
    score = (
        mean
        + float(profile["uncertainty_weight"]) * std
        + float(profile["harm_weight"]) * harm
    )
    return score.astype(np.float32), {
        "uncertainty_weight": float(profile["uncertainty_weight"]),
        "harm_weight": float(profile["harm_weight"]),
    }


def _fit_tiny_mlp(
    features: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    hidden: Sequence[int],
    args: argparse.Namespace,
    seed_offset: int = 0,
) -> tuple[FrozenScalarMLP, list[dict[str, float]], int]:
    member_seed = args.seed + sum(hidden) + 1009 * int(seed_offset)
    _seed_everything(member_seed)
    torch.set_num_threads(args.torch_threads)
    feature_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = np.where(feature_scale > 1e-5, feature_scale, 1.0).astype(np.float32)
    normalized = ((features - feature_mean) / feature_scale).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(normalized),
        torch.from_numpy(np.asarray(target, dtype=np.float32)),
        torch.from_numpy(np.asarray(sample_weight, dtype=np.float32)),
    )
    loader = DataLoader(
        dataset, batch_size=args.mlp_batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(member_seed))
    network = ScalarMLP(features.shape[1], hidden)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=args.mlp_learning_rate,
        weight_decay=args.mlp_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.mlp_epochs, 1))
    history: list[dict[str, float]] = []
    for epoch in range(args.mlp_epochs):
        network.train()
        squared_sum = 0.0
        weight_sum = 0.0
        for batch_features, batch_target, batch_weight in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = network(batch_features)
            squared = (prediction - batch_target).square()
            denominator = batch_weight.sum().clamp_min(1e-8)
            loss = (squared * batch_weight).sum() / denominator
            loss.backward()
            optimizer.step()
            squared_sum += float((squared * batch_weight).sum().detach())
            weight_sum += float(denominator.detach())
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "weighted_mse": squared_sum / max(weight_sum, 1e-8),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        print(
            f"tiny_mlp_{'x'.join(map(str, hidden))} epoch={epoch + 1}/"
            f"{args.mlp_epochs} mse={row['weighted_mse']:.7f}", flush=True)
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    return FrozenScalarMLP(
        features.shape[1], hidden, network.state_dict(), feature_mean,
        feature_scale, args.prediction_chunk), history, parameter_count


def _calibrate_student(
    estimator,
    val_features: np.ndarray,
    validation_errors: np.ndarray,
    windows: np.ndarray,
    native_context: np.ndarray,
    prediction_chunk: int,
) -> tuple[dict, np.ndarray]:
    prediction = np.empty(len(val_features), dtype=np.float32)
    for start in range(0, len(val_features), prediction_chunk):
        stop = min(len(val_features), start + prediction_chunk)
        prediction[start:stop] = estimator.predict(val_features[start:stop])
    shape = (len(validation_errors), len(windows))
    score = prediction.reshape(shape)[:, :-1]
    zeros = np.zeros_like(score)
    profiles, _ = base._calibrate_profiles(
        score, zeros, zeros, validation_errors, windows, native_context,
        uncertainty_weights=(0.0,), harm_weights=(0.0,), n_quantiles=81)
    return profiles, prediction


def _dense_profiles(
    score: np.ndarray,
    validation_errors: np.ndarray,
    windows: np.ndarray,
    native_context: np.ndarray,
    points: int,
    uncertainty_weight: float = 0.0,
    harm_weight: float = 0.0,
) -> dict[str, dict]:
    """Synthetic-only threshold grid for evaluating intermediate policies."""
    score = np.asarray(score, dtype=np.float64).reshape(
        len(validation_errors), len(windows))[:, :-1]
    numeric_windows = windows[:-1]
    eligible = numeric_windows[None, :] < native_context[:, None]
    finite = score[eligible & np.isfinite(score)]
    quantiles = np.linspace(0.0, 1.0, int(points))
    thresholds = np.quantile(finite, quantiles)
    thresholds[0] = -np.inf
    ones = np.ones_like(validation_errors[:, :-1], dtype=np.float64)
    profiles = {}
    for index, (quantile, threshold) in enumerate(zip(quantiles, thresholds)):
        action = base.select_shortest_safe(
            score, float(threshold), numeric_windows, native_context)
        metrics = base.policy_metrics(
            validation_errors[:, :-1], ones, numeric_windows,
            validation_errors[:, -1], np.ones(len(validation_errors)),
            native_context, action)
        profiles[f"dense_{index:02d}"] = {
            "config": {
                "uncertainty_weight": float(uncertainty_weight),
                "harm_weight": float(harm_weight),
                "threshold": float(threshold),
            },
            "validation": metrics,
            "score_quantile": float(quantile),
        }
    return profiles


def _save_candidate(
    name: str,
    family: str,
    estimator,
    fit_details: dict,
    data: dict,
    teacher_train_score: np.ndarray,
    teacher_val_score: np.ndarray,
    teacher_score_config: dict,
    args: argparse.Namespace,
) -> dict:
    profiles, val_prediction = _calibrate_student(
        estimator, data["val_features"], data["validation_errors"],
        data["windows"], data["val_lengths"][data["task_series"]],
        args.prediction_chunk)
    if args.dense_points:
        profiles.update(_dense_profiles(
            val_prediction, data["validation_errors"], data["windows"],
            data["val_lengths"][data["task_series"]], args.dense_points))
    difference = val_prediction - teacher_val_score
    correlation = float(np.corrcoef(val_prediction, teacher_val_score)[0, 1])
    report = {
        "version": VERSION,
        "method": f"{family} distilled from frozen ExtraTrees composite risk",
        "candidate": name,
        "model": args.model_short,
        "model_id": data["meta"]["model_id"],
        "synthetic_dir": args.synthetic_dir,
        "teacher_policy": args.teacher_policy,
        "teacher_score": teacher_score_config,
        "real_labels_used_for_training_or_calibration": False,
        "n_train_series": int(len(data["train_index"])),
        "n_validation_series": int(len(data["val_index"])),
        "n_train_pairs": int(len(data["train_features"])),
        "n_validation_tasks": int(len(data["validation_errors"])),
        "n_features": int(data["train_features"].shape[1]),
        "windows": data["windows"].tolist(),
        "horizons": data["horizons"].tolist(),
        "max_context": int(data["max_context"]),
        "fit_details": fit_details,
        "teacher_fidelity": {
            "validation_rmse": float(np.sqrt(np.mean(difference ** 2))),
            "validation_mae": float(np.mean(np.abs(difference))),
            "validation_correlation": correlation,
            "train_teacher_score_mean": float(np.mean(teacher_train_score)),
            "validation_teacher_score_mean": float(np.mean(teacher_val_score)),
        },
        "profiles": profiles,
        "selected_risk_score": {"uncertainty_weight": 0.0, "harm_weight": 0.0},
        "profile_limits": PROFILE_LIMITS,
    }
    output = Path(args.output_root) / name
    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "version": VERSION,
        "regressor": OneEstimatorEnsemble(estimator),
        "classifier": ZeroHarmClassifier(),
        "meta": data["meta"],
        "profiles": profiles,
        "feature_spec": {
            "scales": [value for value in base.DEFAULT_SCALES
                       if value <= data["max_context"]],
            "lags": list(base.DEFAULT_LAGS),
            "max_context": int(data["max_context"]),
        },
        "training_report": report,
    }
    joblib.dump(bundle, output / "policy.joblib", compress=3)
    report["artifact_bytes"] = int((output / "policy.joblib").stat().st_size)
    (output / "synthetic_calibration.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(
        name,
        "rmse", f"{report['teacher_fidelity']['validation_rmse']:.6f}",
        "balanced_saving", f"{profiles['balanced']['validation']['context_saved_pct']:.2f}%",
        "size", report["artifact_bytes"],
        flush=True,
    )
    return report


def screen(args: argparse.Namespace) -> dict:
    data = _prepare_data(args)
    names = feature_names(data["max_context"])
    if len(names) != data["train_features"].shape[1]:
        raise RuntimeError(
            f"Feature-name mismatch: {len(names)} vs {data['train_features'].shape[1]}")
    print(f"Loading teacher policy {args.teacher_policy}...", flush=True)
    teacher = joblib.load(args.teacher_policy)
    print("Predicting teacher scores for training pairs...", flush=True)
    teacher_train_score, teacher_score_config = _teacher_scores(
        teacher, data["train_features"], args.prediction_chunk)
    print("Predicting teacher scores for validation grid...", flush=True)
    teacher_val_score, _ = _teacher_scores(
        teacher, data["val_features"], args.prediction_chunk)
    if args.dense_points:
        teacher_config = teacher["profiles"]["balanced"]["config"]
        teacher_dense_profiles = _dense_profiles(
            teacher_val_score, data["validation_errors"], data["windows"],
            data["val_lengths"][data["task_series"]], args.dense_points,
            uncertainty_weight=teacher_config["uncertainty_weight"],
            harm_weight=teacher_config["harm_weight"])
        teacher_output = Path(args.output_root) / "teacher_dense"
        teacher_output.mkdir(parents=True, exist_ok=True)
        (teacher_output / "synthetic_calibration.json").write_text(json.dumps({
            "version": VERSION,
            "method": "frozen ExtraTrees teacher with dense synthetic-only thresholds",
            "model": args.model_short,
            "source_policy": args.teacher_policy,
            "real_labels_used_for_training_or_calibration": False,
            "profiles": teacher_dense_profiles,
        }, indent=2) + "\n")
    # A differentiable approximation to the teacher's binary >5%-harm term.
    # This direct target uses only realized synthetic labels and avoids the
    # discontinuity of a hard event label near the 5% boundary.
    smooth_harm = 1.0 / (1.0 + np.exp(
        -np.clip(
            (data["raw_log_risk"] - base.HARM_THRESHOLD)
            / args.direct_harm_temperature,
            -30.0,
            30.0,
        )
    ))
    direct_score = (
        data["raw_log_risk"] + args.direct_harm_weight * smooth_harm
    ).astype(np.float32)
    reports: dict[str, dict] = {}

    for alpha in args.ridge_alphas if "ridge" in args.families else []:
        suffix = f"alpha{alpha:g}".replace(".", "p")
        name = f"distill_ridge_{suffix}"
        estimator = make_pipeline(
            StandardScaler(), Ridge(alpha=float(alpha), random_state=args.seed))
        estimator.fit(
            data["train_features"], teacher_train_score,
            ridge__sample_weight=data["sample_weight"])
        reports[name] = _save_candidate(
            name, "Ridge glass-box score", estimator,
            {"alpha": float(alpha), "type": "linear"}, data,
            teacher_train_score, teacher_val_score, teacher_score_config, args)

        direct_name = f"direct_ridge_{suffix}"
        direct_estimator = make_pipeline(
            StandardScaler(), Ridge(alpha=float(alpha), random_state=args.seed))
        direct_estimator.fit(
            data["train_features"], direct_score,
            ridge__sample_weight=data["sample_weight"])
        reports[direct_name] = _save_candidate(
            direct_name, "direct Ridge glass-box score", direct_estimator,
            {
                "alpha": float(alpha),
                "type": "direct_linear",
                "target": "synthetic log-risk plus smooth harm",
                "harm_temperature": float(args.direct_harm_temperature),
                "harm_weight": float(args.direct_harm_weight),
            },
            data, teacher_train_score, teacher_val_score,
            teacher_score_config, args)

    for depth in args.tree_depths if "tree" in args.families else []:
        name = f"distill_tree_depth{depth}"
        estimator = DecisionTreeRegressor(
            max_depth=int(depth), min_samples_leaf=args.tree_min_samples_leaf,
            random_state=args.seed)
        estimator.fit(
            data["train_features"], teacher_train_score,
            sample_weight=data["sample_weight"])
        nonzero = np.flatnonzero(estimator.feature_importances_ > 0)
        top = sorted(
            ((names[index], float(estimator.feature_importances_[index]))
             for index in nonzero), key=lambda row: -row[1])
        details = {
            "type": "single_decision_tree",
            "max_depth": int(depth),
            "actual_depth": int(estimator.get_depth()),
            "leaves": int(estimator.get_n_leaves()),
            "min_samples_leaf": int(args.tree_min_samples_leaf),
            "features_used": int(len(nonzero)),
            "feature_importances": top,
        }
        reports[name] = _save_candidate(
            name, "single interpretable decision tree", estimator, details, data,
            teacher_train_score, teacher_val_score, teacher_score_config, args)
        output = Path(args.output_root) / name
        (output / "rules.txt").write_text(
            export_text(estimator, feature_names=names, decimals=5) + "\n")

        direct_name = f"direct_tree_depth{depth}"
        direct_estimator = DecisionTreeRegressor(
            max_depth=int(depth), min_samples_leaf=args.tree_min_samples_leaf,
            random_state=args.seed)
        direct_estimator.fit(
            data["train_features"], direct_score,
            sample_weight=data["sample_weight"])
        direct_nonzero = np.flatnonzero(direct_estimator.feature_importances_ > 0)
        direct_top = sorted(
            ((names[index], float(direct_estimator.feature_importances_[index]))
             for index in direct_nonzero), key=lambda row: -row[1])
        direct_details = {
            "type": "direct_single_decision_tree",
            "target": "synthetic log-risk plus smooth harm",
            "harm_temperature": float(args.direct_harm_temperature),
            "harm_weight": float(args.direct_harm_weight),
            "max_depth": int(depth),
            "actual_depth": int(direct_estimator.get_depth()),
            "leaves": int(direct_estimator.get_n_leaves()),
            "min_samples_leaf": int(args.tree_min_samples_leaf),
            "features_used": int(len(direct_nonzero)),
            "feature_importances": direct_top,
        }
        reports[direct_name] = _save_candidate(
            direct_name, "direct single interpretable decision tree",
            direct_estimator, direct_details, data, teacher_train_score,
            teacher_val_score, teacher_score_config, args)
        direct_output = Path(args.output_root) / direct_name
        (direct_output / "rules.txt").write_text(
            export_text(direct_estimator, feature_names=names, decimals=5) + "\n")

    for hidden in args.mlp_hidden if "mlp" in args.families else []:
        widths = tuple(int(value) for value in hidden.split("x") if value)
        estimator, history, parameter_count = _fit_tiny_mlp(
            data["train_features"], teacher_train_score,
            data["sample_weight"], widths, args)
        name = f"distill_tiny_mlp_{hidden}"
        reports[name] = _save_candidate(
            name, "single tiny MLP score", estimator,
            {
                "type": "single_mlp",
                "hidden": list(widths),
                "parameters": int(parameter_count),
                "epochs": int(args.mlp_epochs),
                "training_history": history,
            },
            data, teacher_train_score, teacher_val_score,
            teacher_score_config, args)

        direct_estimator, direct_history, direct_parameter_count = _fit_tiny_mlp(
            data["train_features"], direct_score,
            data["sample_weight"], widths, args, seed_offset=100)
        direct_name = f"direct_tiny_mlp_{hidden}"
        reports[direct_name] = _save_candidate(
            direct_name, "direct single tiny MLP score", direct_estimator,
            {
                "type": "direct_single_mlp",
                "target": "synthetic log-risk plus smooth harm",
                "harm_temperature": float(args.direct_harm_temperature),
                "harm_weight": float(args.direct_harm_weight),
                "hidden": list(widths),
                "parameters": int(direct_parameter_count),
                "epochs": int(args.mlp_epochs),
                "training_history": direct_history,
            },
            data, teacher_train_score, teacher_val_score,
            teacher_score_config, args)

    if "ensemble" in args.families:
        ensemble_widths = tuple(
            int(value) for value in args.ensemble_hidden.split("x") if value)
        ensemble_members = []
        ensemble_histories = []
        ensemble_parameter_count = 0
        for member_index in range(args.ensemble_size):
            member, history, parameter_count = _fit_tiny_mlp(
                data["train_features"], direct_score, data["sample_weight"],
                ensemble_widths, args, seed_offset=200 + member_index)
            ensemble_members.append(member)
            ensemble_histories.append(history)
            ensemble_parameter_count += parameter_count
        ensemble_estimator = MeanScalarEstimator(ensemble_members)
        ensemble_name = f"direct_tiny_mlp_ensemble{args.ensemble_size}_{args.ensemble_hidden}"
        reports[ensemble_name] = _save_candidate(
            ensemble_name, "direct tiny MLP mean ensemble", ensemble_estimator,
            {
                "type": "direct_mlp_ensemble",
                "target": "synthetic log-risk plus smooth harm",
                "harm_temperature": float(args.direct_harm_temperature),
                "harm_weight": float(args.direct_harm_weight),
                "hidden": list(ensemble_widths),
                "members": int(args.ensemble_size),
                "total_parameters": int(ensemble_parameter_count),
                "epochs": int(args.mlp_epochs),
                "training_history": ensemble_histories,
                "deployed_uncertainty_weight": 0.0,
            },
            data, teacher_train_score, teacher_val_score,
            teacher_score_config, args)

    summary = {
        "version": VERSION,
        "model": args.model_short,
        "selection_uses_real_data": False,
        "teacher_score": teacher_score_config,
        "candidates": {
            name: {
                "family": report["fit_details"]["type"],
                "artifact_bytes": report["artifact_bytes"],
                "teacher_fidelity": report["teacher_fidelity"],
                "profiles": {
                    profile: values["validation"]
                    for profile, values in report["profiles"].items()
                },
            }
            for name, report in reports.items()
        },
    }
    output_root = Path(args.output_root)
    (output_root / "screen_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-short", required=True)
    parser.add_argument("--synthetic-dir", required=True)
    parser.add_argument("--teacher-policy", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-train-series", type=int, default=12000)
    parser.add_argument("--max-val-series", type=int, default=2000)
    parser.add_argument("--train-pairs", type=int, default=600000)
    parser.add_argument("--prediction-chunk", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--families", nargs="+",
                        choices=["ridge", "tree", "mlp", "ensemble"],
                        default=["ridge", "tree", "mlp", "ensemble"])
    parser.add_argument("--dense-points", type=int, default=0)
    parser.add_argument("--ridge-alphas", nargs="+", type=float, default=[1.0, 100.0])
    parser.add_argument("--tree-depths", nargs="+", type=int,
                        default=[3, 4, 6, 8, 10])
    parser.add_argument("--tree-min-samples-leaf", type=int, default=64)
    parser.add_argument("--mlp-hidden", nargs="+", default=["32", "64x32"])
    parser.add_argument("--mlp-epochs", type=int, default=16)
    parser.add_argument("--mlp-batch-size", type=int, default=4096)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--torch-threads", type=int, default=16)
    parser.add_argument("--ensemble-hidden", default="64x32")
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--direct-harm-temperature", type=float, default=0.05)
    parser.add_argument("--direct-harm-weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    screen(parse_args())


if __name__ == "__main__":
    # ``python -m`` normally records local classes under ``__main__`` in
    # pickle/joblib artifacts. Re-enter through the stable import path so the
    # compact policies remain loadable from any evaluator process.
    from experiments import distill_calibrated_context_risk as stable_module

    stable_module.main()
