#!/usr/bin/env python3
"""MLP ablation for the synthetic-only calibrated context-risk policy.

This keeps the data split, sampled series/window/horizon pairs, engineered
features, targets, calibration search, and real-data evaluation from
``calibrated_context_risk.py`` fixed.  It replaces the ExtraTrees regressor and
classifier with a small multi-task deep ensemble.  Ensemble variation supplies
the predictive standard deviation consumed by the existing risk calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments import calibrated_context_risk as base


VERSION = 1
PROFILE_LIMITS = {
    **base.PROFILE_LIMITS,
    "extreme": {"harm5_rate": 0.150, "mean_ratio": 1.040},
    "very_extreme": {"harm5_rate": 0.200, "mean_ratio": 1.070},
}
# The published ExtraTrees experiment added these two efficiency endpoints in
# a follow-up calibration pass.  Register the identical limits before calling
# the shared calibration routine so the MLP emits all five paper profiles.
base.PROFILE_LIMITS = PROFILE_LIMITS


class RiskMLP(nn.Module):
    """Shared representation with independent risk and harm heads."""

    def __init__(self, n_features: int, hidden: Sequence[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        width = int(n_features)
        for next_width in hidden:
            layers.extend([
                nn.Linear(width, int(next_width)),
                nn.LayerNorm(int(next_width)),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
            ])
            width = int(next_width)
        self.trunk = nn.Sequential(*layers)
        self.risk_head = nn.Linear(width, 1)
        self.harm_head = nn.Linear(width, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        return self.risk_head(hidden).squeeze(-1), self.harm_head(hidden).squeeze(-1)


class FrozenMLPMember:
    """Pickle-friendly sklearn-like view of one ensemble member."""

    def __init__(
        self,
        n_features: int,
        hidden: Sequence[int],
        dropout: float,
        state_dict: dict[str, torch.Tensor],
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        prediction_batch_size: int,
    ):
        self.n_features = int(n_features)
        self.hidden = tuple(int(value) for value in hidden)
        self.dropout = float(dropout)
        self.state_dict = {key: value.detach().cpu() for key, value in state_dict.items()}
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float32)
        self.prediction_batch_size = int(prediction_batch_size)
        self._network: RiskMLP | None = None

    def _model(self) -> RiskMLP:
        if self._network is None:
            network = RiskMLP(self.n_features, self.hidden, self.dropout)
            network.load_state_dict(self.state_dict)
            network.eval()
            self._network = network
        return self._network

    def _outputs(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(features, dtype=np.float32)
        output_risk = np.empty(len(features), dtype=np.float32)
        output_harm = np.empty(len(features), dtype=np.float32)
        network = self._model()
        with torch.inference_mode():
            for start in range(0, len(features), self.prediction_batch_size):
                stop = min(len(features), start + self.prediction_batch_size)
                normalized = (
                    features[start:stop] - self.feature_mean
                ) / self.feature_scale
                risk, harm = network(torch.from_numpy(normalized))
                output_risk[start:stop] = risk.numpy()
                output_harm[start:stop] = torch.sigmoid(harm).numpy()
        return output_risk, output_harm

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._outputs(features)[0]

    def predict_harm(self, features: np.ndarray) -> np.ndarray:
        return self._outputs(features)[1]


class MLPRegressorEnsemble:
    """Container exposing ``estimators_`` for the established risk scorer."""

    def __init__(self, members: Sequence[FrozenMLPMember]):
        self.estimators_ = list(members)


class MLPClassifierEnsemble:
    """Average member harm probabilities behind sklearn's classifier API."""

    classes_ = np.asarray([0, 1], dtype=np.int64)

    def __init__(self, members: Sequence[FrozenMLPMember]):
        self.members = list(members)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = np.mean(
            [member.predict_harm(features) for member in self.members], axis=0)
        return np.column_stack([1.0 - positive, positive])


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _fit_member(
    features: np.ndarray,
    target: np.ndarray,
    harm: np.ndarray,
    sample_weight: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    args: argparse.Namespace,
    member_index: int,
) -> tuple[FrozenMLPMember, list[dict[str, float]]]:
    seed = int(args.seed) + 1009 * (member_index + 1)
    _seed_everything(seed)
    device = torch.device(args.mlp_device)
    network = RiskMLP(features.shape[1], args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs))

    normalized = ((features - feature_mean) / feature_scale).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(normalized),
        torch.from_numpy(np.asarray(target, dtype=np.float32)),
        torch.from_numpy(np.asarray(harm, dtype=np.float32)),
        torch.from_numpy(np.asarray(sample_weight, dtype=np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    harm_count = float(np.sum(harm))
    non_harm_count = float(len(harm) - harm_count)
    positive_class_weight = non_harm_count / max(harm_count, 1.0)
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        network.train()
        total_regression = 0.0
        total_classification = 0.0
        total_weight = 0.0
        for batch_features, batch_target, batch_harm, batch_weight in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            batch_harm = batch_harm.to(device, non_blocking=True)
            batch_weight = batch_weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predicted_risk, harm_logit = network(batch_features)
            regression_loss = (predicted_risk - batch_target).square()
            class_weight = torch.where(
                batch_harm > 0.5,
                torch.full_like(batch_harm, positive_class_weight),
                torch.ones_like(batch_harm),
            )
            classification_loss = nn.functional.binary_cross_entropy_with_logits(
                harm_logit, batch_harm, reduction="none") * class_weight
            denominator = batch_weight.sum().clamp_min(1e-8)
            weighted_regression = (regression_loss * batch_weight).sum() / denominator
            weighted_classification = (
                classification_loss * batch_weight).sum() / denominator
            loss = weighted_regression + args.harm_loss_weight * weighted_classification
            loss.backward()
            if args.gradient_clip > 0:
                nn.utils.clip_grad_norm_(network.parameters(), args.gradient_clip)
            optimizer.step()
            total_regression += float((regression_loss * batch_weight).sum().detach())
            total_classification += float(
                (classification_loss * batch_weight).sum().detach())
            total_weight += float(denominator.detach())
        scheduler.step()
        row = {
            "epoch": int(epoch + 1),
            "regression_mse": total_regression / max(total_weight, 1e-8),
            "classification_bce": total_classification / max(total_weight, 1e-8),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        print(
            f"member={member_index + 1}/{args.ensemble_size} "
            f"epoch={epoch + 1}/{args.epochs} "
            f"mse={row['regression_mse']:.6f} "
            f"bce={row['classification_bce']:.6f}",
            flush=True,
        )

    member = FrozenMLPMember(
        features.shape[1], args.hidden, args.dropout, network.state_dict(),
        feature_mean, feature_scale, args.mlp_prediction_batch_size)
    return member, history


def train_policy(args: argparse.Namespace) -> dict:
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
    if curves.shape != (len(contexts), len(windows), len(horizons)):
        raise ValueError(f"Unexpected synthetic curve shape {curves.shape}")

    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(contexts))
    n_val = min(int(args.max_val_series), max(1, len(order) // 5))
    n_train = min(int(args.max_train_series), len(order) - n_val)
    val_index = np.sort(order[:n_val])
    train_index = np.sort(order[n_val:n_val + n_train])
    selected = np.concatenate([train_index, val_index])
    prepared = np.asarray(contexts[selected, -max_context:], dtype=np.float32)
    selected_lengths = np.minimum(lengths[selected], max_context).astype(np.int64)
    print(f"Extracting synthetic features for {len(selected):,} series...", flush=True)
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
    target = np.log(raw_error[valid] / native_error[valid])
    target_clipped = np.clip(target, -base.LOG_RATIO_CLIP, base.LOG_RATIO_CLIP)
    train_features = base.make_pair_features(
        train_base[series_pos], train_lengths[series_pos],
        windows[window_pos], horizons[horizon_pos], max_context)
    sample_weight = (
        1.0 + 2.0 * (target > base.HARM_THRESHOLD)
        + 0.25 * np.abs(target_clipped)
    ).astype(np.float32)
    harm = (target > base.HARM_THRESHOLD).astype(np.float32)
    feature_mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = train_features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = np.where(feature_scale > 1e-5, feature_scale, 1.0).astype(np.float32)

    if args.mlp_device == "cpu":
        torch.set_num_threads(args.torch_threads)
    members: list[FrozenMLPMember] = []
    histories: list[list[dict[str, float]]] = []
    for member_index in range(args.ensemble_size):
        member, history = _fit_member(
            train_features, target_clipped, harm, sample_weight,
            feature_mean, feature_scale, args, member_index)
        members.append(member)
        histories.append(history)
    regressor = MLPRegressorEnsemble(members)
    classifier = MLPClassifierEnsemble(members)

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
    predicted_mean, predicted_std, predicted_harm = base.predict_risk(
        regressor, classifier, val_features, args.prediction_chunk)
    shape = (n_tasks, len(windows))
    predicted_mean = predicted_mean.reshape(shape)[:, :-1]
    predicted_std = predicted_std.reshape(shape)[:, :-1]
    predicted_harm = predicted_harm.reshape(shape)[:, :-1]
    validation_errors = np.asarray(
        curves[val_index[task_series], :, task_horizon], dtype=np.float64)
    profiles, candidates = base._calibrate_profiles(
        predicted_mean, predicted_std, predicted_harm,
        validation_errors, windows, val_lengths[task_series],
        args.uncertainty_weights, args.harm_weights,
        args.calibration_quantiles)

    parameter_count = sum(parameter.numel() for parameter in RiskMLP(
        train_features.shape[1], args.hidden, args.dropout).parameters())
    report = {
        "version": VERSION,
        "method": "multi-task MLP deep ensemble + synthetic-only shortest-safe calibration",
        "baseline_control": "calibrated_context_risk ExtraTrees; data, features, targets, split, sampled pairs, and calibration held fixed",
        "model": args.model_short,
        "model_id": meta["model_id"],
        "synthetic_dir": str(model_dir),
        "real_labels_used_for_training_or_calibration": False,
        "target": "log(MAE_window / MAE_native), clipped for regression; >5% harm classified separately",
        "n_train_series": int(len(train_index)),
        "n_validation_series": int(len(val_index)),
        "n_train_pairs": int(len(train_features)),
        "n_validation_tasks": int(n_tasks),
        "n_features": int(train_features.shape[1]),
        "windows": windows.tolist(),
        "horizons": horizons.tolist(),
        "max_context": max_context,
        "architecture": {
            "hidden": list(args.hidden),
            "dropout": float(args.dropout),
            "ensemble_size": int(args.ensemble_size),
            "parameters_per_member": int(parameter_count),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "harm_loss_weight": float(args.harm_loss_weight),
        },
        "training_history": histories,
        "profiles": profiles,
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
            "scales": [value for value in base.DEFAULT_SCALES if value <= max_context],
            "lags": list(base.DEFAULT_LAGS),
            "max_context": max_context,
        },
        "training_report": report,
    }, output / "policy.joblib", compress=3)
    (output / "synthetic_calibration.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value["validation"] for key, value in profiles.items()}, indent=2))
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
    parser.add_argument("--max-val-series", type=int, default=2000)
    parser.add_argument("--train-pairs", type=int, default=600000)
    parser.add_argument("--uncertainty-weights", nargs="+", type=float,
                        default=[0.0, 0.5, 1.0, 1.645])
    parser.add_argument("--harm-weights", nargs="+", type=float,
                        default=[0.0, 0.1, 0.25, 0.5])
    parser.add_argument("--calibration-quantiles", type=int, default=81)
    parser.add_argument("--prediction-chunk", type=int, default=32768)
    parser.add_argument("--prediction-row-batch", type=int, default=256)
    parser.add_argument("--max-real-cells", type=int, default=0)
    parser.add_argument("--max-real-instances-per-cell", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", nargs="+", type=int, default=[256, 128, 64])
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--harm-loss-weight", type=float, default=0.25)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--mlp-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--torch-threads", type=int, default=16)
    parser.add_argument("--mlp-prediction-batch-size", type=int, default=32768)
    args = parser.parse_args()
    if args.stage in {"evaluate", "all"} and not args.cache_roots:
        parser.error("--cache-roots is required for real evaluation")
    if args.calibration_quantiles < 2:
        parser.error("--calibration-quantiles must be >= 2")
    if args.ensemble_size < 2:
        parser.error("--ensemble-size must be >= 2 for uncertainty estimation")
    return args


def main() -> None:
    args = parse_args()
    if args.stage in {"train", "all"}:
        train_policy(args)
    if args.stage in {"evaluate", "all"}:
        base.evaluate_real(args)


if __name__ == "__main__":
    main()
