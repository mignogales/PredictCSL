#!/usr/bin/env python3
"""Policy-aware compact-tree distillation screen.

This experiment keeps all model selection on held-out synthetic data.  It
compares ordinary single-tree score regression with two variants intended to
preserve deployment decisions more faithfully:

* rank: regress the empirical CDF of the ExtraTrees score, because calibrated
  threshold policies depend primarily on score ordering;
* density: retain the raw score but upweight sparsely populated score regions.

Every candidate remains one auditable decision tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text

from experiments import calibrated_context_risk as risk
from experiments import distill_calibrated_context_risk as distill


VERSION = 1
QUALITY_BUDGETS_PCT = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)


def empirical_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    ranks = np.searchsorted(ordered, values, side="right")
    return (ranks / max(len(ordered), 1)).astype(np.float32)


def density_balanced_weight(
    teacher_score: np.ndarray,
    base_weight: np.ndarray,
    bins: int,
    max_factor: float,
) -> np.ndarray:
    score = np.asarray(teacher_score, dtype=np.float64)
    lower, upper = np.quantile(score, [0.001, 0.999])
    edges = np.linspace(lower, upper, int(bins) + 1)
    bucket = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, bins - 1)
    counts = np.bincount(bucket, minlength=bins).astype(np.float64)
    positive = counts[counts > 0]
    reference = float(np.median(positive)) if len(positive) else 1.0
    factor = np.sqrt(reference / np.maximum(counts[bucket], 1.0))
    factor = np.clip(factor, 1.0 / max_factor, max_factor)
    factor /= max(float(np.mean(factor)), 1e-12)
    return (np.asarray(base_weight, dtype=np.float64) * factor).astype(np.float32)


def predict(estimator, features: np.ndarray, chunk: int) -> np.ndarray:
    output = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), chunk):
        stop = min(len(features), start + chunk)
        output[start:stop] = estimator.predict(features[start:stop])
    return output


def policy_fidelity(
    teacher_score: np.ndarray,
    student_score: np.ndarray,
    data: dict,
    quantiles: int,
) -> dict[str, float]:
    n_tasks = len(data["validation_errors"])
    n_windows = len(data["windows"])
    teacher = np.asarray(teacher_score).reshape(n_tasks, n_windows)[:, :-1]
    student = np.asarray(student_score).reshape(n_tasks, n_windows)[:, :-1]
    windows = np.asarray(data["windows"][:-1], dtype=np.int64)
    native = data["val_lengths"][data["task_series"]]
    eligible = windows[None, :] < native[:, None]
    teacher_values = teacher[eligible & np.isfinite(teacher)]
    student_values = student[eligible & np.isfinite(student)]
    agreements = []
    log_errors = []
    for quantile in np.linspace(0.0, 1.0, int(quantiles)):
        teacher_threshold = float(np.quantile(teacher_values, quantile))
        student_threshold = float(np.quantile(student_values, quantile))
        teacher_action = risk.select_shortest_safe(
            teacher, teacher_threshold, windows, native)
        student_action = risk.select_shortest_safe(
            student, student_threshold, windows, native)
        teacher_window = np.where(
            teacher_action >= 0, windows[np.maximum(teacher_action, 0)], native)
        student_window = np.where(
            student_action >= 0, windows[np.maximum(student_action, 0)], native)
        agreements.append(float(np.mean(teacher_window == student_window)))
        log_errors.append(float(np.mean(np.abs(
            np.log2(np.maximum(student_window, 1))
            - np.log2(np.maximum(teacher_window, 1))))))
    return {
        "mean_action_agreement": float(np.mean(agreements)),
        "mean_abs_log2_window_error": float(np.mean(log_errors)),
        "quantile_thresholds": int(quantiles),
    }


def validation_pareto_summary(profiles: dict) -> dict[str, object]:
    dense = [
        values["validation"]
        for name, values in profiles.items()
        if name.startswith("dense_")
    ]
    if not dense:
        raise ValueError("Candidate has no dense validation profiles")
    savings = {}
    for budget in QUALITY_BUDGETS_PCT:
        eligible = [
            metrics for metrics in dense
            if 100.0 * (float(metrics["mean_ratio"]) - 1.0)
            <= budget + 1e-12
        ]
        savings[f"mase_plus_{budget:g}pct"] = (
            max(float(metrics["context_saved_pct"]) for metrics in eligible)
            if eligible else float("nan")
        )
    finite = [value for value in savings.values() if np.isfinite(value)]
    return {
        "context_saved_by_mase_budget_pct": savings,
        "mean_context_saved_across_budgets_pct": float(np.mean(finite)),
    }


def fit_and_save(
    name: str,
    family: str,
    estimator,
    target: np.ndarray,
    target_val: np.ndarray,
    weight: np.ndarray,
    fit_details: dict,
    data: dict,
    raw_teacher_train: np.ndarray,
    raw_teacher_val: np.ndarray,
    teacher_config: dict,
    args: argparse.Namespace,
) -> dict:
    estimator.fit(data["train_features"], target, sample_weight=weight)
    validation_prediction = predict(
        estimator, data["val_features"], args.prediction_chunk)
    fit_details = dict(fit_details)
    fit_details["policy_fidelity"] = policy_fidelity(
        raw_teacher_val, validation_prediction, data,
        args.fidelity_quantiles)
    score_config = dict(teacher_config)
    score_config["student_target"] = fit_details["target"]
    report = distill._save_candidate(
        name, family, estimator, fit_details, data,
        target, target_val, score_config, args)
    report["raw_teacher_policy_fidelity"] = fit_details["policy_fidelity"]
    calibration = Path(args.output_root) / name / "synthetic_calibration.json"
    calibration.write_text(json.dumps(report, indent=2) + "\n")
    return report


def screen(args: argparse.Namespace) -> dict:
    data = distill._prepare_data(args)
    names = distill.feature_names(data["max_context"])
    teacher = joblib.load(args.teacher_policy)
    raw_train, teacher_config = distill._teacher_scores(
        teacher, data["train_features"], args.prediction_chunk)
    raw_val, _ = distill._teacher_scores(
        teacher, data["val_features"], args.prediction_chunk)
    rank_train = empirical_cdf(raw_train, raw_train)
    rank_val = empirical_cdf(raw_train, raw_val)
    density_weight = density_balanced_weight(
        raw_train, data["sample_weight"], args.density_bins,
        args.density_max_factor)

    reports: dict[str, dict] = {}
    targets = {
        "raw": (raw_train, raw_val, data["sample_weight"]),
        "rank": (rank_train, rank_val, data["sample_weight"]),
        "density": (raw_train, raw_val, density_weight),
    }
    for mode in args.modes:
        target, target_val, weight = targets[mode]
        for depth in args.tree_depths:
            for leaf in args.tree_min_samples_leaves:
                name = f"tree_{mode}_d{depth}_leaf{leaf}"
                estimator = DecisionTreeRegressor(
                    max_depth=int(depth), min_samples_leaf=int(leaf),
                    random_state=args.seed)
                details = {
                    "type": "single_decision_tree",
                    "target": mode,
                    "max_depth": int(depth),
                    "min_samples_leaf": int(leaf),
                }
                report = fit_and_save(
                    name, f"policy-aware single decision tree ({mode})",
                    estimator, target, target_val, weight, details, data,
                    raw_train, raw_val, teacher_config, args)
                details = report["fit_details"]
                details.update({
                    "actual_depth": int(estimator.get_depth()),
                    "leaves": int(estimator.get_n_leaves()),
                    "features_used": int(np.count_nonzero(
                        estimator.feature_importances_)),
                })
                output = Path(args.output_root) / name
                (output / "rules.txt").write_text(
                    export_text(estimator, feature_names=names, decimals=5) + "\n")
                (output / "synthetic_calibration.json").write_text(
                    json.dumps(report, indent=2) + "\n")
                reports[name] = report

    rows = []
    for name, report in reports.items():
        fidelity = report["raw_teacher_policy_fidelity"]
        rows.append({
            "candidate": name,
            "type": report["fit_details"]["type"],
            "artifact_bytes": int(report["artifact_bytes"]),
            **fidelity,
            "validation_correlation": float(
                report["teacher_fidelity"]["validation_correlation"]),
            "validation_pareto": validation_pareto_summary(report["profiles"]),
        })
    rows.sort(key=lambda row: (
        -row["validation_pareto"]["mean_context_saved_across_budgets_pct"],
        -row["mean_action_agreement"],
        row["mean_abs_log2_window_error"],
        row["artifact_bytes"],
    ))
    single = [row for row in rows if row["type"] == "single_decision_tree"]
    summary = {
        "version": VERSION,
        "model": args.model_short,
        "selection_uses_real_data": False,
        "selection_metric": (
            "mean best context saving at fixed 0/0.1/0.5/1/2/5% mean-error "
            "budgets on held-out synthetic validation tasks"),
        "best_single_tree": single[0] if single else None,
        "ranking": single,
    }
    output = Path(args.output_root)
    (output / "improved_tree_screen.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
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
    parser.add_argument("--dense-points", type=int, default=101)
    parser.add_argument("--modes", nargs="+", choices=["raw", "rank", "density"],
                        default=["raw", "rank", "density"])
    parser.add_argument("--tree-depths", nargs="+", type=int,
                        default=[8, 10, 12])
    parser.add_argument("--tree-min-samples-leaves", nargs="+", type=int,
                        default=[16, 32, 64])
    parser.add_argument("--density-bins", type=int, default=64)
    parser.add_argument("--density-max-factor", type=float, default=4.0)
    parser.add_argument("--fidelity-quantiles", type=int, default=21)
    return parser.parse_args()


def main() -> None:
    screen(parse_args())


if __name__ == "__main__":
    main()
