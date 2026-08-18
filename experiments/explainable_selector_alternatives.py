#!/usr/bin/env python3
"""Validation-only screen of explainable alternatives to ExtraTrees.

Every candidate is either one axis-aligned decision tree or one decision tree
whose leaves contain linear equations.  Candidate selection uses only held-out
synthetic validation curves; GiftEval is deliberately absent from this module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

from experiments import calibrated_context_risk as risk
from experiments import distill_calibrated_context_risk as distill
from experiments.improve_tree_distillation import (
    density_balanced_weight,
    empirical_cdf,
    fit_and_save,
    validation_pareto_summary,
)


VERSION = 1


def smooth_direct_risk(
    log_ratio: np.ndarray,
    temperature: float = 0.05,
    harm_weight: float = 0.5,
) -> np.ndarray:
    """Continuous synthetic risk target with extra weight near 5% harm."""
    value = np.asarray(log_ratio, dtype=np.float64)
    smooth_harm = 1.0 / (1.0 + np.exp(-np.clip(
        (value - risk.HARM_THRESHOLD) / float(temperature), -30.0, 30.0)))
    return (value + float(harm_weight) * smooth_harm).astype(np.float32)


def validation_log_ratio(data: dict) -> np.ndarray:
    errors = np.asarray(data["validation_errors"], dtype=np.float64)
    native = errors[:, -1][:, None]
    valid = (
        np.isfinite(errors) & np.isfinite(native)
        & (errors > 0.0) & (native > 0.0)
    )
    output = np.zeros_like(errors, dtype=np.float64)
    np.log(np.divide(errors, native, out=np.ones_like(errors), where=valid), out=output)
    return np.clip(output, -risk.LOG_RATIO_CLIP, risk.LOG_RATIO_CLIP).astype(
        np.float32).ravel()


class OrdinalQuantileTree:
    """One classifier tree whose leaf class histogram yields an ordinal score."""

    def __init__(
        self,
        max_depth: int,
        min_samples_leaf: int,
        n_bins: int,
        random_state: int,
    ) -> None:
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.n_bins = int(n_bins)
        self.random_state = int(random_state)

    def fit(self, features, target, sample_weight=None):
        rank = np.clip(np.asarray(target, dtype=np.float64), 0.0, 1.0)
        labels = np.minimum((rank * self.n_bins).astype(np.int64), self.n_bins - 1)
        self.tree_ = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.tree_.fit(features, labels, sample_weight=sample_weight)
        return self

    def predict(self, features):
        probability = self.tree_.predict_proba(features)
        centers = (self.tree_.classes_.astype(np.float64) + 0.5) / self.n_bins
        return (probability @ centers).astype(np.float32)

    @property
    def feature_importances_(self):
        return self.tree_.feature_importances_

    def get_depth(self) -> int:
        return int(self.tree_.get_depth())

    def get_n_leaves(self) -> int:
        return int(self.tree_.get_n_leaves())


class PiecewiseLinearTree:
    """One routing tree followed by an auditable Ridge equation in each leaf."""

    def __init__(
        self,
        routing_depth: int,
        min_samples_leaf: int,
        ridge_alpha: float,
        random_state: int,
    ) -> None:
        self.routing_depth = int(routing_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.ridge_alpha = float(ridge_alpha)
        self.random_state = int(random_state)

    def fit(self, features, target, sample_weight=None):
        features = np.asarray(features, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        weight = (
            np.ones(len(target), dtype=np.float32)
            if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        )
        self.routing_tree_ = DecisionTreeRegressor(
            max_depth=self.routing_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.routing_tree_.fit(features, target, sample_weight=weight)
        leaf = self.routing_tree_.apply(features)
        self.leaf_models_ = {}
        for leaf_id in np.unique(leaf):
            selected = leaf == leaf_id
            model = make_pipeline(
                StandardScaler(), Ridge(alpha=self.ridge_alpha))
            model.fit(
                features[selected], target[selected],
                ridge__sample_weight=weight[selected])
            self.leaf_models_[int(leaf_id)] = model
        return self

    def predict(self, features):
        features = np.asarray(features, dtype=np.float32)
        leaf = self.routing_tree_.apply(features)
        output = np.empty(len(features), dtype=np.float32)
        for leaf_id in np.unique(leaf):
            selected = leaf == leaf_id
            output[selected] = self.leaf_models_[int(leaf_id)].predict(
                features[selected]).astype(np.float32)
        return output

    @property
    def feature_importances_(self):
        return self.routing_tree_.feature_importances_

    def get_depth(self) -> int:
        return int(self.routing_tree_.get_depth())

    def get_n_leaves(self) -> int:
        return int(self.routing_tree_.get_n_leaves())

    def linear_leaf_report(self, feature_names: list[str]) -> dict[str, object]:
        leaves = {}
        for leaf_id, pipeline in self.leaf_models_.items():
            scaler = pipeline.named_steps["standardscaler"]
            ridge = pipeline.named_steps["ridge"]
            coefficient = ridge.coef_ / scaler.scale_
            intercept = float(ridge.intercept_ - np.dot(coefficient, scaler.mean_))
            order = np.argsort(np.abs(coefficient))[::-1]
            leaves[str(leaf_id)] = {
                "intercept": intercept,
                "coefficients": [
                    {"feature": feature_names[int(index)],
                     "coefficient": float(coefficient[int(index)])}
                    for index in order if abs(coefficient[int(index)]) > 1e-10
                ],
            }
        return {"ridge_alpha": self.ridge_alpha, "leaves": leaves}


def _tree_for_rules(estimator):
    if isinstance(estimator, OrdinalQuantileTree):
        return estimator.tree_
    if isinstance(estimator, PiecewiseLinearTree):
        return estimator.routing_tree_
    return estimator


def _save_explanation(
    output_root: str,
    name: str,
    estimator,
    feature_names: list[str],
) -> None:
    output = Path(output_root) / name
    (output / "rules.txt").write_text(
        export_text(_tree_for_rules(estimator), feature_names=feature_names,
                    decimals=5) + "\n")
    if isinstance(estimator, PiecewiseLinearTree):
        (output / "leaf_equations.json").write_text(json.dumps(
            estimator.linear_leaf_report(feature_names), indent=2) + "\n")


def screen(args: argparse.Namespace) -> dict:
    data = distill._prepare_data(args)
    feature_names = distill.feature_names(data["max_context"])
    teacher = joblib.load(args.teacher_policy)
    raw_train, teacher_config = distill._teacher_scores(
        teacher, data["train_features"], args.prediction_chunk)
    raw_val, _ = distill._teacher_scores(
        teacher, data["val_features"], args.prediction_chunk)
    rank_train = empirical_cdf(raw_train, raw_train)
    rank_val = empirical_cdf(raw_train, raw_val)
    rank_density_weight = density_balanced_weight(
        rank_train, data["sample_weight"], args.density_bins,
        args.density_max_factor)
    direct_train = smooth_direct_risk(
        data["raw_log_risk"], args.direct_harm_temperature,
        args.direct_harm_weight)
    direct_val = smooth_direct_risk(
        validation_log_ratio(data), args.direct_harm_temperature,
        args.direct_harm_weight)

    reports = {}

    def fit(name, family, estimator, target, target_val, weight, details):
        report = fit_and_save(
            name, family, estimator, target, target_val, weight, details, data,
            raw_train, raw_val, teacher_config, args)
        _save_explanation(args.output_root, name, estimator, feature_names)
        reports[name] = report

    # Existing controls.
    for mode, target, target_val, weight in (
        ("raw", raw_train, raw_val, data["sample_weight"]),
        ("rank", rank_train, rank_val, data["sample_weight"]),
        ("rank_density", rank_train, rank_val, rank_density_weight),
        ("direct", direct_train, direct_val, data["sample_weight"]),
    ) if "cart" in args.families else ():
        for depth in args.tree_depths:
            for leaf in args.tree_min_samples_leaves:
                name = f"tree_{mode}_d{depth}_leaf{leaf}"
                estimator = DecisionTreeRegressor(
                    max_depth=depth, min_samples_leaf=leaf,
                    random_state=args.seed)
                fit(name, f"single decision tree ({mode})", estimator,
                    target, target_val, weight, {
                        "type": "single_decision_tree", "target": mode,
                        "max_depth": depth, "min_samples_leaf": leaf})

    # Classification impurity gives a genuinely ordinal tree objective.
    for bins in args.ordinal_bins if "ordinal" in args.families else []:
        for depth in args.tree_depths:
            for leaf in args.tree_min_samples_leaves:
                name = f"tree_ordinal{bins}_d{depth}_leaf{leaf}"
                estimator = OrdinalQuantileTree(
                    depth, leaf, bins, args.seed)
                fit(name, "single ordinal quantile tree", estimator,
                    rank_train, rank_val, data["sample_weight"], {
                        "type": "single_ordinal_tree", "target": "teacher_rank_bins",
                        "n_bins": bins, "max_depth": depth,
                        "min_samples_leaf": leaf})

    # Random split thresholds, but still exactly one inspectable tree.
    for seed in args.random_tree_seeds if "random" in args.families else []:
        for depth in args.tree_depths:
            for leaf in args.tree_min_samples_leaves:
                name = f"tree_random_rank_d{depth}_leaf{leaf}_s{seed}"
                estimator = DecisionTreeRegressor(
                    splitter="random", max_depth=depth,
                    min_samples_leaf=leaf, random_state=seed)
                fit(name, "single randomized decision tree", estimator,
                    rank_train, rank_val, data["sample_weight"], {
                        "type": "single_randomized_tree", "target": "rank",
                        "max_depth": depth, "min_samples_leaf": leaf,
                        "random_state": seed})

    # A single rule path plus one linear equation at the reached leaf.
    for target_name, target, target_val in (
        ("rank", rank_train, rank_val),
        ("direct", direct_train, direct_val),
    ) if "model_tree" in args.families else ():
        for routing_depth in args.model_tree_depths:
            for alpha in args.model_tree_alphas:
                suffix = f"{alpha:g}".replace(".", "p")
                name = f"model_tree_{target_name}_d{routing_depth}_a{suffix}"
                estimator = PiecewiseLinearTree(
                    routing_depth, args.model_tree_min_samples_leaf,
                    alpha, args.seed)
                fit(name, "piecewise-linear model tree", estimator,
                    target, target_val, data["sample_weight"], {
                        "type": "piecewise_linear_tree", "target": target_name,
                        "routing_depth": routing_depth,
                        "min_samples_leaf": args.model_tree_min_samples_leaf,
                        "ridge_alpha": alpha})

    ranking = []
    for name, report in reports.items():
        pareto = validation_pareto_summary(report["profiles"])
        ranking.append({
            "candidate": name,
            "type": report["fit_details"]["type"],
            "artifact_bytes": int(report["artifact_bytes"]),
            "validation_pareto": pareto,
            "action_agreement": report["raw_teacher_policy_fidelity"][
                "mean_action_agreement"],
            "mean_abs_log2_window_error": report[
                "raw_teacher_policy_fidelity"]["mean_abs_log2_window_error"],
        })
    ranking.sort(key=lambda row: (
        -row["validation_pareto"]["mean_context_saved_across_budgets_pct"],
        -row["action_agreement"], row["artifact_bytes"]))
    summary = {
        "version": VERSION,
        "model": args.model_short,
        "selection_uses_real_data": False,
        "selection_data": "held-out synthetic validation curves",
        "candidates_are_forests": False,
        "best_candidate": ranking[0],
        "ranking": ranking,
    }
    output = Path(args.output_root)
    (output / "explainable_alternatives_screen.json").write_text(
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
    parser.add_argument(
        "--families", nargs="+",
        choices=["cart", "ordinal", "random", "model_tree"],
        default=["cart", "ordinal", "random", "model_tree"])
    parser.add_argument("--tree-depths", nargs="+", type=int, default=[8, 10])
    parser.add_argument("--tree-min-samples-leaves", nargs="+", type=int,
                        default=[16, 32])
    parser.add_argument("--ordinal-bins", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--random-tree-seeds", nargs="+", type=int,
                        default=[42, 137])
    parser.add_argument("--model-tree-depths", nargs="+", type=int,
                        default=[4, 5])
    parser.add_argument("--model-tree-alphas", nargs="+", type=float,
                        default=[1.0, 100.0])
    parser.add_argument("--model-tree-min-samples-leaf", type=int, default=512)
    parser.add_argument("--density-bins", type=int, default=64)
    parser.add_argument("--density-max-factor", type=float, default=4.0)
    parser.add_argument("--direct-harm-temperature", type=float, default=0.05)
    parser.add_argument("--direct-harm-weight", type=float, default=0.5)
    parser.add_argument("--fidelity-quantiles", type=int, default=21)
    return parser.parse_args()


def main() -> None:
    screen(parse_args())


if __name__ == "__main__":
    # Keep custom estimator class paths stable inside joblib artifacts.
    from experiments import explainable_selector_alternatives as stable_module
    stable_module.main()
