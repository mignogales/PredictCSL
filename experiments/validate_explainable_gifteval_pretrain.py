#!/usr/bin/env python3
"""Validate explainable context selectors on GiftEvalPretrain real curves.

The split is source-disjoint: models are fitted on ``train`` sources, candidate
selection uses only ``val`` sources, and the internal ``test`` sources are read
only after the validation winner has been frozen.  The official GiftEval test
set is never opened here.

This is deliberately a validation screen rather than a paper-result evaluator:
the cached Chronos2-Small curves contain six windows and standardized-continuation
MAE.  A successful candidate can subsequently be frozen and evaluated with the
leaderboard-faithful GiftEval MASE pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor, export_text

from experiments.explainable_selector_alternatives import (
    OrdinalQuantileTree,
    PiecewiseLinearTree,
)
from experiments.train_feature_bounded_selector import series_features


QUALITY_BUDGETS_PCT = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)


def empirical_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    rank = np.searchsorted(ordered, values, side="right")
    return (rank / max(len(ordered), 1)).astype(np.float32)


def pair_features(
    contexts: np.ndarray,
    horizons: np.ndarray,
    windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return task-by-window features and the matching task horizons."""
    base = np.stack([series_features(context) for context in contexts])
    n_series = len(base)
    n_horizons = len(horizons)
    n_windows = len(windows)
    task_base = np.repeat(base, n_horizons, axis=0)
    task_horizon = np.tile(horizons.astype(np.float32), n_series)
    task = np.column_stack([
        task_base,
        np.log1p(task_horizon),
        task_horizon / float(windows[-1]),
    ]).astype(np.float32)
    output = np.repeat(task, n_windows, axis=0)
    candidate = np.tile(windows.astype(np.float32), len(task))
    repeated_horizon = np.repeat(task_horizon, n_windows)
    output = np.column_stack([
        output,
        np.log1p(candidate),
        np.log2(candidate / float(windows[-1])),
        np.log1p(candidate / np.maximum(repeated_horizon, 1.0)),
    ]).astype(np.float32)
    return output, task_horizon


def source_balanced_weights(sources: np.ndarray) -> np.ndarray:
    names, counts = np.unique(sources, return_counts=True)
    lookup = {name: count for name, count in zip(names, counts)}
    weight = np.asarray([1.0 / lookup[name] for name in sources], dtype=np.float64)
    weight /= np.mean(weight)
    return weight.astype(np.float32)


def policy_curve(
    errors: np.ndarray,
    prediction: np.ndarray,
    windows: np.ndarray,
    dense_points: int,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Sweep a predicted-risk tolerance and measure the realized frontier."""
    if thresholds is None:
        non_native = prediction[:, :-1].ravel()
        finite = non_native[np.isfinite(non_native)]
        if not len(finite):
            raise ValueError("No finite non-native predictions")
        quantiles = np.quantile(finite, np.linspace(0.0, 1.0, dense_points))
        epsilon = max(1e-7, 1e-6 * float(np.ptp(finite) + 1.0))
        thresholds = np.unique(np.concatenate((
            [float(np.min(finite) - epsilon)], quantiles,
            [float(np.max(finite) + epsilon)],
        )))
    native = errors[:, -1]
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        safe = prediction <= float(threshold)
        safe[:, -1] = True
        action = np.argmax(safe, axis=1)
        chosen = errors[np.arange(len(errors)), action]
        ratio = chosen / np.maximum(native, 1e-12)
        selected_window = windows[action]
        rows.append({
            "threshold": float(threshold),
            "mean_ratio": float(np.mean(ratio)),
            "geomean_ratio": float(np.exp(np.mean(np.log(
                np.maximum(ratio, 1e-12))))),
            "context_saved_pct": float(100.0 * np.mean(
                1.0 - selected_window / float(windows[-1]))),
            "harm_gt_1pct_rate": float(np.mean(ratio > 1.01)),
            "harm_gt_3pct_rate": float(np.mean(ratio > 1.03)),
            "harm_gt_5pct_rate": float(np.mean(ratio > 1.05)),
            "native_action_rate": float(np.mean(action == len(windows) - 1)),
            "selection_counts": np.bincount(
                action, minlength=len(windows)).tolist(),
        })
    return rows


def select_operating_points(
    curve: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Freeze one validation operating point for each mean-error budget."""
    selected = {}
    for budget in QUALITY_BUDGETS_PCT:
        eligible = [
            row for row in curve
            if 100.0 * (float(row["mean_ratio"]) - 1.0) <= budget + 1e-12
        ]
        if not eligible:
            eligible = [min(curve, key=lambda row: float(row["mean_ratio"]))]
        point = max(eligible, key=lambda row: (
            float(row["context_saved_pct"]), -float(row["mean_ratio"]),
            -float(row["harm_gt_5pct_rate"])))
        selected[f"mae_plus_{budget:g}pct"] = point
    return selected


def pareto_summary(curve: list[dict[str, object]]) -> dict[str, object]:
    operating = select_operating_points(curve)
    by_budget = {
        name: float(row["context_saved_pct"])
        for name, row in operating.items()
    }
    return {
        "context_saved_by_mean_mae_budget_pct": by_budget,
        "mean_context_saved_across_budgets_pct": float(np.mean(
            list(by_budget.values()))),
    }


def frozen_test_report(
    validation_curve: list[dict[str, object]],
    test_curve: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Apply validation-selected thresholds without consulting test labels."""
    validation_points = select_operating_points(validation_curve)
    test_by_threshold = {
        float(row["threshold"]): row for row in test_curve
    }
    output = {}
    for name, validation in validation_points.items():
        threshold = float(validation["threshold"])
        test = test_by_threshold[threshold]
        output[name] = {
            "threshold_selected_on_validation": threshold,
            "validation_mean_mae_change_pct": float(
                100.0 * (float(validation["mean_ratio"]) - 1.0)),
            "validation_context_saved_pct": float(
                validation["context_saved_pct"]),
            "pretrain_test_mean_mae_change_pct": float(
                100.0 * (float(test["mean_ratio"]) - 1.0)),
            "pretrain_test_context_saved_pct": float(
                test["context_saved_pct"]),
            "pretrain_test_harm_gt_5pct_rate": float(
                test["harm_gt_5pct_rate"]),
            "pretrain_test_native_action_rate": float(
                test["native_action_rate"]),
        }
    return output


def compact_report(report: dict[str, object]) -> dict[str, object]:
    return {
        key: value for key, value in report.items()
        if not key.endswith("_curve")
    }


@dataclass
class Candidate:
    name: str
    family: str
    estimator: object
    target: str
    eligible: bool = True


def candidate_grid(args: argparse.Namespace) -> list[Candidate]:
    candidates: list[Candidate] = []
    for target in ("direct", "rank"):
        for depth in args.tree_depths:
            for leaf in args.tree_min_samples_leaves:
                candidates.append(Candidate(
                    f"cart_{target}_d{depth}_leaf{leaf}", "CART",
                    DecisionTreeRegressor(
                        max_depth=depth, min_samples_leaf=leaf,
                        random_state=args.seed), target))
    for bins in args.ordinal_bins:
        for depth in args.ordinal_depths:
            for leaf in args.ordinal_min_samples_leaves:
                candidates.append(Candidate(
                    f"ordinal{bins}_d{depth}_leaf{leaf}", "Ordinal tree",
                    OrdinalQuantileTree(depth, leaf, bins, args.seed), "rank"))
    for seed in args.random_tree_seeds:
        for depth in args.random_tree_depths:
            for leaf in args.random_tree_min_samples_leaves:
                candidates.append(Candidate(
                    f"random_rank_d{depth}_leaf{leaf}_s{seed}",
                    "Randomized tree",
                    DecisionTreeRegressor(
                        splitter="random", max_depth=depth,
                        min_samples_leaf=leaf, random_state=seed), "rank"))
    for target in ("direct", "rank"):
        for depth in args.model_tree_depths:
            for leaf in args.model_tree_min_samples_leaves:
                for alpha in args.model_tree_alphas:
                    suffix = f"{alpha:g}".replace(".", "p")
                    candidates.append(Candidate(
                        f"model_tree_{target}_d{depth}_leaf{leaf}_a{suffix}",
                        "Model tree",
                        PiecewiseLinearTree(depth, leaf, alpha, args.seed),
                        target))
    candidates.append(Candidate(
        "extra_trees_reference", "ExtraTrees reference",
        ExtraTreesRegressor(
            n_estimators=args.extra_trees_estimators,
            min_samples_leaf=args.extra_trees_min_samples_leaf,
            max_features=0.75, n_jobs=args.n_jobs, random_state=args.seed),
        "direct", eligible=False))
    return candidates


def estimator_complexity(estimator) -> dict[str, int]:
    if isinstance(estimator, PiecewiseLinearTree):
        tree = estimator.routing_tree_
    elif isinstance(estimator, OrdinalQuantileTree):
        tree = estimator.tree_
    elif isinstance(estimator, DecisionTreeRegressor):
        tree = estimator
    else:
        return {}
    return {
        "depth": int(tree.get_depth()),
        "leaves": int(tree.get_n_leaves()),
        "features_used": int(np.count_nonzero(tree.feature_importances_)),
    }


def write_curve_csv(path: Path, reports: list[dict[str, object]]) -> None:
    fields = [
        "candidate", "family", "split", "threshold", "mean_ratio",
        "geomean_ratio", "context_saved_pct", "harm_gt_1pct_rate",
        "harm_gt_3pct_rate", "harm_gt_5pct_rate", "native_action_rate",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for split in ("validation", "pretrain_test"):
                for row in report[f"{split}_curve"]:
                    writer.writerow({
                        "candidate": report["candidate"],
                        "family": report["family"], "split": split,
                        **{field: row[field] for field in fields[3:]},
                    })


def plot_selected_families(
    path: Path,
    reports: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    selected = {}
    for report in reports:
        family = str(report["family"])
        score = float(report["validation_pareto"][
            "mean_context_saved_across_budgets_pct"])
        if family not in selected or score > selected[family][0]:
            selected[family] = (score, report)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, split, title in zip(
        axes, ("validation", "pretrain_test"),
        ("GiftEvalPretrain validation sources", "Held-out pretraining sources"),
    ):
        for _, report in selected.values():
            curve = report[f"{split}_curve"]
            x = [100.0 * (float(row["mean_ratio"]) - 1.0) for row in curve]
            y = [float(row["context_saved_pct"]) for row in curve]
            axis.plot(x, y, marker=".", markersize=2, linewidth=1.4,
                      label=str(report["family"]))
        axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_xlim(-2.0, 6.0)
        axis.set_xlabel("Mean MAE change vs 8192 context (%)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean context saved (%)")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Explainable selector validation on source-disjoint real data")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="logs/experiments/gifteval_pretrain_bounded_wide_v5")
    parser.add_argument(
        "--output-dir",
        default="logs/experiments/gifteval_pretrain_explainable_validation_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dense-points", type=int, default=201)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--tree-depths", nargs="+", type=int, default=[6, 8, 10])
    parser.add_argument("--tree-min-samples-leaves", nargs="+", type=int,
                        default=[16, 32, 64])
    parser.add_argument("--ordinal-bins", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--ordinal-depths", nargs="+", type=int, default=[6, 8])
    parser.add_argument("--ordinal-min-samples-leaves", nargs="+", type=int,
                        default=[16, 32])
    parser.add_argument("--random-tree-seeds", nargs="+", type=int,
                        default=[42, 137, 271])
    parser.add_argument("--random-tree-depths", nargs="+", type=int,
                        default=[8, 10])
    parser.add_argument("--random-tree-min-samples-leaves", nargs="+", type=int,
                        default=[16, 32])
    parser.add_argument("--model-tree-depths", nargs="+", type=int,
                        default=[3, 4, 5])
    parser.add_argument("--model-tree-min-samples-leaves", nargs="+", type=int,
                        default=[64, 128])
    parser.add_argument("--model-tree-alphas", nargs="+", type=float,
                        default=[1.0, 100.0])
    parser.add_argument("--extra-trees-estimators", type=int, default=192)
    parser.add_argument("--extra-trees-min-samples-leaf", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "prepared" / "manifest.json").read_text())
    label_meta = json.loads(
        (root / "labels" / "chronos2_small" / "meta.json").read_text())
    windows = np.asarray(label_meta["windows"], dtype=np.int64)
    horizons = np.asarray(label_meta["horizons"], dtype=np.float32)
    contexts = np.load(root / "prepared" / "contexts.npy", mmap_mode="r")
    splits = np.load(root / "prepared" / "splits.npy")
    sources = np.load(root / "prepared" / "sources.npy")
    curves = np.load(
        root / "labels" / "chronos2_small" / "curves_mae.npy")
    if curves.shape != (len(contexts), len(windows), len(horizons)):
        raise ValueError(
            f"Unexpected curve shape {curves.shape}; expected "
            f"{(len(contexts), len(windows), len(horizons))}")

    features, _ = pair_features(contexts, horizons, windows)
    errors = curves.transpose(0, 2, 1).reshape(-1, len(windows)).astype(np.float64)
    task_splits = np.repeat(splits, len(horizons))
    task_sources = np.repeat(sources, len(horizons))
    valid_task = np.all(np.isfinite(errors) & (errors > 0.0), axis=1)
    errors = errors[valid_task]
    task_splits = task_splits[valid_task]
    task_sources = task_sources[valid_task]
    pair_valid = np.repeat(valid_task, len(windows))
    features = features[pair_valid]
    direct = np.log(errors / errors[:, -1:]).astype(np.float32).ravel()
    row_splits = np.repeat(task_splits, len(windows))
    row_sources = np.repeat(task_sources, len(windows))
    train_rows = row_splits == "train"
    rank = empirical_cdf(direct[train_rows], direct)
    weights = source_balanced_weights(row_sources[train_rows])
    task_masks = {name: task_splits == name for name in ("train", "val", "test")}
    if not all(mask.any() for mask in task_masks.values()):
        raise ValueError("train/val/test must all contain valid tasks")

    targets = {"direct": direct, "rank": rank}
    reports: list[dict[str, object]] = []
    for number, candidate in enumerate(candidate_grid(args), 1):
        target = targets[candidate.target]
        candidate.estimator.fit(
            features[train_rows], target[train_rows], sample_weight=weights)
        prediction = candidate.estimator.predict(features).reshape(-1, len(windows))
        validation_curve = policy_curve(
            errors[task_masks["val"]], prediction[task_masks["val"]],
            windows, args.dense_points)
        validation_thresholds = np.asarray([
            row["threshold"] for row in validation_curve], dtype=np.float64)
        test_curve = policy_curve(
            errors[task_masks["test"]], prediction[task_masks["test"]],
            windows, args.dense_points, thresholds=validation_thresholds)
        artifact = output / f"{candidate.name}.joblib"
        joblib.dump(candidate.estimator, artifact)
        report = {
            "candidate": candidate.name,
            "family": candidate.family,
            "target": candidate.target,
            "eligible_explainable_candidate": candidate.eligible,
            "artifact_bytes": int(artifact.stat().st_size),
            "complexity": estimator_complexity(candidate.estimator),
            "validation_pareto": pareto_summary(validation_curve),
            "pretrain_test_diagnostic_envelope": pareto_summary(test_curve),
            "pretrain_test_at_validation_selected_points": frozen_test_report(
                validation_curve, test_curve),
            "validation_curve": validation_curve,
            "pretrain_test_curve": test_curve,
        }
        reports.append(report)
        print(
            f"[{number}] {candidate.name}: val="
            f"{report['validation_pareto']['mean_context_saved_across_budgets_pct']:.3f} "
            f"test-envelope={report['pretrain_test_diagnostic_envelope']['mean_context_saved_across_budgets_pct']:.3f}",
            flush=True)

    eligible = [row for row in reports if row["eligible_explainable_candidate"]]
    ranking = sorted(eligible, key=lambda row: (
        -float(row["validation_pareto"][
            "mean_context_saved_across_budgets_pct"]),
        int(row["artifact_bytes"]), str(row["candidate"])))
    best = ranking[0]
    best_estimator = joblib.load(output / f"{best['candidate']}.joblib")
    if isinstance(best_estimator, PiecewiseLinearTree):
        tree = best_estimator.routing_tree_
    elif isinstance(best_estimator, OrdinalQuantileTree):
        tree = best_estimator.tree_
    else:
        tree = best_estimator
    (output / "selected_rules.txt").write_text(export_text(tree) + "\n")
    family_best = {}
    for row in reports:
        family = str(row["family"])
        score = float(row["validation_pareto"][
            "mean_context_saved_across_budgets_pct"])
        if family not in family_best or score > float(family_best[family][
                "validation_pareto"]["mean_context_saved_across_budgets_pct"]):
            family_best[family] = row
    summary = {
        "version": 2,
        "model": "Chronos2-Small",
        "selection_data": "GiftEvalPretrain source-disjoint validation sources",
        "official_gifteval_test_used": False,
        "metric": "MAE on context-standardized real continuations",
        "windows": windows.tolist(),
        "horizons": horizons.astype(int).tolist(),
        "source_splits": manifest["source_splits"],
        "valid_task_counts": {
            name: int(mask.sum()) for name, mask in task_masks.items()},
        "heldout_policy": (
            "operating thresholds are selected separately for each quality "
            "budget on validation sources and frozen before the pretrain-test audit"),
        "selected_candidate": compact_report(best),
        "best_by_family": {
            name: compact_report(row) for name, row in family_best.items()},
        "validation_ranking": [compact_report(row) for row in ranking],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_curve_csv(output / "all_curves.csv", reports)
    plot_selected_families(output / "pareto_validation_and_pretrain_test.png", reports)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
