#!/usr/bin/env python3
"""Official GiftEval test of selectors frozen on GiftEvalPretrain validation.

Only the validation-selected explainable candidate and the predeclared
ExtraTrees reference are evaluated.  Their thresholds are read from the
GiftEvalPretrain validation report and never recalibrated on GiftEval labels.
All forecast losses come from the existing leaderboard-faithful per-instance
MASE cache, so no TSFM forward pass is required.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed

from experiments.train_feature_bounded_selector import series_features


def standardize_context(values: np.ndarray, max_context: int = 8192) -> np.ndarray:
    """Match GiftEvalPretrain's finite-value context standardization."""
    context = np.asarray(values, dtype=np.float32)[-int(max_context):]
    observed = np.isfinite(context)
    if not observed.any():
        return np.zeros_like(context, dtype=np.float32)
    mean = float(np.mean(context[observed]))
    std = float(np.std(context[observed]))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    output = (context - mean) / std
    return np.nan_to_num(
        output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def usable_context(values: np.ndarray, max_context: int = 8192) -> bool:
    """Apply the same missingness/variance eligibility used in pretraining."""
    context = np.asarray(values, dtype=np.float32)[-int(max_context):]
    observed = np.isfinite(context)
    return bool(
        len(context) > 1
        and observed.mean() >= 0.80
        and observed.any()
        and np.isfinite(np.std(context[observed]))
        and float(np.std(context[observed])) >= 1e-6
    )


def context_feature_row(
    values: np.ndarray, max_context: int = 8192,
) -> tuple[bool, np.ndarray]:
    """Return eligibility and features in one parallelizable operation."""
    valid = usable_context(values, max_context)
    standardized = standardize_context(values, max_context)
    return valid, series_features(standardized)


def pair_features_from_base(
    base: np.ndarray,
    horizon: float,
    windows: np.ndarray,
) -> np.ndarray:
    """Reproduce the exact feature geometry used during pretraining."""
    base = np.asarray(base, dtype=np.float32)
    n_series = len(base)
    task_horizon = np.full(n_series, float(horizon), dtype=np.float32)
    task = np.column_stack([
        base,
        np.log1p(task_horizon),
        task_horizon / float(windows[-1]),
    ]).astype(np.float32)
    output = np.repeat(task, len(windows), axis=0)
    candidate = np.tile(windows.astype(np.float32), n_series)
    repeated_horizon = np.repeat(task_horizon, len(windows))
    return np.column_stack([
        output,
        np.log1p(candidate),
        np.log2(candidate / float(windows[-1])),
        np.log1p(candidate / np.maximum(repeated_horizon, 1.0)),
    ]).astype(np.float32)


def choose_actions(
    prediction: np.ndarray,
    threshold: float,
    windows: np.ndarray,
    native_context: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    """Choose the shortest validation-certified feasible window, else native."""
    eligible = (
        np.asarray(available, dtype=bool)
        & (windows[None, :] < native_context[:, None])
        & np.isfinite(prediction)
        & (prediction <= float(threshold))
    )
    choice = np.argmax(eligible, axis=1)
    return np.where(eligible.any(axis=1), choice, -1).astype(np.int64)


def validation_methods(summary: dict) -> list[dict[str, object]]:
    candidates = [
        ("cart_pretrain", summary["selected_candidate"]),
        ("extra_trees_pretrain", summary["best_by_family"]["ExtraTrees reference"]),
    ]
    methods = []
    for prefix, candidate in candidates:
        points = candidate["pretrain_test_at_validation_selected_points"]
        for budget_name, point in points.items():
            budget = budget_name.removeprefix("mae_plus_").removesuffix("pct")
            methods.append({
                "method": f"{prefix}_budget_{budget}pct",
                "candidate": candidate["candidate"],
                "family": candidate["family"],
                "validation_budget_pct": float(budget),
                "threshold": float(point["threshold_selected_on_validation"]),
            })
    return methods


def plot_report(path: Path, aggregate: dict[str, dict], methods: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.6, 5.2))
    styles = {
        "CART": ("#1f77b4", "o"),
        "ExtraTrees reference": ("#9467bd", "s"),
    }
    for family in styles:
        selected = [row for row in methods if row["family"] == family]
        selected.sort(key=lambda row: row["validation_budget_pct"])
        x = [aggregate[row["method"]]["theoretical_flops_saved_pct"]
             for row in selected]
        y = [100.0 * (aggregate[row["method"]]["geomean_cell_mase_ratio"] - 1.0)
             for row in selected]
        color, marker = styles[family]
        axis.plot(x, y, color=color, marker=marker, linewidth=1.8,
                  markersize=5, label=family)
        labels: dict[tuple[float, float], list[float]] = {}
        for row, px, py in zip(selected, x, y):
            labels.setdefault((round(px, 10), round(py, 10)), []).append(
                float(row["validation_budget_pct"]))
        for (px, py), budgets in labels.items():
            label = (f"{budgets[0]:g}%" if len(budgets) == 1 else
                     f"{min(budgets):g}–{max(budgets):g}%")
            axis.annotate(label, (px, py),
                          xytext=(3, 3), textcoords="offset points", fontsize=8)
    axis.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9)
    axis.set_xlabel("Theoretical TSFM FLOPs saved (%)")
    axis.set_ylabel("Official GiftEval MASE change vs native (%)")
    axis.set_title("GiftEvalPretrain-selected Chronos2-Small policies\n"
                   "Thresholds frozen before official test")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from experiments import calibrated_context_risk as base

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrain-root",
        default="logs/experiments/master_recompute/"
                "gifteval_pretrain_explainable_validation_v1")
    parser.add_argument(
        "--output-dir",
        default="logs/experiments/master_recompute/"
                "gifteval_pretrain_explainable_official_test_v1")
    parser.add_argument("--model-short", default="Chronos2-Small")
    parser.add_argument("--metric", default="mase_gluonts_real")
    parser.add_argument("--cache-roots", nargs="+", default=[
        "logs/experiments/master_recompute/window_ablation_gifteval/general",
        "logs/experiments/master_recompute/window_ablation_gifteval_grid_v2/general",
    ])
    parser.add_argument("--row-batch", type=int, default=1024)
    parser.add_argument("--feature-jobs", type=int, default=12)
    args = parser.parse_args()

    pretrain_root = Path(args.pretrain_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((pretrain_root / "summary.json").read_text())
    if summary.get("official_gifteval_test_used") is not False:
        raise ValueError("Pretraining report does not certify an untouched official test")
    if summary.get("model") != args.model_short:
        raise ValueError("Pretraining report/model mismatch")
    windows = np.asarray(summary["windows"], dtype=np.int64)
    methods = validation_methods(summary)
    models = {
        row["candidate"]: joblib.load(pretrain_root / f"{row['candidate']}.joblib")
        for row in methods
    }

    cell_paths = base._discover_cell_paths(args.cache_roots, args.model_short)
    keys = sorted(set(cell_paths) & set(base._cell_specs()))
    rows: list[dict] = []
    audits: dict[str, list[dict]] = {}
    histogram_rows: list[dict] = []
    skipped = []
    for number, (dataset, term) in enumerate(keys, 1):
        try:
            cell = base.load_real_cell(
                dataset, term, cell_paths[(dataset, term)], args.metric)
            selected_errors = []
            selected_counts = []
            for window in windows:
                hits = np.flatnonzero(cell.windows == window)
                if len(hits) == 1:
                    selected_errors.append(cell.errors[:, int(hits[0])])
                    selected_counts.append(cell.counts[:, int(hits[0])])
                elif window == windows[-1]:
                    # Chronos2-Small's 8192 cap is cached as full_native in
                    # several cells, not duplicated under a w8192 directory.
                    selected_errors.append(cell.native_error.copy())
                    selected_counts.append(cell.native_count.copy())
                else:
                    # Some short-history cells never materialize an unsupported
                    # width. Keep the cell and let the policy's availability
                    # mask reject this action for every row.
                    selected_errors.append(np.full_like(
                        cell.native_error, np.nan, dtype=np.float64))
                    selected_counts.append(np.zeros_like(
                        cell.native_count, dtype=np.float64))
            cell.windows = windows.copy()
            cell.errors = np.column_stack(selected_errors)
            cell.counts = np.column_stack(selected_counts)
            valid_row = (
                np.isfinite(cell.native_error) & (cell.native_error > 0)
                & (cell.native_count > 0)
                & (np.isfinite(cell.errors) & (cell.counts > 0)).any(axis=1)
            )
            index = np.flatnonzero(valid_row)
            cell.errors = cell.errors[index]
            cell.counts = cell.counts[index]
            cell.native_error = cell.native_error[index]
            cell.native_count = cell.native_count[index]
            cell.native_context = np.minimum(cell.native_context[index], windows[-1])
            contexts = [cell.contexts[i] for i in index]
            n = len(index)
            selector_valid = np.empty(n, dtype=bool)
            predictions = {
                candidate: np.empty((n, len(windows)), dtype=np.float32)
                for candidate in models
            }
            for start in range(0, n, args.row_batch):
                stop = min(n, start + args.row_batch)
                extracted = Parallel(
                    n_jobs=args.feature_jobs, prefer="processes", batch_size=32,
                    max_nbytes="1M")(
                        delayed(context_feature_row)(
                            context, int(windows[-1]))
                        for context in contexts[start:stop])
                selector_valid[start:stop] = [row[0] for row in extracted]
                base_features = np.stack([row[1] for row in extracted])
                features = pair_features_from_base(
                    base_features, float(cell.horizon), windows)
                for candidate, estimator in models.items():
                    predictions[candidate][start:stop] = estimator.predict(
                        features).reshape(stop - start, len(windows))
            available = np.isfinite(cell.errors) & (cell.counts > 0)
            # Out-of-distribution missing/constant contexts abstain to native.
            available &= selector_valid[:, None]
            actions = {"full_native": np.full(n, -1, dtype=np.int64)}
            for method in methods:
                actions[method["method"]] = choose_actions(
                    predictions[method["candidate"]], method["threshold"],
                    windows, cell.native_context, available)
            for method_name, action in actions.items():
                row, audit = base._cell_result(cell, method_name, action)
                row["model"] = args.model_short
                rows.append(row)
                audits.setdefault(method_name, []).append(audit)
                requested = np.where(
                    action >= 0, windows[np.maximum(action, 0)],
                    cell.native_context).astype(np.int64)
                for window, count in zip(*np.unique(requested, return_counts=True)):
                    histogram_rows.append({
                        "model": args.model_short, "dataset": cell.dataset,
                        "term": cell.term, "horizon": int(cell.horizon),
                        "method": method_name, "window_size": int(window),
                        "n_instances": int(count), "cell_instances": int(n),
                    })
            print(f"[{number}/{len(keys)}] {dataset}/{term}: n={n}", flush=True)
        except Exception as exc:
            skipped.append({"dataset": dataset, "term": term, "error": repr(exc)})
            print(f"[{number}/{len(keys)}] SKIP {dataset}/{term}: {exc!r}", flush=True)

    if not rows:
        raise RuntimeError("No official GiftEval cells evaluated")
    aggregate = base._aggregate_real(rows, audits)
    report = {
        "version": 1,
        "model": args.model_short,
        "metric": args.metric,
        "method": "GiftEvalPretrain source-disjoint selection; frozen official test",
        "official_gifteval_test_used_for_training_or_calibration": False,
        "official_gifteval_test_used_for_final_evaluation": True,
        "pretrain_summary": str(pretrain_root / "summary.json"),
        "frozen_methods": methods,
        "n_discovered_cells": int(len(keys)),
        "skipped_cells": skipped,
        "aggregate": aggregate,
    }
    (output / "real_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "real_cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "selected_window_histograms.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(histogram_rows[0]))
        writer.writeheader()
        writer.writerows(histogram_rows)
    summary_rows = []
    metadata = {row["method"]: row for row in methods}
    for method_name, metrics in aggregate.items():
        if method_name == "full_native":
            family = "Native"
            budget = 0.0
        else:
            family = str(metadata[method_name]["family"])
            budget = float(metadata[method_name]["validation_budget_pct"])
        summary_rows.append({
            "method": method_name, "family": family,
            "validation_budget_pct": budget,
            "gifteval_mase_change_pct": 100.0 * (
                metrics["geomean_cell_mase_ratio"] - 1.0),
            "theoretical_flops_saved_pct": metrics["theoretical_flops_saved_pct"],
            "instance_harm5_pct": 100.0 * metrics["instance_harm5_rate"],
            "context_saved_pct": metrics["context_saved_pct"],
            "n_cells": metrics["n_cells"], "n_instances": metrics["n_instances"],
        })
    with (output / "official_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    plot_report(output / "official_gifteval_pareto.png", aggregate, methods)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
