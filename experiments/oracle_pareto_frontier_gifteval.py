"""Trace the GiftEval oracle accuracy/compute Pareto frontier.

The deployment unit is one dataset-shared context action per
``(model, dataset, term)`` cell, matching Stage 4. Candidate MASE values use the
same cumulative all-instance cohort and ``mase_gluonts_real`` machinery as the
leaderboard-faithful comparison. Native context is always an action.

For every non-negative Lagrange multiplier the oracle independently chooses
the action minimizing

    mean_cell(log(MASE / SeasonalNaive)) + lambda * total_FLOPs/full_FLOPs.

Enumerating all per-cell line intersections yields the exact *supported*
Pareto frontier (the lower convex envelope of the discrete feasible set). A
multiple-choice discrete problem can contain unsupported nondominated points;
the supported envelope is the reproducible frontier relevant to tuning a
single global accuracy/compute penalty.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.compare_window_strategies_gifteval import (
    DEFAULT_PATCH_SIZES,
    _instance_oracle_from_cache,
    theoretical_flops,
)


@dataclass(frozen=True)
class Action:
    cell: str
    window: str
    mase: float
    naive_mase: float
    flops: float
    is_native: bool = False

    @property
    def normalized_mase(self) -> float:
        return self.mase / self.naive_mase


def _geomean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or np.any(~np.isfinite(array)) or np.any(array <= 0):
        return float("nan")
    return float(np.exp(np.log(array).mean()))


def pareto_prune_actions(actions: Sequence[Action]) -> List[Action]:
    """Remove actions weakly dominated inside one cell."""
    kept: List[Action] = []
    for i, action in enumerate(actions):
        dominated = False
        for j, other in enumerate(actions):
            if i == j:
                continue
            no_worse = other.mase <= action.mase and other.flops <= action.flops
            strictly_better = other.mase < action.mase or other.flops < action.flops
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            kept.append(action)
    return sorted(kept, key=lambda item: (item.flops, item.mase, item.window))


def load_cell_actions(
    run_dir: str,
    comparison_csv: str,
    model_short: str,
    mase_metric: str,
    patch_sizes: Dict[str, int],
    flops_weighting: str = "cell",
) -> Tuple[List[List[Action]], pd.DataFrame, pd.DataFrame]:
    """Load exact Stage-4-comparable grid/native actions for every cell."""
    comparison = pd.read_csv(comparison_csv)
    comparison = comparison[comparison["model_short"] == model_short].copy()
    if comparison.empty:
        raise ValueError(f"No rows for {model_short!r} in {comparison_csv}.")

    curve_key = {
        "mase_gluonts_real": "real_curve_gluonts_real",
        "mase_gluonts": "real_curve_gluonts",
    }[mase_metric]
    compare_dir = os.path.join(
        run_dir, "models", model_short, "compare_real_vs_predicted")
    groups: List[List[Action]] = []
    candidate_rows: List[dict] = []

    for row in comparison.itertuples(index=False):
        cell = f"{row.dataset_display}/t{row.term}"
        flops_multiplier = (
            float(row.n_instances) if flops_weighting == "instances" else 1.0)
        if not np.isfinite(row.naive_mase) or row.naive_mase <= 0:
            raise ValueError(f"Invalid Seasonal-Naive MASE for {cell}.")
        path = os.path.join(
            compare_dir,
            f"compare_{row.dataset_display}_t{row.term}_{model_short}.npz",
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with np.load(path) as data:
            if curve_key not in data.files:
                raise KeyError(f"{path} has no {curve_key!r}.")
            window_grid = np.asarray(data["window_grid"], dtype=np.int64)
            current_valid = np.isfinite(np.asarray(data[curve_key]))
            copied_curve = np.asarray(data[curve_key], dtype=np.float64)

        instance_eval = _instance_oracle_from_cache(
            run_dir,
            str(row.dataset_display),
            model_short,
            str(row.term),
            window_grid,
            current_valid,
            int(row.n_instances),
            mase_metric,
        )
        comparable_curve = (
            np.asarray(instance_eval["comparable_curve"], dtype=np.float64)
            if instance_eval is not None else copied_curve)

        actions = [Action(
            cell=cell,
            window="native",
            mase=float(row.full_mase),
            naive_mase=float(row.naive_mase),
            flops=float(row.full_flops) * flops_multiplier,
            is_native=True,
        )]
        # The diagnostic oracle compares every feasible grid action with native.
        # A cell with only one valid grid value is not selector-eligible, but that
        # value remains a legitimate oracle action and Stage 4 includes it when
        # computing ``best_mase`` / ``best_flops``.
        for idx in np.flatnonzero(np.isfinite(comparable_curve)):
            window = int(window_grid[idx])
            actions.append(Action(
                cell=cell,
                window=str(window),
                mase=float(comparable_curve[idx]),
                naive_mase=float(row.naive_mase),
                flops=float(theoretical_flops(
                    str(row.model), window, int(row.horizon), patch_sizes))
                * flops_multiplier,
            ))

        pruned = pareto_prune_actions(actions)
        groups.append(pruned)
        # ``Action`` is immutable but equal-valued duplicates are possible; use
        # membership equality here because this table is diagnostic only.
        for action in actions:
            candidate_rows.append({
                "cell": action.cell,
                "window": action.window,
                "mase": action.mase,
                "naive_mase": action.naive_mase,
                "normalized_mase": action.normalized_mase,
                "flops": action.flops,
                "flops_weighting": flops_weighting,
                "is_native": action.is_native,
                "within_cell_pareto": action in pruned,
            })

    candidates = pd.DataFrame(candidate_rows)
    return groups, comparison.reset_index(drop=True), candidates


def _aggregate_point(
    groups: Sequence[Sequence[Action]],
    choice_indices: Sequence[int],
    full_flops: float,
    lagrange: float,
) -> dict:
    selected = [group[index] for group, index in zip(groups, choice_indices)]
    total_flops = float(sum(action.flops for action in selected))
    numeric_windows = []
    for action in selected:
        try:
            numeric_windows.append(float(action.window))
        except ValueError:
            # Synthetic tests and future policy actions may use symbolic names.
            pass
    return {
        "lagrange": float(lagrange),
        "normalized_mase": _geomean(
            action.normalized_mase for action in selected),
        "geomean_mase": _geomean(action.mase for action in selected),
        "total_flops": total_flops,
        "flops_ratio": total_flops / full_flops,
        "flops_saved_pct": 100.0 * (1.0 - total_flops / full_flops),
        "native_cells": int(sum(action.is_native for action in selected)),
        "mean_grid_window": (
            float(np.mean(numeric_windows)) if numeric_windows else float("nan")),
        "median_grid_window": (
            float(np.median(numeric_windows)) if numeric_windows else float("nan")),
        "selected_windows": json.dumps({
            action.cell: action.window for action in selected}, sort_keys=True),
        "_choices": tuple(int(index) for index in choice_indices),
    }


def trace_supported_frontier(
    groups: Sequence[Sequence[Action]],
    full_flops: float,
) -> pd.DataFrame:
    """Enumerate every scalarization interval and retain nondominated points."""
    if not groups or full_flops <= 0:
        raise ValueError("At least one action group and positive full_flops required.")
    n_cells = len(groups)
    qualities = [
        np.asarray([
            math.log(action.normalized_mase) / n_cells for action in group
        ], dtype=np.float64)
        for group in groups
    ]
    costs = [
        np.asarray([action.flops / full_flops for action in group], dtype=np.float64)
        for group in groups
    ]

    crossings: List[float] = []
    for qvals, cvals in zip(qualities, costs):
        for left in range(len(qvals)):
            for right in range(left + 1, len(qvals)):
                denom = cvals[left] - cvals[right]
                if abs(denom) <= 1e-18:
                    continue
                value = (qvals[right] - qvals[left]) / denom
                if np.isfinite(value) and value > 0:
                    crossings.append(float(value))
    unique = np.unique(np.asarray(crossings, dtype=np.float64))
    samples: List[float] = [0.0]
    previous = 0.0
    for value in unique:
        if value <= previous:
            continue
        samples.append(
            value * 0.5 if previous == 0 else math.sqrt(previous * value))
        previous = float(value)
    samples.append((float(unique[-1]) * 2.0 + 1.0) if unique.size else 1.0)

    points: List[dict] = []
    seen_choices = set()
    for lagrange in samples:
        choices = tuple(int(np.argmin(q + lagrange * c))
                        for q, c in zip(qualities, costs))
        if choices in seen_choices:
            continue
        seen_choices.add(choices)
        points.append(_aggregate_point(groups, choices, full_flops, lagrange))

    # Sort from minimum compute upward. A point is nondominated exactly when it
    # improves quality over every cheaper point.
    points.sort(key=lambda item: (item["total_flops"], item["normalized_mase"]))
    nondominated: List[dict] = []
    best_quality = float("inf")
    for point in points:
        if point["normalized_mase"] < best_quality - 1e-12:
            nondominated.append(point)
            best_quality = point["normalized_mase"]
    frame = pd.DataFrame(nondominated)
    frame = frame.sort_values("flops_saved_pct").reset_index(drop=True)
    frame.insert(0, "frontier_index", np.arange(len(frame), dtype=int))
    return frame.drop(columns=["_choices"])


def selector_points(
    specs: Sequence[str],
    model_short: str,
    flops_weighting: str = "cell",
    patch_sizes: Dict[str, int] | None = None,
) -> pd.DataFrame:
    patch_sizes = patch_sizes or DEFAULT_PATCH_SIZES
    rows = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Selector spec {spec!r} must be LABEL=/path/comparison.csv.")
        label, path = spec.split("=", 1)
        frame = pd.read_csv(path)
        frame = frame[frame["model_short"] == model_short]
        if frame.empty:
            raise ValueError(f"No {model_short!r} rows in selector file {path}.")
        weights = (
            frame["n_instances"].astype(float)
            if flops_weighting == "instances"
            else pd.Series(1.0, index=frame.index))
        # Stage 4 stores authoritative costs for genuine native/full contexts,
        # including mixed-width batches that cannot be reconstructed from the
        # single representative ``full_window`` column. Prefer those audited
        # values so selector overlays use the same denominator as the frontier.
        pred_costs = (
            frame["pred_flops"].to_numpy(dtype=float)
            if "pred_flops" in frame
            else np.asarray([
                theoretical_flops(
                    str(row.model), int(row.pred_window), int(row.horizon),
                    patch_sizes)
                for row in frame.itertuples(index=False)
            ])
        )
        full_costs = (
            frame["full_flops"].to_numpy(dtype=float)
            if "full_flops" in frame
            else np.asarray([
                theoretical_flops(
                    str(row.model), int(row.full_window), int(row.horizon),
                    patch_sizes)
                for row in frame.itertuples(index=False)
            ])
        )
        rows.append({
            "label": label,
            "normalized_mase": _geomean(
                frame["pred_mase"] / frame["naive_mase"]),
            "flops_saved_pct": 100.0 * (
                1.0 - np.sum(pred_costs * weights.to_numpy())
                / np.sum(full_costs * weights.to_numpy())),
            "source": path,
        })
    return pd.DataFrame(rows)


def key_operating_points(
    frontier: pd.DataFrame,
    full_normalized_mase: float,
) -> pd.DataFrame:
    accuracy_best = float(frontier["normalized_mase"].min())
    records = []
    for label, limit in [
        ("no worse than full", full_normalized_mase),
        ("within 0.1% of oracle", accuracy_best * 1.001),
        ("within 0.5% of oracle", accuracy_best * 1.005),
        ("within 1% of oracle", accuracy_best * 1.01),
        ("within 2% of oracle", accuracy_best * 1.02),
        ("within 5% of oracle", accuracy_best * 1.05),
        ("within 10% of oracle", accuracy_best * 1.10),
    ]:
        eligible = frontier[frontier["normalized_mase"] <= limit + 1e-12]
        if eligible.empty:
            continue
        row = eligible.loc[eligible["flops_saved_pct"].idxmax()]
        records.append({
            "constraint": label,
            "normalized_mase_limit": limit,
            "normalized_mase": float(row.normalized_mase),
            "flops_saved_pct": float(row.flops_saved_pct),
            "native_cells": int(row.native_cells),
            "frontier_index": int(row.frontier_index),
        })
    return pd.DataFrame(records)


def plot_frontier(
    frontier: pd.DataFrame,
    reference_points: pd.DataFrame,
    key_points: pd.DataFrame,
    output_path: str,
    model_short: str,
    flops_weighting: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.0))
    for axis, zoom in zip(axes, (False, True)):
        axis.plot(
            frontier["flops_saved_pct"], frontier["normalized_mase"],
            color="#d95f02", linewidth=2.2, marker="o", markersize=3,
            label="Oracle supported frontier", zorder=2)
        for row in reference_points.itertuples(index=False):
            marker = "*" if row.label == "Unconstrained oracle" else "X"
            size = 150 if marker == "*" else 75
            axis.scatter(row.flops_saved_pct, row.normalized_mase, marker=marker,
                         s=size, label=row.label, zorder=4)
        if not key_points.empty:
            axis.scatter(key_points["flops_saved_pct"],
                         key_points["normalized_mase"], marker="s", s=32,
                         color="#7570b3", label="Quality constraints", zorder=3)
        axis.set_xlabel("Theoretical FLOPs saved vs native/full (%)")
        axis.set_ylabel("GiftEval normalized MASE (lower is better)")
        axis.grid(True, alpha=0.25)
        if zoom:
            upper = max(
                float(reference_points["normalized_mase"].max()) * 1.02,
                float(frontier["normalized_mase"].min()) * 1.08,
            )
            axis.set_ylim(float(frontier["normalized_mase"].min()) * 0.995, upper)
            axis.set_xlim(
                max(0.0, float(frontier["flops_saved_pct"].min()) - 3.0),
                min(100.0, float(key_points["flops_saved_pct"].max()) + 4.0)
                if not key_points.empty else 100.0,
            )
            axis.set_title("Useful accuracy region")
        else:
            axis.set_xlim(-3.0, 100.0)
            axis.set_title("Full supported frontier")
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[1].legend(by_label.values(), by_label.keys(), fontsize=8, loc="best")
    weighting_label = (
        "benchmark-workload FLOPs (weighted by number of series)"
        if flops_weighting == "instances" else "cell-balanced Stage-4 FLOPs")
    fig.suptitle(
        f"{model_short}: GiftEval oracle MASE/FLOPs frontier\n"
        "one oracle context action per dataset/term cell\n"
        f"{weighting_label}",
        fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="Stage-3 run directory containing models/ and datasets/.")
    parser.add_argument("--comparison-csv", required=True,
                        help="Stage-4 comparison.csv used to validate full/oracle endpoints.")
    parser.add_argument("--model", required=True, help="Model display name, e.g. Chronos2-Small.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mase-metric", choices=["mase_gluonts_real", "mase_gluonts"],
        default="mase_gluonts_real")
    parser.add_argument(
        "--patch-sizes", default="{}",
        help="Optional JSON mapping passed to the Stage-4 FLOPs model.")
    parser.add_argument(
        "--flops-weighting", choices=["cell", "instances"], default="cell",
        help=("Cell gives every GiftEval config equal compute weight, matching "
              "the Stage-4 comparison. Instances weights per-forecast FLOPs by "
              "the number of benchmark series in each cell."))
    parser.add_argument(
        "--selector", action="append", default=[],
        help="Overlay LABEL=/path/to/comparison.csv; repeat as needed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_sizes = json.loads(args.patch_sizes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups, comparison, candidates = load_cell_actions(
        args.run_dir, args.comparison_csv, args.model,
        args.mase_metric, patch_sizes, args.flops_weighting)
    flops_weights = (
        comparison["n_instances"].astype(float)
        if args.flops_weighting == "instances"
        else pd.Series(1.0, index=comparison.index))
    # These columns are the Stage-4 audited costs. In particular, native/full
    # can contain several effective context widths, so recomputing it from the
    # representative ``full_window`` changes the endpoint and denominator.
    full_costs = comparison["full_flops"].to_numpy(dtype=float)
    best_costs = comparison["best_flops"].to_numpy(dtype=float)
    full_flops = float(np.sum(full_costs * flops_weights.to_numpy()))
    frontier = trace_supported_frontier(groups, full_flops)

    full_norm = _geomean(comparison["full_mase"] / comparison["naive_mase"])
    expected_oracle_norm = _geomean(
        comparison["best_mase"] / comparison["naive_mase"])
    expected_oracle_saving = 100.0 * (
        1.0 - np.sum(best_costs * flops_weights.to_numpy()) / full_flops)
    accuracy_oracle = frontier.loc[frontier["normalized_mase"].idxmin()]
    if not math.isclose(
            float(accuracy_oracle.normalized_mase), expected_oracle_norm,
            rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError(
            "Frontier accuracy endpoint does not reproduce Stage 4: "
            f"{accuracy_oracle.normalized_mase} vs {expected_oracle_norm}.")
    if not math.isclose(
            float(accuracy_oracle.flops_saved_pct), expected_oracle_saving,
            rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError(
            "Frontier FLOPs endpoint does not reproduce Stage 4: "
            f"{accuracy_oracle.flops_saved_pct} vs {expected_oracle_saving}.")

    overlays = selector_points(
        args.selector, args.model, args.flops_weighting, patch_sizes)
    reference_rows = [
        {"label": "Full/native", "normalized_mase": full_norm,
         "flops_saved_pct": 0.0, "source": args.comparison_csv},
        {"label": "Unconstrained oracle",
         "normalized_mase": float(accuracy_oracle.normalized_mase),
         "flops_saved_pct": float(accuracy_oracle.flops_saved_pct),
         "source": args.comparison_csv},
    ]
    references = pd.concat(
        [pd.DataFrame(reference_rows), overlays], ignore_index=True)
    keys = key_operating_points(frontier, full_norm)

    candidates.to_csv(output_dir / "oracle_cell_actions.csv", index=False)
    frontier.to_csv(output_dir / "oracle_supported_frontier.csv", index=False)
    references.to_csv(output_dir / "reference_points.csv", index=False)
    keys.to_csv(output_dir / "key_operating_points.csv", index=False)
    plot_frontier(
        frontier, references, keys,
        str(output_dir / "oracle_pareto_frontier.png"), args.model,
        args.flops_weighting)

    report = {
        "model": args.model,
        "mase_metric": args.mase_metric,
        "flops_weighting": args.flops_weighting,
        "policy_scope": "one dataset-shared action per GiftEval dataset/term cell",
        "frontier_kind": "exact supported frontier from global linear scalarization",
        "n_cells": int(len(groups)),
        "n_supported_points": int(len(frontier)),
        "full_normalized_mase": full_norm,
        "full_total_flops": full_flops,
        "unconstrained_oracle_normalized_mase": expected_oracle_norm,
        "unconstrained_oracle_flops_saved_pct": expected_oracle_saving,
        "maximum_supported_flops_saved_pct": float(
            frontier["flops_saved_pct"].max()),
        "minimum_compute_normalized_mase": float(
            frontier.loc[frontier["flops_saved_pct"].idxmax(), "normalized_mase"]),
        "key_operating_points": keys.to_dict(orient="records"),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    print(f"Saved frontier artifacts to {output_dir}")


if __name__ == "__main__":
    main()
