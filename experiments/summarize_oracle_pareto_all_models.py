"""Combine per-model GiftEval Oracle Pareto fronts into paper-ready outputs.

The input directories are produced by ``oracle_pareto_frontier_gifteval``.
Absolute normalized MASE is retained in the CSV, while the comparison figure
divides every curve by that model's native/full normalized MASE.  This makes
the y-axis a within-model quality ratio and prevents model accuracy differences
from being mistaken for context-selection trade-offs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WEIGHTINGS = ("cell", "instances")
QUALITY_LIMITS = (
    ("no_worse_than_full", 1.0),
    ("within_0.5pct_of_full", 1.005),
    ("within_1pct_of_full", 1.01),
    ("within_2pct_of_full", 1.02),
    ("within_5pct_of_full", 1.05),
)


def _input_dir(root: Path, model: str, weighting: str) -> Path:
    suffix = "" if weighting == "cell" else "_instance_weighted"
    return root / f"{model}{suffix}"


def collect_frontiers(
    root: Path, models: Iterable[str], strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all fronts and return long-form points and operating summaries."""
    point_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    missing: list[Path] = []

    for model in models:
        for weighting in WEIGHTINGS:
            directory = _input_dir(root, model, weighting)
            frontier_path = directory / "oracle_supported_frontier.csv"
            report_path = directory / "report.json"
            if not frontier_path.is_file() or not report_path.is_file():
                missing.extend(
                    path for path in (frontier_path, report_path)
                    if not path.is_file()
                )
                continue

            frontier = pd.read_csv(frontier_path)
            with report_path.open() as stream:
                report = json.load(stream)
            full_mase = float(report["full_normalized_mase"])
            if not np.isfinite(full_mase) or full_mase <= 0:
                raise ValueError(f"Invalid full MASE in {report_path}: {full_mase}")

            # Per-model files retain the full selected-window JSON. Repeating
            # that large diagnostic payload in the cross-model long table adds
            # tens of megabytes without helping the combined analysis.
            points = frontier.drop(
                columns=["selected_windows"], errors="ignore").copy()
            points.insert(0, "model", model)
            points.insert(1, "flops_weighting", weighting)
            points["mase_ratio_to_full"] = points["normalized_mase"] / full_mase
            point_frames.append(points)

            row = {
                "model": model,
                "flops_weighting": weighting,
                "n_cells": int(report["n_cells"]),
                "n_supported_points": int(report["n_supported_points"]),
                "full_normalized_mase": full_mase,
                "oracle_normalized_mase": float(
                    report["unconstrained_oracle_normalized_mase"]),
                "oracle_quality_gain_vs_full_pct": 100.0 * (
                    1.0
                    - float(report["unconstrained_oracle_normalized_mase"])
                    / full_mase
                ),
                "oracle_flops_saved_pct": float(
                    report["unconstrained_oracle_flops_saved_pct"]),
                "maximum_supported_flops_saved_pct": float(
                    report["maximum_supported_flops_saved_pct"]),
                "minimum_compute_mase_ratio_to_full": float(
                    report["minimum_compute_normalized_mase"] / full_mase),
            }
            for label, ratio_limit in QUALITY_LIMITS:
                eligible = points[
                    points["mase_ratio_to_full"] <= ratio_limit + 1e-12
                ]
                row[f"{label}__flops_saved_pct"] = (
                    float(eligible["flops_saved_pct"].max())
                    if not eligible.empty else float("nan")
                )
            summary_rows.append(row)

    if missing and strict:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing Pareto artifacts:\n{formatted}")
    if not point_frames:
        raise ValueError(f"No Pareto frontiers found below {root}")
    return pd.concat(point_frames, ignore_index=True), pd.DataFrame(summary_rows)


def plot_combined(points: pd.DataFrame, output_path: Path) -> None:
    models = sorted(points["model"].unique())
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 0.95, len(models)))
    color_by_model = dict(zip(models, colors))
    fig, axes = plt.subplots(1, 2, figsize=(17.0, 7.2), sharey=True)

    for axis, weighting in zip(axes, WEIGHTINGS):
        subset = points[points["flops_weighting"] == weighting]
        for model in models:
            curve = subset[subset["model"] == model].sort_values(
                "flops_saved_pct")
            if curve.empty:
                continue
            axis.plot(
                curve["flops_saved_pct"], curve["mase_ratio_to_full"],
                marker="o", markersize=2.5, linewidth=1.55,
                color=color_by_model[model], label=model,
            )
        axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
        axis.set_xlim(-2.0, 100.0)
        axis.set_xlabel("Theoretical FLOPs saved vs native/full (%)")
        axis.grid(alpha=0.22)
        axis.set_title(
            "Cell-balanced compute" if weighting == "cell"
            else "Workload compute (weighted by forecast series)")

    axes[0].set_ylabel("MASE / native MASE (lower is better)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="center left", bbox_to_anchor=(0.885, 0.5),
        fontsize=8.3, frameon=True,
    )
    fig.suptitle(
        "GiftEval context-compute Pareto fronts across forecasting models\n"
        "one context action per dataset/term cell · exact supported fronts",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.88, 0.92))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", action="append", dest="models", required=True)
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="Summarize completed model/weighting pairs instead of failing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points, summary = collect_frontiers(
        args.input_root, args.models, strict=not args.allow_missing)
    points.to_csv(args.output_dir / "all_model_pareto_points.csv", index=False)
    summary.to_csv(args.output_dir / "all_model_operating_points.csv", index=False)
    plot_combined(points, args.output_dir / "all_model_pareto_frontiers.png")
    print(summary.to_string(index=False))
    print(f"Saved combined Pareto outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
