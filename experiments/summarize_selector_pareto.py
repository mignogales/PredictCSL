"""Compare achieved GiftEval compute/quality fronts for two frozen selectors.

The inputs are real-evaluation reports whose thresholds were calibrated only
on held-out synthetic data.  No Oracle action is introduced here: every point
is an actually evaluated policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SELECTORS = ("ExtraTrees", "Depth-8 tree")
QUALITY_BUDGETS = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)


def parse_model_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected DISPLAY=SLUG")
    display, slug = value.split("=", 1)
    if not display or not slug:
        raise argparse.ArgumentTypeError("expected non-empty DISPLAY=SLUG")
    return display, slug


def is_dominated(row: pd.Series, candidates: pd.DataFrame) -> bool:
    no_worse = (
        (candidates["flops_saved_pct"] >= float(row.flops_saved_pct))
        & (candidates["mase_change_pct"] <= float(row.mase_change_pct))
    )
    strict = (
        (candidates["flops_saved_pct"] > float(row.flops_saved_pct) + 1e-12)
        | (candidates["mase_change_pct"] < float(row.mase_change_pct) - 1e-12)
    )
    return bool((no_worse & strict).any())


def pareto_mask(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [not is_dominated(row, frame) for _, row in frame.iterrows()],
        index=frame.index,
        dtype=bool,
    )


def load_report(path: Path, model: str, selector: str) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    if payload.get("model") != model:
        raise ValueError(f"{path} is for {payload.get('model')}, expected {model}")
    rows = []
    for method, metrics in payload["aggregate"].items():
        if not (method.startswith("dense_") or method == "full_native"):
            continue
        ratio = float(metrics["geomean_cell_mase_ratio"])
        rows.append({
            "model": model,
            "selector": selector,
            "method": method,
            "mase_ratio": ratio,
            "mase_change_pct": 100.0 * (ratio - 1.0),
            "flops_saved_pct": float(metrics["theoretical_flops_saved_pct"]),
            "harm5_pct": 100.0 * float(metrics["instance_harm5_rate"]),
            "coverage_pct": 100.0 * float(metrics["coverage"]),
            "n_cells": int(metrics["n_cells"]),
            "n_instances": int(metrics["n_instances"]),
            "source": str(path),
        })
    if not rows:
        raise ValueError(f"No dense achieved policies in {path}")
    frame = pd.DataFrame(rows).sort_values(
        ["flops_saved_pct", "mase_change_pct", "method"])
    # Dense calibration can repeat a threshold. Keep one representative so a
    # selector is not visually rewarded for duplicate operating points.
    return frame.drop_duplicates(
        ["flops_saved_pct", "mase_change_pct"], keep="first")


def summarize_points(
    points: pd.DataFrame,
    selectors: tuple[str, str] = SELECTORS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    points = points.copy()
    points["on_selector_pareto"] = False
    points["on_union_pareto"] = False
    for model in points.model.unique():
        model_mask = points.model == model
        union = points.loc[model_mask]
        points.loc[union.index, "on_union_pareto"] = pareto_mask(union)
        for selector in selectors:
            mask = model_mask & (points.selector == selector)
            subset = points.loc[mask]
            points.loc[subset.index, "on_selector_pareto"] = pareto_mask(subset)

    summary_rows = []
    for (model, selector), group in points.groupby(["model", "selector"]):
        row = {"model": model, "selector": selector}
        for budget in QUALITY_BUDGETS:
            eligible = group[group.mase_change_pct <= budget + 1e-12]
            key = f"max_flops_saved_at_mase_plus_{budget:g}pct"
            row[key] = (
                float(eligible.flops_saved_pct.max())
                if not eligible.empty else float("nan"))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    aggregate_rows = []
    for selector in selectors:
        selected = summary[summary.selector == selector]
        for budget in QUALITY_BUDGETS:
            column = f"max_flops_saved_at_mase_plus_{budget:g}pct"
            values = selected[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            aggregate_rows.append({
                "selector": selector,
                "mase_budget_pct": budget,
                "n_models": int(values.size),
                "median_flops_saved_pct": float(np.median(values)),
                "q25_flops_saved_pct": float(np.quantile(values, 0.25)),
                "q75_flops_saved_pct": float(np.quantile(values, 0.75)),
                "min_flops_saved_pct": float(values.min()),
                "max_flops_saved_pct": float(values.max()),
            })
    return points, summary, pd.DataFrame(aggregate_rows)


def collect(
    extra_root: Path,
    compact_root: Path,
    model_specs: list[tuple[str, str]],
    compact_label: str = "Depth-8 tree",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for model, slug in model_specs:
        frames.append(load_report(
            extra_root / slug / "full_score" / "real_evaluation.json",
            model, "ExtraTrees"))
        frames.append(load_report(
            compact_root / slug / "full_score" / "real_evaluation.json",
            model, compact_label))
    return summarize_points(
        pd.concat(frames, ignore_index=True), ("ExtraTrees", compact_label))


def tirex_unpadded_savings(histogram_path: Path) -> dict[str, float]:
    """Counterfactual TiRex MAC savings if only left padding is removed.

    The public wrapper still defines the quality measurements.  For cost only,
    this keeps TiRex's fixed 928-point future grid and sign-flip TTA while
    allowing the number of context tokens processed by the backbone to vary.
    """
    histogram = pd.read_csv(histogram_path)
    keys = ["dataset", "term", "horizon"]
    native = (
        histogram[histogram.method.astype(str) == "full_native"]
        [keys + ["window_size"]]
        .drop_duplicates()
        .rename(columns={"window_size": "native_window"})
    )
    if native.empty:
        raise ValueError(f"No full_native rows in {histogram_path}")
    frame = histogram.merge(native, on=keys, how="left", validate="many_to_one")
    if frame.native_window.isna().any():
        raise ValueError(f"Missing TiRex native windows in {histogram_path}")

    # Each forecast retains the complete 928-point future region. Patch size is
    # 32. The sign-flip and recursive-call multipliers cancel within a cell but
    # remain here so horizons/cells receive their correct relative weight.
    calls = 2 * np.ceil(frame.horizon.astype(float) / 928.0).astype(int)
    selected_tokens = np.ceil(
        (frame.window_size.astype(float) + 928.0) / 32.0)
    native_tokens = np.ceil(
        (frame.native_window.astype(float) + 928.0) / 32.0)
    frame["selected_counterfactual_macs"] = (
        frame.n_instances.astype(float) * calls * selected_tokens)
    frame["native_counterfactual_macs"] = (
        frame.n_instances.astype(float) * calls * native_tokens)
    totals = frame.groupby("method")[[
        "selected_counterfactual_macs", "native_counterfactual_macs"
    ]].sum()
    savings = 100.0 * (
        1.0
        - totals.selected_counterfactual_macs
        / totals.native_counterfactual_macs
    )
    return {str(method): float(value) for method, value in savings.items()}


def with_tirex_unpadded_counterfactual(
    points: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replace only TiRex's fixed-wrapper cost with unpadded backbone cost."""
    counterfactual = points.copy()
    counterfactual["tirex_unpadded_counterfactual"] = False
    tirex = counterfactual.model.astype(str) == "TiRex2"
    for source in counterfactual.loc[tirex, "source"].unique():
        source_mask = tirex & (counterfactual.source == source)
        savings = tirex_unpadded_savings(
            Path(source).parent / "selected_window_histograms.csv")
        mapped = counterfactual.loc[source_mask, "method"].map(savings)
        if mapped.isna().any():
            missing = counterfactual.loc[source_mask & mapped.isna(), "method"]
            raise ValueError(
                f"Missing TiRex counterfactual costs for {missing.tolist()}")
        counterfactual.loc[source_mask, "flops_saved_pct"] = mapped.to_numpy()
        counterfactual.loc[source_mask, "tirex_unpadded_counterfactual"] = True
    selectors = tuple(dict.fromkeys(counterfactual.selector.astype(str)))
    if len(selectors) != 2:
        raise ValueError(f"Expected two selectors, found {selectors}")
    return summarize_points(counterfactual, selectors)


def plot_facets(
    points: pd.DataFrame,
    output: Path,
    useful_limit: float,
    tirex_counterfactual: bool = False,
) -> None:
    models = list(dict.fromkeys(points.model.tolist()))
    ncols = 3
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(15.5, 3.7 * nrows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    selectors = tuple(dict.fromkeys(points.selector.astype(str)))
    colors = {selectors[0]: "#1f77b4", selectors[1]: "#d95f02"}
    markers = {selectors[0]: "o", selectors[1]: "s"}
    for axis, model in zip(axes, models):
        subset = points[points.model == model]
        for selector in selectors:
            curve = subset[
                (subset.selector == selector) & subset.on_selector_pareto
            ].sort_values("flops_saved_pct")
            axis.plot(
                curve.flops_saved_pct, curve.mase_change_pct,
                color=colors[selector], marker=markers[selector],
                markersize=2.8, linewidth=1.7, label=selector)
            union = curve[curve.on_union_pareto]
            axis.scatter(
                union.flops_saved_pct, union.mase_change_pct,
                color=colors[selector], marker=markers[selector], s=25,
                edgecolor="white", linewidth=0.45, zorder=3)
        axis.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9)
        axis.set_title(model, fontsize=10.5, fontweight="bold")
        axis.grid(alpha=0.22)
        axis.set_xlim(-2.0, 100.0)
        axis.set_ylim(-0.4, useful_limit)
    for axis in axes[len(models):]:
        axis.axis("off")
    for row in range(nrows):
        axes[row * ncols].set_ylabel("GiftEval MASE change vs native (%)")
    for axis in axes[-ncols:]:
        if axis.axison:
            axis.set_xlabel("Theoretical TSFM FLOPs saved (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.952),
        ncol=2, frameon=True)
    subtitle = (
        "TiRex2 cost is counterfactual unpadded-backbone FLOPs; all quality is "
        "from the public wrapper"
        if tirex_counterfactual else
        "thresholds calibrated on held-out synthetic data; GiftEval used only "
        "for evaluation"
    )
    fig.suptitle(
        f"Achieved context-selection Pareto fronts: {selectors[0]} vs "
        f"{selectors[1]}\n" + subtitle,
        fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.905))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate(
    aggregate: pd.DataFrame,
    output: Path,
    tirex_counterfactual: bool = False,
) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 5.4))
    selectors = tuple(dict.fromkeys(aggregate.selector.astype(str)))
    colors = {selectors[0]: "#1f77b4", selectors[1]: "#d95f02"}
    for selector in selectors:
        data = aggregate[aggregate.selector == selector].sort_values(
            "mase_budget_pct")
        budgets = data.mase_budget_pct.to_numpy(dtype=float)
        medians = data.median_flops_saved_pct.to_numpy(dtype=float)
        lower = data.q25_flops_saved_pct.to_numpy(dtype=float)
        upper = data.q75_flops_saved_pct.to_numpy(dtype=float)
        axis.plot(
            budgets, medians,
            marker="o", linewidth=2.2, color=colors[selector], label=selector)
        axis.fill_between(
            budgets, lower, upper, color=colors[selector], alpha=0.16)
    axis.set_xlabel("Allowed GiftEval MASE increase vs native (%)")
    axis.set_ylabel("Median best measured FLOPs saving across models (%)")
    axis.set_xlim(-0.05, max(QUALITY_BUDGETS) + 0.05)
    axis.set_ylim(0.0, 100.0)
    axis.grid(alpha=0.25)
    axis.legend()
    title = "Achieved selector trade-off across 11 models"
    if tirex_counterfactual:
        title += "\nTiRex2 uses counterfactual unpadded cost"
    else:
        title += "\nline: median; band: interquartile range"
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-root", required=True, type=Path)
    parser.add_argument("--compact-root", required=True, type=Path)
    parser.add_argument("--model", action="append", type=parse_model_spec,
                        dest="models", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--useful-mase-limit", type=float, default=5.0)
    parser.add_argument("--compact-label", default="Depth-8 tree")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points, summary, aggregate = collect(
        args.extra_root, args.compact_root, args.models, args.compact_label)
    points.to_csv(args.output_dir / "all_achieved_operating_points.csv", index=False)
    points[points.on_union_pareto].to_csv(
        args.output_dir / "union_pareto_points.csv", index=False)
    summary.to_csv(args.output_dir / "quality_budget_summary.csv", index=False)
    aggregate.to_csv(args.output_dir / "cross_model_summary.csv", index=False)
    plot_facets(
        points, args.output_dir / "selector_pareto_by_model.png",
        args.useful_mase_limit)
    plot_aggregate(aggregate, args.output_dir / "selector_pareto_cross_model.png")
    counterfactual, counterfactual_summary, counterfactual_aggregate = (
        with_tirex_unpadded_counterfactual(points))
    counterfactual.to_csv(
        args.output_dir / "all_achieved_operating_points_tirex_unpadded.csv",
        index=False)
    counterfactual_summary.to_csv(
        args.output_dir / "quality_budget_summary_tirex_unpadded.csv",
        index=False)
    counterfactual_aggregate.to_csv(
        args.output_dir / "cross_model_summary_tirex_unpadded.csv",
        index=False)
    plot_facets(
        counterfactual,
        args.output_dir / "selector_pareto_by_model_tirex_unpadded.png",
        args.useful_mase_limit,
        tirex_counterfactual=True)
    plot_aggregate(
        counterfactual_aggregate,
        args.output_dir / "selector_pareto_cross_model_tirex_unpadded.png",
        tirex_counterfactual=True)
    report = {
        "comparison": f"achieved ExtraTrees versus {args.compact_label}",
        "oracle_actions_used": False,
        "models": [model for model, _ in args.models],
        "n_models": len(args.models),
        "n_points": len(points),
        "n_union_pareto": int(points.on_union_pareto.sum()),
        "threshold_calibration": "held-out synthetic data only",
        "final_evaluation": "GiftEval",
        "counterfactual_outputs": {
            "scope": "TiRex2 cost only",
            "assumption": (
                "remove left-context padding; retain fixed 928-point future "
                "grid, sign-flip TTA, and public-wrapper quality"),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
