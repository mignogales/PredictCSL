"""Generate the diagnostic real-data figures used by Section 3.

The script reads the audited GiftEval exports and writes both vector PDF and
high-resolution PNG versions for Overleaf.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_plot_style import (
    MODEL_COLORS,
    MODEL_LINE_STYLES,
    MODEL_MARKERS,
    MODEL_ORDER,
    PAPER_MPL_STYLE,
)

OUT = Path(__file__).resolve().parent / "figures"
PARETO = ROOT / "logs/experiments/master_recompute/oracle_pareto_frontier/Chronos2-Small"
COMPARISON = (
    ROOT
    / "logs/experiments/master_recompute/window_ablation_gifteval/"
    "Chronos2-Small/strategy_comparison_v4/comparison.csv"
)
MULTIMODEL_SUMMARY = Path(__file__).resolve().parent / "multimodel_oracle_summary.csv"
MULTIMODEL_FRONTIERS = (
    Path(__file__).resolve().parent
    / "data/oracle_pareto_frontier_multimodel"
)

DOMAIN_ORDER = ["Energy", "Nature", "Environment", "Health", "Transport", "Other"]
DOMAIN_KEYWORDS = [
    ("ett", "Energy"),
    ("electricity", "Energy"),
    ("solar", "Energy"),
    ("jenaweather", "Nature"),
    ("saugeen", "Nature"),
    ("kddcup", "Environment"),
    ("temprain", "Environment"),
    ("hospital", "Health"),
    ("covid", "Health"),
    ("usbirths", "Health"),
    ("loopseattle", "Transport"),
    ("sztaxi", "Transport"),
    ("mdense", "Transport"),
]


def domain_for(display: str) -> str:
    base = display.rsplit("-", 1)[0] if "-" in display else display
    key = base.lower().replace("_", "").replace("-", "")
    for keyword, domain in DOMAIN_KEYWORDS:
        if keyword in key:
            return domain
    return "Other"


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    actions = pd.read_csv(PARETO / "oracle_cell_actions.csv")
    cells = pd.read_csv(COMPARISON)
    cells["cell"] = cells["dataset_display"] + "/t" + cells["term"]
    cells["oracle_context"] = cells["best_window"].fillna(cells["full_window"])
    cells["context_ratio"] = cells["oracle_context"] / cells["full_window"]
    cells["relative_change_pct"] = 100.0 * (cells["best_mase"] / cells["full_mase"] - 1.0)
    cells.loc[cells["relative_change_pct"].abs() < 1e-6, "relative_change_pct"] = 0.0
    cells["domain"] = cells["dataset_display"].map(domain_for)
    return actions, cells


def candidate_curves(actions: pd.DataFrame, cells: pd.DataFrame) -> dict[str, pd.DataFrame]:
    native = cells.set_index("cell")
    curves: dict[str, pd.DataFrame] = {}
    for cell, group in actions.groupby("cell", sort=False):
        meta = native.loc[cell]
        numeric = group.loc[~group["is_native"], ["window", "mase"]].copy()
        numeric["window"] = pd.to_numeric(numeric["window"])
        numeric["relative_mase"] = numeric["mase"] / float(meta["full_mase"])
        numeric = numeric.sort_values("window").drop_duplicates("window", keep="last")
        curves[cell] = numeric
    return curves


def choose_cases(curves: dict[str, pd.DataFrame], cells: pd.DataFrame) -> list[tuple[str, str]]:
    """Choose well-populated examples with long native histories."""
    long_history_cases = [
        ("KDDCup2018-H/tlong", "Longer context helps"),
        ("Solar-H/tmedium", "Short context is best"),
        ("JenaWeather-10T/tlong", "Interior optimum"),
        ("BitbrainsRnD-5T/tmedium", "Broad optimum"),
    ]
    if all(cell in curves for cell, _ in long_history_cases):
        return long_history_cases

    # Fallback for exports that do not contain the fixed long-history examples.
    meta = cells.set_index("cell")
    rows: list[dict] = []
    for cell, curve in curves.items():
        if len(curve) < 5 or int(meta.loc[cell, "n_instances"]) < 80:
            continue
        x = np.log2(curve["window"].to_numpy(float))
        y = curve["relative_mase"].to_numpy(float)
        span = float(np.ptp(y))
        span_with_native = float(max(np.max(y), 1.0) - min(np.min(y), 1.0))
        idx = int(np.argmin(y))
        rank_corr = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
        edge_lift = float(min(y[0], y[-1]) / y[idx] - 1.0)
        rows.append(
            {
                "cell": cell,
                "n": int(meta.loc[cell, "n_instances"]),
                "span": span,
                "span_with_native": span_with_native,
                "idx": idx,
                "position": idx / (len(y) - 1),
                "corr": rank_corr,
                "edge_lift": edge_lift,
                "gain": -float(meta.loc[cell, "relative_change_pct"]),
            }
        )
    scores = pd.DataFrame(rows).set_index("cell")

    def select(mask: pd.Series, score: pd.Series) -> str:
        eligible = scores.loc[mask].copy()
        if eligible.empty:
            eligible = scores.copy()
        return str(score.loc[eligible.index].sort_values(ascending=False).index[0])

    # Favor clear, mid-strength examples rather than the largest outliers.
    monotonic = select(
        (scores["corr"] < -0.65) & (scores["position"] >= 0.75) & scores["span"].between(0.015, 0.20),
        -scores["corr"] + scores["span"].clip(upper=0.10),
    )
    short = select(
        (scores["corr"] > 0.55) & (scores["position"] <= 0.25) & scores["span"].between(0.015, 0.20),
        scores["corr"] + scores["span"].clip(upper=0.10),
    )
    interior = select(
        scores["position"].between(0.25, 0.75)
        & (scores["edge_lift"] > 0.008)
        & scores["span"].between(0.015, 0.25),
        scores["edge_lift"].clip(upper=0.10) + scores["span"].clip(upper=0.10),
    )
    used = {monotonic, short, interior}
    flat_pool = scores.loc[(~scores.index.isin(used)) & (scores["gain"] < 1.0)]
    flat = str(
        flat_pool.sort_values(["span_with_native", "n"], ascending=[True, False]).index[0]
    )
    return [
        (monotonic, "Longer context helps"),
        (short, "Short context is best"),
        (interior, "Interior optimum"),
        (flat, "Nearly flat"),
    ]


def plot_case_studies(actions: pd.DataFrame, cells: pd.DataFrame) -> list[tuple[str, str]]:
    curves = candidate_curves(actions, cells)
    cases = choose_cases(curves, cells)
    meta = cells.set_index("cell")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    for ax, (cell, descriptor) in zip(axes.flat, cases):
        curve = curves[cell]
        row = meta.loc[cell]
        x = curve["window"].to_numpy(float)
        y = curve["relative_mase"].to_numpy(float)
        native_x = float(row["full_window"])
        oracle_x = float(row["oracle_context"])
        oracle_y = float(row["best_mase"] / row["full_mase"])
        max_grid = float(np.max(x))

        if max_grid < native_x:
            ax.axvspan(max_grid, native_x, color="0.93", zorder=0)
        ax.plot(x, y, color="#1f77b4", marker="o", ms=3.5, lw=1.5, zorder=2)
        ax.axvline(native_x, color="0.35", ls="--", lw=1.1, zorder=1)
        ax.scatter([oracle_x], [oracle_y], color="#d95f02", edgecolor="white", lw=0.6, s=42, zorder=4)
        ax.axhline(1.0, color="0.72", ls=":", lw=0.9)
        ax.set_xscale("log", base=2)
        ax.grid(axis="y", color="0.9", lw=0.7)
        clean_cell = cell.replace("/t", " / ")
        ax.set_title(
            f"{descriptor}\n{clean_cell} "
            f"($n={int(row['n_instances']):,}$; native={int(native_x):,})",
            fontsize=8.5,
        )
        ax.set_xlabel("Context length")
        ax.set_ylabel("MASE / native MASE")
        ax.tick_params(labelsize=7.5)

    handles = [
        Line2D([0], [0], color="#1f77b4", marker="o", ms=4, lw=1.5, label="Window ablation"),
        Line2D([0], [0], color="0.35", ls="--", lw=1.1, label="Native context"),
        Line2D([0], [0], color="#d95f02", marker="o", markeredgecolor="white", lw=0, ms=6, label="Cell oracle"),
        mpl.patches.Patch(facecolor="0.93", edgecolor="none", label="Not swept (native only)"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False, fontsize=8)
    save(fig, "fig3_context_case_studies")
    return cases


def plot_heterogeneity(cells: pd.DataFrame) -> None:
    ordered_parts = []
    for domain in DOMAIN_ORDER:
        part = cells.loc[cells["domain"] == domain].sort_values(
            ["context_ratio", "relative_change_pct", "dataset_display", "term"]
        )
        ordered_parts.append(part)
    data = pd.concat(ordered_parts, ignore_index=True)
    y = np.arange(len(data))
    ratios = data["context_ratio"].to_numpy(float)
    changes = data["relative_change_pct"].to_numpy(float)
    counts = data["n_instances"].to_numpy(float)
    sizes = 10.0 + 34.0 * (
        (np.log10(counts) - np.log10(counts).min())
        / (np.log10(counts).max() - np.log10(counts).min())
    )

    fig = plt.figure(figsize=(7.2, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[5.6, 1.25], wspace=0.07)
    ax = fig.add_subplot(grid[0, 0])
    hist = fig.add_subplot(grid[0, 1])

    negative = changes[changes < 0]
    lower = float(np.percentile(negative, 5)) if negative.size else -1.0
    norm = TwoSlopeNorm(vmin=min(lower, -0.1), vcenter=-1e-9, vmax=0.1)
    cmap = mpl.colormaps["RdBu_r"]
    colors = cmap(norm(np.minimum(changes, 0.0)))

    ax.hlines(y, np.minimum(ratios, 1.0), np.maximum(ratios, 1.0), color="0.84", lw=0.7, zorder=1)
    scatter = ax.scatter(ratios, y, c=colors, s=sizes, edgecolors="0.25", linewidths=0.25, zorder=3)
    ax.axvline(1.0, color="0.30", ls="--", lw=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xlim(max(ratios.min() * 0.7, 1 / 512), 1.2)
    ax.set_ylim(-1, len(data))
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{d} / {t}" for d, t in zip(data["dataset_display"], data["term"])],
        fontsize=4.8,
    )
    ax.set_xlabel("Cell-oracle context / native context (log scale)")
    ax.grid(axis="x", which="both", color="0.91", lw=0.6)

    boundaries = []
    cursor = 0
    for domain in DOMAIN_ORDER:
        count = int((data["domain"] == domain).sum())
        if count == 0:
            continue
        if cursor:
            ax.axhline(cursor - 0.5, color="0.55", lw=0.8)
        boundaries.append((domain, cursor + (count - 1) / 2))
        cursor += count
    for domain, center in boundaries:
        ax.text(
            -0.34,
            center,
            domain,
            transform=ax.get_yaxis_transform(),
            rotation=0,
            va="center",
            ha="right",
            fontsize=6.5,
            fontweight="bold",
            clip_on=False,
        )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.025, pad=0.025, aspect=45)
    cbar.set_label("Oracle relative MASE change vs. native (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    for n in [10, 100, 1000, 10000]:
        if counts.min() <= n <= counts.max():
            s = 10.0 + 34.0 * (
                (np.log10(n) - np.log10(counts).min())
                / (np.log10(counts).max() - np.log10(counts).min())
            )
            ax.scatter([], [], s=s, facecolor="0.75", edgecolor="0.25", lw=0.25, label=f"{n:,}")
    ax.legend(
        title="Forecast instances",
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=4,
        frameon=False,
        fontsize=6.5,
        title_fontsize=7,
    )

    bins = np.geomspace(max(ratios.min(), 1 / 512), 1.0, 13)
    hist.hist(ratios, bins=bins, orientation="horizontal", color="#5b8db8", edgecolor="white", lw=0.5)
    hist.set_yscale("log", base=2)
    hist.set_ylim(ax.get_xlim())
    hist.invert_yaxis()
    hist.set_xlabel("Cells", fontsize=8)
    hist.set_ylabel("Selected ratio", fontsize=8)
    hist.grid(axis="y", which="both", color="0.92", lw=0.6)
    hist.tick_params(labelsize=7)
    hist.set_title("Marginal", fontsize=8)
    save(fig, "fig4_cell_heterogeneity")


def plot_pareto() -> None:
    frontier = pd.read_csv(PARETO / "oracle_supported_frontier.csv").sort_values("flops_saved_pct")
    refs = pd.read_csv(PARETO / "reference_points.csv")
    ops = pd.read_csv(PARETO / "key_operating_points.csv")
    report = json.loads((PARETO / "report.json").read_text())

    fig, ax = plt.subplots(figsize=(7.2, 4.25), constrained_layout=True)
    ax.plot(
        frontier["flops_saved_pct"],
        frontier["normalized_mase"],
        color="#d95f02",
        lw=2.0,
        label="Cell-shared supported frontier",
    )
    native = refs.loc[refs["label"] == "Full/native"].iloc[0]
    oracle = refs.loc[refs["label"] == "Unconstrained oracle"].iloc[0]
    ax.scatter(native["flops_saved_pct"], native["normalized_mase"], marker="X", s=75, color="#1f77b4", label="Native/full", zorder=4)
    ax.scatter(oracle["flops_saved_pct"], oracle["normalized_mase"], marker="*", s=135, color="#e66101", edgecolor="white", lw=0.5, label="Minimum-MASE oracle", zorder=5)

    labels = {
        "within 0.5% of oracle": ("0.5% tolerance", (-5, 10), "right"),
        "within 1% of oracle": ("1% tolerance", (5, 13), "left"),
        "no worse than full": ("No worse than native", (-5, 12), "right"),
    }
    chosen = ops.loc[ops["constraint"].isin(labels)].copy()
    ax.scatter(chosen["flops_saved_pct"], chosen["normalized_mase"], marker="s", s=42, color="#7570b3", label="Quality-constrained points", zorder=4)
    for row in chosen.itertuples(index=False):
        label, offset, alignment = labels[row.constraint]
        ax.annotate(
            label,
            (row.flops_saved_pct, row.normalized_mase),
            xytext=offset,
            textcoords="offset points",
            ha=alignment,
            fontsize=7,
        )

    endpoint_x = float(report["maximum_supported_flops_saved_pct"])
    endpoint_y = float(report["minimum_compute_normalized_mase"])
    ax.scatter(endpoint_x, endpoint_y, marker="D", s=42, color="0.35", label="Minimum-compute endpoint", zorder=4)
    ax.axhline(float(native["normalized_mase"]), color="#1f77b4", ls="--", lw=0.9, alpha=0.75)
    ax.set_xlabel("Theoretical FLOPs saved vs. native/full (%)")
    ax.set_ylabel("GiftEval normalized MASE (lower is better)")
    ax.set_xlim(-2, 98)
    ax.set_ylim(0.695, 1.065)
    ax.grid(color="0.9", lw=0.7)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    save(fig, "fig5_cell_oracle_pareto")


def plot_multimodel_forest() -> None:
    data = pd.read_csv(MULTIMODEL_SUMMARY).sort_values(
        "relative_mase_change_pct", ascending=True
    ).reset_index(drop=True)
    y = np.arange(len(data))
    changes = data["relative_mase_change_pct"].to_numpy(float)
    savings = data["cell_balanced_flops_saved_pct"].to_numpy(float)
    labels = data["model"].tolist()
    highlight = data["model"].eq("Chronos2-Small").to_numpy()
    colors = np.where(highlight, "#d95f02", "#2b6f9f")

    fig, (quality_ax, compute_ax) = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.8),
        sharey=True,
        gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.10},
    )

    quality_ax.hlines(y, changes, 0.0, color="0.82", lw=1.0, zorder=1)
    quality_ax.scatter(
        changes, y, c=colors, s=38,
        edgecolors="white", linewidths=0.55, zorder=3,
    )
    quality_ax.axvline(0.0, color="0.35", ls="--", lw=0.9)
    quality_ax.set_xlim(min(changes) - 0.65, 0.35)
    quality_ax.set_yticks(y)
    quality_ax.set_yticklabels(labels, fontsize=7.7)
    quality_ax.invert_yaxis()
    quality_ax.set_xlabel("Oracle MASE change vs. native (%)")
    quality_ax.set_title("Accuracy improvement", fontsize=9.5, fontweight="bold")
    quality_ax.grid(axis="x", color="0.91", lw=0.7)
    for value, ypos in zip(changes, y):
        quality_ax.annotate(
            f"{value:.2f}", (value, ypos), xytext=(-5, 0),
            textcoords="offset points", ha="right", va="center", fontsize=6.8,
        )

    compute_ax.hlines(y, 0.0, savings, color="0.82", lw=1.0, zorder=1)
    compute_ax.scatter(
        savings, y, c=colors, s=38,
        edgecolors="white", linewidths=0.55, zorder=3,
    )
    compute_ax.set_xlim(0.0, 60.0)
    compute_ax.set_xlabel("Cell-balanced theoretical FLOPs saved (%)")
    compute_ax.set_title("Compute reduction", fontsize=9.5, fontweight="bold")
    compute_ax.grid(axis="x", color="0.91", lw=0.7)
    compute_ax.tick_params(axis="y", labelleft=False)
    for value, ypos in zip(savings, y):
        compute_ax.annotate(
            f"{value:.1f}", (value, ypos), xytext=(5, 0),
            textcoords="offset points", ha="left", va="center", fontsize=6.8,
        )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d95f02",
               markeredgecolor="white", markersize=6, label="Running example"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2b6f9f",
               markeredgecolor="white", markersize=6, label="Other TSFMs"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=2,
        frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, 0.025),
    )
    fig.suptitle(
        "Cell-shared oracle opportunity across 11 TSFMs",
        fontsize=11, fontweight="bold", y=0.985,
    )
    fig.subplots_adjust(left=0.26, right=0.97, top=0.89, bottom=0.20)
    save(fig, "fig6_multimodel_oracle_forest")


@mpl.rc_context(PAPER_MPL_STYLE)
def plot_multimodel_pareto_overlay() -> None:
    """Overlay the exact supported frontiers after normalizing each native MASE to one."""
    model_order = MODEL_ORDER
    colors = MODEL_COLORS
    line_styles = MODEL_LINE_STYLES

    curves: dict[str, pd.DataFrame] = {}
    reports: dict[str, dict] = {}
    for model in model_order:
        model_dir = MULTIMODEL_FRONTIERS / model
        report = json.loads((model_dir / "report.json").read_text())
        frontier = pd.read_csv(model_dir / "oracle_supported_frontier.csv").copy()
        frontier = frontier.sort_values("flops_saved_pct")
        frontier["relative_mase"] = (
            frontier["normalized_mase"] / float(report["full_normalized_mase"])
        )
        curves[model] = frontier
        reports[model] = report

    all_x = np.concatenate([curve["flops_saved_pct"].to_numpy(float) for curve in curves.values()])
    all_y = np.concatenate([curve["relative_mase"].to_numpy(float) for curve in curves.values()])
    x_max = float(np.max(all_x))
    y_min = float(min(np.min(all_y), 1.0))

    fig, ax = plt.subplots(figsize=(7.8, 4.65))

    # The light wash makes the scientifically desirable region legible at a
    # glance without competing with the eleven model curves.
    ax.fill_between(
        [0.0, x_max + 2.0],
        y_min - 0.006,
        1.0,
        color="#EAF4EF",
        alpha=0.72,
        lw=0,
        zorder=0,
    )

    for model in model_order:
        curve = curves[model]
        report = reports[model]
        x = curve["flops_saved_pct"].to_numpy(float)
        y = curve["relative_mase"].to_numpy(float)
        is_focus = model == "Chronos2-Small"
        lw = 1.35
        zorder = 5 if is_focus else 2
        ax.plot(
            x,
            y,
            color=colors[model],
            ls=line_styles[model],
            lw=lw,
            alpha=1.0 if is_focus else 0.90,
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=zorder,
        )
        # Minimum-MASE oracle: the accuracy-optimal end of each supported curve.
        oracle_idx = int(np.argmin(y))
        ax.scatter(
            [x[oracle_idx]],
            [y[oracle_idx]],
            marker="o",
            s=28 if is_focus else 18,
            facecolor=colors[model],
            edgecolor="white",
            linewidth=0.45,
            zorder=zorder + 1,
        )
        # Furthest supported saving that remains no worse than native accuracy.
        no_worse = next(
            point for point in report["key_operating_points"]
            if point["constraint"] == "no worse than full"
        )
        ax.scatter(
            [float(no_worse["flops_saved_pct"])],
            [float(no_worse["normalized_mase"]) / float(report["full_normalized_mase"])],
            marker="s",
            s=25 if is_focus else 16,
            facecolor=colors[model],
            edgecolor="white",
            linewidth=0.45,
            zorder=zorder + 1,
        )

    ax.axhline(1.0, color="#66737F", ls=(0, (5, 3)), lw=1.0, zorder=1)
    ax.scatter(
        [0.0], [1.0], marker="X", s=46, color="#263746",
        edgecolor="white", linewidth=0.5, zorder=7,
    )
    ax.set_xlim(-2.0, x_max + 2.0)
    ax.set_ylim(y_min - 0.006, 1.05)
    ax.set_xlabel("Theoretical FLOPs saved vs. native/full (%)")
    ax.set_ylabel("Normalized MASE / native MASE (lower is better)")
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8B99A6")
    ax.tick_params(width=0.8)
    ax.text(
        0.985, 0.105, "better than native", transform=ax.transAxes,
        ha="right", va="bottom", color="#557565", fontsize=7.0,
    )

    model_handles = [
        Line2D(
            [0], [0], color=colors[model], ls=line_styles[model],
            marker=MODEL_MARKERS[model], markersize=4.5, lw=1.5,
            label=model,
        )
        for model in model_order
    ]
    marker_handles = [
        Line2D([0], [0], marker="X", color="none", markerfacecolor="0.22",
               markeredgecolor="0.22", markersize=6, label="Native/full"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.45",
               markeredgecolor="white", markersize=5, label="Minimum-MASE oracle"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="0.45",
               markeredgecolor="white", markersize=5, label="No worse than native"),
    ]
    fig.legend(
        handles=model_handles + marker_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=5,
        frameon=True,
        fontsize=7.0,
        handlelength=2.7,
        columnspacing=1.25,
    )
    legend = fig.legends[-1]
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#DDE3E8")
    legend.get_frame().set_linewidth(0.7)
    fig.suptitle(
        "Normalized oracle Pareto frontiers of TSFMs for GiftEval",
        fontsize=11,
        fontweight="semibold",
        color="#263746",
        y=0.985,
    )
    fig.subplots_adjust(left=0.11, right=0.985, top=0.90, bottom=0.25)
    save(fig, "fig7_multimodel_pareto_overlay")


@mpl.rc_context(PAPER_MPL_STYLE)
def plot_multimodel_pareto_overlay_absolute() -> None:
    """Overlay supported frontiers using absolute GiftEval normalized MASE."""
    model_order = MODEL_ORDER
    colors = MODEL_COLORS
    line_styles = MODEL_LINE_STYLES

    curves: dict[str, pd.DataFrame] = {}
    reports: dict[str, dict] = {}
    for model in model_order:
        model_dir = MULTIMODEL_FRONTIERS / model
        reports[model] = json.loads((model_dir / "report.json").read_text())
        curves[model] = pd.read_csv(
            model_dir / "oracle_supported_frontier.csv"
        ).sort_values("flops_saved_pct")

    all_x = np.concatenate([
        curve["flops_saved_pct"].to_numpy(float) for curve in curves.values()
    ])
    all_y = np.concatenate([
        curve["normalized_mase"].to_numpy(float) for curve in curves.values()
    ])
    native_y = np.array([
        float(reports[model]["full_normalized_mase"]) for model in model_order
    ])
    x_max = float(np.max(all_x))
    y_min = float(min(np.min(all_y), np.min(native_y)))
    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    for model in model_order:
        curve = curves[model]
        report = reports[model]
        x = curve["flops_saved_pct"].to_numpy(float)
        y = curve["normalized_mase"].to_numpy(float)
        native = float(report["full_normalized_mase"])
        is_focus = model == "Chronos2-Small"
        lw = 2.5 if is_focus else 1.35
        zorder = 5 if is_focus else 2
        no_worse = next(
            point for point in report["key_operating_points"]
            if point["constraint"] == "no worse than full"
        )
        oracle_idx = int(np.argmin(y))

        ax.plot(
            x,
            y,
            color=colors[model],
            ls=line_styles[model],
            lw=lw,
            alpha=1.0 if is_focus else 0.90,
            zorder=zorder,
        )
        ax.scatter(
            [0.0], [native], marker="X", s=34 if is_focus else 25,
            facecolor=colors[model], edgecolor="white", linewidth=0.45,
            zorder=zorder + 2,
        )
        ax.scatter(
            [x[oracle_idx]], [y[oracle_idx]], marker="o",
            s=28 if is_focus else 18, facecolor=colors[model],
            edgecolor="white", linewidth=0.45, zorder=zorder + 1,
        )
        ax.scatter(
            [float(no_worse["flops_saved_pct"])],
            [float(no_worse["normalized_mase"])],
            marker="s", s=25 if is_focus else 16,
            facecolor=colors[model], edgecolor="white", linewidth=0.45,
            zorder=zorder + 1,
        )

    ax.set_xlim(-2.0, x_max + 2.0)
    ax.set_ylim(y_min - 0.008, 0.85)
    ax.set_xlabel("Theoretical FLOPs saved vs. native/full (%)")
    ax.set_ylabel("GiftEval normalized MASE (absolute; lower is better)")
    ax.grid(color="0.90", lw=0.65)
    ax.tick_params(labelsize=7.5)

    model_handles = [
        Line2D(
            [0], [0], color=colors[model], ls=line_styles[model],
            marker=MODEL_MARKERS[model], markersize=4.5,
            lw=2.4 if model == "Chronos2-Small" else 1.5,
            label=model,
        )
        for model in model_order
    ]
    marker_handles = [
        Line2D([0], [0], marker="X", color="none", markerfacecolor="0.45",
               markeredgecolor="white", markersize=6, label="Native/full"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.45",
               markeredgecolor="white", markersize=5, label="Minimum-MASE oracle"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="0.45",
               markeredgecolor="white", markersize=5, label="No worse than native"),
    ]
    fig.legend(
        handles=model_handles + marker_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=5,
        frameon=False,
        fontsize=7.0,
        handlelength=2.7,
        columnspacing=1.25,
    )
    fig.suptitle(
        "Supported cell-shared Pareto frontiers across 11 TSFMs (absolute nMASE)",
        fontsize=11,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.11, right=0.985, top=0.90, bottom=0.235)
    save(fig, "fig8_multimodel_pareto_overlay_absolute")


def main() -> None:
    mpl.rcParams.update(PAPER_MPL_STYLE)
    actions, cells = load_data()
    cases = plot_case_studies(actions, cells)
    plot_pareto()
    plot_multimodel_forest()
    plot_multimodel_pareto_overlay()
    plot_multimodel_pareto_overlay_absolute()
    print("Selected case studies:")
    for cell, descriptor in cases:
        print(f"  {descriptor}: {cell}")
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
