#!/usr/bin/env python3
"""Regenerate the publication figures for Section 5 from frozen CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "frozen"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

ORANGE = "#D97706"
BLUE = "#2563EB"
TEAL = "#0F766E"
PURPLE = "#7C3AED"
RED = "#B91C1C"
GRAY = "#5B6472"
LIGHT = "#F4F6F8"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _box(ax, xy, width, height, title, body, color, *, dashed=False):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.5, edgecolor=color, facecolor="white",
        linestyle="--" if dashed else "-", zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.67, title, ha="center",
            va="center", weight="bold", color=color, fontsize=9.0)
    ax.text(x + width / 2, y + height * 0.32, body, ha="center",
            va="center", color="#273142", fontsize=7.7, linespacing=1.2)


def _arrow(ax, start, end, color=GRAY):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13,
        linewidth=1.35, color=color, shrinkA=4, shrinkB=4, zorder=1,
    ))


def figure_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.015, 0.92, "Offline synthetic supervision", color=ORANGE,
            weight="bold", fontsize=11)
    ax.text(0.015, 0.44, "Zero-shot real deployment", color=BLUE,
            weight="bold", fontsize=11)

    top_y, bot_y, w, h = 0.59, 0.12, 0.205, 0.22
    xs = [0.02, 0.265, 0.51, 0.755]

    _box(ax, (xs[0], top_y), w, h, "Synthetic series",
         "50,000 non-stationary\ncontexts; no GiftEval data", ORANGE)
    _box(ax, (xs[1], top_y), w, h, "Frozen TSFM labeler",
         "forecast every feasible\nwindow and horizon", ORANGE)
    _box(ax, (xs[2], top_y), w, h, "Performance surface",
         "$E_i(L_k,h)$: measured\nMAE for every action", ORANGE)
    _box(ax, (xs[3], top_y), w, h, "Train + select predictor",
         "patch encoder + horizon head;\nsynthetic validation only", ORANGE)
    for left, right in zip(xs[:-1], xs[1:]):
        _arrow(ax, (left + w, top_y + h / 2), (right, top_y + h / 2))

    _box(ax, (xs[3], bot_y), w, h, "Predictor + real cell",
         "weights + unlabeled\nGiftEval contexts", BLUE)
    _box(ax, (xs[2], bot_y), w, h, "Predict scores",
         "$\\widehat{E}_i(L_k,h)$\nfor each real series", BLUE)
    _box(ax, (xs[1], bot_y), w, h, "Aggregate + select",
         "mean score curve across\nseries in the cell", BLUE)
    _box(ax, (xs[0], bot_y), w, h, "Forecast once",
         "run the TSFM only at\nthe chosen context", BLUE)
    for right, left in zip(xs[:0:-1], xs[-2::-1]):
        _arrow(ax, (right, bot_y + h / 2),
               (left + w, bot_y + h / 2), BLUE)

    weights_x = xs[3] + w / 2
    _arrow(ax, (weights_x, top_y), (weights_x, bot_y + h), ORANGE)
    ax.text(weights_x + 0.018, 0.47, "weights only", rotation=90,
            ha="center", va="center",
            color=ORANGE, fontsize=8.5)
    ax.text(0.50, 0.02,
            "Real forecast outcomes are used only after selection, for evaluation.",
            ha="center", va="bottom", color=GRAY, fontsize=8.5)
    _save(fig, "fig1_zero_shot_pipeline")


def figure_objectives() -> None:
    obj = pd.read_csv(DATA / "chronos2_small_objectives.csv")
    oracle = pd.read_csv(DATA / "oracle_reference_points.csv")
    fig, ax = plt.subplots(figsize=(7.7, 4.7))

    full = float(obj.full_nmase.iloc[0])
    ax.axhline(full, color=GRAY, linewidth=1.1, linestyle="--",
               label="Native/full")
    ax.fill_between([0, 96], 0.698, full, color="#DCFCE7", alpha=0.38,
                    zorder=0)

    attractive = oracle[oracle.label.isin([
        "Minimum-MASE oracle", "Within 0.1% of oracle",
        "Within 0.5% of oracle", "Within 1% of oracle",
        "Within 2% of oracle", "No worse than full",
    ])].sort_values("cell_flops_saved_pct")
    ax.plot(attractive.cell_flops_saved_pct, attractive.normalized_mase,
            color="#94A3B8", linewidth=2.0, marker="o", markersize=4,
            label="Oracle reference points", zorder=1)

    colors = [ORANGE, PURPLE, TEAL, RED, BLUE]
    short = {
        "Mamba curve": "Curve",
        "Soft top-k classification": "Top-k class.",
        "Risk-aware": "Risk",
        "Adjacent pairwise": "Pairwise",
        "3% acceptable set": "3% set",
    }
    offsets = {
        "Mamba curve": (8, -23),
        "Soft top-k classification": (10, 8),
        "Risk-aware": (10, 12),
        "Adjacent pairwise": (-72, -18),
        "3% acceptable set": (10, -18),
    }
    for (_, row), color in zip(obj.iterrows(), colors):
        ax.scatter(row.cell_flops_saved_pct, row.policy_nmase, s=62,
                   color=color, edgecolor="white", linewidth=0.8, zorder=3)
        dx, dy = offsets[row.policy]
        ax.annotate(short[row.policy],
                    (row.cell_flops_saved_pct, row.policy_nmase),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.3, color=color, weight="bold")

    min_oracle = oracle[oracle.label == "Minimum-MASE oracle"].iloc[0]
    ax.annotate("Minimum-MASE oracle",
                (min_oracle.cell_flops_saved_pct, min_oracle.normalized_mase),
                xytext=(-80, -16), textcoords="offset points", fontsize=8,
                color=GRAY, arrowprops={"arrowstyle": "-", "color": GRAY})
    ax.text(1.0, full + 0.00065, "native/full = 0.7267", color=GRAY,
            fontsize=8.2)
    ax.set_xlim(0, 92)
    ax.set_ylim(0.701, 0.731)
    ax.set_xlabel("Cell-balanced theoretical TSFM FLOPs saved (%)")
    ax.set_ylabel("GiftEval normalized MASE (lower is better)")
    ax.set_title("Synthetic-only selector operating points on Chronos2-Small")
    ax.grid(axis="both", color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, loc="lower right", fontsize=8.2)
    _save(fig, "fig2_chronos_objective_operating_points")


def figure_multimodel() -> None:
    df = pd.read_csv(DATA / "zero_shot_multimodel.csv")
    order = [
        "Chronos2-Base", "Chronos2-Small", "Chronos2-Synth",
        "ChronosBolt-Base", "FlowState-R1", "Moirai2-Small",
        "PatchTST-FM-R1", "Sundial-Base-128M", "TiRex2",
        "TimesFM2.5-200M", "Toto-2.0-313m",
    ]
    y = np.arange(len(order))
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.2, 5.6), sharey=True,
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.08},
    )

    styles = {
        "PatchTST": {"color": BLUE, "marker": "o", "label": "Patch-Transformer"},
        "Mamba": {"color": ORANGE, "marker": "s", "label": "Bidirectional Mamba"},
    }
    for variant, sty in styles.items():
        sub = df[df.variant == variant].set_index("model").loc[order]
        offset = -0.11 if variant == "PatchTST" else 0.11
        ax1.scatter(sub.relative_nmase_gain_pct, y + offset, s=38,
                    color=sty["color"], marker=sty["marker"],
                    label=sty["label"], zorder=3)
        ax2.scatter(sub.cell_flops_saved_pct, y + offset, s=38,
                    color=sty["color"], marker=sty["marker"], zorder=3)

    ax1.axvline(0, color=GRAY, linestyle="--", linewidth=1)
    ax1.axvspan(0, 1.05, color="#DCFCE7", alpha=0.35, zorder=0)
    ax1.set_xlim(-3.25, 1.05)
    ax1.set_xlabel("Relative normalized-MASE gain (%)")
    ax1.set_title("Forecast quality")
    ax1.set_yticks(y)
    ax1.set_yticklabels(order)
    ax1.invert_yaxis()
    ax1.grid(axis="x", color="#E5E7EB", linewidth=0.7)

    ax2.set_xlim(0, 70)
    ax2.set_xlabel("Cell-balanced theoretical\nTSFM FLOPs saved (%)")
    ax2.set_title("Context-compute reduction")
    ax2.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax2.tick_params(axis="y", left=False, labelleft=False)

    fig.suptitle("Transfer of zero-shot curve prediction across TSFMs",
                 y=0.995, fontsize=11, weight="bold")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center",
               bbox_to_anchor=(0.53, -0.005), ncol=2, fontsize=8.2)
    fig.subplots_adjust(bottom=0.13)
    _save(fig, "fig3_zero_shot_multimodel")


def main() -> None:
    _style()
    figure_pipeline()
    figure_objectives()
    figure_multimodel()
    print(f"Wrote Section 5 figures to {FIGURES}")


if __name__ == "__main__":
    main()
