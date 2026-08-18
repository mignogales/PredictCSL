#!/usr/bin/env python3
"""Regenerate the vector figures derived from the frozen Section 4 tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "frozen"
FIGURES = ROOT / "figures"


def setup() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", dpi=300,
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def lag_tracking() -> None:
    controls = pd.read_csv(DATA / "exp4_controls.csv")
    summary = pd.read_csv(DATA / "exp4_lag_tracking_summary.csv")
    complete = set(summary["model"])
    data = controls[
        controls["model"].isin(complete)
        & controls["family"].eq("B")
        & controls["design_role"].eq("core")
        & controls["strength"].eq(1.0)
        & ~controls["config_broken"].astype(bool)
    ].copy()

    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h")
    for index, (model, group) in enumerate(data.groupby("model", sort=True)):
        group = group.sort_values("lag")
        ax.plot(group["lag"], group["sufficient_context"],
                marker=markers[index], ms=4.2, lw=1.35,
                color=colors[index], label=model)
    reference = np.array([32, 64, 128, 256])
    ax.plot(reference, reference, ls="--", lw=1.1, color="0.35",
            label=r"$L_{5\%}=d$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(reference, ["32", "64", "128", "256"])
    ax.set_yticks([32, 64, 128, 256, 512, 1024],
                  ["32", "64", "128", "256", "512", "1,024"])
    ax.set_xlabel("Planted dependency lag $d$ (timesteps)")
    ax.set_ylabel(r"Estimated sufficient context $L_{5\%}$")
    ax.grid(True, which="major", alpha=0.22, lw=0.6)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    save(fig, "fig2_sufficient_context_vs_lag_all_models")


def decomposition_reduction() -> None:
    data = pd.read_csv(DATA / "exp7_decomposition_summary.csv")
    data = data[data["metric"].eq("mae")]
    pivot = data.pivot(index="model", columns="variant",
                       values="mean_abs_loss_delta")
    full = pivot["attention_mask/full_history_stats"]
    tail = pivot["attention_mask/tail_matched_stats"]
    reduction = (100 * (1 - tail / full)).sort_values()

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ypos = np.arange(len(reduction))
    ax.hlines(ypos, 40, reduction.values, color="0.78", lw=1.2)
    ax.plot(reduction.values, ypos, "o", color="#3569a8", ms=5)
    for y, value in zip(ypos, reduction.values):
        label = f"{value:.1f}%" if value < 99.95 else "~100%"
        ax.text(min(value + 1.0, 100.4), y, label, va="center", ha="left",
                fontsize=7.5)
    ax.axvline(100, color="0.35", lw=0.9, ls="--")
    ax.set_yticks(ypos, reduction.index)
    ax.set_xlim(40, 107)
    ax.set_xlabel("Reduction in mean absolute masking-slicing MAE gap (%)")
    ax.grid(True, axis="x", alpha=0.22, lw=0.6)
    save(fig, "fig4_tail_stat_gap_reduction_all_models")


def main() -> None:
    setup()
    lag_tracking()
    decomposition_reduction()
    print(f"Wrote Section 4 figures to {FIGURES}")


if __name__ == "__main__":
    main()
