#!/usr/bin/env python3
"""Regenerate the cross-model Exp1 recency profile from saved result rows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_interpretability.analysis import aggregate as agg
from context_interpretability.schema import load_results
from paper_plot_style import (
    FIGURE_WIDTH,
    MODEL_DISPLAY,
    MODEL_ORDER,
    MUTED,
    PAPER_MPL_STYLE,
    add_figure_title,
    model_plot_kwargs,
    style_axes,
    style_legend,
)


def load_profiles(input_root: Path) -> dict[str, tuple[int, object]]:
    profiles = {}
    for model in MODEL_ORDER:
        model_root = input_root / model
        if not model_root.is_dir():
            continue
        frame = load_results(str(model_root))
        if frame.empty:
            continue
        dataset_names = frame["dataset"].astype(str)
        rows = frame[
            frame["method"].eq("perturbation")
            & frame["perturbation_type"].eq("block_mean")
            & ~dataset_names.str.match(r"^fam[ABC]_")
        ]
        if rows.empty:
            continue
        context = int(rows["context_length"].max())
        collapsed = agg.collapse_seeds(rows[rows["context_length"].eq(context)])
        profile = collapsed.groupby("lookback_start")["loss_delta"].apply(
            lambda values: float(np.nanmean(np.abs(values.to_numpy(dtype=float))))
        )
        scale = float(profile.max())
        if scale > 0:
            profiles[model] = (context, profile / scale)
    return profiles


@mpl.rc_context(PAPER_MPL_STYLE)
def plot(profiles: dict[str, tuple[int, object]], output_dir: Path) -> None:
    if not profiles:
        raise RuntimeError("No Exp1 block-mean profiles were found")

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 4.65))
    inset = inset_axes(
        ax, width="47%", height="43%", loc="upper right", borderpad=1.25,
    )
    for model in MODEL_ORDER:
        if model not in profiles:
            continue
        context, profile = profiles[model]
        x = np.maximum(profile.index.to_numpy(dtype=float), 1.0)
        y = profile.to_numpy(dtype=float)
        plot_kwargs = model_plot_kwargs(model)
        ax.plot(
            x, y,
            lw=1.35,
            ms=2.5,
            label=MODEL_DISPLAY[model],
            **plot_kwargs,
        )
        inset.plot(
            x, y, lw=1.05, ms=2.1, **plot_kwargs,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(
        [1, 32, 128, 512, 2048, 8192],
        ["current", "32", "128", "512", "2k", "8k"],
    )
    ax.set_xlim(0.75, 11000)
    ax.set_ylim(-0.015, 1.04)
    ax.set_xlabel("Perturbed-block lookback (timesteps)")
    ax.set_ylabel("Normalized mean |Δ loss|")
    style_axes(ax, grid_axis="y")
    inset.set_xscale("log", base=2)
    inset.set_xlim(16, 11000)
    inset.set_ylim(0.0, 0.4)
    inset.set_xticks([16, 32, 128, 512, 2048, 8192], ["16", "32", "128", "512", "2k", "8k"])
    inset.set_yticks([0.0, 0.2, 0.4])
    inset.tick_params(labelsize=6.2, length=2.5)
    style_axes(inset, grid_axis="y")
    inset.set_title("Zoom: normalized sensitivity 0–0.4", loc="left", fontsize=7.0, pad=3)
    ax.text(
        0.985, 0.965, "profiles normalized per model",
        transform=ax.transAxes, ha="right", va="top",
        color=MUTED, fontsize=7.0,
    )
    add_figure_title(
        fig,
        "Block-mean perturbation sensitivity by lookback",
    )
    legend = fig.legend(
        loc="lower center", bbox_to_anchor=(0.5, 0.012),
        ncol=4, frameon=True, fontsize=6.5,
        handlelength=2.4, columnspacing=1.1,
    )
    style_legend(legend)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "fig1_perturbation_recency_all_models"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(load_profiles(args.input_root), args.output_dir)


if __name__ == "__main__":
    main()
