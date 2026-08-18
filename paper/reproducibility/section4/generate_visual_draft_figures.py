from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_plot_style import (
    FIGURE_WIDTH,
    GRID,
    INK,
    MODEL_COLORS,
    MODEL_DISPLAY,
    MODEL_LINE_STYLES,
    MODEL_MARKERS,
    MODEL_ORDER,
    MUTED,
    PAPER_MPL_STYLE,
    add_figure_title,
    model_legend_handles,
    model_plot_kwargs,
    style_axes,
    style_legend,
)

FROZEN = ROOT / "overleaf_section4" / "data" / "frozen"
RAW = ROOT / "overleaf_section4" / "data" / "frozen_input"
OUT = ROOT / "overleaf_section4" / "figures" / "visual_draft"

DECOMPOSITION_MODEL_ORDER = [
    "Chronos2-Base",
    "Chronos2-Small",
    "Chronos2-Synth",
    "ChronosBolt-Base",
    "Moirai2-Small",
    "PatchTST-FM-R1",
    "Sundial-Base-128M",
    "TimesFM2.5-200M",
]
SLICE = "#263238"
FULL = "#C74B50"
TAIL = "#128C8C"
BLUE = "#2E74B5"
GOLD = "#A97500"


def configure() -> None:
    mpl.rcParams.update(PAPER_MPL_STYLE)
    mpl.rcParams.update({"savefig.bbox": "tight", "savefig.pad_inches": 0.08})


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=260)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def context_error_curve() -> None:
    df = pd.read_csv(FROZEN / "chronos2_small_exp1_curve.csv").sort_values("context_length")
    x = df["context_length"].to_numpy()
    y = df["mean_mae"].to_numpy()
    best_idx = int(np.argmin(y))
    best_x, best_y = int(x[best_idx]), float(y[best_idx])
    threshold = 1.05 * best_y
    enough_x = int(x[np.flatnonzero(y <= threshold)[0]])

    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    ax.plot(x, y, color=BLUE, lw=2.4, marker="o", ms=4.2, zorder=3)
    ax.axhspan(best_y, threshold, color=BLUE, alpha=0.10, lw=0)
    ax.axvline(enough_x, color=GOLD, lw=1.6, ls="--")
    ax.scatter([best_x], [best_y], color=INK, s=42, zorder=4)
    ax.annotate(
        f"first within 5%: {enough_x:,}",
        xy=(enough_x, np.interp(enough_x, x, y)),
        xytext=(740, 0.48),
        arrowprops={"arrowstyle": "-", "color": GOLD, "lw": 1.1},
        color=GOLD,
        fontsize=9.5,
        weight="semibold",
    )
    ax.annotate(
        f"minimum: {best_y:.3f} at {best_x:,}",
        xy=(best_x, best_y),
        xytext=(3600, 0.385),
        arrowprops={"arrowstyle": "-", "color": INK, "lw": 1.0},
        color=INK,
        fontsize=9,
    )
    ax.set_xscale("log", base=2)
    ticks = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    ax.set_xticks(ticks, ["32", "64", "128", "256", "512", "1k", "2k", "4k", "8k"])
    ax.set_xlabel("Context length (timesteps)")
    ax.set_ylabel("Mean forecast MAE")
    ax.grid(True, which="major", color=GRID, lw=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.01,
        0.04,
        "Shaded band: within 5% of the observed minimum",
        transform=ax.transAxes,
        color="#667085",
        fontsize=8.5,
    )
    fig.suptitle(
        "Chronos2 Small: forecast error saturates before the maximum context",
        x=0.075,
        y=1.01,
        ha="left",
        color=INK,
        fontsize=12,
        weight="semibold",
    )
    fig.tight_layout()
    save(fig, "fig0_context_saturation_chronos2_small")


def load_exp7_raw() -> pd.DataFrame:
    frames = []
    for path in sorted(RAW.glob("*/exp7_context_decomposition/synthetic/w*/results.csv")):
        frame = pd.read_csv(path)
        frame = frame[frame["metric"].eq("mae")].copy()
        frame["visible_suffix"] = frame["block_index"].astype(int)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No frozen Exp7 result files were found")
    return pd.concat(frames, ignore_index=True)


@mpl.rc_context(PAPER_MPL_STYLE)
def slicing_vs_masking() -> None:
    raw = load_exp7_raw()
    agg = (
        raw.groupby(
            ["model", "context_length", "visible_suffix", "perturbation_type"],
            as_index=False,
        )[["clean_loss", "intervened_loss"]]
        .mean()
    )

    fig, axes = plt.subplots(2, 4, figsize=(FIGURE_WIDTH, 4.9), constrained_layout=False)
    for ax, model in zip(axes.flat, DECOMPOSITION_MODEL_ORDER):
        sub = agg[agg["model"].eq(model)].sort_values("visible_suffix")
        if sub.empty:
            ax.set_visible(False)
            continue
        full = sub[sub["perturbation_type"].eq("attention_mask/full_history_stats")]
        tail = sub[sub["perturbation_type"].eq("attention_mask/tail_matched_stats")]
        clean = (
            sub.groupby("visible_suffix", as_index=False)["clean_loss"]
            .mean()
            .sort_values("visible_suffix")
        )
        markevery = max(1, len(clean) // 6)
        ax.plot(
            clean["visible_suffix"], clean["clean_loss"],
            lw=1.75, ms=2.5, markevery=markevery,
            label="Physical slice", zorder=5,
            **model_plot_kwargs(model),
        )
        ax.plot(
            full["visible_suffix"], full["intervened_loss"],
            color=MODEL_COLORS[model], lw=1.45, alpha=0.48,
            label="Mask: full-history stats", zorder=3,
        )
        ax.plot(
            tail["visible_suffix"],
            tail["intervened_loss"],
            color=MODEL_COLORS[model],
            lw=1.55,
            ls="--",
            marker=MODEL_MARKERS[model],
            ms=2.4,
            markevery=markevery,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label="Mask: tail-matched stats",
            zorder=7,
        )
        ax.fill_between(
            full["visible_suffix"].to_numpy(),
            clean.set_index("visible_suffix").loc[full["visible_suffix"], "clean_loss"].to_numpy(),
            full["intervened_loss"].to_numpy(),
            color=MODEL_COLORS[model],
            alpha=0.08,
            lw=0,
        )
        ax.set_xscale("log", base=2)
        xmax = int(sub["visible_suffix"].max())
        tick_candidates = [32, 128, 512, 2048, 8192]
        ticks = [t for t in tick_candidates if t <= xmax]
        labels = ["32", "128", "512", "2k", "8k"][: len(ticks)]
        ax.set_xticks(ticks, labels)
        style_axes(ax, grid_axis="y")
        ax.set_title(
            MODEL_DISPLAY[model], loc="left",
            color=MODEL_COLORS[model], pad=5,
        )
        ax.text(
            0.98,
            0.94,
            f"W={int(sub['context_length'].max()):,}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=MUTED,
            fontsize=6.8,
        )
        ax.margins(x=0.03, y=0.08)

    for ax in axes[:, 0]:
        ax.set_ylabel("Mean MAE")
    for ax in axes[-1, :]:
        ax.set_xlabel("Visible suffix")

    method_handles = [
        mpl.lines.Line2D(
            [0], [0], color=INK, lw=1.75, marker="o", ms=3.5,
            label="Physical slice",
        ),
        mpl.lines.Line2D(
            [0], [0], color=INK, lw=1.45, alpha=0.48,
            label="Mask: full-history stats",
        ),
        mpl.lines.Line2D(
            [0], [0], color=INK, lw=1.55, ls="--", marker="o",
            markerfacecolor="white", ms=3.5,
            label="Mask: tail-matched stats",
        ),
    ]
    legend = fig.legend(
        handles=method_handles, loc="lower center", ncol=3, frameon=True,
        bbox_to_anchor=(0.5, 0.012), handlelength=2.5, columnspacing=1.5,
    )
    style_legend(legend)
    add_figure_title(fig, "Tail-stat matching narrows the slicing–masking gap")
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.16, top=0.89, wspace=0.30, hspace=0.38)
    save(fig, "fig3_slicing_vs_masking_all_models")


@mpl.rc_context(PAPER_MPL_STYLE)
def lag_tracking_with_toto_placeholder() -> None:
    controls = pd.read_csv(FROZEN / "exp4_controls.csv")
    summary = pd.read_csv(FROZEN / "exp4_lag_tracking_summary.csv")
    complete = set(summary["model"])
    models = [model for model in MODEL_ORDER if model in complete]
    data = controls[
        controls["model"].isin(complete)
        & controls["family"].eq("B")
        & controls["design_role"].eq("core")
        & controls["strength"].isin([0.5, 1.0])
        & ~controls["config_broken"].astype(bool)
    ].copy()

    short_lags = np.array([32, 64, 128, 256])
    fig = plt.figure(figsize=(FIGURE_WIDTH, 5.35))
    grid = fig.add_gridspec(
        2, 2, width_ratios=(1.22, 1.0), wspace=0.27, hspace=0.25,
    )
    short_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])]
    long_ax = fig.add_subplot(grid[:, 1])

    for panel, ax, strength in zip(("A", "B"), short_axes, (0.5, 1.0)):
        subset = data[data["strength"].eq(strength)]
        for model in models:
            group = subset[subset["model"].eq(model)].sort_values("lag")
            ax.plot(
                group["lag"], group["sufficient_context"],
                ms=3.3, lw=1.25,
                label=MODEL_DISPLAY.get(model, model),
                **model_plot_kwargs(model),
            )
        ax.plot(
            short_lags, short_lags, ls=(0, (4, 3)), lw=1.0,
            color=MUTED, label="Exact recovery",
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_yticks(
            [32, 64, 128, 256, 512, 1024],
            ["32", "64", "128", "256", "512", "1,024"],
        )
        ax.set_ylabel("5% sufficient context")
        style_axes(ax, grid_axis="y")
        ax.text(
            0.015, 0.93, f"Short protocol · α = {strength:g}",
            transform=ax.transAxes, ha="left", va="top", color=INK,
            fontsize=8.2, weight="semibold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2.5},
        )
        ax.text(
            -0.22, 1.04, panel, transform=ax.transAxes, color=INK,
            fontsize=10.5, weight="bold", va="top",
        )

    short_axes[-1].set_xticks(short_lags, ["32", "64", "128", "256"])
    short_axes[-1].set_xlabel("Planted dependency lag (timesteps)")
    short_axes[0].tick_params(labelbottom=False)

    long_data = pd.read_csv(FROZEN / "exp4_long_lag_all_models.csv")
    long_lags = np.array([128, 512, 1024, 2048])
    strong = long_data[long_data["strength"].eq(1.0)]
    for model in MODEL_ORDER:
        group = strong[strong["model"].eq(model)].sort_values("lag")
        if group.empty:
            continue
        long_ax.plot(
            group["lag"], group["sufficient_context"],
            ms=3.5, lw=1.3,
            label=MODEL_DISPLAY.get(model, model), zorder=4,
            **model_plot_kwargs(model),
        )
        flagged = group[group["limitation_flag"].astype(bool)]
        if not flagged.empty:
            long_ax.scatter(
                flagged["lag"], flagged["sufficient_context"],
                s=38, facecolors="none", edgecolors="#B42318",
                linewidths=1.0, zorder=7,
            )

    toto_weak = long_data[
        long_data["model"].eq("Toto-2.0-313m")
        & long_data["strength"].eq(0.5)
    ].sort_values("lag")
    long_ax.plot(
        toto_weak["lag"], toto_weak["sufficient_context"],
        color=MODEL_COLORS["Toto-2.0-313m"],
        marker=MODEL_MARKERS["Toto-2.0-313m"],
        ms=3.2, lw=1.15, ls=MODEL_LINE_STYLES["Toto-2.0-313m"],
        markerfacecolor="white", markeredgewidth=0.9, alpha=0.72,
        label="Toto α=0.5", zorder=3,
    )
    weak_flagged = toto_weak[toto_weak["limitation_flag"].astype(bool)]
    long_ax.scatter(
        weak_flagged["lag"], weak_flagged["sufficient_context"],
        s=38, facecolors="none", edgecolors="#B42318",
        linewidths=1.0, zorder=7,
    )
    long_ax.plot(
        long_lags, long_lags, ls=(0, (4, 3)), lw=1.0,
        color=MUTED, label="Exact recovery", zorder=2,
    )
    long_ax.set_xscale("log", base=2)
    long_ax.set_yscale("log", base=2)
    long_ax.set_xticks(long_lags, ["128", "512", "1,024", "2,048"])
    long_ax.set_yticks(
        [128, 256, 512, 1024, 2048, 4096],
        ["128", "256", "512", "1,024", "2,048", "4,096"],
    )
    long_ax.set_ylim(112, 4600)
    long_ax.set_xlabel("Planted dependency lag (timesteps)")
    long_ax.set_ylabel("5% sufficient context")
    style_axes(long_ax, grid_axis="y")
    long_ax.text(
        -0.18, 1.04, "C", transform=long_ax.transAxes, color=INK,
        fontsize=10.5, weight="bold", va="top", ha="left",
    )
    long_ax.text(
        0.02, 0.98, "Long protocol · α = 1",
        transform=long_ax.transAxes, ha="left", va="top",
        color=INK, fontsize=8.2, weight="semibold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2.5},
    )

    legend = fig.legend(
        handles=model_legend_handles(MODEL_ORDER),
        loc="lower center", ncol=6, frameon=True,
        bbox_to_anchor=(0.5, 0.012), fontsize=6.5,
        handlelength=2.25, columnspacing=1.0,
    )
    style_legend(legend)

    semantic_handles = [
        mpl.lines.Line2D(
            [0], [0], color=MODEL_COLORS["Toto-2.0-313m"],
            marker=MODEL_MARKERS["Toto-2.0-313m"], ms=3.2,
            markerfacecolor="white", ls=MODEL_LINE_STYLES["Toto-2.0-313m"],
            lw=1.15, alpha=0.72,
            label="Toto α=0.5",
        ),
        mpl.lines.Line2D(
            [0], [0], color=MUTED, ls=(0, (4, 3)), lw=1.0,
            label="Exact recovery",
        ),
        mpl.lines.Line2D(
            [0], [0], color="#B42318", marker="o", ms=5,
            markerfacecolor="none", lw=0, label="Limitation flag",
        ),
    ]
    long_ax.legend(
        handles=semantic_handles, loc="upper right", frameon=False,
        fontsize=6.5, handlelength=2.1,
    )
    add_figure_title(
        fig,
        "Adaptive context reach across short- and long-lag controls",
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.18, top=0.90)
    save(fig, "fig2_sufficient_context_vs_lag_with_toto_placeholder")


def masking_gap_heatmap() -> None:
    profiles = pd.read_csv(FROZEN / "exp7_decomposition_profiles.csv")
    profiles = profiles[profiles["metric"].eq("mae")].copy()
    suffixes = sorted(profiles["visible_suffix"].unique())
    variants = [
        ("attention_mask/full_history_stats", "Mask retains full-history statistics"),
        ("attention_mask/tail_matched_stats", "Mask uses tail-matched statistics"),
    ]
    matrices = []
    for variant, _ in variants:
        pivot = (
            profiles[profiles["variant"].eq(variant)]
            .pivot(index="model", columns="visible_suffix", values="mean_abs_loss_delta")
            .reindex(index=DECOMPOSITION_MODEL_ORDER, columns=suffixes)
        )
        matrices.append(pivot.to_numpy(dtype=float))

    finite = np.concatenate([m[np.isfinite(m)] for m in matrices])
    positive = finite[finite > 0]
    vmin = max(1e-5, float(np.nanpercentile(positive, 1)))
    vmax = float(np.nanpercentile(positive, 99))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"wspace": 0.07})
    cmap = mpl.colormaps["magma"].copy()
    cmap.set_bad("#EEF1F5")
    images = []
    for idx, (ax, matrix, (_, title)) in enumerate(zip(axes, matrices, variants)):
        img = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        images.append(img)
        ax.set_title(title, loc="left", color=INK, pad=9)
        ax.set_xticks(np.arange(len(suffixes)))
        tick_labels = []
        for s in suffixes:
            if s >= 1024:
                tick_labels.append(f"{s // 1024}k" if s % 1024 == 0 else f"{s / 1024:g}k")
            else:
                tick_labels.append(str(s))
        ax.set_xticklabels(tick_labels, rotation=55, ha="right")
        ax.set_xlabel("Visible suffix")
        ax.set_yticks(np.arange(len(DECOMPOSITION_MODEL_ORDER)))
        if idx == 0:
            ax.set_yticklabels([MODEL_DISPLAY[m] for m in DECOMPOSITION_MODEL_ORDER])
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.set_xticks(np.arange(-0.5, len(suffixes), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(DECOMPOSITION_MODEL_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.55, alpha=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(left=0.19, right=0.89, bottom=0.19, top=0.84)
    cax = fig.add_axes([0.915, 0.20, 0.015, 0.64])
    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.set_label("Mean |MAE(mask) − MAE(slice)|", color=INK)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        "Tail-matching removes most of the masking–slicing gap",
        x=0.07,
        y=1.02,
        ha="left",
        color=INK,
        fontsize=13,
        weight="semibold",
    )
    fig.text(
        0.07,
        0.96,
        "Log color scale; pale cells at the right are unavailable suffixes for shorter-context models.",
        color="#667085",
        fontsize=9,
    )
    save(fig, "fig4_masking_gap_heatmap_all_models")


def main() -> None:
    configure()
    context_error_curve()
    lag_tracking_with_toto_placeholder()
    slicing_vs_masking()
    masking_gap_heatmap()
    print(f"Wrote visual-draft figures to {OUT}")


if __name__ == "__main__":
    main()
