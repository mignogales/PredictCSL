"""Create publication-ready figures from oracle-distribution analysis outputs.

This is pure post-processing and never loads a TSFM or predictor.

Example
-------
python -m experiments.plot_oracle_results \
  --analysis-dir logs/experiments/master_recompute/window_ablation_gifteval/general/oracle_distribution_analysis

Outputs are written under ``<analysis-dir>/paper_figures`` by default:

* ``oracle_story.{png,pdf}`` — fraction distribution, cross-model agreement,
  and split-half stability.
* ``oracle_absolute_windows.{png,pdf}`` — per-model distribution over absolute
  oracle window choices, useful when choosing one fixed-window baseline.
* ``oracle_global_summary.csv`` — plotted model-level summary statistics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


FRACTION_EDGES = np.asarray(
    [0.0, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 1.0 + 1e-12],
    dtype=np.float64,
)
FRACTION_LABELS = [
    "≤1/32", "1/32–1/16", "1/16–1/8", "1/8–1/4",
    "1/4–1/2", "1/2–3/4", "3/4–1",
]

PREFERRED_ORDER = [
    "Chronos2-Small", "Chronos2-Base", "Chronos2-Synth",
    "ChronosBolt-Base", "Moirai2-Small", "TimesFM2.5-200M",
    "PatchTST-FM-R1", "Sundial-Base-128M", "Toto-2.0-313m",
    "FlowState-R1", "TiRex2",
]


def _model_order(models: Iterable[str]) -> List[str]:
    available = set(models)
    return [m for m in PREFERRED_ORDER if m in available] + sorted(
        available.difference(PREFERRED_ORDER)
    )


def _read_instance_distributions(
    path: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Counter], Counter, Counter]:
    fraction_counts: Dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(len(FRACTION_LABELS), dtype=np.int64)
    )
    window_counts: Dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    native: Counter = Counter()
    columns = [
        "model", "oracle_fraction", "oracle_requested_context",
        "native_selected",
    ]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250_000):
        for model, group in chunk.groupby("model", sort=False):
            fractions = group["oracle_fraction"].to_numpy(dtype=np.float64)
            finite = fractions[np.isfinite(fractions)]
            fraction_counts[model] += np.histogram(
                finite, bins=FRACTION_EDGES
            )[0]
            fixed = group.loc[
                ~group["native_selected"].astype(bool), "oracle_requested_context"
            ]
            for window, count in fixed.value_counts().items():
                if np.isfinite(window):
                    window_counts[model][int(round(float(window)))] += int(count)
            totals[model] += len(group)
            native[model] += int(group["native_selected"].sum())
    return fraction_counts, window_counts, totals, native


def _matrix_from_pairs(
    pairs: pd.DataFrame, models: List[str], column: str,
) -> np.ndarray:
    index = {model: i for i, model in enumerate(models)}
    matrix = np.full((len(models), len(models)), np.nan, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0 if "rate" in column else 0.0)
    for row in pairs.itertuples(index=False):
        left = index.get(row.model_left)
        right = index.get(row.model_right)
        if left is None or right is None:
            continue
        value = float(getattr(row, column))
        matrix[left, right] = value
        matrix[right, left] = value
    return matrix


def _weighted_stability(stability: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        group = stability[stability["model"] == model]
        group = group[np.isfinite(group["split_half_js_mean"])]
        if group.empty:
            continue
        values = group["split_half_js_mean"].to_numpy(dtype=np.float64)
        weights = group["n_instances"].to_numpy(dtype=np.float64)
        rows.append({
            "model": model,
            "weighted_split_half_js": float(np.average(values, weights=weights)),
            "median_cell_split_half_js": float(np.median(values)),
            "q95_cell_split_half_js": float(np.quantile(values, 0.95)),
            "n_finite_cells": int(len(group)),
        })
    return pd.DataFrame(rows)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _annotated_heatmap(
    ax: plt.Axes, values: np.ndarray, xlabels: List[str], ylabels: List[str],
    title: str, colorbar_label: str, percent: bool = True,
) -> None:
    image = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(xlabels)), xlabels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    ax.set_title(title, loc="left", fontweight="bold")
    threshold = 0.55
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isfinite(value):
                continue
            label = f"{100 * value:.0f}" if percent else f"{value:.2f}"
            ax.text(
                column, row, label, ha="center", va="center", fontsize=6.5,
                color="white" if value > threshold else "#202020",
            )
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label(colorbar_label)
    if percent:
        bar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def plot_story(
    output: Path,
    models: List[str],
    fraction_counts: Dict[str, np.ndarray],
    pairs: pd.DataFrame,
    stability_summary: pd.DataFrame,
) -> None:
    fractions = np.vstack([
        fraction_counts[model] / max(1, fraction_counts[model].sum())
        for model in models
    ])
    agreement = _matrix_from_pairs(
        pairs, models, "weighted_mean_within_factor_2_rate"
    )

    figure = plt.figure(figsize=(15.2, 5.9), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=[1.2, 1.4, 1.05])
    ax_fraction = figure.add_subplot(grid[0, 0])
    ax_agreement = figure.add_subplot(grid[0, 1])
    ax_stability = figure.add_subplot(grid[0, 2])

    _annotated_heatmap(
        ax_fraction, fractions, FRACTION_LABELS, models,
        "a  Oracle context is broadly distributed",
        "Share of instances (%)",
    )
    ax_fraction.set_xlabel("Oracle context / available history")

    _annotated_heatmap(
        ax_agreement, agreement, models, models,
        "b  Cross-model per-instance agreement",
        "Within-factor-two agreement (%)",
    )
    ax_agreement.set_xlabel("Model")

    lookup = stability_summary.set_index("model")
    y = np.arange(len(models))
    weighted = np.asarray([lookup.loc[m, "weighted_split_half_js"] for m in models])
    median = np.asarray([lookup.loc[m, "median_cell_split_half_js"] for m in models])
    q95 = np.asarray([lookup.loc[m, "q95_cell_split_half_js"] for m in models])
    ax_stability.hlines(y, weighted, q95, color="#a8a8a8", linewidth=1.2)
    ax_stability.scatter(q95, y, marker="|", s=55, color="#777777", label="95th percentile cell")
    ax_stability.scatter(median, y, s=25, color="#d97706", label="Median cell")
    ax_stability.scatter(weighted, y, s=30, color="#2563a6", label="Instance-weighted mean", zorder=3)
    ax_stability.set_yticks(y, models)
    ax_stability.invert_yaxis()
    ax_stability.set_xlim(left=0.0)
    ax_stability.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax_stability.set_xlabel("Split-half Jensen–Shannon distance (lower is better)")
    ax_stability.set_title(
        "c  Within-dataset distributions are reproducible",
        loc="left", fontweight="bold",
    )
    ax_stability.legend(frameon=False, fontsize=7.5, loc="lower right")

    figure.savefig(output / "oracle_story.png")
    figure.savefig(output / "oracle_story.pdf")
    plt.close(figure)


def plot_absolute_windows(
    output: Path,
    models: List[str],
    window_counts: Dict[str, Counter],
    totals: Counter,
    native: Counter,
) -> None:
    windows = sorted({window for model in models for window in window_counts[model]})
    matrix = np.zeros((len(models), len(windows) + 1), dtype=np.float64)
    for row, model in enumerate(models):
        total = totals[model]
        if total:
            matrix[row, :-1] = [
                window_counts[model][window] / total for window in windows
            ]
            matrix[row, -1] = native[model] / total

    figure, ax = plt.subplots(figsize=(14.0, 4.8), constrained_layout=True)
    vmax = max(0.01, float(np.nanquantile(matrix, 0.99)))
    image = ax.imshow(
        matrix, aspect="auto", cmap="Blues", norm=Normalize(vmin=0.0, vmax=vmax)
    )
    ax.set_yticks(np.arange(len(models)), models)
    ax.set_xticks(
        np.arange(len(windows) + 1),
        [str(window) for window in windows] + ["native"],
        rotation=45, ha="right",
    )
    ax.set_xlabel("Oracle requested context length")
    ax.set_title(
        "Absolute oracle windows: the global mode is short, but probability mass is diffuse",
        loc="left", fontweight="bold",
    )
    bar = figure.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    bar.set_label("Share of instances")
    bar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    figure.savefig(output / "oracle_absolute_windows.png")
    figure.savefig(output / "oracle_absolute_windows.pdf")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    analysis = Path(args.analysis_dir)
    required = {
        "instances": analysis / "instance_oracles.csv.gz",
        "pairs": analysis / "model_pair_summary.csv",
        "stability": analysis / "intra_dataset_stability.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise SystemExit("Missing oracle analysis outputs:\n  " + "\n  ".join(missing))

    output = Path(args.output_dir or analysis / "paper_figures")
    output.mkdir(parents=True, exist_ok=True)
    fraction_counts, window_counts, totals, native = _read_instance_distributions(
        required["instances"]
    )
    models = _model_order(fraction_counts)
    pairs = pd.read_csv(required["pairs"])
    stability = pd.read_csv(required["stability"])
    stability_summary = _weighted_stability(stability, models)

    _style()
    plot_story(output, models, fraction_counts, pairs, stability_summary)
    plot_absolute_windows(output, models, window_counts, totals, native)

    summary = stability_summary.set_index("model")
    summary["n_instances"] = pd.Series(totals)
    summary["native_selected_rate"] = pd.Series({
        model: native[model] / totals[model] for model in models
    })
    summary["global_oracle_window_mode"] = pd.Series({
        model: window_counts[model].most_common(1)[0][0] for model in models
    })
    summary["global_mode_mass"] = pd.Series({
        model: window_counts[model].most_common(1)[0][1] / totals[model]
        for model in models
    })
    summary.loc[models].reset_index().to_csv(
        output / "oracle_global_summary.csv", index=False
    )
    print(f"Oracle paper figures -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        default=("logs/experiments/master_recompute/window_ablation_gifteval/"
                 "general/oracle_distribution_analysis"),
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
