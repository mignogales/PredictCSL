"""STD-focused alignment diagnostics for synthetic parameter sweeps.

The report separates:

* alignment STD: dispersion across parameter bins after each bin curve is
  normalized by its attainable minimum and interpolated onto a common
  context/parameter grid;
* series CV: within-bin standard deviation across generated series divided by
  the corresponding mean MAE.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.plot_synth_sweep_results import (
    DEFAULT_MODELS,
    DEFAULT_ROOT,
    EXPERIMENTS,
    EXPERIMENT_TITLES,
    DASHBOARD_EXPERIMENTS,
    dashboard_xlim,
    _nanmean,
    _save_figure,
    _selected_horizon_index,
    closest_agreement_window,
    open_inset_bounds,
    coverage_rows,
)


@dataclass(frozen=True)
class AlignmentCell:
    model: str
    experiment: str
    ratios: np.ndarray
    aligned_mean: np.ndarray
    alignment_std: np.ndarray
    series_cv_median: np.ndarray
    actual_ratios: list[np.ndarray]
    relative_bin_curves: list[np.ndarray]
    bin_labels: list[str]
    bin_clamped: list[bool]

    @property
    def median_alignment_std_pct(self) -> float:
        return float(100 * np.nanmedian(self.alignment_std))

    @property
    def median_series_cv_pct(self) -> float:
        return float(100 * np.nanmedian(self.series_cv_median))

    @property
    def clamped(self) -> bool:
        return any(self.bin_clamped)


def _nanstd(array: np.ndarray, axis: int) -> np.ndarray:
    mean = _nanmean(array, axis=axis)
    expanded = np.expand_dims(mean, axis=axis)
    squared = (array - expanded) ** 2
    count = np.sum(np.isfinite(array), axis=axis)
    variance = np.divide(
        np.nansum(squared, axis=axis),
        count,
        out=np.full(np.shape(mean), np.nan, dtype=float),
        where=count > 0,
    )
    return np.sqrt(variance)


def _unique_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique_x = np.unique(x)
    unique_y = np.array([np.nanmean(y[x == value]) for value in unique_x])
    return unique_x, unique_y


def _log_interpolate(x: np.ndarray, y: np.ndarray,
                     grid: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if np.sum(valid) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    x, y = _unique_curve(x[valid], y[valid])
    if len(x) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(np.log2(grid), np.log2(x), y, left=np.nan, right=np.nan)


def load_alignment_cell(path: Path) -> AlignmentCell:
    model, experiment = path.parts[-3:-1]
    with np.load(path) as data:
        curves = data["curves_mae"].astype(float)
        contexts = data["contexts"].astype(float)
        norms = data["norms"].astype(float)
        ratios = data["ratios"].astype(float)
        horizon_idx = _selected_horizon_index(data["eval_horizons"])
        selected = curves[:, :, :, horizon_idx]
        means = _nanmean(selected, axis=1)
        stds = _nanstd(selected, axis=1)

        actual_ratios: list[np.ndarray] = []
        relative_curves: list[np.ndarray] = []
        bin_clamped: list[bool] = []
        aligned_rows = []
        series_cv_rows = []
        for bin_idx in range(means.shape[0]):
            valid = (
                (contexts[bin_idx] >= 0)
                & np.isfinite(means[bin_idx])
                & (means[bin_idx] > 0)
            )
            x = contexts[bin_idx, valid] / norms[bin_idx]
            y = means[bin_idx, valid]
            relative = y / np.min(y) if len(y) else np.array([], dtype=float)
            cv = stds[bin_idx, valid] / y if len(y) else np.array([], dtype=float)
            actual_ratios.append(x)
            relative_curves.append(relative)
            raw_contexts = contexts[bin_idx, valid]
            bin_clamped.append(bool(
                len(raw_contexts) >= 2
                and np.any(np.diff(raw_contexts[-4:]) == 0)))
            aligned_rows.append(_log_interpolate(x, relative, ratios))
            series_cv_rows.append(_log_interpolate(x, cv, ratios))

    meta_path = path.with_name("done.json")
    labels: list[str] = []
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        labels = [str(item.get("label", f"bin {idx + 1}"))
                  for idx, item in enumerate(meta.get("bins", []))]
    if len(labels) != len(actual_ratios):
        labels = [f"bin {idx + 1}" for idx in range(len(actual_ratios))]

    aligned = np.asarray(aligned_rows, dtype=float)
    series_cv = np.asarray(series_cv_rows, dtype=float)
    return AlignmentCell(
        model=model,
        experiment=experiment,
        ratios=ratios,
        aligned_mean=_nanmean(aligned, axis=0),
        alignment_std=_nanstd(aligned, axis=0),
        series_cv_median=np.ma.median(
            np.ma.masked_invalid(series_cv), axis=0).filled(np.nan),
        actual_ratios=actual_ratios,
        relative_bin_curves=relative_curves,
        bin_labels=labels,
        bin_clamped=bin_clamped,
    )


def _tail_subset(x: np.ndarray, y: np.ndarray,
                 n_points: int = 4) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[valid], y[valid]
    if not len(x):
        return x, y
    # Several requested ratios can collapse onto the same attainable context.
    # Deduplicate before selecting the tail so the inset shows the last four
    # genuinely distinct points rather than overplotting four copies at x_max.
    x, y = _unique_curve(x, y)
    return x[-min(n_points, len(x)):], y[-min(n_points, len(y)):]


def _window_subset(x: np.ndarray, y: np.ndarray, minimum_x: float,
                   maximum_x: float) -> tuple[np.ndarray, np.ndarray]:
    """Unique attainable points inside the shared agreement zoom window."""
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[valid], y[valid]
    if not len(x):
        return x, y
    x, y = _unique_curve(x, y)
    keep = ((x >= minimum_x * (1 - 1e-12))
            & (x <= maximum_x * (1 + 1e-12)))
    return x[keep], y[keep]


def _finish_tail_inset(inset, title: str) -> None:
    inset.set_xscale("log", base=2)
    inset.grid(alpha=0.18)
    inset.tick_params(labelsize=6, length=2)
    inset.set_title(title, fontsize=7, pad=2)
    inset.set_facecolor((1, 1, 1, 0.94))
    for spine in inset.spines.values():
        spine.set_color("#6B7280")
        spine.set_linewidth(0.8)


def _zoom_y_to_lines(inset, ys: list[np.ndarray]) -> None:
    parts = [y[np.isfinite(y)] for y in ys if np.isfinite(y).any()]
    if not parts:
        return
    finite = np.concatenate(parts)
    low, high = float(np.min(finite)), float(np.max(finite))
    pad = max((high - low) * 0.12, max(abs(low), abs(high), 1.0) * 0.015)
    inset.set_ylim(low - pad, high + pad)


def collect_alignment(root: Path) -> dict[tuple[str, str], AlignmentCell]:
    cells = {}
    for path in sorted(root.glob("*/*/results.npz")):
        cell = load_alignment_cell(path)
        cells[(cell.model, cell.experiment)] = cell
    return cells


def _metric_matrix(cells: dict[tuple[str, str], AlignmentCell],
                   models: Sequence[str], experiments: Sequence[str],
                   attribute: str) -> np.ndarray:
    matrix = np.full((len(models), len(experiments)), np.nan)
    for row, model in enumerate(models):
        for col, experiment in enumerate(experiments):
            cell = cells.get((model, experiment))
            if cell is not None:
                matrix[row, col] = float(getattr(cell, attribute))
    return matrix


def plot_metric_heatmap(
    cells: dict[tuple[str, str], AlignmentCell], models: Sequence[str],
    experiments: Sequence[str], attribute: str, title: str, stem: str,
    output_dir: Path, formats: Sequence[str],
) -> None:
    import matplotlib.pyplot as plt

    matrix = _metric_matrix(cells, models, experiments, attribute)
    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.nanpercentile(finite, 95)) if len(finite) else 1.0
    fig, ax = plt.subplots(figsize=(16, 7.4))
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="magma",
                      vmin=0, vmax=max(vmax, 1e-6))
    ax.set_xticks(range(len(experiments)),
                  [EXPERIMENT_TITLES[x] for x in experiments],
                  rotation=35, ha="right")
    ax.set_yticks(range(len(models)), models)
    midpoint = float(np.nanmedian(finite)) if len(finite) else 0
    for row in range(len(models)):
        for col in range(len(experiments)):
            value = matrix[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.0f}%", ha="center", va="center",
                        fontsize=7,
                        color="white" if value >= midpoint else "black")
            else:
                ax.text(col, row, "missing", ha="center", va="center",
                        fontsize=6.5, color="#9CA3AF", rotation=25)
    fig.colorbar(image, ax=ax, label="median variability (%)", shrink=0.82)
    ax.set_title(title, fontsize=15)
    fig.tight_layout()
    _save_figure(fig, output_dir, stem, formats)
    plt.close(fig)


def plot_std_dashboard(
    cells: dict[tuple[str, str], AlignmentCell], models: Sequence[str],
    experiments: Sequence[str], output_dir: Path, formats: Sequence[str],
) -> None:
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(models)))
    fig, axes = plt.subplots(3, 4, figsize=(17, 11), sharex=False)
    for ax, experiment in zip(axes.flat, experiments):
        panel_cells = []
        for model, color in zip(models, colors):
            cell = cells.get((model, experiment))
            if cell is None:
                continue
            valid = np.isfinite(cell.alignment_std)
            ax.plot(cell.ratios[valid], 100 * cell.alignment_std[valid],
                    marker="o", ms=2.5, lw=1.3, color=color, label=model)
            panel_cells.append((cell, color))
        ax.set_xscale("log", base=2)
        ax.set_xlim(*dashboard_xlim(experiment))
        ax.set_title(EXPERIMENT_TITLES[experiment])
        ax.set_xlabel("context / parameter")
        ax.set_ylabel("cross-bin STD (%)")
        ax.grid(alpha=0.22)
        clamped = [cell for cell, _ in panel_cells if cell.clamped]
        if clamped:
            window = closest_agreement_window([
                (cell.ratios, 100 * cell.alignment_std)
                for cell, _ in panel_cells])
            if window is None:
                continue
            zoom_min, _, zoom_max = window
            inset = ax.inset_axes(open_inset_bounds(
                ax, [(cell.ratios, 100 * cell.alignment_std)
                     for cell, _ in panel_cells]))
            inset_values = []
            for cell, color in panel_cells:
                x, y = _window_subset(
                    cell.ratios, 100 * cell.alignment_std, zoom_min, zoom_max)
                if not len(x):
                    continue
                inset.plot(x, y, marker="o", ms=1.8, lw=1, color=color)
                inset_values.append(y)
            _zoom_y_to_lines(inset, inset_values)
            _finish_tail_inset(inset, "closest-agreement zoom")
            inset.tick_params(labelsize=5.5, length=2)
            inset.title.set_fontsize(6.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6,
               fontsize=8, frameon=False)
    fig.suptitle("Alignment dispersion across normalized parameter bins",
                 fontsize=16)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    _save_figure(fig, output_dir, "05_alignment_std_dashboard", formats)
    plt.close(fig)


def plot_mean_std_dashboard(
    cells: dict[tuple[str, str], AlignmentCell], models: Sequence[str],
    experiments: Sequence[str], output_dir: Path, formats: Sequence[str],
) -> None:
    """Combined view: aligned mean lines with restrained ±1 STD ribbons."""
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(models)))
    fig, axes = plt.subplots(3, 4, figsize=(17, 11), sharex=False)
    for ax, experiment in zip(axes.flat, experiments):
        panel_cells = []
        pending_bands = []
        for model, color in zip(models, colors):
            cell = cells.get((model, experiment))
            if cell is None:
                continue
            valid = (np.isfinite(cell.aligned_mean)
                     & np.isfinite(cell.alignment_std)
                     & (cell.aligned_mean > 0))
            x = cell.ratios[valid]
            mean = cell.aligned_mean[valid]
            std = cell.alignment_std[valid]
            if not len(x):
                continue
            ax.plot(x, mean, marker="o", ms=2.5, lw=1.35, color=color,
                    label=model, zorder=2)
            panel_cells.append((cell, color))
            pending_bands.append((x, mean, std, color))
        # At this point only the mean lines contribute to Matplotlib's data
        # limits, exactly as in the original mean-only dashboard.
        mean_limits = ax.get_ylim()
        ax.axhline(1.02, color="#6B7280", ls="--", lw=0.9, zorder=0)
        ax.set_xscale("log", base=2)
        ax.set_xlim(*dashboard_xlim(experiment))
        ax.set_yscale("log")
        ax.set_title(EXPERIMENT_TITLES[experiment])
        ax.set_xlabel("context / parameter")
        ax.set_ylabel("mean relative MAE")
        ax.grid(alpha=0.22)
        # Add uncertainty without letting it change the original line-only
        # visual scale.
        for x, mean, std, color in pending_bands:
            ax.fill_between(x, np.maximum(mean - std, 1e-4), mean + std,
                            color=color, alpha=0.055, lw=0, zorder=1)
        ax.set_ylim(mean_limits)

        if any(cell.clamped for cell, _ in panel_cells):
            window = closest_agreement_window([
                (cell.ratios, cell.aligned_mean) for cell, _ in panel_cells])
            if window is None:
                continue
            zoom_min, _, zoom_max = window
            inset = ax.inset_axes(open_inset_bounds(
                ax, [(cell.ratios, cell.aligned_mean)
                     for cell, _ in panel_cells]))
            inset_means = []
            for cell, color in panel_cells:
                x, mean = _window_subset(
                    cell.ratios, cell.aligned_mean, zoom_min, zoom_max)
                _, std = _window_subset(
                    cell.ratios, cell.alignment_std, zoom_min, zoom_max)
                if not len(x):
                    continue
                inset.fill_between(x, np.maximum(mean - std, 1e-4), mean + std,
                                   color=color, alpha=0.075, lw=0)
                inset.plot(x, mean, marker="o", ms=1.8, lw=1, color=color)
                inset_means.append(mean)
            # Keep the zoom's vertical scale comparable to its parent: bands
            # communicate dispersion but never expand the y-range.
            _zoom_y_to_lines(inset, inset_means)
            _finish_tail_inset(inset, "closest-agreement zoom")
            inset.tick_params(labelsize=5.5, length=2)
            inset.title.set_fontsize(6.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6,
               fontsize=8, frameon=False)
    fig.suptitle("Synthetic sweeps — aligned mean with ±1 cross-bin STD",
                 fontsize=16)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    _save_figure(fig, output_dir, "08_mean_std_dashboard", formats)
    plt.close(fig)


def plot_cell_detail(cell: AlignmentCell, output_dir: Path,
                     formats: Sequence[str]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    cmap = plt.get_cmap("viridis")
    for idx, (x, y, label) in enumerate(zip(
            cell.actual_ratios, cell.relative_bin_curves, cell.bin_labels)):
        if not len(x):
            continue
        axes[0].plot(x, y, color=cmap(idx / max(len(cell.bin_labels) - 1, 1)),
                     alpha=0.55, lw=1, marker="o", ms=2, label=label)
    valid = np.isfinite(cell.aligned_mean) & np.isfinite(cell.alignment_std)
    x = cell.ratios[valid]
    mean = cell.aligned_mean[valid]
    std = cell.alignment_std[valid]
    axes[0].plot(x, mean, color="#111827", lw=2.4, label="aligned mean")
    axes[0].fill_between(x, np.maximum(mean - std, 1e-6), mean + std,
                         color="#64748B", alpha=0.28, label="±1 cross-bin STD")
    axes[0].axhline(1.02, color="#9CA3AF", ls="--", lw=0.9)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MAE / bin minimum")
    axes[0].grid(alpha=0.22)
    if len(cell.bin_labels) <= 8:
        axes[0].legend(fontsize=7, ncol=2,
                       loc="upper left" if cell.clamped else "best")

    axes[1].plot(cell.ratios, 100 * cell.alignment_std, marker="o",
                 color="#DC2626", label="cross-bin alignment STD")
    axes[1].plot(cell.ratios, 100 * cell.series_cv_median, marker="o",
                 color="#2563EB", label="median within-bin series CV")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("context / generating parameter")
    axes[1].set_ylabel("variability (%)")
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=8)

    if cell.clamped:
        window = closest_agreement_window([
            (raw_x, raw_y)
            for raw_x, raw_y in zip(cell.actual_ratios, cell.relative_bin_curves)
        ])
        if window is None:
            return
        zoom_min, _, zoom_max = window
        mean_inset = axes[0].inset_axes([0.57, 0.53, 0.40, 0.40])
        inset_ys: list[np.ndarray] = []
        for idx, (raw_x, raw_y) in enumerate(zip(
                cell.actual_ratios, cell.relative_bin_curves)):
            window_x, window_y = _window_subset(raw_x, raw_y, zoom_min, zoom_max)
            if not len(window_x):
                continue
            mean_inset.plot(
                window_x, window_y,
                color=cmap(idx / max(len(cell.bin_labels) - 1, 1)),
                alpha=0.75, lw=1, marker="o", ms=2,
            )
            inset_ys.append(window_y)
        mean_x, mean_y = _window_subset(
            cell.ratios, cell.aligned_mean, zoom_min, zoom_max)
        mean_inset.plot(mean_x, mean_y, color="#111827", lw=2)
        inset_ys.append(mean_y)
        _zoom_y_to_lines(mean_inset, inset_ys)
        _finish_tail_inset(mean_inset, "closest-agreement zoom")

        std_inset = axes[1].inset_axes([0.57, 0.49, 0.40, 0.44])
        std_x, std_y = _window_subset(
            cell.ratios, 100 * cell.alignment_std, zoom_min, zoom_max)
        cv_x, cv_y = _window_subset(
            cell.ratios, 100 * cell.series_cv_median, zoom_min, zoom_max)
        std_inset.plot(std_x, std_y, color="#DC2626", marker="o", ms=2)
        std_inset.plot(cv_x, cv_y, color="#2563EB", marker="o", ms=2)
        _zoom_y_to_lines(std_inset, [std_y, cv_y])
        _finish_tail_inset(std_inset, "closest-agreement zoom")
        axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{cell.model} — {EXPERIMENT_TITLES[cell.experiment]} alignment")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    target = output_dir / "alignment_cells" / cell.experiment
    target.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, target, cell.model, formats)
    plt.close(fig)


def write_alignment_report(
    root: Path, output_dir: Path, models: Sequence[str],
    experiments: Sequence[str], formats: Sequence[str], per_cell: bool,
    strict: bool,
) -> None:
    import matplotlib
    matplotlib.use("Agg")

    output_dir.mkdir(parents=True, exist_ok=True)
    cells = {
        key: value for key, value in collect_alignment(root).items()
        if key[0] in models and key[1] in experiments
    }
    coverage = coverage_rows(root, models, experiments)
    missing = [row for row in coverage if row["status"] != "complete"]

    rows = []
    for model in models:
        for experiment in experiments:
            cell = cells.get((model, experiment))
            if cell is not None:
                rows.append({
                    "model": model,
                    "experiment": experiment,
                    "median_alignment_std_pct": cell.median_alignment_std_pct,
                    "median_within_bin_series_cv_pct": cell.median_series_cv_pct,
                })
    if rows:
        with (output_dir / "alignment_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    dashboard_experiments = [
        experiment for experiment in DASHBOARD_EXPERIMENTS
        if experiment in experiments]
    plot_std_dashboard(cells, models, dashboard_experiments, output_dir, formats)
    plot_mean_std_dashboard(
        cells, models, dashboard_experiments, output_dir, formats)
    plot_metric_heatmap(
        cells, models, experiments, "median_alignment_std_pct",
        "Cross-bin alignment STD after context/parameter normalization",
        "06_alignment_std_heatmap", output_dir, formats,
    )
    plot_metric_heatmap(
        cells, models, experiments, "median_series_cv_pct",
        "Within-bin variability across generated series",
        "07_within_bin_series_cv_heatmap", output_dir, formats,
    )
    if per_cell:
        for cell in cells.values():
            if cell.model in models and cell.experiment in experiments:
                plot_cell_detail(cell, output_dir, formats)

    print(f"Alignment report: {len(cells)}/{len(models) * len(experiments)} cells")
    print(f"Outputs: {output_dir}")
    if strict and missing:
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cross-bin and within-bin STD for synthetic sweeps.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS,
                        default=EXPERIMENTS)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"),
                        default=("png", "pdf"))
    parser.add_argument("--per-cell", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "summary_plots"
    write_alignment_report(root, output_dir, args.models, args.experiments,
                           args.formats, args.per_cell, args.strict)


if __name__ == "__main__":
    main()
