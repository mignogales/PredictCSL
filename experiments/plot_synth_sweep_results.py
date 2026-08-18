"""Consolidated plots and tables for the synthetic parameter sweeps.

This is intentionally inference-free. It reads the resumable ``results.npz``
cells written by :mod:`experiments.synth_param_sweeps`, tolerates an incomplete
matrix, and can be rerun after new model cells arrive.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DEFAULT_ROOT = os.environ.get(
    "PREDICTCSL_SWEEP_ROOT", "logs/experiments/synth_param_sweeps")
DEFAULT_MODELS = [
    "Chronos2-Small",
    "Moirai2-Small",
    "TimesFM2.5-200M",
    "PatchTST-FM-R1",
    "Sundial-Base-128M",
    "Chronos2-Synth",
    "Chronos2-Base",
    "ChronosBolt-Base",
    "Toto-2.0-313m",
    "FlowState-R1",
    "TiRex2",
]
EXPERIMENTS = [
    "period",
    "seasonality",
    "ar_order",
    "memory",
    "delay",
    "regime",
    "horizon",
    "break_age",
    "snr",
    "multiscale",
    "period_drift",
    "missing_gap",
]
EXPERIMENT_TITLES = {
    "period": "Period",
    "seasonality": "Seasonality",
    "ar_order": "AR order",
    "memory": "Memory",
    "delay": "Delay",
    "regime": "Regime",
    "horizon": "Horizon",
    "break_age": "Break age",
    "snr": "SNR",
    "multiscale": "Multiscale",
    "period_drift": "Period drift",
    "missing_gap": "Missing gap",
}
# Put the three long-tail sweeps together in the right column of the 3x4
# dashboards, making their extended x-ranges easy to compare vertically.
DASHBOARD_EXPERIMENTS = [
    "period", "seasonality", "ar_order", "delay",
    "memory", "regime", "break_age", "horizon",
    "multiscale", "period_drift", "missing_gap", "snr",
]
LONG_TAIL_EXPERIMENTS = frozenset({"delay", "horizon", "snr"})
STANDARD_XLIM = (0.25, 16.0)
LONG_TAIL_XLIM = (0.25, 64.0)


def dashboard_xlim(experiment: str) -> tuple[float, float]:
    """Requested normalized-context range shown for a dashboard panel."""
    return LONG_TAIL_XLIM if experiment in LONG_TAIL_EXPERIMENTS else STANDARD_XLIM


@dataclass(frozen=True)
class CellSummary:
    model: str
    experiment: str
    n_bins: int
    n_series: int
    evaluation_horizon: int
    best_nominal_ratio: float
    saturation_ratio_2pct: float
    short_context_penalty_pct: float
    longest_context_penalty_pct: float
    best_mae_over_naive: float


@dataclass(frozen=True)
class PlotCurve:
    ratios: np.ndarray
    values: np.ndarray
    clamped: bool


def _nanmean(array: np.ndarray, axis=None) -> np.ndarray:
    count = np.sum(np.isfinite(array), axis=axis)
    total = np.nansum(array, axis=axis)
    return np.divide(
        total,
        count,
        out=np.full(np.shape(total), np.nan, dtype=float),
        where=count > 0,
    )


def _selected_horizon_index(eval_horizons: np.ndarray) -> int:
    values = [int(x) for x in eval_horizons]
    return values.index(64) if 64 in values else 0


def summarize_cell(path: Path) -> tuple[CellSummary, np.ndarray, np.ndarray]:
    model, experiment = path.parts[-3:-1]
    with np.load(path) as data:
        curves = data["curves_mae"].astype(float)
        contexts = data["contexts"].astype(int)
        ratios = data["ratios"].astype(float)
        eval_horizons = data["eval_horizons"].astype(int)
        horizon_idx = _selected_horizon_index(eval_horizons)

        # Average over series, normalize every bin by its own attainable best,
        # then average bins. This matches the existing all-model overlay.
        mean_by_bin = _nanmean(curves[:, :, :, horizon_idx], axis=1)
        relative_by_bin = np.full_like(mean_by_bin, np.nan, dtype=float)
        best_by_bin = np.full(mean_by_bin.shape[0], np.nan, dtype=float)
        for bin_idx, values in enumerate(mean_by_bin):
            valid = (contexts[bin_idx] >= 0) & np.isfinite(values)
            if not np.any(valid):
                continue
            best = float(np.min(values[valid]))
            best_by_bin[bin_idx] = best
            if best > 0:
                relative_by_bin[bin_idx, valid] = values[valid] / best
        relative_curve = _nanmean(relative_by_bin, axis=0)
        finite = np.flatnonzero(np.isfinite(relative_curve))
        if not len(finite):
            raise ValueError(f"no finite MAE curve in {path}")
        best_idx = int(finite[np.argmin(relative_curve[finite])])
        threshold = float(relative_curve[best_idx]) * 1.02
        saturation_idx = next(
            int(idx) for idx in finite if relative_curve[idx] <= threshold)

        naive = _nanmean(data["naive_mae"].astype(float), axis=1)
        selected_naive = naive[:, horizon_idx]
        valid_naive = (
            np.isfinite(best_by_bin)
            & np.isfinite(selected_naive)
            & (selected_naive > 0)
        )
        best_over_naive = float(np.nanmedian(
            best_by_bin[valid_naive] / selected_naive[valid_naive]))

    summary = CellSummary(
        model=model,
        experiment=experiment,
        n_bins=int(curves.shape[0]),
        n_series=int(curves.shape[1]),
        evaluation_horizon=int(eval_horizons[horizon_idx]),
        best_nominal_ratio=float(ratios[best_idx]),
        saturation_ratio_2pct=float(ratios[saturation_idx]),
        short_context_penalty_pct=float(
            100 * (relative_curve[finite[0]] / relative_curve[best_idx] - 1)),
        longest_context_penalty_pct=float(
            100 * (relative_curve[finite[-1]] / relative_curve[best_idx] - 1)),
        best_mae_over_naive=best_over_naive,
    )
    return summary, ratios, relative_curve


def collect_results(
    root: Path,
) -> tuple[list[CellSummary], dict[tuple[str, str], PlotCurve]]:
    summaries: list[CellSummary] = []
    curves: dict[tuple[str, str], PlotCurve] = {}
    for path in sorted(root.glob("*/*/results.npz")):
        summary, ratios, curve = summarize_cell(path)
        summaries.append(summary)
        with np.load(path) as data:
            contexts = data["contexts"].astype(float)
            norms = data["norms"].astype(float)
        clamped = False
        for bin_idx in range(len(contexts)):
            valid = contexts[bin_idx] >= 0
            raw = contexts[bin_idx, valid]
            if len(raw) < 2 or not np.any(np.diff(raw[-4:]) == 0):
                continue
            clamped = True
        curves[(summary.model, summary.experiment)] = PlotCurve(
            ratios=ratios,
            values=curve,
            clamped=clamped,
        )
    return summaries, curves


def coverage_rows(
    root: Path, models: Sequence[str], experiments: Sequence[str]
) -> list[dict[str, object]]:
    rows = []
    for model in models:
        for experiment in experiments:
            cell = root / model / experiment
            results = cell / "results.npz"
            done = cell / "done.json"
            status = (
                "complete" if results.is_file() and done.is_file()
                else "results_without_done" if results.is_file()
                else "done_without_results" if done.is_file()
                else "missing"
            )
            rows.append({
                "model": model,
                "experiment": experiment,
                "status": status,
                "results_path": str(results),
            })
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig, output_dir: Path, stem: str, formats: Sequence[str]) -> None:
    for extension in formats:
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=180,
                    bbox_inches="tight")


def closest_agreement_window(
    lines: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, float] | None:
    """Window around minimum spread: one point before, two points after."""
    prepared = []
    for x, y in lines:
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
        if not np.any(valid):
            continue
        x, y = x[valid], y[valid]
        unique_x = np.unique(x)
        unique_y = np.array([np.nanmean(y[x == value]) for value in unique_x])
        prepared.append((unique_x, unique_y))
    if not prepared:
        return None
    candidates = np.unique(np.concatenate([x for x, _ in prepared]))
    spreads = np.full(len(candidates), np.nan, dtype=float)
    for idx, candidate in enumerate(candidates):
        values = []
        for x, y in prepared:
            if x[0] <= candidate <= x[-1]:
                values.append(float(np.interp(np.log2(candidate), np.log2(x), y)))
        if len(values) >= 2:
            values = np.asarray(values)
            spreads[idx] = np.mean(np.abs(values[:, None] - values[None, :]))
    finite = np.flatnonzero(np.isfinite(spreads))
    if not len(finite):
        return None
    centre_idx = int(finite[np.argmin(spreads[finite])])
    return (float(candidates[max(0, centre_idx - 1)]),
            float(candidates[centre_idx]),
            float(candidates[min(len(candidates) - 1, centre_idx + 2)]))


def open_inset_bounds(ax, lines: Sequence[tuple[np.ndarray, np.ndarray]],
                      width: float = 0.41, height: float = 0.40) -> list[float]:
    """Choose the dashboard inset corner with the fewest plotted points.

    The candidates include a top-centre location, which is often clear for a
    U-shaped curve such as the regime sweep.  Coordinates are assessed in the
    active log/linear axis scales, so the decision follows the plotted space.
    """
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    if xlo <= 0 or xhi <= xlo or yhi <= ylo:
        return [0.56, 0.53, width, height]
    x_log = ax.get_xscale() == "log"
    y_log = ax.get_yscale() == "log" and ylo > 0
    def scaled(values: np.ndarray, lo: float, hi: float, log: bool) -> np.ndarray:
        if log:
            return (np.log(values) - np.log(lo)) / (np.log(hi) - np.log(lo))
        return (values - lo) / (hi - lo)
    candidates = [
        [0.04, 0.54, width, height], [0.30, 0.54, width, height],
        [0.56, 0.54, width, height], [0.04, 0.08, width, height],
        [0.30, 0.08, width, height], [0.56, 0.08, width, height],
    ]
    scores = []
    for left, bottom, box_width, box_height in candidates:
        score = 0
        for x, y in lines:
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
            if y_log:
                valid &= y > 0
            if not np.any(valid):
                continue
            sx = scaled(x[valid], xlo, xhi, x_log)
            sy = scaled(y[valid], ylo, yhi, y_log)
            score += int(np.sum((sx >= left) & (sx <= left + box_width)
                                & (sy >= bottom) & (sy <= bottom + box_height)))
        scores.append(score)
    return candidates[int(np.argmin(scores))]


def plot_experiment_dashboard(
    curves: dict[tuple[str, str], PlotCurve],
    models: Sequence[str],
    experiments: Sequence[str],
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(17, 11), sharex=False)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(models)))
    for ax, experiment in zip(axes.flat, experiments):
        panel_items = []
        for model, color in zip(models, colors):
            item = curves.get((model, experiment))
            if item is None:
                continue
            valid = np.isfinite(item.values) & (item.values > 0)
            ax.plot(item.ratios[valid], item.values[valid], marker="o", ms=2.5,
                    lw=1.35, color=color, label=model)
            panel_items.append((item, color))
        ax.axhline(1.02, color="#6B7280", ls="--", lw=0.9)
        ax.set_xscale("log", base=2)
        ax.set_xlim(*dashboard_xlim(experiment))
        ax.set_yscale("log")
        ax.grid(alpha=0.22)
        ax.set_title(EXPERIMENT_TITLES[experiment])
        ax.set_xlabel("context / parameter")
        ax.set_ylabel("MAE / attainable minimum")
        clamped = [item for item, _ in panel_items if item.clamped]
        if clamped:
            window = closest_agreement_window([
                (item.ratios, item.values) for item, _ in panel_items])
            if window is None:
                continue
            zoom_min, _, zoom_max = window
            inset = ax.inset_axes(open_inset_bounds(
                ax, [(item.ratios, item.values) for item, _ in panel_items]))
            inset_values = []
            for item, color in panel_items:
                keep = (
                    np.isfinite(item.values)
                    & (item.ratios >= zoom_min * (1 - 1e-12))
                    & (item.ratios <= zoom_max * (1 + 1e-12))
                )
                if not np.any(keep):
                    continue
                inset.plot(item.ratios[keep], item.values[keep], marker="o",
                           ms=1.8, lw=1, color=color)
                inset_values.append(item.values[keep])
            parts = [values[np.isfinite(values)] for values in inset_values
                     if np.isfinite(values).any()]
            if parts:
                finite = np.concatenate(parts)
                low, high = float(np.min(finite)), float(np.max(finite))
                pad = max((high - low) * 0.12,
                          max(abs(low), abs(high), 1.0) * 0.015)
                inset.set_ylim(low - pad, high + pad)
            inset.set_xscale("log", base=2)
            inset.grid(alpha=0.18)
            inset.tick_params(labelsize=5.5, length=2)
            inset.set_title("closest-agreement zoom", fontsize=6.5, pad=1)
            inset.set_facecolor((1, 1, 1, 0.94))
            for spine in inset.spines.values():
                spine.set_color("#6B7280")
                spine.set_linewidth(0.7)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6,
               fontsize=8, frameon=False)
    fig.suptitle("Synthetic context sweeps — normalized error curves", fontsize=16)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    _save_figure(fig, output_dir, "01_experiment_dashboard", formats)
    plt.close(fig)


def _summary_matrix(
    summaries: Sequence[CellSummary], models: Sequence[str],
    experiments: Sequence[str], field: str,
) -> np.ndarray:
    lookup = {(s.model, s.experiment): s for s in summaries}
    matrix = np.full((len(models), len(experiments)), np.nan)
    for row, model in enumerate(models):
        for col, experiment in enumerate(experiments):
            item = lookup.get((model, experiment))
            if item is not None:
                matrix[row, col] = float(getattr(item, field))
    return matrix


def plot_heatmap(
    summaries: Sequence[CellSummary], models: Sequence[str],
    experiments: Sequence[str], field: str, title: str, colorbar_label: str,
    stem: str, output_dir: Path, formats: Sequence[str], log_color: bool = False,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    matrix = _summary_matrix(summaries, models, experiments, field)
    shown = np.ma.masked_invalid(matrix)
    fig, ax = plt.subplots(figsize=(16, 7.4))
    finite = matrix[np.isfinite(matrix)]
    norm = None
    if log_color and len(finite):
        norm = LogNorm(vmin=max(float(np.min(finite)), 0.25),
                       vmax=max(float(np.max(finite)), 0.5))
    image = ax.imshow(shown, aspect="auto", cmap="viridis", norm=norm)
    ax.set_xticks(range(len(experiments)),
                  [EXPERIMENT_TITLES[x] for x in experiments], rotation=35,
                  ha="right")
    ax.set_yticks(range(len(models)), models)
    for row in range(len(models)):
        for col in range(len(experiments)):
            value = matrix[row, col]
            if np.isfinite(value):
                label = f"{value:g}" if field.endswith("ratio_2pct") else f"{value:.0f}%"
                ax.text(col, row, label, ha="center", va="center",
                        fontsize=7, color="white" if value > np.nanmedian(finite) else "black")
            else:
                ax.text(col, row, "missing", ha="center", va="center",
                        fontsize=6.5, color="#9CA3AF", rotation=25)
    fig.colorbar(image, ax=ax, label=colorbar_label, shrink=0.82)
    ax.set_title(title, fontsize=15)
    fig.tight_layout()
    _save_figure(fig, output_dir, stem, formats)
    plt.close(fig)


def plot_penalty_summary(
    summaries: Sequence[CellSummary], experiments: Sequence[str],
    output_dir: Path, formats: Sequence[str],
) -> None:
    import matplotlib.pyplot as plt

    short = []
    long = []
    for experiment in experiments:
        group = [s for s in summaries if s.experiment == experiment]
        short.append(float(np.nanmedian(
            [s.short_context_penalty_pct for s in group])) if group else np.nan)
        long.append(float(np.nanmedian(
            [s.longest_context_penalty_pct for s in group])) if group else np.nan)
    positions = np.arange(len(experiments))
    width = 0.39
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(positions - width / 2, short, width, label="Shortest context",
           color="#2563EB")
    ax.bar(positions + width / 2, long, width, label="Longest context",
           color="#F97316")
    ax.axhline(0, color="#111827", lw=0.8)
    ax.set_xticks(positions, [EXPERIMENT_TITLES[x] for x in experiments],
                  rotation=35, ha="right")
    ax.set_ylabel("Median excess MAE versus best context (%)")
    ax.set_title("Cost of using boundary context lengths")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, output_dir, "04_boundary_context_penalties", formats)
    plt.close(fig)


def write_outputs(
    root: Path, output_dir: Path, models: Sequence[str],
    experiments: Sequence[str], formats: Sequence[str], strict: bool,
) -> None:
    import matplotlib
    matplotlib.use("Agg")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries, curves = collect_results(root)
    coverage = coverage_rows(root, models, experiments)
    completed = sum(row["status"] == "complete" for row in coverage)
    expected = len(coverage)
    missing = [row for row in coverage if row["status"] != "complete"]

    _write_csv(output_dir / "coverage.csv", coverage)
    _write_csv(output_dir / "cell_summary.csv",
               [asdict(summary) for summary in summaries])
    experiment_rows = []
    for experiment in experiments:
        group = [s for s in summaries if s.experiment == experiment]
        if not group:
            continue
        experiment_rows.append({
            "experiment": experiment,
            "n_models": len(group),
            "median_best_nominal_ratio": float(np.nanmedian(
                [s.best_nominal_ratio for s in group])),
            "median_saturation_ratio_2pct": float(np.nanmedian(
                [s.saturation_ratio_2pct for s in group])),
            "median_short_context_penalty_pct": float(np.nanmedian(
                [s.short_context_penalty_pct for s in group])),
            "median_longest_context_penalty_pct": float(np.nanmedian(
                [s.longest_context_penalty_pct for s in group])),
            "median_best_mae_over_naive": float(np.nanmedian(
                [s.best_mae_over_naive for s in group])),
        })
    _write_csv(output_dir / "experiment_summary.csv", experiment_rows)
    report = {
        "root": str(root),
        "completed_cells": completed,
        "expected_cells": expected,
        "coverage_fraction": completed / expected if expected else 0,
        "missing_cells": [
            {"model": row["model"], "experiment": row["experiment"],
             "status": row["status"]}
            for row in missing
        ],
        "definitions": {
            "relative_curve": (
                "At h=64 (or the experiment-specific horizon), mean MAE over "
                "series is divided by each bin's attainable minimum and then "
                "averaged over bins."
            ),
            "saturation_ratio_2pct": (
                "Smallest nominal context/parameter ratio within 2% of the "
                "minimum normalized curve."
            ),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    if summaries:
        dashboard_experiments = [
            experiment for experiment in DASHBOARD_EXPERIMENTS
            if experiment in experiments]
        plot_experiment_dashboard(
            curves, models, dashboard_experiments, output_dir, formats)
        plot_heatmap(
            summaries, models, experiments, "saturation_ratio_2pct",
            "Where each model reaches within 2% of its best context",
            "nominal context / parameter", "02_saturation_ratio_heatmap",
            output_dir, formats, log_color=True,
        )
        plot_heatmap(
            summaries, models, experiments, "short_context_penalty_pct",
            "Penalty from using the shortest available context",
            "excess MAE (%)", "03_short_context_penalty_heatmap",
            output_dir, formats,
        )
        plot_penalty_summary(summaries, experiments, output_dir, formats)

    print(f"Synthetic sweep report: {completed}/{expected} cells complete")
    print(f"Outputs: {output_dir}")
    if missing:
        print("Missing cells:")
        for row in missing:
            print(f"  - {row['model']}/{row['experiment']}: {row['status']}")
    if strict and missing:
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate consolidated synthetic-sweep plots and tables.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS,
                        default=EXPERIMENTS)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"),
                        default=("png", "pdf"))
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero unless every requested cell is complete.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "summary_plots"
    write_outputs(root, output_dir, args.models, args.experiments,
                  args.formats, args.strict)


if __name__ == "__main__":
    main()
