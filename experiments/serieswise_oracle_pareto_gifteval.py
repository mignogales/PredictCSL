"""Trace a series-wise GiftEval Oracle MASE/FLOPs envelope.

This analysis consumes the aligned Phase-6 audit matrices written by
``evaluate_instance_windows.py``.  Every forecast instance may choose its own
numeric context window or native/full context.  Quality is still aggregated by
the official experiment convention:

1. valid-count-weighted MASE inside each ``(dataset, term)`` cell;
2. geometric mean of ``cell MASE / cell SeasonalNaive MASE`` over cells.

The instance action space is enormous (roughly 19**366,000 for Chronos2-Small),
and the geometric mean couples decisions within a cell.  Therefore this module
does not claim exhaustive discrete Pareto optimality.  It builds a dense,
reproducible *feasible Oracle envelope* from:

* exact minimum-MASE and minimum-compute endpoints;
* per-series relative-regret constraints; and
* supported sweeps of two first-order linearisations of the official log-MASE
  objective (around the full and minimum-MASE policies).

The lower nondominated envelope of those policies is reported in the official
normalized MASE.  Existing dataset-wise and series-wise Mamba policies are
overlaid without rerunning a TSFM or Mamba: their per-series score curves and
selected actions are already stored in the Phase-6 audits.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.compare_window_strategies_gifteval import (
    DEFAULT_PATCH_SIZES,
    theoretical_flops,
)


@dataclass
class CellActions:
    key: str
    naive_mase: float
    errors: np.ndarray
    costs: np.ndarray
    valid_counts: np.ndarray
    windows: np.ndarray
    selection_eligible: bool
    audit_path: Path

    @property
    def n_instances(self) -> int:
        return int(self.errors.shape[0])

    @property
    def native_index(self) -> int:
        return int(self.errors.shape[1] - 1)


def _geomean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or np.any(~np.isfinite(array)) or np.any(array <= 0):
        return float("nan")
    return float(np.exp(np.log(array).mean()))


def _flops_for_contexts(
    model_id: str, contexts: np.ndarray, horizon: int,
) -> np.ndarray:
    contexts = np.asarray(contexts, dtype=np.int64)
    result = np.full(contexts.shape, np.nan, dtype=np.float64)
    valid = contexts > 0
    if not valid.any():
        return result
    unique = np.unique(contexts[valid])
    lookup = {
        int(context): theoretical_flops(
            model_id, int(context), int(horizon), DEFAULT_PATCH_SIZES)
        for context in unique
    }
    for context, value in lookup.items():
        result[contexts == context] = value
    return result


def load_cells(
    instance_dir: Path,
    comparison_csv: Path,
    model_short: str,
) -> Tuple[List[CellActions], pd.DataFrame]:
    comparison = pd.read_csv(comparison_csv)
    comparison = comparison[comparison["model_short"] == model_short].copy()
    if comparison.empty:
        raise ValueError(
            f"No rows for model_short={model_short!r} in {comparison_csv}.")
    comparison["term"] = comparison["term"].astype(str)

    cells: List[CellActions] = []
    for row in comparison.itertuples(index=False):
        key = f"{row.dataset_display}/t{row.term}"
        audit_path = (
            instance_dir / "cells"
            / f"{model_short}__{row.dataset_display}__t{row.term}.npz"
        )
        if not audit_path.is_file():
            raise FileNotFoundError(audit_path)
        with np.load(audit_path) as data:
            grid_windows = np.asarray(data["window_grid"], dtype=np.int64)
            grid_errors = np.asarray(data["grid_mase"], dtype=np.float64)
            native_error = np.asarray(data["native_mase"], dtype=np.float64)
            native_valid_count = np.asarray(
                data["native_valid_count"], dtype=np.float64)
            grid_valid_count = (
                np.asarray(data["grid_valid_count"], dtype=np.float64)
                if "grid_valid_count" in data.files
                else np.broadcast_to(
                    native_valid_count[:, None], grid_errors.shape).copy()
            )
            native_context = np.asarray(
                data["native_effective_context"], dtype=np.int64)

        if grid_errors.shape != (native_error.size, grid_windows.size):
            raise ValueError(f"Misaligned grid errors in {audit_path}.")
        if native_valid_count.shape != native_error.shape:
            raise ValueError(f"Misaligned native valid counts in {audit_path}.")
        if grid_valid_count.shape != grid_errors.shape:
            raise ValueError(f"Misaligned grid valid counts in {audit_path}.")
        if native_context.shape != native_error.shape:
            raise ValueError(f"Misaligned native contexts in {audit_path}.")

        grid_cost = np.asarray([
            theoretical_flops(
                str(row.model), int(window), int(row.horizon),
                DEFAULT_PATCH_SIZES)
            for window in grid_windows
        ], dtype=np.float64)
        native_cost = _flops_for_contexts(
            str(row.model), native_context, int(row.horizon))
        if np.any(~np.isfinite(native_cost)):
            fallback_context = int(getattr(row, "full_window", 8192))
            fallback = theoretical_flops(
                str(row.model), fallback_context, int(row.horizon),
                DEFAULT_PATCH_SIZES)
            native_cost = np.where(np.isfinite(native_cost), native_cost, fallback)

        errors = np.column_stack([grid_errors, native_error])
        valid_counts = np.column_stack([
            grid_valid_count, native_valid_count,
        ])
        costs = np.column_stack([
            np.broadcast_to(grid_cost, grid_errors.shape), native_cost,
        ])
        windows = np.column_stack([
            np.broadcast_to(grid_windows, grid_errors.shape), native_context,
        ])
        cells.append(CellActions(
            key=key,
            naive_mase=float(row.naive_mase),
            errors=errors,
            costs=costs,
            valid_counts=valid_counts,
            windows=windows,
            selection_eligible=bool(row.selection_eligible),
            audit_path=audit_path,
        ))
    return cells, comparison.reset_index(drop=True)


def _valid_actions(cell: CellActions) -> np.ndarray:
    # Oracle policies may use every measured action, including the sole grid
    # action in a one-grid cell.  Learned selectors separately respect
    # ``selection_eligible`` because a one-class cell cannot support selection.
    return (
        np.isfinite(cell.errors) & np.isfinite(cell.valid_counts)
        & (cell.valid_counts > 0)
    )


def select_accuracy_oracle(cell: CellActions) -> np.ndarray:
    valid = _valid_actions(cell)
    choices = np.argmin(np.where(valid, cell.errors, np.inf), axis=1)
    choices[~valid.any(axis=1)] = cell.native_index
    return choices.astype(np.int16)


def select_minimum_compute(cell: CellActions) -> np.ndarray:
    valid = _valid_actions(cell)
    choices = np.argmin(np.where(valid, cell.costs, np.inf), axis=1)
    choices[~valid.any(axis=1)] = cell.native_index
    return choices.astype(np.int16)


def select_relative_tolerance(
    cell: CellActions, tolerance: float,
) -> np.ndarray:
    """Cheapest action no more than ``tolerance`` above each row's best MASE."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    valid = _valid_actions(cell)
    best = np.min(np.where(valid, cell.errors, np.inf), axis=1)
    if math.isinf(tolerance):
        allowed = valid
    else:
        # Multiplicative tolerance is meaningful for MASE.  The tiny absolute
        # allowance avoids numerical instability for an exactly-zero best row.
        limit = best * (1.0 + float(tolerance)) + 1e-12
        allowed = valid & (cell.errors <= limit[:, None])
    choices = np.argmin(np.where(allowed, cell.costs, np.inf), axis=1)
    choices[~allowed.any(axis=1)] = cell.native_index
    return choices.astype(np.int16)


def _selected_cell_mase(cell: CellActions, choices: np.ndarray) -> float:
    rows = np.arange(cell.n_instances)
    errors = cell.errors[rows, choices]
    counts = cell.valid_counts[rows, choices]
    ok = (
        np.isfinite(errors) & np.isfinite(counts) & (counts > 0)
    )
    if not ok.any():
        return float("nan")
    return float(np.average(errors[ok], weights=counts[ok]))


def aggregate_policy(
    cells: Sequence[CellActions],
    choices: Sequence[np.ndarray],
    full_flops: float,
    family: str,
    parameter: float,
) -> dict:
    cell_mase: List[float] = []
    total_flops = 0.0
    native_selected = 0
    selected_contexts: List[np.ndarray] = []
    for cell, selected in zip(cells, choices):
        rows = np.arange(cell.n_instances)
        cell_mase.append(_selected_cell_mase(cell, selected))
        total_flops += float(cell.costs[rows, selected].sum())
        native_selected += int(np.sum(selected == cell.native_index))
        selected_contexts.append(cell.windows[rows, selected])
    normalized = _geomean(
        mase / cell.naive_mase for mase, cell in zip(cell_mase, cells))
    contexts = np.concatenate(selected_contexts)
    return {
        "family": family,
        "parameter": float(parameter),
        "normalized_mase": normalized,
        "geomean_mase": _geomean(cell_mase),
        "total_flops": total_flops,
        "flops_ratio": total_flops / full_flops,
        "flops_saved_pct": 100.0 * (1.0 - total_flops / full_flops),
        "native_selected": int(native_selected),
        "mean_context": float(np.mean(contexts)),
        "median_context": float(np.median(contexts)),
    }


def select_linearized(
    cell: CellActions,
    reference_mase: float,
    lagrange: float,
    n_cells: int,
    full_flops: float,
) -> np.ndarray:
    """Select one action per row under a first-order official-log-MASE score."""
    if reference_mase <= 0 or not np.isfinite(reference_mase):
        raise ValueError("reference_mase must be finite and positive")
    valid = _valid_actions(cell)
    # Use the reference policy's cell denominator as the first-order scale.
    # Valid forecast counts are action-specific in GluonTS, so each candidate
    # contributes its own numerator and denominator weight.  Counts are nearly
    # always identical across actions; retaining them exactly also handles the
    # few partially-defined horizons without silently changing the cohort.
    native_counts = cell.valid_counts[:, cell.native_index]
    total_weight = float(native_counts[
        np.isfinite(native_counts) & (native_counts > 0)].sum())
    # d log(N/D) = dN/N - dD/D.  Writing N = reference_mase * D gives each
    # discrete candidate the first-order contribution below.  When all actions
    # have equal valid counts, the ``-1`` term is constant within a row.
    quality = (
        cell.valid_counts * (cell.errors / float(reference_mase) - 1.0)
        / (float(n_cells) * total_weight)
    )
    score = quality + float(lagrange) * cell.costs / float(full_flops)
    choices = np.argmin(np.where(valid, score, np.inf), axis=1)
    choices[~valid.any(axis=1)] = cell.native_index
    return choices.astype(np.int16)


def pareto_envelope(points: pd.DataFrame) -> pd.DataFrame:
    """Keep observed policies not dominated in total FLOPs and MASE."""
    required = {"normalized_mase", "total_flops"}
    if not required.issubset(points.columns):
        raise ValueError(f"points must contain {sorted(required)}")
    valid = points[
        np.isfinite(points["normalized_mase"])
        & np.isfinite(points["total_flops"])
    ].copy()
    valid = valid.sort_values(
        ["total_flops", "normalized_mase", "family", "parameter"])
    valid = valid.drop_duplicates("total_flops", keep="first")
    keep: List[int] = []
    best_quality = float("inf")
    for index, row in valid.iterrows():
        quality = float(row.normalized_mase)
        if quality < best_quality - 1e-12:
            keep.append(index)
            best_quality = quality
    result = valid.loc[keep].copy()
    return result.sort_values("flops_saved_pct").reset_index(drop=True)


def _choices_from_stored_method(
    cell: CellActions, method: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    with np.load(cell.audit_path) as data:
        mase_key = f"{method}__mase"
        window_key = f"{method}__window"
        count_key = f"{method}__valid_count"
        if not all(key in data.files for key in (mase_key, window_key, count_key)):
            return None
        return (
            np.asarray(data[mase_key], dtype=np.float64),
            np.asarray(data[window_key], dtype=np.int64),
            np.asarray(data[count_key], dtype=np.float64),
        )


def _reference_from_stored_method(
    cells: Sequence[CellActions], method: str, full_flops: float,
    label: str,
) -> Optional[dict]:
    cell_mase = []
    total_flops = 0.0
    total_native = 0
    all_contexts = []
    for cell in cells:
        stored = _choices_from_stored_method(cell, method)
        if stored is None:
            return None
        errors, contexts, counts = stored
        ok = np.isfinite(errors) & np.isfinite(counts) & (counts > 0)
        if not ok.any():
            return None
        cell_mase.append(float(np.average(errors[ok], weights=counts[ok])))
        contexts = np.where(contexts > 0, contexts, cell.windows[:, -1])
        context_cost = np.empty(contexts.shape, dtype=np.float64)
        # The stored method can mix grid and native widths.  Costs are already
        # present for every unique context in the cell's action matrix.
        for context in np.unique(contexts):
            matches = cell.windows == int(context)
            values = cell.costs[matches]
            if values.size:
                context_cost[contexts == context] = float(values[0])
            else:
                raise ValueError(
                    f"Stored context {context} absent from {cell.key} actions.")
        total_flops += float(context_cost.sum())
        all_contexts.append(contexts)
        # A native action can share a numeric width with a grid action, so this
        # count is diagnostic only and deliberately conservative.
        total_native += int(np.sum(contexts == cell.windows[:, -1]))
    contexts = np.concatenate(all_contexts)
    return {
        "label": label,
        "normalized_mase": _geomean(
            mase / cell.naive_mase for mase, cell in zip(cell_mase, cells)),
        "geomean_mase": _geomean(cell_mase),
        "total_flops": total_flops,
        "flops_saved_pct": 100.0 * (1.0 - total_flops / full_flops),
        "native_selected": total_native,
        "mean_context": float(np.mean(contexts)),
        "median_context": float(np.median(contexts)),
        "source": f"Phase-6 stored method {method}",
    }


def _raw_mamba_reference(
    cells: Sequence[CellActions], full_flops: float,
) -> Optional[dict]:
    cell_mase = []
    total_flops = 0.0
    native_count = 0
    all_contexts = []
    for cell in cells:
        with np.load(cell.audit_path) as data:
            if "mamba_curve__scores" not in data.files:
                return None
            scores = np.asarray(data["mamba_curve__scores"], dtype=np.float64)
            native_score_supported = (
                bool(np.asarray(data["native_score_supported"]).item())
                if "native_score_supported" in data.files else True
            )
        if scores.shape != cell.errors[:, :-1].shape:
            raise ValueError(f"Misaligned Mamba scores in {cell.audit_path}.")
        if not cell.selection_eligible:
            choices = np.full(cell.n_instances, cell.native_index, dtype=np.int16)
        else:
            native_score = (
                np.nextafter(scores[:, -1], -np.inf)
                if native_score_supported
                else np.full(scores.shape[0], np.nan, dtype=np.float64)
            )
            candidate_scores = np.column_stack([scores, native_score])
            valid = _valid_actions(cell) & np.isfinite(candidate_scores)
            choices = np.argmin(
                np.where(valid, candidate_scores, np.inf), axis=1)
            choices[~valid.any(axis=1)] = cell.native_index
        rows = np.arange(cell.n_instances)
        cell_mase.append(_selected_cell_mase(cell, choices))
        total_flops += float(cell.costs[rows, choices].sum())
        native_count += int(np.sum(choices == cell.native_index))
        all_contexts.append(cell.windows[rows, choices])
    contexts = np.concatenate(all_contexts)
    return {
        "label": "Mamba series-wise raw",
        "normalized_mase": _geomean(
            mase / cell.naive_mase for mase, cell in zip(cell_mase, cells)),
        "geomean_mase": _geomean(cell_mase),
        "total_flops": total_flops,
        "flops_saved_pct": 100.0 * (1.0 - total_flops / full_flops),
        "native_selected": native_count,
        "mean_context": float(np.mean(contexts)),
        "median_context": float(np.median(contexts)),
        "source": "raw per-series Mamba curve argmin with native action",
    }


def _dataset_references(
    comparison: pd.DataFrame, full_flops: float,
) -> List[dict]:
    specs = [
        ("Full/native", "full_mase", "full_window"),
        ("Dataset-wise accuracy Oracle", "best_mase", "best_window"),
        ("Mamba dataset-shared", "pred_mamba_mase", "pred_mamba_window"),
    ]
    references = []
    weights = comparison["n_instances"].to_numpy(dtype=np.float64)
    naive = comparison["naive_mase"].to_numpy(dtype=np.float64)
    for label, mase_column, window_column in specs:
        if mase_column not in comparison or window_column not in comparison:
            continue
        mase = comparison[mase_column].to_numpy(dtype=np.float64)
        flops = np.asarray([
            theoretical_flops(
                str(row.model), int(getattr(row, window_column)),
                int(row.horizon), DEFAULT_PATCH_SIZES)
            for row in comparison.itertuples(index=False)
        ])
        valid = (
            np.isfinite(mase) & np.isfinite(naive) & (mase > 0) & (naive > 0)
            & np.isfinite(flops) & np.isfinite(weights)
        )
        if not valid.all():
            continue
        total = float(np.sum(flops * weights))
        references.append({
            "label": label,
            "normalized_mase": _geomean(mase / naive),
            "geomean_mase": _geomean(mase),
            "total_flops": total,
            "flops_saved_pct": 100.0 * (1.0 - total / full_flops),
            "native_selected": float("nan"),
            "mean_context": float("nan"),
            "median_context": float("nan"),
            "source": f"Stage-4 {mase_column}/{window_column}; FLOPs recomputed",
        })
    return references


def _risk_references(
    specs: Sequence[str], full_point: dict, full_flops: float,
) -> List[dict]:
    """Load series-wise risk-policy points from calibrated-risk reports.

    Syntax is ``LABEL=/path/to/real_evaluation.json#method``. The report's
    saving percentage is cohort-relative and therefore transfers exactly to
    the common full-workload axis even when its eligible instance count differs.
    """
    rows = []
    for spec in specs:
        if "=" not in spec or "#" not in spec:
            raise ValueError(
                "Risk reference must be LABEL=/path/real_evaluation.json#method")
        label, source = spec.split("=", 1)
        path_text, method = source.rsplit("#", 1)
        path = Path(path_text)
        with path.open() as handle:
            report = json.load(handle)
        aggregate = report.get("aggregate", {})
        if method not in aggregate:
            raise KeyError(f"{path} has no aggregate method {method!r}")
        point = aggregate[method]
        ratio = float(point["geomean_cell_mase_ratio"])
        saved = float(point["theoretical_flops_saved_pct"])
        rows.append({
            "label": label,
            "normalized_mase": float(full_point["normalized_mase"]) * ratio,
            "geomean_mase": float(full_point["geomean_mase"]) * ratio,
            "total_flops": full_flops * (1.0 - saved / 100.0),
            "flops_saved_pct": saved,
            "native_selected": float("nan"),
            "mean_context": float(point.get("mean_selected_context", float("nan"))),
            "median_context": float("nan"),
            "source": f"{path}#{method}",
        })
    return rows


def key_operating_points(
    frontier: pd.DataFrame, full_mase: float,
) -> pd.DataFrame:
    accuracy_best = float(frontier["normalized_mase"].min())
    constraints = [("no worse than full", full_mase)] + [
        (f"within {percent:g}% of series Oracle",
         accuracy_best * (1.0 + percent / 100.0))
        for percent in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
    ]
    rows = []
    for label, limit in constraints:
        eligible = frontier[frontier["normalized_mase"] <= limit + 1e-12]
        if eligible.empty:
            continue
        point = eligible.loc[eligible["flops_saved_pct"].idxmax()]
        rows.append({
            "constraint": label,
            "normalized_mase_limit": limit,
            "normalized_mase": float(point.normalized_mase),
            "flops_saved_pct": float(point.flops_saved_pct),
            "family": str(point.family),
            "parameter": float(point.parameter),
            "native_selected": int(point.native_selected),
        })
    return pd.DataFrame(rows)


def plot_frontier(
    frontier: pd.DataFrame,
    references: pd.DataFrame,
    key_points: pd.DataFrame,
    output: Path,
    model_short: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    colors = {
        "Full/native": "#222222",
        "Dataset-wise accuracy Oracle": "#1f77b4",
        "Series-wise accuracy Oracle": "#2ca02c",
        "Mamba dataset-shared": "#9467bd",
        "Mamba series-wise (shrinkage=0.25)": "#d62728",
        "Mamba series-wise raw": "#ff7f0e",
        "Balanced risk": "#8c564b",
        "Aggressive risk": "#e377c2",
        "Very aggressive risk": "#bcbd22",
        "Very-very aggressive risk": "#17becf",
        "Extreme risk": "#7f7f7f",
        "Very extreme risk": "#d62728",
    }
    markers = {
        "Full/native": "X",
        "Dataset-wise accuracy Oracle": "D",
        "Series-wise accuracy Oracle": "P",
        "Mamba dataset-shared": "o",
        "Mamba series-wise (shrinkage=0.25)": "s",
        "Mamba series-wise raw": "^",
        "Balanced risk": "v",
        "Aggressive risk": "<",
        "Very aggressive risk": ">",
        "Very-very aggressive risk": "h",
        "Extreme risk": "*",
        "Very extreme risk": "8",
    }
    for axis in axes:
        axis.plot(
            frontier["flops_saved_pct"], frontier["normalized_mase"],
            color="#087E8B", linewidth=2.1,
            label="Series-wise Oracle feasible envelope",
        )
        for row in references.itertuples(index=False):
            axis.scatter(
                row.flops_saved_pct, row.normalized_mase,
                marker=markers.get(row.label, "o"), s=62,
                color=colors.get(row.label, "#555555"),
                edgecolor="white", linewidth=0.7, zorder=4, label=row.label,
            )
        if not key_points.empty:
            axis.scatter(
                key_points["flops_saved_pct"], key_points["normalized_mase"],
                marker="s", s=25, color="#087E8B", zorder=3,
                label="Accuracy constraints",
            )
        axis.set_xlabel("Theoretical TSFM FLOPs saved vs native (%)")
        axis.set_ylabel("GiftEval normalized MASE (lower is better)")
        axis.grid(alpha=0.22)

    axes[0].set_xlim(-2, 100)
    axes[0].set_title("Full feasible envelope")
    useful = frontier[frontier["normalized_mase"] <= max(
        0.82, float(references["normalized_mase"].max()) * 1.03)]
    if useful.empty:
        useful = frontier
    axes[1].set_xlim(
        max(-1.0, float(useful["flops_saved_pct"].min()) - 2.0),
        min(100.0, float(useful["flops_saved_pct"].max()) + 2.0),
    )
    lower = min(
        float(useful["normalized_mase"].min()),
        float(references["normalized_mase"].min()),
    )
    upper = max(
        float(useful["normalized_mase"].max()),
        float(references["normalized_mase"].max()),
    )
    axes[1].set_ylim(lower * 0.995, upper * 1.01)
    axes[1].set_title("Useful accuracy region")

    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[1].legend(
        by_label.values(), by_label.keys(), loc="best", fontsize=7.4,
        frameon=True,
    )
    fig.suptitle(
        f"{model_short} · GiftEval series-wise Oracle MASE/FLOPs envelope\n"
        "one context action per forecast instance · 97 dataset/term cells\n"
        "official cell-normalized geometric MASE; workload-summed FLOPs",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output, dpi=190)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cells, comparison = load_cells(
        Path(args.instance_dir), Path(args.comparison_csv), args.model_short)
    n_instances = int(sum(cell.n_instances for cell in cells))
    full_choices = [
        np.full(cell.n_instances, cell.native_index, dtype=np.int16)
        for cell in cells
    ]
    full_flops = float(sum(
        cell.costs[:, cell.native_index].sum() for cell in cells))
    full_point = aggregate_policy(
        cells, full_choices, full_flops, "full", 0.0)

    accuracy_choices = [select_accuracy_oracle(cell) for cell in cells]
    accuracy_point = aggregate_policy(
        cells, accuracy_choices, full_flops, "accuracy_oracle", 0.0)
    min_compute_choices = [select_minimum_compute(cell) for cell in cells]
    min_compute_point = aggregate_policy(
        cells, min_compute_choices, full_flops, "minimum_compute", math.inf)

    candidate_rows = [full_point, accuracy_point, min_compute_point]

    tolerances = np.concatenate([
        np.asarray([0.0]),
        np.geomspace(args.min_tolerance, args.max_tolerance,
                     args.n_tolerances),
        np.asarray([math.inf]),
    ])
    for tolerance in tolerances:
        choices = [
            select_relative_tolerance(cell, float(tolerance)) for cell in cells
        ]
        candidate_rows.append(aggregate_policy(
            cells, choices, full_flops, "relative_tolerance", tolerance))

    full_reference = [
        _selected_cell_mase(cell, choice)
        for cell, choice in zip(cells, full_choices)
    ]
    accuracy_reference = [
        _selected_cell_mase(cell, choice)
        for cell, choice in zip(cells, accuracy_choices)
    ]
    lagranges = np.concatenate([
        np.asarray([0.0]),
        np.geomspace(args.min_lagrange, args.max_lagrange,
                     args.n_lagranges),
    ])
    for family, references in (
        ("linearized_full", full_reference),
        ("linearized_accuracy", accuracy_reference),
    ):
        for lagrange in lagranges:
            choices = [
                select_linearized(
                    cell, reference, float(lagrange), len(cells), full_flops)
                for cell, reference in zip(cells, references)
            ]
            candidate_rows.append(aggregate_policy(
                cells, choices, full_flops, family, lagrange))

    candidates = pd.DataFrame(candidate_rows)
    frontier = pareto_envelope(candidates)

    references = _dataset_references(comparison, full_flops)
    references.append({
        "label": "Series-wise accuracy Oracle",
        **{key: accuracy_point[key] for key in (
            "normalized_mase", "geomean_mase", "total_flops",
            "flops_saved_pct", "native_selected", "mean_context",
            "median_context")},
        "source": "minimum realised MASE per forecast instance",
    })
    stored_mamba = _reference_from_stored_method(
        cells, "mamba_curve_instance", full_flops,
        "Mamba series-wise (shrinkage=0.25)")
    if stored_mamba is not None:
        references.append(stored_mamba)
    raw_mamba = _raw_mamba_reference(cells, full_flops)
    if raw_mamba is not None:
        references.append(raw_mamba)
    references.extend(_risk_references(
        args.risk_reference, full_point, full_flops))
    reference_frame = pd.DataFrame(references)

    key_points = key_operating_points(
        frontier, float(full_point["normalized_mase"]))
    comparison_rows = []
    for row in reference_frame.itertuples(index=False):
        eligible = frontier[
            frontier["flops_saved_pct"] >= float(row.flops_saved_pct) - 1e-12]
        if eligible.empty:
            oracle_mase = float("nan")
            gap = float("nan")
        else:
            oracle_mase = float(eligible["normalized_mase"].min())
            gap = 100.0 * (float(row.normalized_mase) / oracle_mase - 1.0)
        comparison_rows.append({
            "label": row.label,
            "normalized_mase": float(row.normalized_mase),
            "flops_saved_pct": float(row.flops_saved_pct),
            "envelope_mase_at_equal_or_greater_saving": oracle_mase,
            "relative_mase_gap_pct": gap,
        })
    gaps = pd.DataFrame(comparison_rows)

    candidates.to_csv(output / "serieswise_oracle_candidates.csv", index=False)
    frontier.to_csv(output / "serieswise_oracle_pareto_envelope.csv", index=False)
    reference_frame.to_csv(output / "reference_points.csv", index=False)
    key_points.to_csv(output / "key_operating_points.csv", index=False)
    gaps.to_csv(output / "reference_gaps.csv", index=False)
    plot_frontier(
        frontier, reference_frame, key_points,
        output / "serieswise_oracle_pareto.png", args.model_short)

    report = {
        "model_short": args.model_short,
        "n_cells": len(cells),
        "n_forecast_instances": n_instances,
        "n_mase_valid_instances": int(sum(
            np.sum(
                np.isfinite(cell.valid_counts[:, cell.native_index])
                & (cell.valid_counts[:, cell.native_index] > 0))
            for cell in cells)),
        "aggregation": (
            "valid-count-weighted MASE per dataset/term, then geometric mean "
            "of cell MASE / SeasonalNaive MASE"),
        "compute": "sum of per-forecast theoretical TSFM FLOPs",
        "frontier_claim": (
            "nondominated feasible envelope from relative-regret and two "
            "linearized supported sweeps; not exhaustive discrete frontier"),
        "n_candidate_policies": len(candidates),
        "n_envelope_points": len(frontier),
        "full_normalized_mase": float(full_point["normalized_mase"]),
        "series_oracle_normalized_mase": float(
            accuracy_point["normalized_mase"]),
        "series_oracle_flops_saved_pct": float(
            accuracy_point["flops_saved_pct"]),
        "minimum_compute_normalized_mase": float(
            min_compute_point["normalized_mase"]),
        "maximum_flops_saved_pct": float(
            min_compute_point["flops_saved_pct"]),
        "references": reference_frame.to_dict(orient="records"),
        "key_operating_points": key_points.to_dict(orient="records"),
    }
    with open(output / "report.json", "w") as handle:
        json.dump(report, handle, indent=2, allow_nan=True)

    print(json.dumps(report, indent=2, allow_nan=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-dir", required=True)
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--model-short", default="Chronos2-Small")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--risk-reference", action="append", default=[],
        help=("Overlay LABEL=/path/to/real_evaluation.json#method; "
              "repeat for multiple calibrated-risk profiles."),
    )
    parser.add_argument("--n-tolerances", type=int, default=96)
    parser.add_argument("--min-tolerance", type=float, default=1e-5)
    parser.add_argument("--max-tolerance", type=float, default=1e2)
    parser.add_argument("--n-lagranges", type=int, default=96)
    parser.add_argument("--min-lagrange", type=float, default=1e-5)
    parser.add_argument("--max-lagrange", type=float, default=1e3)
    args = parser.parse_args()
    if args.n_tolerances < 1 or args.n_lagranges < 1:
        parser.error("sweep sizes must be positive")
    if args.min_tolerance <= 0 or args.max_tolerance <= args.min_tolerance:
        parser.error("tolerance bounds must be positive and increasing")
    if args.min_lagrange <= 0 or args.max_lagrange <= args.min_lagrange:
        parser.error("Lagrange bounds must be positive and increasing")
    return args


if __name__ == "__main__":
    run(parse_args())
