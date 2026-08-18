"""Compact all-model sweep of hierarchical instance-curve shrinkage.

This reuses Phase-6's authoritative loading and action-selection functions but
does not write its large per-cell audit matrices or mutate Stage-4 reports.
Each cell is loaded once and all requested shrinkage weights are evaluated in
memory.  The output contains official normalized GiftEval MASE and
workload-summed theoretical TSFM FLOPs savings for every model/weight pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.compare_window_strategies_gifteval import (
    DEFAULT_PATCH_SIZES,
    theoretical_flops,
)
from experiments.evaluate_instance_windows import (
    _choose_regularized_scores,
    _ground_tree,
    _regularize_scores,
    discover_cells,
    evaluate_cell,
)


def _geomean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(array) & (array > 0)
    return float(np.exp(np.mean(np.log(array[valid]))))


def _context_flops(model_id: str, contexts: np.ndarray, horizon: int) -> float:
    contexts = np.asarray(contexts, dtype=np.int64)
    lookup = {
        int(context): theoretical_flops(
            model_id, int(context), int(horizon), DEFAULT_PATCH_SIZES)
        for context in np.unique(contexts[contexts > 0])
    }
    return float(sum(lookup[int(context)] for context in contexts if context > 0))


def run(args: argparse.Namespace) -> pd.DataFrame:
    root = str(Path(args.ablation_root))
    ground_tree = _ground_tree(root)
    if ground_tree is None:
        raise SystemExit(f"No ground datasets tree below {root}")
    cells = discover_cells(root, args.models)
    if not cells:
        raise SystemExit("No predictor cells found")

    base_frames: dict[str, pd.DataFrame] = {}
    aggregates: dict[tuple[str, float, str], dict] = {}
    cell_rows: list[dict] = []

    def accumulate(
        model: str, dataset: str, term: str, alpha: float, policy: str,
        values: np.ndarray,
        selected_w: np.ndarray, selected_native: np.ndarray,
        selected_counts: np.ndarray, native_context: np.ndarray,
        naive_mase: float, model_id: str, horizon: int,
    ) -> None:
        valid = np.isfinite(values) & (selected_counts > 0)
        cell_mase = float(np.average(values[valid], weights=selected_counts[valid]))
        contexts = np.where(selected_w > 0, selected_w, native_context)
        selected_flops = _context_flops(model_id, contexts[valid], horizon)
        reference_flops = _context_flops(
            model_id, native_context[valid], horizon)
        key = (model, float(alpha), policy)
        agg = aggregates.setdefault(key, {
            "ratios": [], "selected_flops": 0.0, "full_flops": 0.0,
            "native_selected": 0, "instances": 0, "cells": 0,
        })
        agg["ratios"].append(cell_mase / naive_mase)
        agg["selected_flops"] += selected_flops
        agg["full_flops"] += reference_flops
        agg["native_selected"] += int(np.sum(selected_native & valid))
        agg["instances"] += int(np.sum(valid))
        agg["cells"] += 1
        cell_rows.append({
            "model": model,
            "dataset_display": dataset,
            "term": term,
            "alpha": float(alpha),
            "fallback_policy": policy,
            "mase": cell_mase,
            "naive_mase": naive_mase,
            "normalized_mase": cell_mase / naive_mase,
            "flops_saved_pct": 100.0 * (
                1.0 - selected_flops / reference_flops),
            "selected_flops": selected_flops,
            "full_native_flops": reference_flops,
            "native_selected": int(np.sum(selected_native & valid)),
            "n_instances": int(np.sum(valid)),
        })

    for cell in tqdm(cells, desc="Compact shrinkage sweep", unit="cell"):
        if cell.model not in base_frames:
            path = Path(root) / cell.model / "strategy_comparison_v4" / "comparison.csv"
            frame = pd.read_csv(path)
            frame["term"] = frame["term"].astype(str)
            base_frames[cell.model] = frame
        base = base_frames[cell.model]
        match = base[
            (base["dataset_display"].astype(str) == str(cell.dataset))
            & (base["term"] == str(cell.term))
        ]
        if len(match) != 1:
            raise ValueError(f"Missing/duplicate base row for {cell}")
        base_row = match.iloc[0]

        _records, audit = evaluate_cell(
            cell, root, ground_tree, mase_field=args.mase_field,
            instance_weight=0.0,
        )
        if "mamba_curve__scores" not in audit:
            continue

        errors = audit["grid_mase"]
        counts = audit["grid_valid_count"]
        windows = audit["window_grid"]
        native = audit["native_mase"]
        native_counts = audit["native_valid_count"]
        native_context = audit["native_effective_context"]
        scores = audit["mamba_curve__scores"]
        native_supported = bool(audit["native_score_supported"])
        eligible = bool(audit["selection_eligible"])
        model_id = str(base_row.get("model", cell.model))
        horizon = int(base_row.get("horizon", 1))
        naive_mase = float(base_row["naive_mase"])
        for alpha in args.weights:
            accumulate(
                cell.model, cell.dataset, cell.term, alpha, "full_native",
                native, native_context,
                np.ones(native.shape, dtype=bool), native_counts,
                native_context, naive_mase, model_id, horizon)
            if eligible:
                values, selected_w, selected_native, regularized = _choose_regularized_scores(
                    scores, errors, windows, native, native_context,
                    instance_weight=alpha,
                    native_score_supported=native_supported,
                )
            else:
                values = native.copy()
                selected_w = native_context.copy()
                selected_native = np.ones(native.shape, dtype=bool)
                regularized = _regularize_scores(scores, alpha)

            selected_counts = native_counts.copy()
            for j, window in enumerate(windows):
                use = (~selected_native) & (selected_w == int(window))
                selected_counts[use] = counts[use, j]
            accumulate(
                cell.model, cell.dataset, cell.term, alpha,
                "feasible_backoff", values, selected_w,
                selected_native, selected_counts, native_context, naive_mase,
                model_id, horizon)

            # Alternative requested policy: first choose the preferred action
            # without a row feasibility mask. If that action is unavailable for
            # the row, retain its actual native context instead of backing off
            # to another supported grid point.
            if eligible:
                native_score = regularized[:, -1]
                if not native_supported:
                    native_score = np.full(native_score.shape, np.inf)
                candidates = np.column_stack([regularized, native_score])
                preferred = np.argmin(candidates, axis=1)
                rows = np.arange(native.size)
                preferred_grid = preferred < windows.size
                grid_idx = np.minimum(preferred, windows.size - 1)
                grid_available = preferred_grid & np.isfinite(
                    errors[rows, grid_idx])
                native_values = np.where(
                    grid_available, errors[rows, grid_idx], native)
                native_w = np.where(
                    grid_available, windows[grid_idx], native_context)
                native_selected_alt = ~grid_available
                native_counts_alt = native_counts.copy()
                for j, window in enumerate(windows):
                    use = grid_available & (native_w == int(window))
                    native_counts_alt[use] = counts[use, j]
            else:
                native_values = native.copy()
                native_w = native_context.copy()
                native_selected_alt = np.ones(native.shape, dtype=bool)
                native_counts_alt = native_counts.copy()
            accumulate(
                cell.model, cell.dataset, cell.term, alpha,
                "native_fallback", native_values, native_w,
                native_selected_alt, native_counts_alt, native_context,
                naive_mase, model_id, horizon)

            # Cost-aware near-optimal set: among actions whose predicted score
            # lies within a fraction of the cell score range from the predicted
            # optimum, choose the action with minimum audited TSFM MACs.
            for tolerance in args.score_tolerances:
                if not eligible:
                    tolerance_values = native.copy()
                    tolerance_w = native_context.copy()
                    tolerance_native = np.ones(native.shape, dtype=bool)
                    tolerance_counts = native_counts.copy()
                else:
                    native_score = regularized[:, -1]
                    if not native_supported:
                        native_score = np.full(native_score.shape, np.inf)
                    candidate_scores = np.column_stack([regularized, native_score])
                    candidate_values = np.column_stack([errors, native])
                    candidate_counts = np.column_stack([counts, native_counts])
                    finite_scores = np.where(
                        np.isfinite(candidate_scores), candidate_scores, np.nan)
                    score_scale = np.nanmax(finite_scores, axis=1) - np.nanmin(
                        finite_scores, axis=1)
                    score_scale = np.where(score_scale > 0, score_scale, 1.0)
                    feasible = np.isfinite(candidate_values) & np.isfinite(
                        candidate_scores)
                    best_score = np.min(
                        np.where(feasible, candidate_scores, np.inf), axis=1)
                    acceptable = feasible & (
                        candidate_scores <= best_score[:, None]
                        + float(tolerance) * score_scale[:, None])

                    grid_cost = np.asarray([
                        theoretical_flops(
                            model_id, int(window), horizon, DEFAULT_PATCH_SIZES)
                        for window in windows
                    ], dtype=np.float64)
                    native_cost = np.asarray([
                        theoretical_flops(
                            model_id, int(context), horizon, DEFAULT_PATCH_SIZES)
                        for context in native_context
                    ], dtype=np.float64)
                    costs = np.column_stack([
                        np.broadcast_to(grid_cost, errors.shape), native_cost])
                    choice = np.argmin(
                        np.where(acceptable, costs, np.inf), axis=1)
                    rows = np.arange(native.size)
                    has_choice = acceptable.any(axis=1)
                    choice = np.where(has_choice, choice, windows.size)
                    tolerance_native = choice == windows.size
                    tolerance_values = candidate_values[rows, choice]
                    tolerance_counts = candidate_counts[rows, choice]
                    tolerance_w = np.where(
                        tolerance_native, native_context,
                        windows[np.minimum(choice, windows.size - 1)])
                accumulate(
                    cell.model, cell.dataset, cell.term, alpha,
                    f"score_tolerance_{float(tolerance):g}",
                    tolerance_values, tolerance_w, tolerance_native,
                    tolerance_counts, native_context, naive_mase, model_id,
                    horizon)

    rows = []
    for (model, alpha, policy), agg in sorted(aggregates.items()):
        rows.append({
            "model": model,
            "alpha": alpha,
            "fallback_policy": policy,
            "normalized_mase": _geomean(agg["ratios"]),
            "flops_saved_pct": 100.0 * (
                1.0 - agg["selected_flops"] / agg["full_flops"]),
            "native_selected": agg["native_selected"],
            "n_instances": agg["instances"],
            "n_cells": agg["cells"],
        })
    result = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    cell_output = (
        Path(args.cell_output) if args.cell_output else
        output.with_name(f"{output.stem}_cells.csv"))
    pd.DataFrame(cell_rows).to_csv(cell_output, index=False)
    print(result.to_string(index=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cell-output", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--weights", nargs="+", type=float,
        default=[0.0, 0.05, 0.1, 0.25, 1.0],
    )
    parser.add_argument(
        "--score-tolerances", nargs="+", type=float, default=[],
        help=("Fractions of each row's predicted score range admitted above "
              "the predicted optimum before choosing the cheapest action."),
    )
    parser.add_argument(
        "--mase-field", choices=["mase_gluonts_real", "mase_gluonts"],
        default="mase_gluonts_real",
    )
    args = parser.parse_args()
    if any(weight < 0 or weight > 1 for weight in args.weights):
        parser.error("all --weights must lie in [0, 1]")
    if any(tolerance < 0 for tolerance in args.score_tolerances):
        parser.error("all --score-tolerances must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
