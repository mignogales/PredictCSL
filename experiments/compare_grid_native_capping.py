"""Compare grid capping with native-context capping for a shared policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.compare_window_strategies_gifteval import (
    DEFAULT_PATCH_SIZES,
    theoretical_flops,
)


def _geomean(values: list[float]) -> float:
    values_array = np.asarray(values, dtype=np.float64)
    return float(np.exp(np.mean(np.log(values_array))))


def _total_flops(model: str, contexts: np.ndarray, horizon: int) -> float:
    contexts = np.asarray(contexts, dtype=np.int64)
    lookup = {
        int(context): theoretical_flops(
            model, int(context), int(horizon), DEFAULT_PATCH_SIZES)
        for context in np.unique(contexts[contexts > 0])
    }
    return float(sum(lookup[int(context)] for context in contexts))


def run(args: argparse.Namespace) -> dict:
    instance_dir = Path(args.instance_dir)
    comparison = pd.read_csv(args.comparison_csv)
    comparison["term"] = comparison["term"].astype(str)

    grid_ratios: list[float] = []
    native_ratios: list[float] = []
    total_grid_flops = 0.0
    total_native_cap_flops = 0.0
    total_full_flops = 0.0
    changed = 0
    changed_to_native = 0
    instances = 0
    cells_changed = 0
    cell_rows = []

    for row in comparison.itertuples(index=False):
        path = instance_dir / "cells" / (
            f"{args.model_short}__{row.dataset_display}__t{row.term}.npz")
        with np.load(path) as audit:
            windows = np.asarray(audit["window_grid"], dtype=np.int64)
            errors = np.asarray(audit["grid_mase"], dtype=np.float64)
            counts = np.asarray(audit["grid_valid_count"], dtype=np.float64)
            native_error = np.asarray(audit["native_mase"], dtype=np.float64)
            native_count = np.asarray(
                audit["native_valid_count"], dtype=np.float64)
            native_context = np.asarray(
                audit["native_effective_context"], dtype=np.int64)
            scores = np.asarray(
                audit[f"{args.variant}__scores"], dtype=np.float64)
            selection_eligible = bool(audit["selection_eligible"])
            grid_error = np.asarray(
                audit[f"{args.variant}_dataset__mase"], dtype=np.float64)
            grid_window = np.asarray(
                audit[f"{args.variant}_dataset__window"], dtype=np.int64)
            grid_count = np.asarray(
                audit[f"{args.variant}_dataset__valid_count"], dtype=np.float64)

        globally_valid = np.isfinite(errors).any(axis=0)
        target_index = int(np.argmin(np.mean(scores, axis=0)))
        if not globally_valid[target_index]:
            target_index = int(np.flatnonzero(globally_valid)[-1])
        target = int(windows[target_index])

        # Use the exact shared target where it is measurable. If the series is
        # too short (or that action is otherwise unavailable), use its actual
        # native context rather than backing off to an earlier grid point.
        if selection_eligible:
            target_available = np.isfinite(errors[:, target_index])
            native_selected = ~target_available
            native_cap_error = np.where(
                target_available, errors[:, target_index], native_error)
            native_cap_count = np.where(
                target_available, counts[:, target_index], native_count)
            native_cap_window = np.where(
                target_available, target, native_context)
        else:
            # Phase 6 deliberately makes cells with fewer than two grid actions
            # native-only; retain that safety rule in both capping policies.
            native_selected = np.ones(native_error.shape, dtype=bool)
            native_cap_error = native_error.copy()
            native_cap_count = native_count.copy()
            native_cap_window = native_context.copy()

        valid_grid = np.isfinite(grid_error) & (grid_count > 0)
        valid_native = np.isfinite(native_cap_error) & (native_cap_count > 0)
        if not np.array_equal(valid_grid, valid_native):
            raise ValueError(f"Coverage differs in {row.dataset_display}/t{row.term}")
        valid = valid_grid
        grid_cell = float(np.average(grid_error[valid], weights=grid_count[valid]))
        native_cell = float(np.average(
            native_cap_error[valid], weights=native_cap_count[valid]))
        naive = float(row.naive_mase)
        grid_ratios.append(grid_cell / naive)
        native_ratios.append(native_cell / naive)

        model_id = str(row.model)
        horizon = int(row.horizon)
        grid_context = np.where(grid_window > 0, grid_window, native_context)
        grid_flops = _total_flops(model_id, grid_context[valid], horizon)
        native_flops = _total_flops(
            model_id, native_cap_window[valid], horizon)
        full_flops = _total_flops(model_id, native_context[valid], horizon)
        total_grid_flops += grid_flops
        total_native_cap_flops += native_flops
        total_full_flops += full_flops

        cell_changed = valid & (grid_context != native_cap_window)
        n_changed = int(cell_changed.sum())
        changed += n_changed
        changed_to_native += int((cell_changed & native_selected).sum())
        instances += int(valid.sum())
        cells_changed += int(n_changed > 0)
        cell_rows.append({
            "dataset_display": row.dataset_display,
            "term": row.term,
            "target_window": target,
            "n_instances": int(valid.sum()),
            "n_changed": n_changed,
            "grid_cap_mase": grid_cell,
            "native_cap_mase": native_cell,
            "delta_mase": native_cell - grid_cell,
        })

    grid_mase = _geomean(grid_ratios)
    native_mase = _geomean(native_ratios)
    # Phase 6 reconstructs cell MASE from aligned per-instance vectors. A few
    # legacy cells differ minutely from their authoritative Stage-4 scalar.
    # Apply the paired capping ratio to the official Stage-4 aggregate so the
    # headline keeps leaderboard parity while preserving the exact paired
    # effect measured above.
    official_grid_mase = _geomean(
        (comparison["pred_mase"] / comparison["naive_mase"]).tolist())
    official_native_mase = official_grid_mase * native_mase / grid_mase
    report = {
        "model": args.model_short,
        "variant": args.variant,
        "policy": "dataset-shared predictor target",
        "official_grid_cap_normalized_mase": official_grid_mase,
        "official_native_cap_normalized_mase": official_native_mase,
        "phase6_grid_cap_normalized_mase": grid_mase,
        "phase6_native_cap_normalized_mase": native_mase,
        "native_cap_relative_mase_change_pct": 100.0 * (native_mase / grid_mase - 1.0),
        "grid_cap_flops_saved_pct": 100.0 * (1.0 - total_grid_flops / total_full_flops),
        "native_cap_flops_saved_pct": 100.0 * (
            1.0 - total_native_cap_flops / total_full_flops),
        "n_instances": instances,
        "n_changed": changed,
        "changed_rate": changed / instances,
        "n_changed_to_native": changed_to_native,
        "n_cells_changed": cells_changed,
        "n_cells": len(cell_rows),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    pd.DataFrame(cell_rows).to_csv(output_dir / "cell_results.csv", index=False)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-dir", required=True)
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-short", required=True)
    parser.add_argument("--variant", default="mamba_curve")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
