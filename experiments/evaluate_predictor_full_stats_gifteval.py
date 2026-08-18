"""Evaluate full-history RevIN at the context predictor's chosen window.

The context-length predictor and its selected grid action are held fixed. For
each GIFT-Eval cell we forecast the same selected suffix in two conditions:

1. ``selected_stats``: the TSFM's ordinary RevIN over the selected suffix;
2. ``full_stats``: RevIN mean/scale from the largest context available to that
   predictor, while the TSFM still receives only the selected suffix.

Only Chronos-Bolt and PatchTST-FM are supported because their normalization and
inverse-normalization paths have been inspected in the installed runtimes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Iterable

import numpy as np
import torch

from experiments import models_config
from experiments.context_normalization_override import (
    normalization_reference_override,
)
from experiments.gifteval_reference import published_naive_record
from experiments.test_window_ablation_gifteval_v5 import (
    DATASETS,
    GiftEvalCache,
    GiftEvalDataset,
    _forecast_cell,
    _merge_grouped,
    load_handle,
    preserves_missing,
)


SUPPORTED = {"chronos_bolt", "patchtst_fm"}


def _model_spec(display: str):
    matches = [spec for spec in models_config.CATALOG if spec.display == display]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous model display {display!r}")
    spec = matches[0]
    if spec.family not in SUPPORTED:
        raise ValueError(
            f"{display} uses {spec.family!r}; verified families are "
            f"{sorted(SUPPORTED)}")
    return spec


def _comparison_path(root: str, model: str, dataset: str, term: str) -> str:
    return os.path.join(
        root, "models", model, "compare_real_vs_predicted",
        f"compare_{dataset}_t{term}_{model}.npz")


def _reference_rows(
    contexts: list[np.ndarray], indices: Iterable[int], width: int,
) -> list[np.ndarray]:
    return [np.asarray(contexts[int(i)], dtype=np.float32)[-int(width):]
            for i in indices]


def _forecast_condition(
    family: str,
    handle,
    model_id: str,
    groups,
    cache: GiftEvalCache,
    reference_width: int,
    horizon: int,
    device: str,
    batch_size: int,
    *,
    use_full_stats: bool,
):
    results = []
    for width, batches, _x, _y, indices in groups:
        if use_full_stats:
            offset = 0
            for batch in batches:
                size = int(batch["x"].shape[0])
                batch_indices = indices[offset:offset + size]
                references = _reference_rows(
                    cache.contexts_raw, batch_indices, reference_width)
                with normalization_reference_override(
                        family, handle, references):
                    forecast, targets = _forecast_cell(
                        family, handle, model_id, [batch], width, horizon,
                        device, batch_size,
                        flowstate_scale=cache.flowstate_scale)
                results.append((batch_indices, forecast, targets))
                offset += size
            if offset != len(indices):
                raise RuntimeError(
                    f"Forecast batches covered {offset}/{len(indices)} rows")
        else:
            forecast, targets = _forecast_cell(
                family, handle, model_id, batches, width, horizon,
                device, batch_size, flowstate_scale=cache.flowstate_scale)
            results.append((indices, forecast, targets))
    return _merge_grouped(results, cache.n_total, horizon, device)


def _per_sample(prediction: np.ndarray, target: np.ndarray,
                seasonal_error: np.ndarray) -> dict[str, np.ndarray]:
    valid = np.isfinite(prediction) & np.isfinite(target)
    count = valid.sum(axis=1).astype(np.float64)
    absolute = np.where(valid, np.abs(prediction - target), 0.0)
    mae = np.divide(absolute.sum(axis=1), count,
                    out=np.full(len(count), np.nan), where=count > 0)
    ok = (count > 0) & np.isfinite(seasonal_error) & (seasonal_error != 0)
    mase = np.divide(mae, seasonal_error,
                     out=np.full(len(count), np.nan), where=ok)
    return {"mae": mae, "mase": mase, "valid_count": count}


def _aggregate_mase(metrics: dict[str, np.ndarray]) -> float:
    ok = np.isfinite(metrics["mase"]) & (metrics["valid_count"] > 0)
    if not ok.any():
        return float("nan")
    return float(np.average(
        metrics["mase"][ok], weights=metrics["valid_count"][ok]))


def _geomean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array) & (array > 0)]
    return float(np.exp(np.mean(np.log(array)))) if array.size else float("nan")


def _bootstrap_delta(rows: list[dict], n_bootstrap: int, seed: int) -> list[float]:
    if not rows or n_bootstrap <= 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    standard = np.asarray(
        [row["selected_stats_normalized_mase"] for row in rows])
    proposed = np.asarray([row["full_stats_normalized_mase"] for row in rows])
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, len(rows), len(rows))
        deltas[b] = 100.0 * (
            _geomean(proposed[idx]) / _geomean(standard[idx]) - 1.0)
    return [float(v) for v in np.quantile(deltas, [0.025, 0.975])]


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict], n_bootstrap: int, seed: int) -> dict:
    standard = _geomean(
        row["selected_stats_normalized_mase"] for row in rows)
    proposed = _geomean(row["full_stats_normalized_mase"] for row in rows)
    relative = 100.0 * (proposed / standard - 1.0)
    return {
        "n_cells": len(rows),
        "selected_stats_geomean_normalized_mase": standard,
        "full_stats_geomean_normalized_mase": proposed,
        "full_stats_relative_change_pct": relative,
        "full_stats_relative_change_95pct_cell_bootstrap": _bootstrap_delta(
            rows, n_bootstrap, seed),
        "full_stats_better_cells": sum(
            row["full_stats_mase"] < row["selected_stats_mase"]
            for row in rows),
        "full_stats_worse_cells": sum(
            row["full_stats_mase"] > row["selected_stats_mase"]
            for row in rows),
        "full_stats_equal_cells": sum(
            row["full_stats_mase"] == row["selected_stats_mase"]
            for row in rows),
        "interpretation": "negative relative change means full-history stats help",
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--comparison-root",
        default="logs/experiments/master_recompute/window_ablation_gifteval/general_v4")
    parser.add_argument(
        "--output-root",
        default="logs/experiments/predictor_full_stats_gifteval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def main():
    args = parse_args()
    spec = _model_spec(args.model)
    selected_datasets = [
        row for row in DATASETS
        if args.datasets is None or row[2] in set(args.datasets)
    ]
    if args.max_cells is not None:
        selected_datasets = selected_datasets[:max(0, args.max_cells)]
    output_dir = os.path.join(args.output_root, args.model)
    cells_dir = os.path.join(output_dir, "cells")
    os.makedirs(cells_dir, exist_ok=True)

    device = args.device
    torch.set_grad_enabled(False)
    if str(device).startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    handle = load_handle(spec.family, spec.model_id, device)
    rows = []

    for ge_name, term, dataset_display, to_univariate in selected_datasets:
        comparison = _comparison_path(
            args.comparison_root, args.model, dataset_display, term)
        if not os.path.isfile(comparison):
            print(f"SKIP {dataset_display}/t{term}: missing {comparison}")
            continue
        cell_path = os.path.join(cells_dir, f"{dataset_display}_t{term}.npz")
        if os.path.isfile(cell_path):
            with np.load(cell_path) as cached:
                row = json.loads(str(cached["summary_json"].item()))
            rows.append(row)
            print(f"CACHED {dataset_display}/t{term}")
            continue

        with np.load(comparison) as data:
            grid = np.asarray(data["window_grid"], dtype=np.int64)
            predicted_mean = np.asarray(data["predicted_mean"], dtype=np.float64)
        selected_width = int(grid[int(np.argmin(predicted_mean))])
        reference_width = int(grid.max())

        dataset = GiftEvalDataset(
            name=ge_name, term=term, to_univariate=to_univariate)
        cache = GiftEvalCache(dataset, dataset_display)
        groups = cache.build_batches_padded(
            selected_width, args.batch_size, device,
            pin_memory=str(device).startswith("cuda"),
            window_grid=grid.tolist(),
            preserve_missing=preserves_missing(spec.family))

        ordinary, targets = _forecast_condition(
            spec.family, handle, spec.model_id, groups, cache, reference_width,
            cache.horizon, device, args.batch_size, use_full_stats=False)
        full_stats, targets_full = _forecast_condition(
            spec.family, handle, spec.model_id, groups, cache, reference_width,
            cache.horizon, device, args.batch_size, use_full_stats=True)
        np.testing.assert_allclose(
            targets.detach().cpu().numpy(),
            targets_full.detach().cpu().numpy(), equal_nan=True)

        target_np = targets.detach().cpu().numpy()
        ordinary_np = ordinary.median.detach().cpu().numpy()
        full_stats_np = full_stats.median.detach().cpu().numpy()
        ordinary_metrics = _per_sample(
            ordinary_np, target_np, cache.seasonal_errors_gluonts)
        full_metrics = _per_sample(
            full_stats_np, target_np, cache.seasonal_errors_gluonts)
        ordinary_mase = _aggregate_mase(ordinary_metrics)
        full_mase = _aggregate_mase(full_metrics)
        naive = float(published_naive_record(
            ge_name, term)["mase_gluonts_real"])
        row = {
            "model": args.model,
            "dataset": dataset_display,
            "term": str(term),
            "n_instances": cache.n_total,
            "selected_width": selected_width,
            "reference_width": reference_width,
            "selected_stats_mase": ordinary_mase,
            "full_stats_mase": full_mase,
            "selected_stats_normalized_mase": ordinary_mase / naive,
            "full_stats_normalized_mase": full_mase / naive,
            "full_stats_relative_change_pct": 100.0 * (
                full_mase / ordinary_mase - 1.0),
            "mean_absolute_prediction_change": float(
                np.nanmean(np.abs(full_stats_np - ordinary_np))),
        }
        np.savez_compressed(
            cell_path,
            selected_stats_prediction=ordinary_np,
            full_stats_prediction=full_stats_np,
            targets=target_np,
            selected_stats_mase=ordinary_metrics["mase"],
            full_stats_mase=full_metrics["mase"],
            valid_count=ordinary_metrics["valid_count"],
            summary_json=np.asarray(json.dumps(row)),
        )
        rows.append(row)
        print(
            f"DONE {dataset_display}/t{term}: L={selected_width} "
            f"Wstats={reference_width} MASE {ordinary_mase:.5f} -> "
            f"{full_mase:.5f} ({row['full_stats_relative_change_pct']:+.2f}%)")

    _write_csv(os.path.join(output_dir, "cells.csv"), rows)
    report = _summarize(rows, args.bootstrap, args.seed)
    report.update({
        "model": args.model,
        "comparison_root": args.comparison_root,
        "policy": "dataset-shared argmin of predicted_mean",
        "reference_definition": "largest grid context available to predictor",
    })
    with open(os.path.join(output_dir, "report.json"), "w") as handle_out:
        json.dump(report, handle_out, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
