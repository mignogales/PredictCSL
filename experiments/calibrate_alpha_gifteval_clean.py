#!/usr/bin/env python3
"""Calibrate shrinkage alpha on GIFT-Eval's clean pre-test validation slice.

The Mamba context selector is frozen.  For every dataset cell, this script uses
the official training/validation split to forecast one horizon immediately
before the official test origins.  Alpha is selected without reading test
labels, then the selected value can be evaluated with the cached test sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from gift_eval.data import Dataset as GiftEvalDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import datasets_config, models_config
from experiments.compare_window_strategies_gifteval import (
    DEFAULT_PATCH_SIZES,
    theoretical_flops,
)
from experiments.test_window_ablation_gifteval_v5 import flowstate_scale_factor
from experiments.train_gifteval_clean_worth_gate import (
    _load_forecaster,
    _load_frozen_selector,
    _median_forecasts,
    _per_instance_mase,
    prepare_cell_oracle,
)


DEFAULT_ALPHAS = (0.0, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 1.0)


def _paths(root: Path, display: str, term: str) -> tuple[Path, Path]:
    safe = display.replace("/", "_")
    return (
        root / "prepared_cells" / f"{safe}__{term}.npz",
        root / "alpha_cells" / f"{safe}__{term}.npz",
    )


def _selected_contexts(
    curves: np.ndarray,
    windows: np.ndarray,
    native_context: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    consensus = np.mean(curves, axis=0, keepdims=True)
    selected = np.empty((len(alphas), len(curves)), dtype=np.int64)
    for index, alpha in enumerate(alphas):
        scores = float(alpha) * curves + (1.0 - float(alpha)) * consensus
        action = np.argmin(scores, axis=1)
        width = np.minimum(windows[action], native_context)
        # The final supervised curve output represents native/full.  The test
        # evaluator resolves its exact tie with the largest grid action in
        # favor of native, so reproduce that behavior here.
        width = np.where(action == len(windows) - 1, native_context, width)
        selected[index] = width
    return selected


def forecast_cell(
    spec,
    forecaster,
    model_id: str,
    family: str,
    root: Path,
    device: str,
    batch_size: int,
    alphas: np.ndarray,
    force: bool,
) -> dict:
    ge_name, term, display, _to_univariate = spec
    prepared_path, output_path = _paths(root, display, term)
    cell = f"{display}/{term}"
    if output_path.exists() and not force:
        with np.load(output_path) as data:
            return {"cell": cell, "status": "cached", "n": len(data["native_mase"])}
    if not prepared_path.exists():
        return {"cell": cell, "status": "missing_prepared", "n": 0}

    with np.load(prepared_path, allow_pickle=True) as data:
        split = data["split"].astype(str)
        keep = np.flatnonzero(split == "val")
        contexts = [np.asarray(data["contexts"][i], dtype=np.float32) for i in keep]
        labels = [np.asarray(data["labels"][i], dtype=np.float32) for i in keep]
        curves = data["predicted_curves"][keep].astype(np.float64)
        native_context = data["native_context"][keep].astype(np.int64)
        horizon = int(data["horizon"][keep][0]) if len(keep) else 0
        items = data["item"][keep].astype(str)
    if not contexts:
        return {"cell": cell, "status": "empty", "n": 0}

    selector_config = json.loads(
        (root / "selector_config.json").read_text()
    )
    windows = np.asarray(selector_config["window_grid"], dtype=np.int64)
    selected_context = _selected_contexts(
        curves, windows, native_context, alphas
    )

    job_rows: list[int] = []
    job_widths: list[int] = []
    lookup: dict[tuple[int, int], int] = {}
    for row in range(len(contexts)):
        for width in np.unique(np.r_[native_context[row], selected_context[:, row]]):
            lookup[(row, int(width))] = len(job_rows)
            job_rows.append(row)
            job_widths.append(int(width))

    forecast_contexts = [
        contexts[row][-width:] for row, width in zip(job_rows, job_widths)
    ]
    forecast_labels = [labels[row] for row in job_rows]
    dataset = GiftEvalDataset(
        name=ge_name, term=term, to_univariate=spec[3]
    )
    started = time.time()
    forecasts = _median_forecasts(
        forecaster,
        model_id,
        family,
        forecast_contexts,
        forecast_labels,
        horizon,
        batch_size,
        device,
        flowstate_scale_factor(dataset.freq, ge_name),
    )
    job_mase, job_count = _per_instance_mase(
        forecasts,
        forecast_labels,
        [contexts[row] for row in job_rows],
        dataset.freq,
    )

    native_mase = np.empty(len(contexts), dtype=np.float64)
    native_count = np.empty(len(contexts), dtype=np.int64)
    selected_mase = np.empty(selected_context.shape, dtype=np.float64)
    selected_count = np.empty(selected_context.shape, dtype=np.int64)
    for row in range(len(contexts)):
        job = lookup[(row, int(native_context[row]))]
        native_mase[row] = job_mase[job]
        native_count[row] = job_count[job]
        for alpha_index in range(len(alphas)):
            job = lookup[(row, int(selected_context[alpha_index, row]))]
            selected_mase[alpha_index, row] = job_mase[job]
            selected_count[alpha_index, row] = job_count[job]

    native_flops = np.asarray([
        theoretical_flops(model_id, int(width), horizon, DEFAULT_PATCH_SIZES)
        for width in native_context
    ], dtype=np.float64)
    selected_flops = np.asarray([
        [theoretical_flops(model_id, int(width), horizon, DEFAULT_PATCH_SIZES)
         for width in row]
        for row in selected_context
    ], dtype=np.float64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        alphas=alphas,
        selected_context=selected_context,
        selected_mase=selected_mase,
        selected_count=selected_count,
        selected_flops=selected_flops,
        native_context=native_context,
        native_mase=native_mase,
        native_count=native_count,
        native_flops=native_flops,
        item=items,
        cell=np.asarray([cell] * len(contexts)),
        horizon=np.asarray([horizon] * len(contexts)),
    )
    return {
        "cell": cell,
        "status": "built",
        "n": len(contexts),
        "unique_forecasts": len(job_rows),
        "seconds": time.time() - started,
    }


def aggregate(root: Path, tolerance: float) -> dict:
    paths = sorted((root / "alpha_cells").glob("*.npz"))
    if not paths:
        raise RuntimeError(f"No alpha cells below {root / 'alpha_cells'}")
    cell_rows = []
    alphas = None
    for path in paths:
        with np.load(path) as data:
            current_alphas = data["alphas"].astype(np.float64)
            if alphas is None:
                alphas = current_alphas
            elif not np.array_equal(alphas, current_alphas):
                raise ValueError(f"Alpha grid mismatch in {path}")
            native = data["native_mase"]
            native_count = data["native_count"]
            selected = data["selected_mase"]
            selected_count = data["selected_count"]
            valid = (
                np.isfinite(native) & (native > 0) & (native_count > 0)
                & np.all(np.isfinite(selected), axis=0)
                & np.all(selected_count > 0, axis=0)
            )
            if not valid.any():
                continue
            native_cell = float(np.average(native[valid], weights=native_count[valid]))
            full_flops = float(np.sum(data["native_flops"][valid]))
            cell = str(data["cell"][0])
            for index, alpha in enumerate(alphas):
                mase = float(np.average(
                    selected[index, valid], weights=selected_count[index, valid]
                ))
                chosen_flops = float(np.sum(data["selected_flops"][index, valid]))
                cell_rows.append({
                    "cell": cell,
                    "alpha": float(alpha),
                    "mase_ratio_vs_native": mase / native_cell,
                    "selected_flops": chosen_flops,
                    "native_flops": full_flops,
                    "n": int(valid.sum()),
                })

    summary = []
    for alpha in alphas:
        rows = [row for row in cell_rows if row["alpha"] == float(alpha)]
        ratio = math.exp(sum(math.log(row["mase_ratio_vs_native"]) for row in rows) / len(rows))
        selected_flops = sum(row["selected_flops"] for row in rows)
        native_flops = sum(row["native_flops"] for row in rows)
        summary.append({
            "alpha": float(alpha),
            "validation_mase_ratio_vs_native": ratio,
            "validation_flops_saved_pct": 100.0 * (1.0 - selected_flops / native_flops),
            "n_cells": len(rows),
            "n_instances": sum(row["n"] for row in rows),
        })
    best = min(summary, key=lambda row: row["validation_mase_ratio_vs_native"])
    feasible = [
        row for row in summary
        if row["validation_mase_ratio_vs_native"]
        <= best["validation_mase_ratio_vs_native"] * (1.0 + tolerance)
    ]
    selected = max(
        feasible,
        key=lambda row: (
            row["validation_flops_saved_pct"],
            -row["validation_mase_ratio_vs_native"],
        ),
    )
    for row in summary:
        row["accuracy_best"] = row is best
        row["selected_with_tolerance"] = row is selected

    summary_path = root / "validation_alpha_sweep.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (root / "validation_alpha_cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cell_rows[0]))
        writer.writeheader()
        writer.writerows(cell_rows)
    report = {
        "selection_data": "GIFT-Eval pre-test validation horizon only",
        "official_test_labels_used": False,
        "selection_rule": (
            "maximize audited FLOPs saving among alphas within "
            f"{100.0 * tolerance:g}% relative MASE of validation-best alpha"
        ),
        "accuracy_best": best,
        "selected": selected,
        "n_candidate_alphas": len(summary),
    }
    (root / "validation_alpha_selection.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-short", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--legacy-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--stage", choices=["prepare", "forecast", "aggregate", "all"],
        default="all",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-items-per-cell", type=int, default=64)
    parser.add_argument("--min-context", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=0.001)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.legacy_config).read_text())
    (root / "selector_config.json").write_text(json.dumps(config, indent=2) + "\n")
    alphas = np.asarray(args.alphas, dtype=np.float64)
    if np.any((alphas < 0) | (alphas > 1)):
        parser.error("alphas must lie in [0, 1]")
    if args.tolerance < 0:
        parser.error("tolerance must be non-negative")

    model_specs = {
        display: (model_id, family)
        for model_id, family, display in models_config.models_to_run()
    }
    model_id, family = model_specs[args.model_short]
    specs = datasets_config.datasets_to_run()
    if args.datasets:
        wanted = set(args.datasets)
        specs = [
            spec for spec in specs
            if spec[2] in wanted or f"{spec[2]}/{spec[1]}" in wanted
        ]

    if args.stage in {"prepare", "all"}:
        selector, bundle = _load_frozen_selector(
            args.checkpoint, args.legacy_config, args.device
        )
        manifest = []
        for number, spec in enumerate(specs, 1):
            try:
                result = prepare_cell_oracle(
                    spec, selector, bundle, family, root, args.device,
                    origins_per_series=0,
                    max_items=args.max_items_per_cell,
                    min_context=args.min_context,
                    seed=args.seed,
                    force=args.force,
                )
            except Exception as exc:
                result = {
                    "cell": f"{spec[2]}/{spec[1]}",
                    "status": "error",
                    "error": repr(exc),
                }
            manifest.append(result)
            (root / "prepare_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            print(f"[{number}/{len(specs)}] {json.dumps(result)}", flush=True)
        del selector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.stage in {"forecast", "all"}:
        forecaster = _load_forecaster(model_id, family, args.device)
        manifest = []
        for number, spec in enumerate(specs, 1):
            try:
                result = forecast_cell(
                    spec, forecaster, model_id, family, root, args.device,
                    args.batch_size, alphas, args.force,
                )
            except Exception as exc:
                result = {
                    "cell": f"{spec[2]}/{spec[1]}",
                    "status": "error",
                    "error": repr(exc),
                }
            manifest.append(result)
            (root / "forecast_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            print(f"[{number}/{len(specs)}] {json.dumps(result)}", flush=True)
        del forecaster
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.stage in {"aggregate", "all"}:
        print(json.dumps(aggregate(root, args.tolerance), indent=2), flush=True)


if __name__ == "__main__":
    main()
