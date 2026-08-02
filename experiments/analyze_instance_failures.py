"""Drill into harmful per-instance context choices.

Consumes ``evaluate_instance_windows`` outputs only.  It ranks failing
dataset/term/method cells, then expands their audit NPZs into instance rows so
we can distinguish broad model misspecification from a small harmful tail.

Example::

    python -m experiments.analyze_instance_failures \
      --instance-dir logs/experiments/master_recompute/instance_window_evaluation \
      --output-dir logs/experiments/master_recompute/instance_failure_analysis \
      --models Chronos2-Small \
      --datasets BizITObsL2C-5T JenaWeather-10T JenaWeather-H JenaWeather-D
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable

import numpy as np
import pandas as pd


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return (float(np.average(values[ok], weights=weights[ok]))
            if ok.any() else float("nan"))


def _audit_path(root: str, model: str, dataset: str, term: str) -> str:
    return os.path.join(root, "cells", f"{model}__{dataset}__t{term}.npz")


def _selected_rows(
    frame: pd.DataFrame, models: Iterable[str] | None,
    datasets: Iterable[str] | None, top_cells: int,
) -> pd.DataFrame:
    rows = frame[frame["method_kind"] == "predictor_instance"].copy()
    if models:
        rows = rows[rows["model"].isin(set(models))]
    if datasets:
        rows = rows[rows["dataset_display"].isin(set(datasets))]
    rows = rows.sort_values("delta_vs_full", ascending=False)
    return rows.head(top_cells) if not datasets else rows


def run(args: argparse.Namespace) -> None:
    source = pd.read_csv(os.path.join(args.instance_dir, "cell_results.csv"))
    cells = _selected_rows(
        source, args.models, args.datasets, args.top_cells)
    if cells.empty:
        raise SystemExit("No matching predictor-instance cells.")

    summaries = []
    details = []
    for cell in cells.itertuples(index=False):
        path = _audit_path(
            args.instance_dir, cell.model, cell.dataset_display, str(cell.term))
        if not os.path.isfile(path):
            continue
        with np.load(path) as data:
            selected = np.asarray(data[f"{cell.method}__mase"], dtype=float)
            selected_w = np.asarray(data[f"{cell.method}__window"], dtype=int)
            full = np.asarray(data["full_native__mase"], dtype=float)
            full_w = np.asarray(data["full_native__window"], dtype=int)
            oracle = np.asarray(data["oracle_instance__mase"], dtype=float)
            counts = np.asarray(data["native_valid_count"], dtype=float)
            context = np.asarray(data["native_effective_context"], dtype=int)
            grid = np.asarray(data["window_grid"], dtype=int)
            grid_error = np.asarray(data["grid_mase"], dtype=float)

            prefix = cell.method.removesuffix("_instance")
            score_key = f"{prefix}__scores"
            scores = (np.asarray(data[score_key], dtype=float)
                      if score_key in data.files else None)

        delta = selected - full
        relative = delta / np.maximum(np.abs(full), 1e-12)
        regret = (selected - oracle) / np.maximum(np.abs(oracle), 1e-12)
        valid = np.isfinite(selected) & np.isfinite(full) & (counts > 0)
        harmed = valid & (delta > 0)
        helped = valid & (delta < 0)
        native_selected = valid & (selected_w == full_w)

        margin = np.full(selected.shape, np.nan)
        full_score_gap = np.full(selected.shape, np.nan)
        if scores is not None and scores.shape == grid_error.shape:
            feasible = np.isfinite(grid_error) & np.isfinite(scores)
            candidate = np.where(feasible, scores, np.inf)
            ordered = np.sort(candidate, axis=1)
            enough = np.isfinite(ordered[:, 1])
            margin[enough] = ordered[enough, 1] - ordered[enough, 0]
            full_score_gap = scores[:, -1] - np.nanmin(candidate, axis=1)

        summaries.append({
            "model": cell.model,
            "dataset_display": cell.dataset_display,
            "term": cell.term,
            "method": cell.method,
            "n_instances": int(valid.sum()),
            "aggregate_mase": _weighted_mean(selected, counts),
            "full_mase": _weighted_mean(full, counts),
            "aggregate_rel_change_pct": 100 * (
                _weighted_mean(selected, counts)
                / _weighted_mean(full, counts) - 1),
            "harmed_instance_rate": float(harmed.sum() / max(valid.sum(), 1)),
            "helped_instance_rate": float(helped.sum() / max(valid.sum(), 1)),
            "native_selected_rate": float(
                native_selected.sum() / max(valid.sum(), 1)),
            "weighted_harm_pct": 100 * _weighted_mean(relative, counts),
            "median_instance_change_pct": float(
                100 * np.nanmedian(relative[valid])),
            "p90_instance_harm_pct": float(
                100 * np.nanquantile(relative[valid], 0.90)),
            "median_regret_pct": float(100 * np.nanmedian(regret[valid])),
            "median_context": float(np.median(context[valid])),
            "median_selected_window": float(np.median(selected_w[valid])),
            "median_choice_margin": float(np.nanmedian(margin[valid])),
            "median_full_score_gap": float(np.nanmedian(full_score_gap[valid])),
        })

        for i in np.flatnonzero(valid):
            details.append({
                "model": cell.model,
                "dataset_display": cell.dataset_display,
                "term": cell.term,
                "method": cell.method,
                "instance_index": int(i),
                "valid_count": float(counts[i]),
                "native_context": int(context[i]),
                "selected_window": int(selected_w[i]),
                "selected_mase": float(selected[i]),
                "full_mase": float(full[i]),
                "oracle_mase": float(oracle[i]),
                "change_pct": float(100 * relative[i]),
                "regret_pct": float(100 * regret[i]),
                "choice_margin": float(margin[i]),
                "full_score_gap": float(full_score_gap[i]),
            })

    os.makedirs(args.output_dir, exist_ok=True)
    summary = pd.DataFrame(summaries).sort_values(
        "aggregate_rel_change_pct", ascending=False)
    detail = pd.DataFrame(details).sort_values("change_pct", ascending=False)
    summary.to_csv(os.path.join(args.output_dir, "failure_summary.csv"), index=False)
    detail.to_csv(os.path.join(args.output_dir, "instance_details.csv"), index=False)
    with open(os.path.join(args.output_dir, "failure_summary.json"), "w") as handle:
        json.dump(summary.to_dict("records"), handle, indent=2, allow_nan=True)
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--top-cells", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
