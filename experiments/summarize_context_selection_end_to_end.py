#!/usr/bin/env python3
"""Combine measured selector overhead with sampled TSFM throughput timings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def selector_us_per_series(
    report: dict, label: str, requested_batch: int,
) -> tuple[float, int]:
    batches = report["policies"][label]["batches"]
    available = np.asarray([int(value) for value in batches], dtype=int)
    chosen = int(available[np.argmin(np.abs(available - requested_batch))])
    return float(batches[str(chosen)]["selector_median_us_per_series"]), chosen


def policy_timing(
    histogram: pd.DataFrame, timing: pd.DataFrame, model: str, method: str,
) -> dict:
    selected = histogram[
        (histogram["model"].astype(str) == model)
        & (histogram["method"].astype(str) == method)
    ].copy()
    if selected.empty:
        raise ValueError(f"No histogram rows for {model}/{method}")
    timed = timing[timing["model_short"].astype(str) == model].copy()
    if method == "full_native":
        timed = timed[timed["timing_kind"].astype(str) == "full_native"]
        # Histogram window_size is only a legacy numeric description of the
        # baseline cap. The measurement itself lives in wfull_native and serves
        # each sampled series with its own genuine history length.
        merged = selected.merge(
            timed,
            left_on=["dataset", "term", "horizon"],
            right_on=["dataset_display", "term", "horizon"],
            how="left", validate="many_to_one", suffixes=("", "_timed"),
        )
    else:
        timed = timed[timed["timing_kind"].astype(str) == "numeric_window"]
        selected["window_size"] = pd.to_numeric(
            selected["window_size"], errors="raise").astype("Int64")
        timed["window_size"] = pd.to_numeric(
            timed["window_size"], errors="raise").astype("Int64")
        merged = selected.merge(
            timed,
            left_on=["dataset", "term", "horizon", "window_size"],
            right_on=["dataset_display", "term", "horizon", "window_size"],
            how="left", validate="many_to_one",
        )
    covered = merged["mean_s"].notna() & (merged["n_timed_series"] > 0)
    estimated = float((
        merged.loc[covered, "n_instances"]
        * merged.loc[covered, "mean_s"]
        / merged.loc[covered, "n_timed_series"]
    ).sum())
    total_assignments = int(merged["n_instances"].sum())
    covered_assignments = int(merged.loc[covered, "n_instances"].sum())
    return {
        "estimated_tsfm_s": estimated,
        "n_instances": total_assignments,
        "timing_coverage": float(
            covered_assignments / max(total_assignments, 1)),
        "missing_windows": merged.loc[
            ~covered, ["dataset", "term", "window_size"]
        ].drop_duplicates().to_dict(orient="records"),
        "peak_cuda_allocated_gb": float(
            merged.loc[covered, "cuda_peak_allocated_gb"].max())
            if covered.any() else float("nan"),
        "incremental_peak_cuda_allocated_gb": float(
            merged.loc[covered, "cuda_incremental_peak_allocated_gb"].max())
            if covered.any() else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timing-summary", required=True, type=Path)
    parser.add_argument("--selector-overhead", required=True, type=Path)
    parser.add_argument(
        "--policy", action="append", nargs=4, required=True,
        metavar=("NAME", "HISTOGRAM_CSV", "METHOD", "OVERHEAD_LABEL"),
        help="One selected policy to roll up.",
    )
    parser.add_argument("--native-histogram", required=True, type=Path)
    parser.add_argument("--selector-batch-size", type=int, default=256)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    timing = pd.read_csv(args.timing_summary)
    overhead = json.loads(args.selector_overhead.read_text())
    native_histogram = pd.read_csv(args.native_histogram)
    native = policy_timing(native_histogram, timing, args.model, "full_native")
    if native["timing_coverage"] < 1.0:
        raise RuntimeError(
            f"Incomplete full-native timing for {args.model}: "
            f"coverage={native['timing_coverage']:.3f}")
    native_total = native["estimated_tsfm_s"]

    policies = {}
    for name, histogram_path, method, overhead_label in args.policy:
        histogram = pd.read_csv(histogram_path)
        forecast = policy_timing(histogram, timing, args.model, method)
        if forecast["timing_coverage"] < 1.0:
            raise RuntimeError(
                f"Incomplete timing for {args.model}/{name}: "
                f"coverage={forecast['timing_coverage']:.3f}")
        overhead_us, actual_batch = selector_us_per_series(
            overhead, overhead_label, args.selector_batch_size)
        selector_s = overhead_us * forecast["n_instances"] / 1e6
        total_s = forecast["estimated_tsfm_s"] + selector_s
        policies[name] = {
            **forecast,
            "histogram": histogram_path,
            "method": method,
            "overhead_label": overhead_label,
            "selector_batch_size": actual_batch,
            "selector_us_per_series": overhead_us,
            "selector_total_s": selector_s,
            "estimated_end_to_end_s": total_s,
            "estimated_end_to_end_saved_pct": float(
                100.0 * (1.0 - total_s / max(native_total, 1e-12))),
        }
    report = {
        "model": args.model,
        "timing_summary": str(args.timing_summary),
        "methodology": (
            "measured selector stages plus deterministic context-stratified "
            "TSFM throughput samples (up to max_series per cell), scaled from "
            "measured seconds per series; full-native uses genuine variable "
            "history lengths"
        ),
        "native": native,
        "policies": policies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
