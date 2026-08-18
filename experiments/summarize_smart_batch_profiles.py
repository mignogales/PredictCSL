#!/usr/bin/env python3
"""Roll up exact smart-batch profile buckets into policy speed estimates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("balanced", "efficiency", "max_efficiency")


def scheduled_time_ms(assignments: pd.DataFrame, profile: pd.DataFrame,
                      latency_column: str) -> float:
    counts = assignments.groupby(
        ["horizon", "window_size"], as_index=False)["n_instances"].sum()
    joined = counts.merge(
        profile[["horizon", "window_size", latency_column,
                 "effective_batch_size"]],
        on=["horizon", "window_size"], how="left", validate="one_to_one")
    if joined[latency_column].isna().any():
        missing = joined.loc[
            joined[latency_column].isna(), ["horizon", "window_size"]]
        raise ValueError(f"Missing smart-batch profiles:\n{missing}")
    batches = np.ceil(
        joined["n_instances"].to_numpy(float)
        / joined["effective_batch_size"].to_numpy(float)
    )
    return float(np.sum(batches * joined[latency_column].to_numpy(float)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", required=True, type=Path)
    parser.add_argument("--histogram-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for histogram_path in sorted(args.histogram_root.glob(
            "*/full_score/selected_window_histograms.csv")):
        histogram = pd.read_csv(histogram_path)
        model = str(histogram["model"].iloc[0])
        profile_path = args.profile_dir / f"{model}_forward_profile.csv"
        if not profile_path.exists():
            continue
        profile = pd.read_csv(profile_path)
        if model == "FlowState-R1" and "flowstate_scale" not in profile.columns:
            # FlowState changes its internal forecast grid using a per-dataset
            # cadence scale.  Consequently (horizon, window) is not a complete
            # execution-bucket key for this family.  The original microprofile
            # used the default scale and must not enter a cross-dataset rollup.
            print(
                "SKIP FlowState-R1: profile lacks dataset-specific "
                "flowstate_scale buckets"
            )
            continue
        requested_values = profile["requested_batch_size"].astype(int).unique()
        if len(requested_values) != 1:
            raise ValueError(
                f"Mixed requested batch sizes for {model}: {requested_values}")
        batch_size = int(requested_values[0])
        effective = profile["effective_batch_size"].astype(int)
        native = histogram[histogram["method"] == "full_native"]
        for latency_column, timing_scope in (
            ("wall_ms", "official_call_wall"),
            ("isolated_core_ms", "isolated_core"),
        ):
            native_ms = scheduled_time_ms(
                native, profile, latency_column)
            for method in METHODS:
                selected = histogram[histogram["method"] == method]
                policy_ms = scheduled_time_ms(
                    selected, profile, latency_column)
                rows.append({
                    "model": model,
                    "method": method,
                    "timing_scope": timing_scope,
                    "profile_batch_size": batch_size,
                    "min_effective_batch_size": int(effective.min()),
                    "max_effective_batch_size": int(effective.max()),
                    "n_instances": int(selected["n_instances"].sum()),
                    "native_cap_s": native_ms / 1000.0,
                    "policy_s": policy_ms / 1000.0,
                    "speedup_x": native_ms / max(policy_ms, 1e-12),
                    "time_saved_pct": 100.0 * (
                        1.0 - policy_ms / max(native_ms, 1e-12)),
                })

    frame = pd.DataFrame(rows)
    totals = []
    total_label = f"TOTAL_{frame['model'].nunique()}_MODELS"
    for (method, timing_scope), group in frame.groupby(
            ["method", "timing_scope"], sort=False):
        native = float(group["native_cap_s"].sum())
        policy = float(group["policy_s"].sum())
        totals.append({
            "model": total_label,
            "method": method,
            "timing_scope": timing_scope,
            "profile_batch_size": 32,
            "min_effective_batch_size": int(
                group["min_effective_batch_size"].min()),
            "max_effective_batch_size": int(
                group["max_effective_batch_size"].max()),
            "n_instances": int(group["n_instances"].sum()),
            "native_cap_s": native,
            "policy_s": policy,
            "speedup_x": native / max(policy, 1e-12),
            "time_saved_pct": 100.0 * (1.0 - policy / max(native, 1e-12)),
        })
    frame = pd.concat([frame, pd.DataFrame(totals)], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame[
        (frame["timing_scope"] == "official_call_wall")
    ][["model", "method", "speedup_x", "time_saved_pct"]].to_string(
        index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
