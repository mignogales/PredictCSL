#!/usr/bin/env python3
"""Estimate GiftEval policy speedups from isolated-forward latency curves.

This is an interim estimate, not an end-to-end benchmark.  Predictor window
histograms are joined to batch-32 CUDA forward profiles.  Latency is linearly
interpolated between measured context anchors and clamped below the shortest
anchor.  The native baseline uses the per-cell ``full_native`` context reported
by the policy evaluator; therefore it represents a native-cap approximation,
not genuine variable-history batching.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METHODS = ("balanced", "efficiency", "max_efficiency")


def latency_ms(curves: pd.DataFrame, model: str, horizon: int,
               windows: np.ndarray) -> np.ndarray:
    model_curves = curves[curves["model"].astype(str) == model]
    if model_curves.empty:
        raise ValueError(f"Missing forward curve for {model}")
    requested_windows = np.asarray(windows, dtype=float)
    profile_horizons = sorted(model_curves["horizon"].astype(int).unique())
    by_horizon = []
    for profile_horizon in profile_horizons:
        curve = model_curves[
            model_curves["horizon"].astype(int) == profile_horizon
        ].sort_values("window_size")
        x = curve["window_size"].to_numpy(dtype=float)
        y = curve["isolated_core_ms"].to_numpy(dtype=float)
        by_horizon.append(np.interp(
            requested_windows, x, y, left=float(y[0]), right=float(y[-1])))
    values = np.stack(by_horizon, axis=0)
    return np.asarray([
        np.interp(
            float(horizon), profile_horizons, values[:, column],
            left=float(values[0, column]), right=float(values[-1, column]),
        )
        for column in range(values.shape[1])
    ])


def estimate_model_method(curves: pd.DataFrame, histogram: pd.DataFrame,
                          model: str, method: str) -> dict[str, object]:
    policy = histogram[histogram["method"].astype(str) == method].copy()
    native = histogram[histogram["method"].astype(str) == "full_native"].copy()
    keys = ["dataset", "term", "horizon"]
    policy_cells = policy.groupby(keys, as_index=False)["n_instances"].sum()
    paired = policy_cells.merge(
        native[keys + ["window_size", "n_instances"]], on=keys,
        how="inner", validate="one_to_one", suffixes=("_policy", "_native"),
    )
    if len(paired) != len(policy_cells) or len(paired) != len(native):
        raise ValueError(f"Unpaired policy/native cells for {model}/{method}")
    if not np.array_equal(
        paired["n_instances_policy"].to_numpy(),
        paired["n_instances_native"].to_numpy(),
    ):
        raise ValueError(f"Instance-count mismatch for {model}/{method}")

    batch_size_values = curves[
        curves["model"].astype(str) == model
    ]["effective_batch_size"].dropna().astype(int).unique()
    if len(batch_size_values) != 1:
        raise ValueError(
            f"Expected one profiled batch size for {model}, got "
            f"{batch_size_values.tolist()}")
    batch_size = int(batch_size_values[0])

    native_ideal_ms = native_grouped_ms = 0.0
    policy_ideal_ms = policy_grouped_ms = 0.0
    below_anchor_instances = 0
    total_instances = int(policy["n_instances"].sum())
    for horizon, group in native.groupby("horizon"):
        lat = latency_ms(
            curves, model, int(horizon), group["window_size"].to_numpy())
        counts = group["n_instances"].to_numpy(dtype=float)
        native_ideal_ms += float(np.sum(counts * lat / batch_size))
        native_grouped_ms += float(np.sum(np.ceil(counts / batch_size) * lat))

    for horizon, group in policy.groupby("horizon"):
        lat = latency_ms(
            curves, model, int(horizon), group["window_size"].to_numpy())
        counts = group["n_instances"].to_numpy(dtype=float)
        policy_ideal_ms += float(np.sum(counts * lat / batch_size))
        policy_grouped_ms += float(np.sum(np.ceil(counts / batch_size) * lat))
        minimum = float(curves[
            curves["model"].astype(str) == model
        ]["window_size"].min())
        below_anchor_instances += int(group.loc[
            group["window_size"] < minimum, "n_instances"].sum())

    return {
        "model": model,
        "method": method,
        "n_instances": total_instances,
        "profile_batch_size": batch_size,
        "native_cap_ideal_s": native_ideal_ms / 1000.0,
        "policy_ideal_s": policy_ideal_ms / 1000.0,
        "ideal_speedup_x": native_ideal_ms / max(policy_ideal_ms, 1e-12),
        "ideal_time_saved_pct": 100.0 * (
            1.0 - policy_ideal_ms / max(native_ideal_ms, 1e-12)),
        "native_cap_grouped_s": native_grouped_ms / 1000.0,
        "policy_grouped_s": policy_grouped_ms / 1000.0,
        "grouped_speedup_x": native_grouped_ms / max(policy_grouped_ms, 1e-12),
        "grouped_time_saved_pct": 100.0 * (
            1.0 - policy_grouped_ms / max(native_grouped_ms, 1e-12)),
        "below_shortest_anchor_pct": (
            100.0 * below_anchor_instances / max(total_instances, 1)),
        "selector_overhead_included": False,
        "baseline_kind": "per-cell native-context cap approximation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-profile", required=True, type=Path)
    parser.add_argument("--histogram-root", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    curves = pd.read_csv(args.forward_profile)
    rows = []
    for histogram_path in sorted(args.histogram_root.glob(
            "*/full_score/selected_window_histograms.csv")):
        histogram = pd.read_csv(histogram_path)
        model = str(histogram["model"].iloc[0])
        if model not in set(curves["model"].astype(str)):
            continue
        for method in args.methods:
            rows.append(estimate_model_method(
                curves, histogram, model, method))

    frame = pd.DataFrame(rows)
    totals = []
    for method, group in frame.groupby("method", sort=False):
        native_ideal = float(group["native_cap_ideal_s"].sum())
        policy_ideal = float(group["policy_ideal_s"].sum())
        native_grouped = float(group["native_cap_grouped_s"].sum())
        policy_grouped = float(group["policy_grouped_s"].sum())
        totals.append({
            "model": "TOTAL_10_MODELS",
            "method": method,
            "n_instances": int(group["n_instances"].sum()),
            "profile_batch_size": 32,
            "native_cap_ideal_s": native_ideal,
            "policy_ideal_s": policy_ideal,
            "ideal_speedup_x": native_ideal / max(policy_ideal, 1e-12),
            "ideal_time_saved_pct": 100 * (1 - policy_ideal / native_ideal),
            "native_cap_grouped_s": native_grouped,
            "policy_grouped_s": policy_grouped,
            "grouped_speedup_x": native_grouped / max(policy_grouped, 1e-12),
            "grouped_time_saved_pct": 100 * (
                1 - policy_grouped / native_grouped),
            "below_shortest_anchor_pct": float(np.average(
                group["below_shortest_anchor_pct"],
                weights=group["n_instances"])),
            "selector_overhead_included": False,
            "baseline_kind": "per-cell native-context cap approximation",
        })
    frame = pd.concat([frame, pd.DataFrame(totals)], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    columns = [
        "model", "method", "ideal_speedup_x", "grouped_speedup_x",
        "ideal_time_saved_pct", "grouped_time_saved_pct",
        "below_shortest_anchor_pct",
    ]
    print(frame[columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
