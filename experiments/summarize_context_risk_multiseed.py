#!/usr/bin/env python3
"""Aggregate calibrated-risk seeds with paired dataset-cluster uncertainty.

The bootstrap samples predictor seeds and GIFT-Eval dataset displays with
replacement.  Terms/frequencies belonging to one display stay together, so the
interval does not treat correlated cells or hundreds of thousands of forecast
rows as independent evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_seed_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SEED=PATH")
    seed, path = value.split("=", 1)
    return int(seed), Path(path)


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    ratio = frame["cell_mase_ratio"].to_numpy(dtype=float)
    n = frame["n_instances"].to_numpy(dtype=float)
    selected = frame["selected_theoretical_macs"].to_numpy(dtype=float)
    native = frame["native_theoretical_macs"].to_numpy(dtype=float)
    geomean = float(np.exp(np.mean(np.log(ratio))))
    return {
        "geomean_mase_ratio": geomean,
        "mase_change_pct": float(100.0 * (geomean - 1.0)),
        "harm5_rate": float(np.sum(frame["harm5_rate"].to_numpy(dtype=float) * n)
                            / np.maximum(n.sum(), 1.0)),
        "flops_saved_pct": float(100.0 * (1.0 - selected.sum()
                                           / max(native.sum(), 1e-12))),
        "coverage": float(np.sum(frame["coverage"].to_numpy(dtype=float) * n)
                          / np.maximum(n.sum(), 1.0)),
    }


def _resample_clusters(
    frame: pd.DataFrame, sampled: np.ndarray,
) -> pd.DataFrame:
    parts = []
    for draw, cluster in enumerate(sampled):
        part = frame[frame["dataset"] == cluster].copy()
        part["_bootstrap_cluster"] = draw
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def summarize(
    runs: list[tuple[int, Path]], output_dir: Path,
    repeats: int, seed: int,
) -> dict:
    frames: dict[int, pd.DataFrame] = {}
    for run_seed, path in runs:
        frame = pd.read_csv(path)
        required = {
            "dataset", "method", "cell_mase_ratio", "n_instances",
            "harm5_rate", "coverage", "selected_theoretical_macs",
            "native_theoretical_macs",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames[run_seed] = frame
    methods = sorted(set.intersection(
        *(set(frame["method"].astype(str)) for frame in frames.values())))
    clusters = sorted(set.intersection(
        *(set(frame["dataset"].astype(str)) for frame in frames.values())))
    if not methods or not clusters:
        raise RuntimeError("No common methods or dataset clusters across seeds")

    seed_rows = []
    for run_seed, frame in frames.items():
        for method in methods:
            row = metrics(frame[frame["method"] == method])
            seed_rows.append({"seed": run_seed, "method": method, **row})

    rng = np.random.default_rng(seed)
    seed_values = np.asarray(sorted(frames), dtype=int)
    draws: dict[str, dict[str, list[float]]] = {
        method: {name: [] for name in metrics(
            frames[int(seed_values[0])][
                frames[int(seed_values[0])]["method"] == method]).keys()}
        for method in methods
    }
    cluster_values = np.asarray(clusters, dtype=object)
    for _ in range(int(repeats)):
        sampled_clusters = rng.choice(
            cluster_values, size=len(cluster_values), replace=True)
        sampled_seeds = rng.choice(
            seed_values, size=len(seed_values), replace=True)
        for method in methods:
            per_seed = []
            for sampled_seed in sampled_seeds:
                source = frames[int(sampled_seed)]
                source = source[source["method"] == method]
                per_seed.append(metrics(
                    _resample_clusters(source, sampled_clusters)))
            for metric_name in draws[method]:
                draws[method][metric_name].append(float(np.mean(
                    [row[metric_name] for row in per_seed])))

    summary_rows = []
    seed_frame = pd.DataFrame(seed_rows)
    for method in methods:
        for metric_name, samples in draws[method].items():
            values = seed_frame.loc[
                seed_frame["method"] == method, metric_name].to_numpy(dtype=float)
            samples_array = np.asarray(samples, dtype=float)
            summary_rows.append({
                "method": method,
                "metric": metric_name,
                "estimate": float(values.mean()),
                "seed_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "ci95_low": float(np.quantile(samples_array, 0.025)),
                "ci95_high": float(np.quantile(samples_array, 0.975)),
                "n_seeds": int(len(values)),
                "n_dataset_clusters": int(len(clusters)),
                "bootstrap_repeats": int(repeats),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(seed_rows).to_csv(output_dir / "per_seed_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "cluster_bootstrap_summary.csv", index=False)
    report = {
        "seeds": sorted(frames),
        "methods": methods,
        "dataset_clusters": clusters,
        "bootstrap_repeats": int(repeats),
        "bootstrap_seed": int(seed),
        "unit": "GIFT-Eval dataset display; all terms/frequencies retained together",
        "summary": summary_rows,
    }
    (output_dir / "cluster_bootstrap_summary.json").write_text(
        json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=parse_seed_path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    args = parser.parse_args()
    summarize(args.run, args.output_dir, args.bootstrap_repeats, args.bootstrap_seed)


if __name__ == "__main__":
    main()
