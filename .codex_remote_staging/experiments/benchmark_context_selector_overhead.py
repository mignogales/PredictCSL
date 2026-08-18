#!/usr/bin/env python3
"""Benchmark end-to-end selector overhead for ExtraTrees and compact policies.

The timed selector path includes series-feature extraction, candidate geometry,
model prediction, risk-score construction, and shortest-safe selection.  TSFM
forward timing remains in ``benchmark_window_timing_gifteval`` and is combined
with these measurements by ``summarize_context_selection_end_to_end``.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np

from experiments import calibrated_context_risk as risk


def current_rss_mb() -> float:
    """Best-effort Linux resident-set measurement; NaN on other platforms."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def parse_policy(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=POLICY.joblib")
    label, path = value.split("=", 1)
    return label, Path(path)


def timing(callable_, warmups: int, repeats: int) -> dict[str, float | list[float]]:
    for _ in range(warmups):
        callable_()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        callable_()
        samples.append(time.perf_counter() - start)
    array = np.asarray(samples, dtype=np.float64)
    return {
        "mean_s": float(array.mean()),
        "std_s": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median_s": float(np.median(array)),
        "p95_s": float(np.quantile(array, 0.95)),
        "samples_s": samples,
    }


def benchmark_policy(
    label: str, path: Path, contexts: np.ndarray, lengths: np.ndarray,
    windows: np.ndarray, horizon: int, max_context: int,
    batch_sizes: list[int], warmups: int, repeats: int,
) -> dict:
    gc.collect()
    rss_before = current_rss_mb()
    bundle = joblib.load(path)
    rss_after = current_rss_mb()
    profile = bundle["profiles"]["balanced"]
    config = profile["config"]
    output = {
        "label": label,
        "path": str(path),
        "artifact_bytes": int(path.stat().st_size),
        "resident_rss_before_load_mb": rss_before,
        "resident_rss_after_load_mb": rss_after,
        "resident_rss_load_delta_mb": float(rss_after - rss_before),
        "profile": "balanced",
        "profile_config": config,
        "batches": {},
    }
    for requested_batch in batch_sizes:
        count = min(int(requested_batch), len(contexts))

        def prepare_only():
            return (
                np.array(
                    contexts[:count, -max_context:], dtype=np.float32, copy=True),
                np.minimum(lengths[:count], max_context).astype(np.int64),
            )

        prepared, valid = prepare_only()

        def features_only():
            return risk.extract_series_features(
                prepared, valid, max_context,
                bundle["feature_spec"]["scales"],
                bundle["feature_spec"]["lags"],
            )

        base_features = features_only()

        def pairs_only():
            return risk.make_pair_features(
                np.repeat(base_features, len(windows), axis=0),
                np.repeat(valid, len(windows)),
                np.tile(windows, count),
                np.full(count * len(windows), horizon),
                max_context,
            )

        pair_features = pairs_only()

        def prediction_only():
            return risk.predict_risk(
                bundle["regressor"], bundle["classifier"], pair_features)

        predicted = prediction_only()

        def decision_only():
            mean, std, harm = (
                value.reshape(count, len(windows)) for value in predicted)
            score = (
                mean
                + float(config["uncertainty_weight"]) * std
                + float(config["harm_weight"]) * harm
            )
            return risk.select_shortest_safe(
                score, float(config["threshold"]), windows, valid)

        def complete_selector():
            local_prepared, local_valid = prepare_only()
            base_values = risk.extract_series_features(
                local_prepared, local_valid, max_context,
                bundle["feature_spec"]["scales"],
                bundle["feature_spec"]["lags"],
            )
            pairs = risk.make_pair_features(
                np.repeat(base_values, len(windows), axis=0),
                np.repeat(local_valid, len(windows)),
                np.tile(windows, count),
                np.full(count * len(windows), horizon),
                max_context,
            )
            mean, std, harm = risk.predict_risk(
                bundle["regressor"], bundle["classifier"], pairs)
            score = (
                mean.reshape(count, len(windows))
                + float(config["uncertainty_weight"])
                  * std.reshape(count, len(windows))
                + float(config["harm_weight"])
                  * harm.reshape(count, len(windows))
            )
            return risk.select_shortest_safe(
                score, float(config["threshold"]), windows, local_valid)

        stages = {
            "input_preparation": timing(prepare_only, warmups, repeats),
            "feature_extraction": timing(features_only, warmups, repeats),
            "pair_feature_construction": timing(pairs_only, warmups, repeats),
            "predictor_inference": timing(prediction_only, warmups, repeats),
            "risk_decision": timing(decision_only, warmups, repeats),
            "complete_selector": timing(complete_selector, warmups, repeats),
        }
        complete_median = float(stages["complete_selector"]["median_s"])
        output["batches"][str(count)] = {
            "n_series": count,
            "n_candidate_rows": int(count * len(windows)),
            "stages": stages,
            "selector_median_us_per_series": float(
                1e6 * complete_median / max(count, 1)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-dir", required=True, type=Path)
    parser.add_argument("--policy", action="append", required=True, type=parse_policy)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32, 256])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    meta = json.loads((args.synthetic_dir / "meta.json").read_text())
    context_path, length_path, _ = risk._synthetic_paths(args.synthetic_dir)
    contexts = np.load(context_path, mmap_mode="r")
    lengths = np.load(length_path, mmap_mode="r")
    windows = np.asarray(meta["window_grid"], dtype=np.int64)
    max_context = int(meta["max_window"])
    horizon = int(meta["horizon_grid"][0])
    report = {
        "model": meta["model_display"],
        "synthetic_dir": str(args.synthetic_dir),
        "windows": windows.tolist(),
        "horizon": horizon,
        "warmups": int(args.warmups),
        "repeats": int(args.repeats),
        "policies": {},
    }
    for label, path in args.policy:
        report["policies"][label] = benchmark_policy(
            label, path, contexts, lengths, windows, horizon, max_context,
            args.batch_sizes, args.warmups, args.repeats,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
