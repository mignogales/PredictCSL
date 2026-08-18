#!/usr/bin/env python3
"""Benchmark frozen calibrated-risk predictors on identical feature batches."""

import argparse
import json
import statistics
import time
from pathlib import Path

import joblib
import numpy as np

from experiments import calibrated_context_risk as risk


def timed(callable_, repeats: int, warmups: int = 2) -> list[float]:
    for _ in range(warmups):
        callable_()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        callable_()
        values.append(time.perf_counter() - start)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-dir", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--series", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    model_dir = Path(args.synthetic_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    contexts_path, lengths_path, _ = risk._synthetic_paths(model_dir)
    contexts = np.load(contexts_path, mmap_mode="r")
    lengths = np.load(lengths_path, mmap_mode="r")
    count = min(args.series, len(contexts))
    max_context = int(meta["max_window"])
    prepared = np.asarray(contexts[:count, -max_context:], dtype=np.float32)
    valid_lengths = np.minimum(lengths[:count], max_context).astype(np.int64)
    windows = np.asarray(meta["window_grid"], dtype=np.int64)
    horizon = int(meta["horizon_grid"][0])

    single_feature_times = timed(
        lambda: risk.extract_series_features(
            prepared[:1], valid_lengths[:1], max_context),
        repeats=max(args.repeats, 20), warmups=3)
    base_features = risk.extract_series_features(
        prepared, valid_lengths, max_context)
    pair_features = risk.make_pair_features(
        np.repeat(base_features, len(windows), axis=0),
        np.repeat(valid_lengths, len(windows)),
        np.tile(windows, count),
        np.full(count * len(windows), horizon),
        max_context,
    )
    single_pair_features = pair_features[:len(windows)]
    output = {
        "n_features": int(pair_features.shape[1]),
        "windows_per_series": int(len(windows)),
        "batch_rows": int(len(pair_features)),
        "feature_extraction_single_series_median_ms": float(
            1000.0 * statistics.median(single_feature_times)),
        "models": {},
    }
    for path_text in args.models:
        path = Path(path_text)
        bundle = joblib.load(path)
        one_times = timed(
            lambda: risk.predict_risk(
                bundle["regressor"], bundle["classifier"],
                single_pair_features),
            repeats=args.repeats)
        batch_times = timed(
            lambda: risk.predict_risk(
                bundle["regressor"], bundle["classifier"], pair_features),
            repeats=args.repeats)
        output["models"][path.parent.name] = {
            "artifact_bytes": int(path.stat().st_size),
            "single_series_median_ms": float(1000.0 * statistics.median(one_times)),
            "batch_median_ms": float(1000.0 * statistics.median(batch_times)),
            "batch_microseconds_per_series": float(
                1e6 * statistics.median(batch_times) / count),
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
