#!/usr/bin/env python3
"""Profile isolated Chronos2 model-forward latency versus pipeline overhead.

The official ``Chronos2Pipeline.predict`` path is retained so preprocessing and
long-horizon unrolling exactly match the benchmark recipe. CUDA events attached
to ``pipeline.model.forward`` isolate the neural forward(s), while nested events
split mutually exclusive model stages. Wall time outside those events is
reported separately as pipeline/wrapper overhead.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from experiments.test_window_ablation_gifteval_v5 import (
    load_chronos2,
    predict_chronos2,
)


MODEL_IDS = {
    "Chronos2-Base": "autogluon/chronos-2",
    "Chronos2-Small": "autogluon/chronos-2-small",
    "Chronos2-Synth": "autogluon/chronos-2-synth",
}


class ForwardRecorder:
    def __init__(self) -> None:
        self.enabled = False
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = (
            defaultdict(list)
        )

    def reset(self) -> None:
        self.events.clear()

    def record(self, stage: str, function: Callable, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function(*args, **kwargs)
        end.record()
        self.events[stage].append((start, end))
        return result

    def milliseconds(self, stage: str) -> float:
        return float(sum(start.elapsed_time(end)
                         for start, end in self.events.get(stage, [])))


def _patch_method(
    owner, method_name: str, stage: str, recorder: ForwardRecorder,
) -> Callable:
    original = getattr(owner, method_name)

    def wrapped(*args, **kwargs):
        return recorder.record(stage, original, *args, **kwargs)

    setattr(owner, method_name, wrapped)
    return original


def _install_stage_hooks(pipeline, recorder: ForwardRecorder) -> list[tuple[object, str, Callable]]:
    model = pipeline.model
    targets = [
        (model, "forward", "model_forward"),
        (model, "_prepare_patched_context", "context_normalize_patch"),
        (model, "_prepare_patched_future", "future_patch"),
        (model.input_patch_embedding, "forward", "input_embedding"),
        (model.encoder, "forward", "encoder"),
        (model.output_patch_embedding, "forward", "output_projection"),
        (model.instance_norm, "inverse", "inverse_scale"),
    ]
    originals = []
    for owner, method_name, stage in targets:
        originals.append((
            owner, method_name,
            _patch_method(owner, method_name, stage, recorder),
        ))
    return originals


def _restore_hooks(originals: list[tuple[object, str, Callable]]) -> None:
    for owner, method_name, original in reversed(originals):
        setattr(owner, method_name, original)


def _one_pass(pipeline, x: torch.Tensor, y: torch.Tensor, horizon: int,
              batch_size: int, recorder: ForwardRecorder) -> dict[str, float]:
    batches = [{"x": x, "y": y}]
    recorder.reset()
    torch.cuda.synchronize()
    recorder.enabled = True
    start = time.perf_counter()
    try:
        forecast, targets = predict_chronos2(
            pipeline, batches, horizon, "cuda", batch_size)
        # Force the final wrapper outputs to be materialized before stopping.
        _ = forecast.median.sum() + targets.sum()
        torch.cuda.synchronize()
    finally:
        recorder.enabled = False
    wall_ms = 1000.0 * (time.perf_counter() - start)

    model_ms = recorder.milliseconds("model_forward")
    context_ms = recorder.milliseconds("context_normalize_patch")
    future_ms = recorder.milliseconds("future_patch")
    embedding_ms = recorder.milliseconds("input_embedding")
    encoder_ms = recorder.milliseconds("encoder")
    projection_ms = recorder.milliseconds("output_projection")
    inverse_ms = recorder.milliseconds("inverse_scale")
    accounted_model_ms = (
        context_ms + future_ms + embedding_ms + encoder_ms
        + projection_ms + inverse_ms
    )
    return {
        "wall_ms": wall_ms,
        "isolated_model_forward_ms": model_ms,
        "pipeline_wrapper_ms": max(0.0, wall_ms - model_ms),
        "context_normalize_patch_ms": context_ms,
        "future_patch_ms": future_ms,
        "input_embedding_ms": embedding_ms,
        "encoder_ms": encoder_ms,
        "output_projection_ms": projection_ms,
        "inverse_scale_ms": inverse_ms,
        "model_misc_ms": max(0.0, model_ms - accounted_model_ms),
        "n_model_forwards": float(len(recorder.events.get("model_forward", []))),
    }


def _median_record(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.median([record[key] for record in records]))
        for key in records[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODEL_IDS))
    parser.add_argument("--windows", nargs="+", type=int,
                        default=[128, 512, 2048, 8192])
    parser.add_argument("--horizons", nargs="+", type=int,
                        default=[48, 480, 720])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    unknown = sorted(set(args.models) - set(MODEL_IDS))
    if unknown:
        raise ValueError(f"Unknown Chronos2 model(s): {unknown}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for isolated forward profiling")

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(args.seed)
    rows = []

    for model_short in args.models:
        pipeline = load_chronos2(MODEL_IDS[model_short], "cuda")
        recorder = ForwardRecorder()
        originals = _install_stage_hooks(pipeline, recorder)
        try:
            for horizon in args.horizons:
                for window in args.windows:
                    values = rng.standard_normal(
                        (args.batch_size, window), dtype=np.float32)
                    x = torch.from_numpy(values).unsqueeze(-1).pin_memory()
                    y = torch.zeros(
                        args.batch_size, horizon, 1,
                        dtype=torch.float32,
                    ).pin_memory()
                    for _ in range(args.warmup):
                        _one_pass(
                            pipeline, x, y, horizon, args.batch_size, recorder)
                    measured = [
                        _one_pass(
                            pipeline, x, y, horizon, args.batch_size, recorder)
                        for _ in range(args.repeats)
                    ]
                    result = _median_record(measured)
                    result.update({
                        "model": model_short,
                        "window_size": window,
                        "horizon": horizon,
                        "batch_size": args.batch_size,
                        "warmup": args.warmup,
                        "repeats": args.repeats,
                        "model_forward_fraction": (
                            result["isolated_model_forward_ms"]
                            / max(result["wall_ms"], 1e-12)),
                        "encoder_fraction_of_model": (
                            result["encoder_ms"]
                            / max(result["isolated_model_forward_ms"], 1e-12)),
                    })
                    rows.append(result)
                    print(
                        f"{model_short} W={window} H={horizon}: "
                        f"wall={result['wall_ms']:.3f}ms "
                        f"model={result['isolated_model_forward_ms']:.3f}ms "
                        f"({100 * result['model_forward_fraction']:.1f}%) "
                        f"encoder={result['encoder_ms']:.3f}ms"
                    )
        finally:
            _restore_hooks(originals)
            del pipeline
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "chronos2_forward_profile.csv", index=False)
    summary = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(0),
        "models": args.models,
        "windows": args.windows,
        "horizons": args.horizons,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rows": rows,
    }
    (args.output_dir / "chronos2_forward_profile.json").write_text(
        json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
