#!/usr/bin/env python3
"""Measure CUDA time inside each TSFM's neural inference call.

The experiment deliberately keeps the public/official forecasting wrapper used
by the GIFT-Eval ablation, but places CUDA events around the lowest stable model
entry point reached by that wrapper.  This separates neural execution from CPU
preprocessing, Python dispatch, output conversion, and metric plumbing.

For ordinary PyTorch models the entry point is ``model.forward``.  TimesFM 2.5
is the one exception: its public wrapper uses a torch-compiled closure rather
than ``nn.Module.forward``.  We therefore time ``compiled_decode`` after one
untimed compilation for the exact (window, horizon, batch) shape.  That scope
includes its two flip-invariant decodes and the small device-side normalization
and quantile postprocessing around them, but excludes compilation and NumPy
input preparation.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from experiments import models_config
from experiments.test_window_ablation_gifteval_v5 import (
    _is_accelerator_oom,
    _moirai_max_context,
    build_forward,
    load_handle,
)


MODEL_SPECS = {
    display: (model_id, family)
    for model_id, family, display in models_config.models_to_run()
}


def _anchor_windows(family: str, horizon: int) -> list[int]:
    """Short, middle, and native-cap anchors for a scaling curve."""
    cap = models_config.context_limit(family)
    if family == "moirai":
        cap = min(cap, _moirai_max_context(horizon))
    return sorted({w for w in (128, 512, 2048, cap) if 0 < w <= cap})


class MethodCudaRecorder:
    """Record every invocation of one bound method using CUDA events."""

    def __init__(self) -> None:
        self.enabled = False
        self.events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def reset(self) -> None:
        self.events.clear()

    def wrap(self, owner: object, method_name: str) -> Callable:
        original = getattr(owner, method_name)

        @wraps(original)
        def wrapped(*args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            self.events.append((start, end))
            return result

        setattr(owner, method_name, wrapped)
        return original

    def milliseconds(self) -> float:
        return float(sum(start.elapsed_time(end) for start, end in self.events))


def _core_target(family: str, handle: object) -> tuple[object, str, str]:
    if family in {"chronos2", "chronos_bolt"}:
        return handle.model, "forward", "nn_module_forward"
    if family in {"moirai", "patchtst_fm", "flowstate"}:
        return handle, "forward", "nn_module_forward"
    if family == "tirex":
        # The public ForecastModel adapter performs input/output conversion,
        # while its nested TiRex2 module contains the actual neural forward.
        return handle.model, "forward", "nn_module_forward"
    # These public generation methods directly invoke their transformer stacks
    # and do not reliably route through the root module's forward method.
    if family == "toto":
        return handle, "forecast", "model_forecast_method"
    if family == "sundial":
        return handle, "generate", "model_generate_method"
    raise ValueError(f"No ordinary forward target for family {family!r}")


def _materialize(result: object) -> None:
    forecast, targets = result
    value = forecast.median.sum() + targets.sum()
    # Do not call item(): the following synchronize is the explicit boundary.
    del value


def _one_pass(
    forward: Callable[[], object], recorder: MethodCudaRecorder,
) -> dict[str, float]:
    recorder.reset()
    torch.cuda.synchronize()
    recorder.enabled = True
    started = time.perf_counter()
    try:
        result = forward()
        _materialize(result)
        torch.cuda.synchronize()
    finally:
        recorder.enabled = False
    wall_ms = 1000.0 * (time.perf_counter() - started)
    core_ms = recorder.milliseconds()
    if not recorder.events:
        raise RuntimeError(
            "The official forecast path never reached the configured core method")
    return {
        "wall_ms": wall_ms,
        "isolated_core_ms": core_ms,
        "outside_core_ms": max(0.0, wall_ms - core_ms),
        "n_core_calls": float(len(recorder.events)),
    }


def _median(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.median([record[key] for record in records]))
        for key in records[0]
    }


def _make_batch(batch_size: int, window: int, horizon: int, seed: int):
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(
        rng.standard_normal((batch_size, window, 1), dtype=np.float32)
    ).pin_memory()
    y = torch.zeros(batch_size, horizon, 1, dtype=torch.float32).pin_memory()
    return [{"x": x, "y": y}]


def _measure_cell(
    *, model_short: str, model_id: str, family: str, handle: object,
    window: int, horizon: int, requested_batch_size: int, warmup: int,
    repeats: int, seed: int,
) -> dict[str, object]:
    batch_size = requested_batch_size
    while True:
        batches = _make_batch(batch_size, window, horizon, seed)
        forward = teardown = None
        recorder = MethodCudaRecorder()
        owner = original = None
        timesfm_original_compile = None
        try:
            forward, teardown = build_forward(
                family, handle, model_id, batches, window, horizon, "cuda",
                batch_size,
            )
            if family == "timesfm":
                # Compile and execute once without measurement.  Repeated calls
                # in the official helper request the identical compile config;
                # suppress that dispatch so our events surround only the
                # already-compiled device path.
                forward()
                torch.cuda.synchronize()
                timesfm_original_compile = handle.compile
                handle.compile = lambda *args, **kwargs: None
                owner, method_name, scope = (
                    handle, "compiled_decode", "timesfm_compiled_decode")
            else:
                owner, method_name, scope = _core_target(family, handle)

            original = recorder.wrap(owner, method_name)
            for _ in range(warmup):
                _one_pass(forward, recorder)
            measured = [_one_pass(forward, recorder) for _ in range(repeats)]
            result: dict[str, object] = _median(measured)
            result.update({
                "model": model_short,
                "model_family": family,
                "window_size": window,
                "horizon": horizon,
                "requested_batch_size": requested_batch_size,
                "effective_batch_size": batch_size,
                "warmup": warmup,
                "repeats": repeats,
                "isolation_scope": scope,
                "core_fraction": float(result["isolated_core_ms"])
                / max(float(result["wall_ms"]), 1e-12),
                "status": "complete",
            })
            return result
        except Exception as exc:
            if not _is_accelerator_oom(exc) or batch_size <= 1:
                raise
            next_batch = max(1, batch_size // 2)
            print(
                f"{model_short} W={window} H={horizon}: OOM at B={batch_size}; "
                f"retrying B={next_batch}",
                flush=True,
            )
            batch_size = next_batch
        finally:
            if owner is not None and original is not None:
                setattr(owner, method_name, original)
            if timesfm_original_compile is not None:
                handle.compile = timesfm_original_compile
            if teardown is not None:
                teardown()
            del batches
            torch.cuda.empty_cache()


def _write_model_output(
    output_dir: Path, model_short: str, rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["horizon", "window_size"])
    frame.to_csv(output_dir / f"{model_short}_forward_profile.csv", index=False)
    payload = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(0),
        "model": model_short,
        "requested_batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rows": frame.to_dict(orient="records"),
    }
    (output_dir / f"{model_short}_forward_profile.json").write_text(
        json.dumps(payload, indent=2) + "\n")


def _combine(output_dir: Path) -> None:
    paths = sorted(output_dir.glob("*_forward_profile.csv"))
    # The earlier detailed Chronos2 run uses one shared filename; include it.
    chronos2 = output_dir / "chronos2_forward_profile.csv"
    if chronos2.exists() and chronos2 not in paths:
        paths.append(chronos2)
    frames = []
    for path in paths:
        if path.name == "all_models_forward_profile.csv":
            continue
        frame = pd.read_csv(path)
        if "isolated_model_forward_ms" in frame.columns:
            frame = frame.rename(columns={
                "isolated_model_forward_ms": "isolated_core_ms",
                "pipeline_wrapper_ms": "outside_core_ms",
                "n_model_forwards": "n_core_calls",
                "batch_size": "effective_batch_size",
                "model_forward_fraction": "core_fraction",
            })
            frame["requested_batch_size"] = frame["effective_batch_size"]
            frame["model_family"] = "chronos2"
            frame["isolation_scope"] = "nn_module_forward_detailed"
            frame["status"] = "complete"
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No forward profiles found in {output_dir}")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(
        ["model", "window_size", "horizon"], keep="last")
    combined.sort_values(["model", "horizon", "window_size"]).to_csv(
        output_dir / "all_models_forward_profile.csv", index=False)
    print(f"Combined {len(combined)} cells from {combined.model.nunique()} models")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS))
    parser.add_argument("--windows", nargs="+", type=int)
    parser.add_argument("--horizons", nargs="+", type=int,
                        default=[48, 480, 720])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bucket-histogram", type=Path,
        help=("Profile only the distinct (horizon, window_size) execution "
              "buckets requested by this policy histogram."),
    )
    parser.add_argument(
        "--bucket-methods", nargs="+",
        default=["full_native", "balanced", "efficiency", "max_efficiency"],
        help="Histogram methods whose execution buckets are profiled.",
    )
    parser.add_argument("--combine-only", action="store_true")
    args = parser.parse_args()

    if args.combine_only:
        _combine(args.output_dir)
        return
    if not args.models:
        parser.error("--models is required unless --combine-only is used")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for isolated forward profiling")

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    for model_short in args.models:
        model_id, family = MODEL_SPECS[model_short]
        requested_pairs = None
        if args.bucket_histogram is not None:
            histogram = pd.read_csv(args.bucket_histogram)
            histogram = histogram[
                (histogram["model"].astype(str) == model_short)
                & histogram["method"].astype(str).isin(args.bucket_methods)
            ]
            if histogram.empty:
                raise ValueError(
                    f"No requested buckets for {model_short} in "
                    f"{args.bucket_histogram}")
            requested_pairs = sorted({
                (int(row.horizon), int(row.window_size))
                for row in histogram.itertuples(index=False)
            })
        handle = load_handle(family, model_id, "cuda")
        rows: list[dict[str, object]] = []
        output_csv = args.output_dir / f"{model_short}_forward_profile.csv"
        if output_csv.exists():
            rows = pd.read_csv(output_csv).to_dict(orient="records")
        done = {
            (int(row["window_size"]), int(row["horizon"])) for row in rows
            if row.get("status") == "complete"
        }
        try:
            if requested_pairs is None:
                pairs = [
                    (horizon, window)
                    for horizon in args.horizons
                    for window in (
                        sorted(set(args.windows)) if args.windows
                        else _anchor_windows(family, horizon)
                    )
                ]
            else:
                pairs = requested_pairs
            for horizon, window in pairs:
                cap = models_config.context_limit(family)
                if family == "moirai":
                    cap = min(cap, _moirai_max_context(horizon))
                if window > cap or (window, horizon) in done:
                    continue
                row = _measure_cell(
                    model_short=model_short, model_id=model_id,
                    family=family, handle=handle, window=window,
                    horizon=horizon,
                    requested_batch_size=args.batch_size,
                    warmup=args.warmup, repeats=args.repeats,
                    seed=args.seed + window + horizon,
                )
                rows.append(row)
                _write_model_output(args.output_dir, model_short, rows, args)
                print(
                    f"{model_short} W={window} H={horizon} "
                    f"B={row['effective_batch_size']}: "
                    f"wall={row['wall_ms']:.3f}ms "
                    f"core={row['isolated_core_ms']:.3f}ms "
                    f"({100 * row['core_fraction']:.1f}%, "
                    f"calls={row['n_core_calls']:.0f})",
                    flush=True,
                )
        finally:
            del handle
            torch.cuda.empty_cache()

    _combine(args.output_dir)


if __name__ == "__main__":
    main()
