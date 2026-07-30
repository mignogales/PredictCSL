#!/usr/bin/env python3
"""Reproduce the TimesFM 2.5 context-limit failure on a tiny GiftEval subset.

This intentionally uses the real GiftEval rolling windows and the shared
TimesFM inference wrapper.  It is also suitable for rerunning on the server:

  GIFT_EVAL=/path/to/GiftEval \
  TIMESFM_2P5_CHECKPOINT=/path/to/checkpoint \
  python -m experiments.diagnose_timesfm_boundary
"""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from experiments.timesfm_gifteval import forecast_quantiles, load_model


MODEL_ID = "google/timesfm-2.5-200m-pytorch"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Match the main GiftEval experiment entry points: their dataset location is
# normally supplied by the repository .env file rather than an exported shell
# variable.
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_gift_eval_root(explicit: str | None) -> tuple[Path | None, list[Path]]:
    """Find the existing GiftEval root used by the main experiment scripts."""
    raw_candidates = [
        explicit,
        os.environ.get("GIFT_EVAL"),
        PROJECT_ROOT / "GiftEval",
        PROJECT_ROOT / "data" / "GiftEval",
        PROJECT_ROOT.parent / "GiftEval",
        PROJECT_ROOT.parent / "data" / "GiftEval",
    ]
    candidates: list[Path] = []
    for raw in raw_candidates:
        if not raw:
            continue
        candidate = Path(raw).expanduser().resolve()
        if candidate in candidates:
            continue
        candidates.append(candidate)
        if (candidate / "electricity" / "15T").is_dir():
            return candidate, candidates
    return None, candidates


def _parse_indices(value: str) -> list[int]:
    indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not indices or min(indices) < 0:
        raise argparse.ArgumentTypeError(
            "indices must be a non-empty comma-separated list of nonnegative integers")
    return indices


def _parse_cases(value: str) -> list[tuple[int, int]]:
    cases = []
    for part in value.split(","):
        try:
            width, max_horizon = map(int, part.strip().split(":"))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "cases must use width:max_horizon, e.g. 15360:128") from exc
        cases.append((width, max_horizon))
    return cases


def _load_contexts(indices: list[int], term: str, max_width: int
                   ) -> tuple[list[np.ndarray], list[int], int]:
    from gift_eval.data import Dataset

    base = Dataset(name="electricity/15T", term=term, to_univariate=False)
    dataset = Dataset(
        name="electricity/15T", term=term,
        to_univariate=base.target_dim > 1,
    )
    wanted = set(indices)
    contexts: dict[int, np.ndarray] = {}
    full_lengths: dict[int, int] = {}
    for row, entry in enumerate(itertools.islice(
            dataset.test_data.input, max(indices) + 1)):
        if row in wanted:
            context = np.asarray(entry["target"], dtype=np.float32)
            full_lengths[row] = int(context.shape[0])
            contexts[row] = context[-max_width:].copy()
    missing = [row for row in indices if row not in contexts]
    if missing:
        raise IndexError(f"GiftEval input rows do not exist: {missing}")
    return ([contexts[row] for row in indices],
            [full_lengths[row] for row in indices],
            int(dataset.prediction_length))


def _run_case(tfm, contexts: list[np.ndarray], source_rows: list[int],
              width: int, max_horizon: int, prediction_length: int,
              batch_size: int):
    inputs = [context[-width:] for context in contexts]
    started = time.perf_counter()
    try:
        forecasts = forecast_quantiles(
            tfm,
            inputs,
            prediction_length=prediction_length,
            batch_size=batch_size,
            max_horizon=max_horizon,
            forecast_row_indices=source_rows,
        )
    except FloatingPointError as exc:
        return None, {
            "finite": False,
            "error": str(exc),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
    return forecasts, {
        "finite": bool(np.isfinite(forecasts).all()),
        "shape": list(forecasts.shape),
        "minimum": float(np.min(forecasts)),
        "maximum": float(np.max(forecasts)),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gift-eval",
        help=("GiftEval root containing electricity/15T; defaults to GIFT_EVAL "
              "from the repository .env, then common project data locations"))
    parser.add_argument(
        "--checkpoint", default=(os.environ.get("TIMESFM_2P5_CHECKPOINT")
                                  or os.environ.get("TIMESFM_CHECKPOINT")),
        help="Local TimesFM 2.5 checkpoint directory")
    parser.add_argument(
        "--indices", type=_parse_indices, default=_parse_indices("0,20"),
        help="GiftEval test-input row indices (default: 0,20)")
    parser.add_argument("--term", choices=("short", "medium", "long"),
                        default="short")
    parser.add_argument(
        "--all-rows", action="store_true",
        help="Run all 7,400 Electricity-15T short input rows")
    parser.add_argument(
        "--cases", type=_parse_cases,
        default=_parse_cases("15360:128,15360:1024,12288:128,12288:1024"),
        help="Comma-separated width:max_horizon cases")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    gift_eval_root, searched = _resolve_gift_eval_root(args.gift_eval)
    if gift_eval_root is None:
        parser.error(
            "could not find GiftEval electricity/15T data. Checked: "
            + ", ".join(map(str, searched)))
    os.environ["GIFT_EVAL"] = str(gift_eval_root)
    if args.checkpoint:
        os.environ["TIMESFM_2P5_CHECKPOINT"] = str(args.checkpoint)

    if args.all_rows:
        args.indices = list(range(7400))
    max_width = max(width for width, _ in args.cases)
    contexts, full_lengths, prediction_length = _load_contexts(
        args.indices, args.term, max_width)
    context_details = [
        {
            "source_row": row,
            "full_length": full_length,
            "nan_count_by_width": {
                str(width): int(np.isnan(context[-width:]).sum())
                for width, _ in args.cases
            },
        }
        for row, context, full_length in zip(args.indices, contexts, full_lengths)
    ]
    if len(context_details) <= 20:
        context_summary = context_details
    else:
        context_summary = {
            "count": len(context_details),
            "full_length_min": min(full_lengths),
            "full_length_max": max(full_lengths),
            "nan_by_width": {},
        }
        for width in sorted({width for width, _ in args.cases}):
            nan_rows = [
                item["source_row"] for item in context_details
                if item["nan_count_by_width"][str(width)]]
            context_summary["nan_by_width"][str(width)] = {
                "row_count": len(nan_rows),
                "row_preview": nan_rows[:100],
                "maximum_nan_count": max(
                    item["nan_count_by_width"][str(width)]
                    for item in context_details),
            }

    tfm = load_model(MODEL_ID)
    report = {
        "python": platform.python_version(),
        "timesfm": importlib.metadata.version("timesfm"),
        "torch": importlib.metadata.version("torch"),
        "torch_device": str(tfm.model.device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "term": args.term,
        "prediction_length": prediction_length,
        "model_context_limit": int(tfm.model.config.context_limit),
        "input_patch_size": int(tfm.model.p),
        "output_patch_size": int(tfm.model.o),
        "source_rows": (args.indices if len(args.indices) <= 100
                        else {"count": len(args.indices),
                              "preview": args.indices[:100]}),
        "contexts": context_summary,
        "cases": {},
    }
    arrays = {}
    for width, max_horizon in args.cases:
        key = f"w{width}_max_horizon{max_horizon}"
        display_rows = args.indices if len(args.indices) <= 20 else args.indices[:20]
        suffix = "" if len(args.indices) <= 20 else f" (of {len(args.indices)})"
        print(f"Running {key} on source rows {display_rows}{suffix}...", flush=True)
        arrays[key], report["cases"][key] = _run_case(
            tfm, contexts, args.indices, width, max_horizon,
            prediction_length, args.batch_size)
        print(json.dumps(report["cases"][key], indent=2), flush=True)

    report["comparisons"] = {}
    for width in sorted({width for width, _ in args.cases}):
        horizons = sorted({
            max_horizon for case_width, max_horizon in args.cases
            if case_width == width})
        if len(horizons) != 2:
            continue
        left_key = f"w{width}_max_horizon{horizons[0]}"
        right_key = f"w{width}_max_horizon{horizons[1]}"
        left, right = arrays[left_key], arrays[right_key]
        comparison_key = (
            f"w{width}_first{prediction_length}_max_horizon"
            f"{horizons[0]}_vs_{horizons[1]}")
        if left is None or right is None:
            report["comparisons"][comparison_key] = {
                "available": False,
                "reason": "one or both forecasts were non-finite",
            }
        else:
            difference = np.abs(left - right)
            report["comparisons"][comparison_key] = {
                "available": True,
                "allclose_rtol1e-5_atol1e-5": bool(np.allclose(
                    left, right, rtol=1e-5, atol=1e-5)),
                "max_absolute_difference": float(np.max(difference)),
                "mean_absolute_difference": float(np.mean(difference)),
            }

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
