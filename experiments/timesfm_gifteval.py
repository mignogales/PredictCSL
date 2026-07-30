"""Shared, official TimesFM 2.5 inference recipe for GiftEval.

Both the clean-room leaderboard sanity check and the master window-ablation
pipeline call this module.  Keeping the checkpoint loader and ``forecast()``
configuration in one place prevents the two numerators from drifting again.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import numpy as np


# Stored in every TimesFM cell cache.  Bump this whenever anything that can
# change a forecast (loader, missing-value handling, compile config, output head)
# changes; stage 3 will then recompute old cells instead of silently reusing them.
TIMESFM_GIFTEVAL_RECIPE = "official_timesfm2p5_patch_bucket_batching_v4"
QUANTILE_LEVELS = tuple(i / 10.0 for i in range(1, 10))

_MODEL_CACHE: dict[str, object] = {}


def model_context(entry_target) -> np.ndarray:
    """Prepare one context exactly as the official GiftEval wrapper does."""
    arr = np.asarray(entry_target, dtype=np.float32)
    # TimesFM interpolates partial missing values itself.  Current builds need a
    # finite fallback only for empty/all-missing contexts.
    if arr.size == 0:
        return np.zeros(1, dtype=np.float32)
    if np.isnan(arr).all():
        return np.zeros_like(arr, dtype=np.float32)
    return arr


def load_model(model_name: str):
    """Load TimesFM through the official 2.5 PyTorch checkpoint path."""
    if model_name not in _MODEL_CACHE:
        from timesfm.timesfm_2p5 import timesfm_2p5_torch

        cls = timesfm_2p5_torch.TimesFM_2p5_200M_torch
        tfm = cls()
        try:
            tfm.load_checkpoint()
        except TypeError as exc:
            message = str(exc)
            if "path" not in message and "required positional argument" not in message:
                raise
            checkpoint = (
                os.environ.get("TIMESFM_2P5_CHECKPOINT")
                or os.environ.get("TIMESFM_CHECKPOINT")
            )
            if checkpoint and hasattr(cls, "from_pretrained"):
                # The PyTorch 2.5 implementation inherits an abstract
                # ``load_checkpoint(path)`` from the base wrapper.  Its actual
                # safetensors loader is wired through the HF mixin instead.
                try:
                    tfm = cls.from_pretrained(checkpoint)
                except TypeError as hub_exc:
                    # timesfm 2.0.0 + huggingface-hub 0.36 passes the removed
                    # ``proxies`` kwarg through the mixin.  Reproduce TimesFM's
                    # local _from_pretrained implementation directly.
                    if "proxies" not in str(hub_exc) or not os.path.isdir(checkpoint):
                        raise
                    weights_name = getattr(cls, "WEIGHTS_FILENAME", "model.safetensors")
                    weights_path = os.path.join(checkpoint, weights_name)
                    tfm.model.load_checkpoint(
                        weights_path,
                        torch_compile=getattr(tfm, "torch_compile", True),
                    )
            elif checkpoint:
                tfm.load_checkpoint(checkpoint)
            elif hasattr(cls, "from_pretrained"):
                tfm = cls.from_pretrained(model_name)
            else:
                raise RuntimeError(
                    "This TimesFM build requires load_checkpoint(path) and does "
                    "not expose from_pretrained(). Set TIMESFM_2P5_CHECKPOINT "
                    "to the local google/timesfm-2.5-200m-pytorch checkpoint."
                ) from exc
        _MODEL_CACHE[model_name] = tfm
    return _MODEL_CACHE[model_name]


def forecast_quantiles(
    tfm,
    contexts: Sequence[np.ndarray] | Iterable[np.ndarray],
    prediction_length: int,
    batch_size: int = 1024,
    max_context_knob: Optional[int] = None,
    max_horizon: Optional[int] = None,
    forecast_row_indices: Optional[Sequence[int]] = None,
    safe_variable_length_batching: bool = True,
) -> np.ndarray:
    """Return q0.1..q0.9 as ``(series, quantile, horizon)``.

    This is the model-facing part of the official TimesFM GiftEval predictor:
    variable-length series stay variable length, compile bounds are rounded to
    the model patch, and the public ``forecast`` method is used instead of the
    lower-level ``compiled_decode`` shortcut. Before batching, contexts are
    pre-stripped of leading NaNs (the first step of TimesFM's preprocessing)
    and grouped by their patch-rounded effective length. Inputs in one model
    batch therefore need less than one patch of left padding. This avoids fully
    masked attention rows, which produce all-NaN forecasts with torch 2.4 CUDA.
    """
    from timesfm import configs

    contexts = list(contexts)
    if not contexts:
        return np.empty((0, len(QUANTILE_LEVELS), prediction_length), dtype=np.float32)
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    prediction_length = int(prediction_length)
    if prediction_length < 1:
        raise ValueError(
            f"prediction_length must be positive, got {prediction_length}")
    if max_horizon is None:
        output_patch_size = int(tfm.model.o)
        max_horizon = (
            (prediction_length + output_patch_size - 1) // output_patch_size
        ) * output_patch_size
    else:
        max_horizon = int(max_horizon)
    if max_horizon < prediction_length:
        raise ValueError(
            "TimesFM max_horizon must cover prediction_length; "
            f"got max_horizon={max_horizon}, prediction_length={prediction_length}")
    if forecast_row_indices is None:
        forecast_row_indices = tuple(range(len(contexts)))
    elif len(forecast_row_indices) != len(contexts):
        raise ValueError(
            "forecast_row_indices must have one entry per context; "
            f"got {len(forecast_row_indices)} indices for {len(contexts)} contexts")

    outputs = np.empty(
        (len(contexts), len(QUANTILE_LEVELS), prediction_length),
        dtype=np.float32,
    )
    patch_size = int(tfm.model.p)
    context_limit = 15360
    prepared = []
    for call_row, raw in enumerate(contexts):
        arr = model_context(raw)
        if max_context_knob is not None:
            arr = model_context(arr[-int(max_context_knob):])
        raw_nan_count = int(np.isnan(arr).sum())
        leading_nans_stripped = 0
        if safe_variable_length_batching and arr.size and np.isnan(arr[0]):
            valid = np.flatnonzero(~np.isnan(arr))
            if valid.size:
                leading_nans_stripped = int(valid[0])
                arr = arr[leading_nans_stripped:]
        effective_length = min(context_limit, int(arr.shape[0]))
        compiled_context = min(
            context_limit,
            ((effective_length + patch_size - 1) // patch_size) * patch_size,
        )
        prepared.append({
            "call_row": int(call_row),
            "input": arr,
            "raw_nan_count": raw_nan_count,
            "leading_nans_stripped": leading_nans_stripped,
            "compiled_context": compiled_context,
        })

    if safe_variable_length_batching:
        patch_buckets: dict[int, list[dict[str, object]]] = {}
        for item in prepared:
            patch_buckets.setdefault(int(item["compiled_context"]), []).append(item)
        model_batches = [
            bucket[start:start + int(batch_size)]
            for bucket in patch_buckets.values()
            for start in range(0, len(bucket), int(batch_size))
        ]
    else:
        model_batches = [
            prepared[start:start + int(batch_size)]
            for start in range(0, len(prepared), int(batch_size))
        ]

    model_batch_index = 0
    for model_batch in model_batches:
        model_inputs = [item["input"] for item in model_batch]
        call_rows = np.asarray(
            [item["call_row"] for item in model_batch], dtype=np.int64)
        compiled_context = max(
            int(item["compiled_context"]) for item in model_batch)
        tfm.compile(
            forecast_config=configs.ForecastConfig(
                max_context=compiled_context,
                max_horizon=max_horizon,
                infer_is_positive=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
                force_flip_invariance=True,
                return_backcast=False,
                normalize_inputs=True,
                per_core_batch_size=max(1, min(128, len(model_inputs))),
            ),
        )
        _, full_predictions = tfm.forecast(
            horizon=prediction_length, inputs=model_inputs)
        full_predictions = np.asarray(full_predictions)
        requested_predictions = full_predictions[:, :prediction_length, :]
        bad = ~np.isfinite(requested_predictions)
        if np.any(bad):
            bad_local_rows = np.flatnonzero(np.any(bad, axis=(1, 2)))
            bad_call_rows = call_rows[bad_local_rows]
            bad_source_rows = [
                int(forecast_row_indices[int(i)]) for i in bad_call_rows]
            locations = np.argwhere(bad)
            preview = [tuple(map(int, loc)) for loc in locations[:20]]
            input_bad = [
                {
                    "batch_row": int(i),
                    "call_row": int(call_rows[int(i)]),
                    "source_row": int(forecast_row_indices[
                        int(call_rows[int(i)])]),
                    "length": int(model_inputs[int(i)].shape[0]),
                    "raw_nan_count": int(model_batch[int(i)][
                        "raw_nan_count"]),
                    "leading_nans_stripped": int(model_batch[int(i)][
                        "leading_nans_stripped"]),
                    "remaining_nan_count": int(np.isnan(
                        model_inputs[int(i)]).sum()),
                    "inf_count": int(np.isinf(model_inputs[int(i)]).sum()),
                }
                for i in bad_local_rows
            ]
            raise FloatingPointError(
                "TimesFM forecast contains non-finite values; evaluation was "
                "stopped without dropping or replacing forecasts. "
                f"batch={model_batch_index} "
                f"call_rows={bad_call_rows.tolist()} "
                f"batch_rows={bad_local_rows.tolist()} "
                f"source_rows={bad_source_rows} "
                f"nan_count={int(np.isnan(requested_predictions).sum())} "
                f"inf_count={int(np.isinf(requested_predictions).sum())} "
                f"locations(batch_row,horizon_step,channel)={preview} "
                f"compiled_max_context={compiled_context} "
                f"compiled_max_horizon={max_horizon} "
                f"input_rows={input_bad}"
            )
        quantiles = np.asarray(
            requested_predictions[:, :, 1:], dtype=np.float32)
        if quantiles.shape[2] != len(QUANTILE_LEVELS):
            raise ValueError(
                "TimesFM forecast output must contain mean + q0.1..q0.9; "
                f"got shape {tuple(np.asarray(full_predictions).shape)}"
            )
        outputs[call_rows] = quantiles.transpose((0, 2, 1))
        model_batch_index += 1

    return outputs
