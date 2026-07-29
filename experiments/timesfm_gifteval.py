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
TIMESFM_GIFTEVAL_RECIPE = "official_timesfm2p5_forecast_v1"
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
            if checkpoint:
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
) -> np.ndarray:
    """Return q0.1..q0.9 as ``(series, quantile, horizon)``.

    This is the model-facing part of the official TimesFM GiftEval predictor:
    variable-length series stay variable length, partial NaNs are retained,
    compile bounds are rounded to the model patch, and the public ``forecast``
    method is used instead of the lower-level ``compiled_decode`` shortcut.
    """
    from timesfm import configs

    contexts = list(contexts)
    if not contexts:
        return np.empty((0, len(QUANTILE_LEVELS), prediction_length), dtype=np.float32)
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    outputs = []
    for start in range(0, len(contexts), int(batch_size)):
        model_inputs = []
        max_context = 0
        for raw in contexts[start:start + int(batch_size)]:
            arr = model_context(raw)
            if max_context_knob is not None:
                arr = model_context(arr[-int(max_context_knob):])
            max_context = max(max_context, int(arr.shape[0]))
            model_inputs.append(arr)

        patch_size = int(tfm.model.p)
        compiled_context = ((max_context + patch_size - 1) // patch_size) * patch_size
        tfm.compile(
            forecast_config=configs.ForecastConfig(
                max_context=min(15360, compiled_context),
                max_horizon=1024,
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
            horizon=int(prediction_length), inputs=model_inputs)
        quantiles = np.asarray(
            full_predictions[:, :int(prediction_length), 1:], dtype=np.float32)
        if quantiles.shape[2] != len(QUANTILE_LEVELS):
            raise ValueError(
                "TimesFM forecast output must contain mean + q0.1..q0.9; "
                f"got shape {tuple(np.asarray(full_predictions).shape)}"
            )
        outputs.append(quantiles.transpose((0, 2, 1)))

    return np.concatenate(outputs, axis=0)
