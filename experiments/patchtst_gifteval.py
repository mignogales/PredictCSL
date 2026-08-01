"""Official PatchTST-FM inference helpers for GIFT-Eval.

PatchTST-FM has had two public Granite APIs.  The leaderboard-producing branch
accepts CPU ``inputs`` tensors and owns their device placement; newer Granite
releases accept device-resident ``past_values``.  Moving the legacy inputs to
CUDA in the caller changes the padded-context execution path and measurably
changes forecasts, so the distinction below is deliberate.

Both paths preserve the model's original attention mask and use ``no_grad``, as
the official GIFT-Eval predictor does.  We do not modify fully masked padding
rows: runtimes that cannot evaluate the official mask finitely must fail loudly
instead of silently producing a different PatchTST variant.
"""

from __future__ import annotations

import inspect
from typing import Sequence


def _uses_legacy_inputs_api(model) -> bool:
    """Whether ``model.forward`` is the leaderboard-era ``inputs`` API."""
    params = inspect.signature(model.forward).parameters
    if "inputs" in params and "past_values" not in params:
        return True
    if "past_values" in params:
        return False
    raise RuntimeError(
        "Unsupported PatchTST-FM API: expected forward(..., inputs=...) or "
        "forward(..., past_values=...)."
    )


def _official_preprocess(contexts: Sequence, *, device: str, legacy: bool):
    """Port ``PatchTSTFMEvalPredictor.preprocess`` without changing placement."""
    import numpy as np
    import torch

    target = []
    for context in contexts:
        if isinstance(context, torch.Tensor):
            values = context.detach().cpu().numpy()
        else:
            values = np.asarray(context)
        if np.isnan(values).any():
            if np.isnan(values).all():
                values = np.zeros_like(values)
            else:
                values = np.nan_to_num(values, nan=np.nanmean(values))
        row = torch.from_numpy(np.asarray(values)).float()
        # The published legacy wrapper passes CPU tensors and lets the model
        # place them. The current public wrapper explicitly moves past_values.
        if not legacy:
            row = row.to(device)
        target.append(row)
    return target


def forecast_patchtst_quantiles_official(
    model, contexts: Sequence, horizon: int, device: str,
    quantile_levels: Sequence[float],
):
    """Run either public Granite API with its official input-placement recipe."""
    import torch

    legacy = _uses_legacy_inputs_api(model)
    target = _official_preprocess(contexts, device=device, legacy=legacy)
    with torch.no_grad():
        if legacy:
            output = model(
                inputs=target,
                prediction_length=horizon,
                quantile_levels=list(quantile_levels),
            )
            raw = output.quantile_predictions
        else:
            output = model(
                past_values=target,
                prediction_length=horizon,
                quantile_levels=list(quantile_levels),
            )
            raw = output.quantile_outputs

    if isinstance(raw, (list, tuple)):
        raw = torch.stack([torch.as_tensor(item) for item in raw], dim=0)
    else:
        raw = torch.as_tensor(raw)
    if raw.dim() == 4 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    if raw.dim() != 3:
        raise ValueError(
            f"Unexpected PatchTST quantile output shape {tuple(raw.shape)}")

    q_count = len(quantile_levels)
    if raw.shape[1] == q_count:                         # (B, Q, H)
        quantiles = raw[:, :, :horizon]
    elif raw.shape[2] == q_count:                       # (B, H, Q)
        quantiles = raw[:, :horizon, :].permute(0, 2, 1)
    else:
        raise ValueError(
            "PatchTST quantile output has no requested quantile axis: "
            f"{tuple(raw.shape)}")
    if quantiles.shape[2] != horizon:
        raise ValueError(
            f"PatchTST returned horizon={quantiles.shape[2]}, expected {horizon}")
    return require_finite_patchtst_forecast(
        quantiles.to(device=device, dtype=torch.float32))


def require_finite_patchtst_forecast(forecast):
    """Reject a poisoned official forecast before it can enter cell caches."""
    import torch

    bad = ~torch.isfinite(forecast)
    if bool(bad.any()):
        raise RuntimeError(
            "The official PatchTST-FM inference path returned non-finite "
            f"forecasts ({int(bad.sum().item())}/{forecast.numel()} values). "
            "Use the leaderboard-compatible PatchTST environment; do not "
            "change the padding mask to make this runtime appear finite."
        )
    return forecast
