"""Compatibility helpers for TiRex2's FlashRNN runtime backend."""

import os

import numpy as np
import torch


TIREX_BACKEND_ENV = "PREDICTCSL_TIREX_BACKEND"
VALID_TIREX_BACKENDS = {
    "vanilla",
    "vanilla_fwbw",
    "cuda",
    "cuda_fused",
    "triton_fused",
}


def tirex_backend_for_device(device: str) -> str:
    """Return the configured FlashRNN backend for a runtime device."""
    # TiRex2's CUDA extension fails to build on the server, while its Triton
    # kernel requires more shared memory than the installed GPU exposes. The
    # vanilla implementation is ordinary torch code and still runs on CUDA.
    default = "vanilla"
    backend = os.environ.get(TIREX_BACKEND_ENV, default)
    if backend not in VALID_TIREX_BACKENDS:
        choices = ", ".join(sorted(VALID_TIREX_BACKENDS))
        raise ValueError(f"Invalid {TIREX_BACKEND_ENV}={backend!r}; choose one of: {choices}")
    return backend


def configure_tirex_backend(device: str) -> str:
    """Select a FlashRNN backend that does not require the CUDA C++ JIT."""
    backend = tirex_backend_for_device(device)

    # tirex-2 0.1.1 hardcodes device="cuda" to FlashRNN's C++/NVCC backend.
    from tirex2.model.component import flashrnn_slstm

    flashrnn_slstm._flashrnn_backend = lambda _device: backend
    return backend


def forecast_tirex_medians(
    model,
    sequences: torch.Tensor,
    prediction_length: int,
    median_quantile_idx: int = 4,
) -> np.ndarray:
    """Forecast an arbitrary horizon by rolling TiRex2's finite output window."""
    from tirex2 import TimeseriesType

    max_chunk = int(model.future_len)
    max_context = int(model.context_len)
    if max_chunk < 1 or prediction_length < 1:
        raise ValueError("TiRex forecast lengths must be positive")

    context = sequences
    chunks = []
    remaining = prediction_length
    while remaining:
        chunk_length = min(remaining, max_chunk)
        series = [
            TimeseriesType(
                target=row.unsqueeze(0),
                past_covariates=None,
                future_covariates=None,
            )
            for row in context
        ]
        forecasts = model.forecast(
            series,
            prediction_length=chunk_length,
            output_type="numpy",
        )
        medians = np.stack(
            [forecast[0, median_quantile_idx, :chunk_length] for forecast in forecasts],
            axis=0,
        )
        if medians.shape != (sequences.shape[0], chunk_length):
            raise RuntimeError(
                "TiRex2 returned an unexpected median shape "
                f"{medians.shape}; expected {(sequences.shape[0], chunk_length)}"
            )
        chunks.append(medians)
        remaining -= chunk_length
        if remaining:
            extension = torch.as_tensor(medians, dtype=context.dtype, device=context.device)
            context = torch.cat((context, extension), dim=1)[:, -max_context:]

    return np.concatenate(chunks, axis=1)
