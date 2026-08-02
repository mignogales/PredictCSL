"""Compatibility helpers for TiRex2's FlashRNN runtime backend."""

import os

import numpy as np
import torch


TIREX_BACKEND_ENV = "PREDICTCSL_TIREX_BACKEND"
TIREX_GIFTEVAL_MODEL_ID = "NX-AI/TiRex-2-gifteval-zs"
TIREX_MODEL_CONFIG = "model-config.yaml"
VALID_TIREX_BACKENDS = {
    "vanilla",
    "vanilla_fwbw",
    "cuda",
    "cuda_fused",
    "triton_fused",
}


class TirexCheckpointAccessError(RuntimeError):
    """The authenticated Hugging Face account cannot read a TiRex checkpoint."""


def _is_huggingface_access_denied(exc: BaseException) -> bool:
    """Recognize gated 401/403 errors even when Hub wraps them as cache misses."""
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    current: BaseException | None = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, GatedRepoError):
            return True
        if isinstance(current, HfHubHTTPError):
            response = getattr(current, "response", None)
            if getattr(response, "status_code", None) in {401, 403}:
                return True
            message = str(current).lower()
            if "gated" in message and ("403" in message or "permission" in message):
                return True
        current = current.__cause__ or current.__context__
    return False


def require_tirex_checkpoint_access(
    model_id: str = TIREX_GIFTEVAL_MODEL_ID,
) -> str:
    """Resolve TiRex's small config file, failing with actionable gate guidance.

    ``hf_hub_download`` reuses the normal Hugging Face token and cache. Checking
    the config is enough to verify gated-repository authorization without
    downloading the 330 MB model checkpoint during preflight.
    """
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(
            repo_id=model_id,
            filename=TIREX_MODEL_CONFIG,
            repo_type="model",
        )
    except Exception as exc:
        if not _is_huggingface_access_denied(exc):
            raise
        raise TirexCheckpointAccessError(
            f"TiREX checkpoint access denied for {model_id}.\n"
            f"1. Sign in at https://huggingface.co/{model_id} and accept the "
            "access conditions.\n"
            "2. In the token's fine-grained settings, enable access to public "
            "gated repositories (or explicitly grant this model).\n"
            "3. On the server, verify that the same account is active with "
            "`huggingface-cli whoami`; use `huggingface-cli login` if needed.\n"
            "   If HF_TOKEN is set in the shell or project .env, update it too; "
            "it overrides the saved login.\n"
            "4. Rerun this command after access is granted.\n"
            "Do not substitute NX-AI/TiRex-2 for this benchmark: that generic "
            "checkpoint is not the decontaminated GIFT-Eval zero-shot model."
        ) from exc


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
