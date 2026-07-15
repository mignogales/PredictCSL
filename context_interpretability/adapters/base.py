"""
Model-agnostic adapter interface (spec §5.2 + §10.1).

An :class:`InterpretabilityAdapter` is the ONLY thing experiment code touches.
It declares capabilities up front (so unsupported methods are skipped
explicitly) and exposes:

  * forecasting (point / optionally quantile), batched, no-grad;
  * a differentiable forward for integrated gradients;
  * attention-span restriction (Experiment 0);
  * layer listing, activation capture and activation patching (Experiment 2);
  * the frozen-head forecast lens (Experiment 3);
  * input-block <-> internal-token index mapping.

Block convention (spec §3.3): blocks are indexed relative to the forecast
origin — block 0 is the MOST RECENT block; lookback_start/end are timestep
distances from the origin, so block b of length P covers input positions
``[W - end, W - start)`` of a length-W context.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class CapabilityError(RuntimeError):
    """Raised when a declared-supported method turns out unusable at runtime.
    Callers convert this into an explicit RunMeta skip — never a silent pass."""


@dataclasses.dataclass
class AdapterCapabilities:
    supports_attention_masking: bool = False
    supports_activation_patching: bool = False
    supports_forecast_lens: bool = False
    supports_integrated_gradients: bool = False
    uses_patches: bool = False
    patch_length: Optional[int] = None
    maximum_context_length: int = 8192
    # context + horizon share the length budget (timemoe / moirai)
    max_includes_horizon: bool = False
    # model always consumes its full native context (patchtst_fm NaN-padding)
    fixed_native_context: bool = False

    def supported(self, method: str) -> bool:
        return bool(getattr(self, f"supports_{method}", False))


@dataclasses.dataclass(frozen=True)
class TemporalBlock:
    index: int              # 0 = most recent
    lookback_start: int     # timesteps from forecast origin (inclusive)
    lookback_end: int       # exclusive; end - start == block length

    def input_slice(self, context_length: int) -> slice:
        """Positional slice of this block within a (…, W) input array."""
        return slice(context_length - self.lookback_end,
                     context_length - self.lookback_start)


def align_context_lengths(requested: Sequence[int], patch_length: Optional[int],
                          maximum: int) -> List[Tuple[int, int]]:
    """Spec §3.2: drop unsupported lengths, align DOWN to whole patches.

    Returns ``[(requested, effective), ...]`` with duplicates (after alignment)
    removed keeping the first occurrence.
    """
    out: List[Tuple[int, int]] = []
    seen = set()
    p = int(patch_length) if patch_length else 1
    for req in requested:
        if req > maximum:
            continue
        eff = max(p, (int(req) // p) * p)
        if eff > maximum or eff in seen:
            continue
        seen.add(eff)
        out.append((int(req), eff))
    return out


def blocks_for_context(context_length: int, block_length: int,
                       include_partial: bool = False) -> List[TemporalBlock]:
    """Enumerate temporal blocks of a length-W context, most recent first."""
    W, P = int(context_length), int(block_length)
    n_full = W // P
    blocks = [TemporalBlock(b, b * P, (b + 1) * P) for b in range(n_full)]
    rem = W - n_full * P
    if rem and include_partial:
        blocks.append(TemporalBlock(n_full, n_full * P, W))
    return blocks


def thin_blocks(blocks: List[TemporalBlock], max_blocks: int
                ) -> Tuple[List[TemporalBlock], bool]:
    """Cap the evaluated block count, keeping recent blocks dense and thinning
    distant ones geometrically. Returns (subset, was_thinned) — callers must
    record thinning in run_meta (spec §11: no silent size reduction)."""
    if len(blocks) <= max_blocks:
        return blocks, False
    n = len(blocks)
    dense = max_blocks // 2
    keep = set(range(dense))
    # geometric spacing over the remaining lookback range
    n_geo = max_blocks - dense
    geo = np.unique(np.round(np.geomspace(dense + 1, n, n_geo)).astype(int) - 1)
    keep.update(int(g) for g in geo)
    subset = [b for b in blocks if b.index in keep]
    return subset, True


class InterpretabilityAdapter:
    """Abstract adapter. Concrete implementations: adapters/tsfm.py (the real
    TSFM zoo) and tests/dummy_adapter.py (a tiny local model for CI)."""

    def __init__(self, name: str, capabilities: AdapterCapabilities,
                 horizon: int, device: str = "cpu", batch_size: int = 16):
        self.name = name
        self.capabilities = capabilities
        self.horizon = int(horizon)
        self.device = device
        self.batch_size = int(batch_size)

    # -- context geometry ------------------------------------------------------

    @property
    def patch_length(self) -> Optional[int]:
        return self.capabilities.patch_length

    def max_context(self) -> int:
        cap = self.capabilities.maximum_context_length
        if self.capabilities.max_includes_horizon:
            cap -= self.horizon
        return cap

    def effective_context_lengths(self, requested: Sequence[int]
                                  ) -> List[Tuple[int, int]]:
        return align_context_lengths(requested, self.capabilities.patch_length,
                                     self.max_context())

    def block_length(self, block_mode: str, fixed_length: int) -> int:
        """Resolve the temporal-block length (spec §3.3): the model patch when
        available and requested, else the configured fixed length."""
        if (block_mode == "model_patch" and self.capabilities.uses_patches
                and self.capabilities.patch_length
                and self.capabilities.patch_length > 1):
            return int(self.capabilities.patch_length)
        return int(fixed_length)

    # -- forecasting -----------------------------------------------------------

    def forecast(self, contexts: np.ndarray) -> np.ndarray:
        """Point (median) forecast. ``contexts``: (N, W) float32 -> (N, H)."""
        raise NotImplementedError

    def forecast_quantiles(self, contexts: np.ndarray
                           ) -> Optional[Tuple[np.ndarray, List[float]]]:
        """(N, Q, H) quantile forecasts + levels, or None (point-only model)."""
        return None

    # -- Experiment 0: attention masking ----------------------------------------

    def forecast_attention_masked(self, contexts: np.ndarray,
                                  visible_timesteps: int) -> np.ndarray:
        """Full-window forecast with attention restricted to the last
        ``visible_timesteps`` (input tensor untouched)."""
        raise CapabilityError(f"{self.name}: attention masking unsupported")

    # -- Experiment 2: activation capture / patching ----------------------------

    def get_layer_names(self) -> List[str]:
        raise CapabilityError(f"{self.name}: layer access unsupported")

    def capture_activations(self, contexts: np.ndarray, layers: List[str]
                            ) -> Dict[str, "object"]:
        """Run a clean forward; return {layer_name: activation tensor} for the
        PREFILL pass over the full context (batch dim first, token dim -2)."""
        raise CapabilityError(f"{self.name}: activation capture unsupported")

    def run_with_activation_patch(self, contexts: np.ndarray, layer_name: str,
                                  token_indices: List[int],
                                  replacement_activation) -> np.ndarray:
        """Forward ``contexts`` while overwriting ``layer_name``'s output at
        ``token_indices`` (token axis -2) with the clean activation slice."""
        raise CapabilityError(f"{self.name}: activation patching unsupported")

    def input_block_to_token_indices(self, block_start: int, block_end: int,
                                     context_length: int) -> List[int]:
        """Map an input block [W-block_end, W-block_start) to internal token
        positions of the prefill sequence."""
        p = self.capabilities.patch_length or 1
        n_tokens = math.ceil(context_length / p)
        tok_lo = (context_length - block_end) // p
        tok_hi = math.ceil((context_length - block_start) / p)
        return list(range(max(0, tok_lo), min(n_tokens, tok_hi)))

    def forecast_token_indices(self, context_length: int) -> Optional[List[int]]:
        """Dedicated forecast/query/summary token positions (spec §5.5), or
        None when the architecture has none (then forecast-token patching is
        skipped, logged)."""
        return None

    # -- Experiment 3: forecast lens --------------------------------------------

    def forecast_from_layer(self, contexts: np.ndarray, layer_name: str
                            ) -> np.ndarray:
        """Frozen-head forecast read at intermediate layer ``layer_name``
        (identity-skip of deeper residual blocks — documented per adapter)."""
        raise CapabilityError(f"{self.name}: forecast lens unsupported")

    # -- Experiment 5: integrated gradients --------------------------------------

    def forecast_differentiable(self, contexts_tensor):
        """Differentiable point forecast: torch (N, W) -> torch (N, H) with a
        grad_fn reaching the input. Raise CapabilityError when the model path
        cannot support it (sampling / compiled decode / pipeline no_grad)."""
        raise CapabilityError(f"{self.name}: integrated gradients unsupported")

    # -- lifecycle ---------------------------------------------------------------

    def load(self) -> None:
        """Idempotent heavyweight load (weights on device)."""

    def close(self) -> None:
        """Release device memory."""

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "horizon": self.horizon,
            "device": str(self.device),
            "capabilities": dataclasses.asdict(self.capabilities),
        }
