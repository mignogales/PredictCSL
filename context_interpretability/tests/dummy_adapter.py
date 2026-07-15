"""
A tiny deterministic patch forecaster implementing the FULL adapter interface
on CPU, so every experiment (0–5) has an end-to-end test without any TSFM.

Architecture: patch embedding -> residual token-wise blocks -> (norm) ->
recency-weighted pooling -> linear head. Because blocks act token-wise until
the pooling step, the model has clean, analytically checkable properties:

  * pooling weights decay with token age  -> recent blocks matter more;
  * patching the corrupted block's tokens with clean activations at ANY layer
    fully restores the clean forecast    -> recovery score ~= 1;
  * ``linear=True`` removes every nonlinearity -> integrated gradients are
    EXACT (completeness error ~ float tolerance);
  * masking = renormalizing the pooling over the visible suffix — a faithful
    stand-in for "attend only to the last L timesteps".
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from context_interpretability.adapters.base import (
    AdapterCapabilities, InterpretabilityAdapter)
from context_interpretability.adapters.tsfm import resolve_token_indices


class _Block(nn.Module):
    def __init__(self, d: int, linear: bool):
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.linear = linear

    def forward(self, x):
        h = self.fc(x)
        return x + (h if self.linear else torch.tanh(h))


class DummyForecaster(nn.Module):
    def __init__(self, patch: int = 8, d: int = 16, n_layers: int = 3,
                 horizon: int = 8, linear: bool = False, decay: float = 0.15,
                 seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.patch = patch
        self.embed = nn.Linear(patch, d)
        self.blocks = nn.ModuleList(_Block(d, linear) for _ in range(n_layers))
        self.norm = nn.Identity() if linear else nn.LayerNorm(d)
        self.head = nn.Linear(d, horizon)
        self.decay = decay
        self.linear = linear

    def _pool_weights(self, n_tokens: int, visible_tokens: Optional[int]
                      ) -> torch.Tensor:
        age = torch.arange(n_tokens - 1, -1, -1, dtype=torch.float32)
        w = torch.exp(-self.decay * age)
        if visible_tokens is not None:
            w = w.clone()
            w[:max(0, n_tokens - visible_tokens)] = 0.0
        return w / w.sum()

    def forward(self, x: torch.Tensor, visible_tokens: Optional[int] = None,
                stop_layer: Optional[int] = None,
                patch_spec: Optional[dict] = None) -> torch.Tensor:
        """x: (B, W) with W a multiple of patch. ``patch_spec``:
        {layer, token_indices, replacement (B, T, d)} activation patch."""
        B, W = x.shape
        tokens = x.view(B, W // self.patch, self.patch)
        h = self.embed(tokens)                          # (B, T, d)
        for li, blk in enumerate(self.blocks):
            if stop_layer is not None and li > stop_layer:
                break                                    # identity-skip lens
            h = blk(h)
            if patch_spec is not None and patch_spec["layer"] == li:
                idx = resolve_token_indices(patch_spec["token_indices"],
                                            h.shape[-2])
                h = h.clone()
                h[:, idx, :] = patch_spec["replacement"][:, idx, :].to(h.dtype)
        h = self.norm(h)
        w = self._pool_weights(h.shape[1], visible_tokens)
        pooled = (h * w[None, :, None]).sum(dim=1)
        return self.head(pooled)                         # (B, H)


class DummyAdapter(InterpretabilityAdapter):
    def __init__(self, horizon: int = 8, patch: int = 8, n_layers: int = 3,
                 max_context: int = 64, linear: bool = False, seed: int = 0,
                 batch_size: int = 4):
        caps = AdapterCapabilities(
            supports_attention_masking=True,
            supports_activation_patching=True,
            supports_forecast_lens=True,
            supports_integrated_gradients=True,
            uses_patches=True, patch_length=patch,
            maximum_context_length=max_context)
        super().__init__("DummyModel", caps, horizon, "cpu", batch_size)
        self.model = DummyForecaster(patch=patch, n_layers=n_layers,
                                     horizon=horizon, linear=linear, seed=seed)
        self.model.eval()

    # -- helpers ---------------------------------------------------------------

    def _t(self, contexts: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            np.ascontiguousarray(contexts, dtype=np.float32))

    # -- interface ---------------------------------------------------------------

    def forecast(self, contexts: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.model(self._t(contexts)).numpy()

    def forecast_attention_masked(self, contexts: np.ndarray,
                                  visible_timesteps: int) -> np.ndarray:
        vis_tokens = max(1, int(visible_timesteps) // self.model.patch)
        with torch.no_grad():
            return self.model(self._t(contexts),
                              visible_tokens=vis_tokens).numpy()

    def get_layer_names(self) -> List[str]:
        return [f"blocks.{i}" for i in range(len(self.model.blocks))]

    def _layer_index(self, layer_name: str) -> int:
        names = self.get_layer_names()
        if layer_name not in names:
            raise KeyError(layer_name)
        return names.index(layer_name)

    def capture_activations(self, contexts: np.ndarray, layers: List[str]
                            ) -> Dict[str, torch.Tensor]:
        x = self._t(contexts)
        out: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            B, W = x.shape
            h = self.model.embed(x.view(B, W // self.model.patch,
                                        self.model.patch))
            for li, blk in enumerate(self.model.blocks):
                h = blk(h)
                name = f"blocks.{li}"
                if name in layers:
                    out[name] = h.clone()
        return out

    def run_with_activation_patch(self, contexts: np.ndarray, layer_name: str,
                                  token_indices, replacement_activation
                                  ) -> np.ndarray:
        spec = {"layer": self._layer_index(layer_name),
                "token_indices": token_indices,
                "replacement": replacement_activation}
        with torch.no_grad():
            return self.model(self._t(contexts), patch_spec=spec).numpy()

    def forecast_token_indices(self, context_length: int):
        return [-1]                                # most recent token

    def forecast_from_layer(self, contexts: np.ndarray, layer_name: str
                            ) -> np.ndarray:
        with torch.no_grad():
            return self.model(self._t(contexts),
                              stop_layer=self._layer_index(layer_name)).numpy()

    def forecast_differentiable(self, contexts_tensor: torch.Tensor
                                ) -> torch.Tensor:
        return self.model(contexts_tensor)


def make_data(n: int = 6, T: int = 64, horizon: int = 8, seed: int = 1):
    """Small AR(1)+seasonal pool as an ExperimentData."""
    from context_interpretability.experiments.common import ExperimentData
    rng = np.random.default_rng(seed)
    total = T + horizon
    x = np.zeros((n, total), dtype=np.float32)
    for i in range(n):
        eps = rng.normal(0, 0.3, total)
        for t in range(1, total):
            x[i, t] = 0.7 * x[i, t - 1] + 0.5 * np.sin(2 * np.pi * t / 16) \
                + eps[t]
    return ExperimentData(name="dummy", contexts=x[:, :T],
                          targets=x[:, T:], sample_ids=[f"s{i}" for i in
                                                        range(n)])


def make_config(tmp_root: str) -> dict:
    return {
        "output_root": tmp_root,
        "seed": 0,
        "horizon": 8,
        "batch_size": 4,
        "max_samples": 6,
        "primary_metric": "mae",
        "context_lengths": [16, 32, 64, 128],   # 128 dropped (max_context 64)
        "block_mode": "model_patch",
        "block_length": 8,
        "perturb_partial_patches": False,
        "max_blocks_per_context": 64,
        "perturbation": {
            "methods": ["block_mean", "permutation", "matched_block", "noise"],
            "mean_scope": "block",
            "noise_scales": [0.5],
            "noise_std_floor": 1e-3,
            "n_seeds": 2,
            "matched": {"same_series_tolerance": 0.25},
        },
        "activation_patching": {
            "corruption": "permutation",
            "granularity": ["block_token", "forecast_token"],
            "pilot_samples": 3,
            "layers": "all",
            "full_layers": "every:2",
        },
        "forecast_lens": {"stability_tau": 0.05},
        "integrated_gradients": {
            "steps": 8, "convergence_steps": [4, 8],
            "internal_batch_size": 4,
            "baselines": ["context_mean", "random_sample"],
            "n_random_baselines": 1,
            "target": "mean_forecast",
            "supplementary_loss_target": True,
        },
        "synthetic_controls": {
            "n_instances": 12,
            "series_length": 64,
            "local_order": 4,
            "local_kinds": ["linear"],
            "distant_kind": "linear",
            "distant_lags": [24],
            "dependency_strengths": [0.0, 1.0],
            "noise_levels": [0.1],
            "oracle_ridge_alpha": 1.0,
            "run_methods": ["perturbation"],
            "sufficient_context_tolerance": 0.05,
        },
        "analysis": {"bootstrap_samples": 200, "alpha": 0.05,
                     "top_k_blocks": 3},
        "datasets": {"source": "synthetic"},
        "experiments": {},
    }
