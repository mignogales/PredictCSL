"""
Shared experiment plumbing: dataset container, the clean-baseline cache
(spec §3.6) and block-grid helpers.

Every intervention result is computed against the SAME cached clean run —
identical input slice, identical preprocessing — never a separately normalized
baseline. Clean forecasts/losses per (dataset, context length) are cached to
disk so re-runs and the other experiments reuse them (spec §11).
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from context_interpretability.adapters.base import (
    InterpretabilityAdapter, TemporalBlock, blocks_for_context, thin_blocks)
from context_interpretability.metrics import forecast_metrics as fm
from context_interpretability.metrics import prediction_distance as pdist


@dataclasses.dataclass
class ExperimentData:
    """One evaluation pool: contexts are tail-aligned (forecast origin = last
    column), targets are the true continuation."""
    name: str
    contexts: np.ndarray            # (N, T) float32, NaN-free
    targets: np.ndarray             # (N, H) float32
    sample_ids: List[str]
    season_length: int = 1
    metadata: Optional[dict] = None

    def __post_init__(self):
        if self.contexts.ndim != 2 or self.targets.ndim != 2:
            raise ValueError("contexts/targets must be 2-D (N, T)/(N, H)")
        if self.contexts.shape[0] != self.targets.shape[0]:
            raise ValueError("contexts/targets sample counts differ")
        if len(self.sample_ids) != self.contexts.shape[0]:
            raise ValueError("sample_ids length mismatch")

    @property
    def n(self) -> int:
        return self.contexts.shape[0]

    def window(self, context_length: int) -> np.ndarray:
        """The last ``context_length`` timesteps (the model input at W)."""
        if context_length > self.contexts.shape[1]:
            raise ValueError(
                f"context_length {context_length} > pool length "
                f"{self.contexts.shape[1]}")
        return np.ascontiguousarray(self.contexts[:, -context_length:])

    def subset(self, n: int) -> "ExperimentData":
        n = min(n, self.n)
        return ExperimentData(self.name, self.contexts[:n], self.targets[:n],
                              self.sample_ids[:n], self.season_length,
                              self.metadata)


class CleanCache:
    """Per-(adapter, dataset) clean forecasts + losses, disk-backed.

    ``get(W)`` returns ``(clean_pred (N, H), clean_loss (N,))`` for the
    effective context length W, computing + caching on first use.
    """

    def __init__(self, adapter: InterpretabilityAdapter, data: ExperimentData,
                 metric: str, cache_dir: Optional[str] = None):
        self.adapter = adapter
        self.data = data
        self.metric = metric
        self.cache_dir = cache_dir
        self._mem: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _path(self, W: int) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"clean_w{W}.npz")

    def loss(self, pred: np.ndarray) -> np.ndarray:
        return fm.compute_loss(self.metric, pred, self.data.targets,
                               context=self.data.contexts,
                               season_length=self.data.season_length)

    def get(self, W: int) -> Tuple[np.ndarray, np.ndarray]:
        if W in self._mem:
            return self._mem[W]
        path = self._path(W)
        if path and os.path.exists(path):
            z = np.load(path)
            if (z["pred"].shape[0] == self.data.n
                    and str(z["metric"]) == self.metric):
                self._mem[W] = (z["pred"], z["loss"])
                return self._mem[W]
        pred = self.adapter.forecast(self.data.window(W))
        loss = self.loss(pred)
        self._mem[W] = (pred, loss)
        if path:
            tmp = path + ".tmp.npz"
            np.savez(tmp, pred=pred, loss=loss, metric=self.metric)
            os.replace(tmp, path)
        return self._mem[W]

    def error_curve(self, effective_lengths: Sequence[int]
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """(per-sample loss matrix (N, K), mean curve (K,)) over the grid."""
        mat = np.stack([self.get(W)[1] for W in effective_lengths], axis=1)
        return mat, np.nanmean(mat, axis=0)


def block_grid(adapter: InterpretabilityAdapter, context_length: int,
               config: dict) -> Tuple[List[TemporalBlock], bool, int]:
    """Blocks for one context length under the configured block mode.

    Returns ``(blocks, was_thinned, block_length)``; thinning must be recorded
    in run_meta by the caller (spec §11).
    """
    P = adapter.block_length(config.get("block_mode", "model_patch"),
                             config.get("block_length", 32))
    blocks = blocks_for_context(context_length, P,
                                include_partial=config.get(
                                    "perturb_partial_patches", False))
    blocks, thinned = thin_blocks(blocks,
                                  int(config.get("max_blocks_per_context", 64)))
    return blocks, thinned, P


def intervention_effects(clean_pred: np.ndarray, clean_loss: np.ndarray,
                         pert_pred: np.ndarray, cache: CleanCache) -> dict:
    """Per-sample Δloss + prediction distances for one intervention forward."""
    pert_loss = cache.loss(pert_pred)
    return {
        "clean_loss": clean_loss,
        "intervened_loss": pert_loss,
        "loss_delta": pert_loss - clean_loss,
        "prediction_distance": pdist.l1_distance(pert_pred, clean_pred),
        "prediction_distance_norm": pdist.normalized_distance(clean_pred,
                                                              pert_pred),
    }
