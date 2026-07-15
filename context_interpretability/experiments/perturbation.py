"""
Experiment 1 — temporal-block perturbation heatmaps (spec §4).

For each (context length, block, perturbation, seed): perturb ONLY that block
of the clean input (total length and positions unchanged), forecast, and score
Δloss + prediction distance against the cached clean run. Mean replacement
alone is never treated as conclusive — all four methods run by default.

Perturbations operate on (N, W) arrays (univariate; the channel-wise wording of
the spec reduces to per-series here). Stochastic methods are seeded per
(sample, block, seed) so results are reproducible and seeds are aggregated
WITHIN a sample before statistics (spec §12).

Output per cell (``.../exp1_perturbation/<dataset>/w<W>/``):
    results.csv       one row per (sample, block, method, severity, seed)
    heatmap rows feed analysis/figures.py (Heatmaps A/B + line plots + stats).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from context_interpretability.adapters.base import (
    InterpretabilityAdapter, TemporalBlock)
from context_interpretability.experiments.common import (
    CleanCache, ExperimentData, block_grid, intervention_effects)
from context_interpretability.schema import ResultsWriter, cell_done

METHOD = "perturbation"


# ------------------------------------------------------------------------------
#  Perturbation primitives (pure functions on a copy of the window)
# ------------------------------------------------------------------------------

def block_mean_replace(window: np.ndarray, blk: slice,
                       scope: str = "block") -> np.ndarray:
    """Replace the block with its own mean (or the context-wide mean)."""
    out = window.copy()
    src = window[:, blk] if scope == "block" else window
    out[:, blk] = np.nanmean(src, axis=1, keepdims=True)
    return out

def within_block_permutation(window: np.ndarray, blk: slice,
                             seed: int) -> np.ndarray:
    """Independently permute time positions inside the block per sample
    (value distribution preserved; deterministic per (seed, sample))."""
    out = window.copy()
    n = window.shape[0]
    width = blk.stop - blk.start
    for i in range(n):
        rng = np.random.default_rng((seed, i))
        out[i, blk] = window[i, blk][rng.permutation(width)]
    return out


def matched_block_replace(window: np.ndarray, blk: slice, seed: int,
                          tolerance: float = 0.25) -> np.ndarray:
    """Replace the block with a matched block (spec §4.2C preferred order):
    1) another equal-length block of the SAME series with similar mean/var;
    2) fallback: the same block position from another series in the batch.
    (Seasonal-phase matching needs seasonal metadata; datasets without it use
    the first two tiers — recorded by the caller.)"""
    out = window.copy()
    n, W = window.shape
    width = blk.stop - blk.start
    starts = np.arange(0, W - width + 1, width)
    tgt_mean = np.nanmean(window[:, blk], axis=1)
    tgt_std = np.nanstd(window[:, blk], axis=1)
    for i in range(n):
        rng = np.random.default_rng((seed, i, 1))
        cands = []
        for s in starts:
            if s == blk.start:
                continue
            seg = window[i, s:s + width]
            m, sd = np.nanmean(seg), np.nanstd(seg)
            scale = max(abs(tgt_mean[i]), tgt_std[i], 1e-6)
            if (abs(m - tgt_mean[i]) <= tolerance * scale
                    and abs(sd - tgt_std[i]) <= tolerance * scale):
                cands.append(s)
        if cands:
            s = int(rng.choice(cands))
            out[i, blk] = window[i, s:s + width]
        else:                                # tier 2: another series, same slot
            j = int(rng.integers(0, n - 1))
            j = j + 1 if j >= i else j
            out[i, blk] = window[j, blk]
    return out


def additive_noise(window: np.ndarray, blk: slice, scale: float, seed: int,
                   std_floor: float = 1e-3) -> np.ndarray:
    """x_b + scale * sigma_b * eps, sigma_b per series from the block, floored."""
    out = window.copy()
    rng = np.random.default_rng((seed, 7))
    sigma = np.maximum(np.nanstd(window[:, blk], axis=1, keepdims=True),
                       std_floor)
    out[:, blk] = window[:, blk] + scale * sigma * rng.standard_normal(
        window[:, blk].shape).astype(window.dtype)
    return out


def apply_perturbation(window: np.ndarray, blk: slice, kind: str, seed: int,
                       severity: float, pcfg: dict) -> np.ndarray:
    if kind == "block_mean":
        return block_mean_replace(window, blk, pcfg.get("mean_scope", "block"))
    if kind == "permutation":
        return within_block_permutation(window, blk, seed)
    if kind == "matched_block":
        tol = (pcfg.get("matched") or {}).get("same_series_tolerance", 0.25)
        return matched_block_replace(window, blk, seed, tol)
    if kind == "noise":
        return additive_noise(window, blk, severity, seed,
                              pcfg.get("noise_std_floor", 1e-3))
    raise ValueError(f"Unknown perturbation {kind!r}")


_STOCHASTIC = {"permutation", "matched_block", "noise"}


# ------------------------------------------------------------------------------
#  Runner
# ------------------------------------------------------------------------------

def run(adapter: InterpretabilityAdapter, data: ExperimentData, config: dict,
        out_dir: str, run_meta=None, seed: int = 0) -> List[str]:
    """Run the full perturbation grid; returns the finished cell dirs.

    Resumable per context length (one cell dir per W with a done-marker)."""
    pcfg = config.get("perturbation") or {}
    methods = pcfg.get("methods",
                       ["block_mean", "permutation", "matched_block", "noise"])
    noise_scales = pcfg.get("noise_scales", [0.1, 0.5, 1.0])
    n_seeds = int(pcfg.get("n_seeds", 3))
    metric = config.get("primary_metric", "mae")

    cache = CleanCache(adapter, data, metric,
                       cache_dir=os.path.join(out_dir, "clean_cache"))
    pairs = adapter.effective_context_lengths(config["context_lengths"])
    if run_meta:
        run_meta.note("effective_context_lengths",
                      {str(r): e for r, e in pairs})
        run_meta.note_samples("exp1_perturbation", data.n)
    cells: List[str] = []

    for requested, W in pairs:
        cell = os.path.join(out_dir, data.name, f"w{W}")
        cells.append(cell)
        if cell_done(cell):
            continue
        window = data.window(W)
        clean_pred, clean_loss = cache.get(W)
        blocks, thinned, P = block_grid(adapter, W, config)
        if thinned and run_meta:
            run_meta.note_subsampling(
                "exp1_perturbation",
                f"w{W}: {len(blocks)} of {W // P} blocks (geometric thinning)")
        writer = ResultsWriter(cell)

        for blk_obj in blocks:
            blk = blk_obj.input_slice(W)
            for kind in methods:
                sevs = noise_scales if kind == "noise" else [np.nan]
                seeds = range(n_seeds) if kind in _STOCHASTIC else [0]
                for severity in sevs:
                    for s in seeds:
                        pert = apply_perturbation(window, blk, kind,
                                                  seed + s, severity, pcfg)
                        assert pert.shape == window.shape  # length preserved
                        pred = adapter.forecast(pert)
                        eff = intervention_effects(clean_pred, clean_loss,
                                                   pred, cache)
                        _write_block_rows(writer, adapter, data, requested, W,
                                          blk_obj, kind, severity, seed + s,
                                          metric, eff)
        writer.finalize({"context_length": W, "block_length": P,
                         "n_blocks": len(blocks), "thinned": thinned,
                         "methods": methods})
        print(f"[exp1][{adapter.name}] {data.name} w{W}: "
              f"{len(blocks)} blocks x {len(methods)} methods done")
    return cells


def _write_block_rows(writer: ResultsWriter, adapter, data: ExperimentData,
                      requested: int, W: int, blk: TemporalBlock, kind: str,
                      severity: float, seed: int, metric: str,
                      eff: Dict[str, np.ndarray]) -> None:
    for i, sid in enumerate(data.sample_ids):
        writer.add(
            model=adapter.name, dataset=data.name, sample_id=sid,
            context_length=W, requested_context_length=requested,
            horizon=adapter.horizon, block_index=blk.index,
            lookback_start=blk.lookback_start, lookback_end=blk.lookback_end,
            method=METHOD, perturbation_type=kind, severity=severity,
            metric=metric, seed=seed,
            clean_loss=eff["clean_loss"][i],
            intervened_loss=eff["intervened_loss"][i],
            loss_delta=eff["loss_delta"][i],
            prediction_distance=eff["prediction_distance"][i],
            prediction_distance_norm=eff["prediction_distance_norm"][i],
        )
