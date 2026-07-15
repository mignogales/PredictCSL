"""
Experiment 2 — activation patching (spec §5).

For a sample and a temporal block: run the clean input (capturing activations),
corrupt the block (within-block permutation or matched replacement — NEVER
noise, §5.4), run the corrupted input, then rerun it with the clean activations
patched back in at one layer. Clean and corrupted inputs have identical length,
positions, normalization and masks by construction (the corruption edits values
in place).

Two granularities (spec §5.5):
  * block_token     — patch only the tokens covering the corrupted block;
  * forecast_token  — patch the model's dedicated forecast/readout tokens
                      (skipped + logged when the architecture has none).

Recovery scores (spec §5.6):
  * prediction recovery R = 1 - D(patched, clean) / (D(corrupted, clean) + eps)
  * loss recovery        = (L_corr - L_patched) / (L_corr - L_clean + eps)

Two-phase efficiency schedule (spec §11): all layers on a pilot subset, then
the configured representative layers on the full evaluation subset.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from context_interpretability.adapters.base import (
    CapabilityError, InterpretabilityAdapter)
from context_interpretability.experiments.common import (
    CleanCache, ExperimentData, block_grid)
from context_interpretability.experiments.perturbation import (
    matched_block_replace, within_block_permutation)
from context_interpretability.metrics import prediction_distance as pdist
from context_interpretability.schema import ResultsWriter, cell_done

METHOD = "activation_patching"
EPS = 1e-8


def _select_layers(all_layers: List[str], spec) -> List[str]:
    if spec == "all" or spec is None:
        return list(all_layers)
    if isinstance(spec, str) and spec.startswith("every:"):
        k = max(1, int(spec.split(":", 1)[1]))
        picked = all_layers[::k]
        if all_layers[-1] not in picked:      # always keep the final layer
            picked.append(all_layers[-1])
        return picked
    if isinstance(spec, (list, tuple)):
        return [l for l in all_layers if l in set(spec)]
    raise ValueError(f"Bad layer spec {spec!r}")


def _corrupt(window: np.ndarray, blk: slice, kind: str, seed: int,
             tolerance: float) -> np.ndarray:
    if kind == "permutation":
        return within_block_permutation(window, blk, seed)
    if kind == "matched_block":
        return matched_block_replace(window, blk, seed, tolerance)
    raise ValueError(
        f"corruption {kind!r} not allowed for activation patching "
        "(spec §5.4: permutation | matched_block)")


def recovery_score(patched: np.ndarray, clean: np.ndarray,
                   corrupted: np.ndarray) -> np.ndarray:
    d_pc = pdist.l1_distance(patched, clean)
    d_cc = pdist.l1_distance(corrupted, clean)
    return 1.0 - d_pc / (d_cc + EPS)


def loss_recovery(l_patched: np.ndarray, l_clean: np.ndarray,
                  l_corr: np.ndarray) -> np.ndarray:
    return (l_corr - l_patched) / (l_corr - l_clean + EPS)


def run(adapter: InterpretabilityAdapter, data: ExperimentData, config: dict,
        out_dir: str, run_meta=None, seed: int = 0) -> List[str]:
    if not adapter.capabilities.supports_activation_patching:
        raise CapabilityError(f"{adapter.name}: activation patching unsupported")

    acfg = config.get("activation_patching") or {}
    corruption = acfg.get("corruption", "permutation")
    tolerance = ((config.get("perturbation") or {}).get("matched") or {}).get(
        "same_series_tolerance", 0.25)
    granularity = acfg.get("granularity", ["block_token", "forecast_token"])
    metric = config.get("primary_metric", "mae")
    all_layers = adapter.get_layer_names()
    pilot_n = int(acfg.get("pilot_samples", 32))

    cache = CleanCache(adapter, data, metric,
                       cache_dir=os.path.join(out_dir, "clean_cache"))
    pairs = adapter.effective_context_lengths(config["context_lengths"])
    cells: List[str] = []

    phases = [
        ("pilot", data.subset(pilot_n),
         _select_layers(all_layers, acfg.get("layers", "all"))),
        ("full", data,
         _select_layers(all_layers, acfg.get("full_layers", "every:2"))),
    ]
    if run_meta:
        run_meta.note_samples("exp2_activation_patching", data.n)
        run_meta.note_subsampling(
            "exp2_activation_patching",
            f"pilot: {min(pilot_n, data.n)} samples x {len(phases[0][2])} "
            f"layers; full: {data.n} samples x {len(phases[1][2])} layers")

    for phase, pdata, layers in phases:
        pcache = cache if pdata.n == data.n else CleanCache(
            adapter, pdata, metric,
            cache_dir=os.path.join(out_dir, f"clean_cache_{phase}"))
        for requested, W in pairs:
            cell = os.path.join(out_dir, data.name, phase, f"w{W}")
            cells.append(cell)
            if cell_done(cell):
                continue
            _run_cell(adapter, pdata, pcache, config, cell, requested, W,
                      layers, granularity, corruption, tolerance, metric,
                      seed, run_meta, phase)
    return cells


def _run_cell(adapter, data: ExperimentData, cache: CleanCache, config: dict,
              cell: str, requested: int, W: int, layers: List[str],
              granularity: List[str], corruption: str, tolerance: float,
              metric: str, seed: int, run_meta, phase: str) -> None:
    window = data.window(W)
    clean_pred, clean_loss = cache.get(W)
    blocks, thinned, P = block_grid(adapter, W, config)
    if thinned and run_meta:
        run_meta.note_subsampling(
            "exp2_activation_patching",
            f"{phase} w{W}: {len(blocks)} of {W // P} blocks")
    clean_acts = adapter.capture_activations(window, layers)
    fc_tokens = adapter.forecast_token_indices(W)
    if "forecast_token" in granularity and fc_tokens is None and run_meta:
        run_meta.skip("exp2_activation_patching/forecast_token",
                      f"{adapter.name}: no dedicated forecast token at w{W}")
    writer = ResultsWriter(cell)

    for blk in blocks:
        corrupted = _corrupt(window, blk.input_slice(W), corruption, seed,
                             tolerance)
        corr_pred = adapter.forecast(corrupted)
        corr_loss = cache.loss(corr_pred)
        token_idx = adapter.input_block_to_token_indices(
            blk.lookback_start, blk.lookback_end, W)
        grans = [("block_token", token_idx)]
        if "forecast_token" in granularity and fc_tokens is not None:
            grans.append(("forecast_token", fc_tokens))

        for gran_name, tokens in grans:
            for layer in layers:
                patched_pred = adapter.run_with_activation_patch(
                    corrupted, layer, tokens, clean_acts[layer])
                patched_loss = cache.loss(patched_pred)
                rec = recovery_score(patched_pred, clean_pred, corr_pred)
                lrec = loss_recovery(patched_loss, clean_loss, corr_loss)
                for i, sid in enumerate(data.sample_ids):
                    writer.add(
                        model=adapter.name, dataset=data.name, sample_id=sid,
                        context_length=W, requested_context_length=requested,
                        horizon=adapter.horizon, block_index=blk.index,
                        lookback_start=blk.lookback_start,
                        lookback_end=blk.lookback_end,
                        method=METHOD,
                        perturbation_type=f"{corruption}/{gran_name}",
                        layer=layer, metric=metric, seed=seed,
                        clean_loss=clean_loss[i],
                        intervened_loss=patched_loss[i],
                        loss_delta=corr_loss[i] - clean_loss[i],
                        prediction_distance=pdist.l1_distance(
                            patched_pred, clean_pred)[i],
                        prediction_distance_norm=pdist.normalized_distance(
                            clean_pred, patched_pred)[i],
                        recovery_score=rec[i],
                        loss_recovery=lrec[i],
                    )
    writer.finalize({"context_length": W, "phase": phase, "layers": layers,
                     "corruption": corruption, "granularity": granularity,
                     "n_blocks": len(blocks)})
    print(f"[exp2][{adapter.name}] {data.name} {phase} w{W}: "
          f"{len(blocks)} blocks x {len(layers)} layers done")
