"""
Experiment 0 — the EXISTING attention-masking analysis, integrated (spec §9).

The mechanism is untouched: ``experiments.context_attention_mask`` restricts
the attention span to the last L timesteps of a full W-length input (positions
+ normalization stats stay full-window; see that module's docstring and
``experiments/masking_vs_slicing.py`` / ``test_window_masking_gifteval.py`` for
the original studies). What this wrapper adds is the COMMON evaluation: same
datasets, same context lengths, same metrics, same tabular schema as the five
new experiments, so masking can be compared with them directly.

Row semantics: masking to visible length L hides every block with
lookback >= L, so a row records the HIDDEN span —
``lookback_start = L``, ``lookback_end = W``, ``block_index`` = index of the
first hidden block. Δloss is masked-vs-clean at the SAME full window W.
"""

from __future__ import annotations

import os
from typing import List

from context_interpretability.adapters.base import (
    CapabilityError, InterpretabilityAdapter)
from context_interpretability.experiments.common import (
    CleanCache, ExperimentData, intervention_effects)
from context_interpretability.schema import ResultsWriter, cell_done

METHOD = "attention_masking"


def run(adapter: InterpretabilityAdapter, data: ExperimentData, config: dict,
        out_dir: str, run_meta=None, seed: int = 0) -> List[str]:
    if not adapter.capabilities.supports_attention_masking:
        raise CapabilityError(f"{adapter.name}: attention masking unsupported")

    metric = config.get("primary_metric", "mae")
    cache = CleanCache(adapter, data, metric,
                       cache_dir=os.path.join(out_dir, "clean_cache"))
    pairs = adapter.effective_context_lengths(config["context_lengths"])
    if run_meta:
        run_meta.note_samples("exp0_attention_masking", data.n)
    P = adapter.block_length(config.get("block_mode", "model_patch"),
                             config.get("block_length", 32))
    cells: List[str] = []

    for requested, W in pairs:
        cell = os.path.join(out_dir, data.name, f"w{W}")
        cells.append(cell)
        if cell_done(cell):
            continue
        window = data.window(W)
        clean_pred, clean_loss = cache.get(W)
        writer = ResultsWriter(cell)

        # every effective grid length strictly below W is a visible-span level
        visibles = [e for _r, e in pairs if e < W]
        for L in visibles:
            masked_pred = adapter.forecast_attention_masked(window, L)
            eff = intervention_effects(clean_pred, clean_loss, masked_pred,
                                       cache)
            first_hidden_block = L // P
            for i, sid in enumerate(data.sample_ids):
                writer.add(
                    model=adapter.name, dataset=data.name, sample_id=sid,
                    context_length=W, requested_context_length=requested,
                    horizon=adapter.horizon,
                    block_index=first_hidden_block,
                    lookback_start=L, lookback_end=W,
                    method=METHOD, perturbation_type="attention_masking",
                    metric=metric, seed=seed,
                    clean_loss=eff["clean_loss"][i],
                    intervened_loss=eff["intervened_loss"][i],
                    loss_delta=eff["loss_delta"][i],
                    prediction_distance=eff["prediction_distance"][i],
                    prediction_distance_norm=eff["prediction_distance_norm"][i],
                )
        writer.finalize({"context_length": W, "visible_lengths": visibles,
                         "block_length": P})
        print(f"[exp0][{adapter.name}] {data.name} w{W}: "
              f"{len(visibles)} mask levels done")
    return cells
