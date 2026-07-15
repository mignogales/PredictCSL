"""
Experiment 3 — layer-wise forecast lens (spec §6).

At which layer does the forecast stabilize, and do longer contexts change early
representations without changing the final forecast?

``adapter.forecast_from_layer(x, layer)`` reuses the model's OWN final
normalization + frozen forecasting head on the layer-l representation (the
TSFM adapter implements this as identity-skip of deeper residual blocks; no
learned probes anywhere, spec §6.2). Models where that transformation is
architecturally invalid declare ``supports_forecast_lens: false`` and are
skipped explicitly.

Per (context length W, layer l) we record three references (spec §6.3):
  * loss( yhat^(l, W), Y )                     — ground-truth error;
  * D( yhat^(l, W), yhat^(L, W_max) )          — distance to the final
                                                  full-context forecast;
  * D( yhat^(l, W), yhat^(L, W) )              — distance to the final
                                                  same-context forecast
                                                  (depth-only convergence).

The saturation layer per W is the first l whose normalized same-context
distance drops below the configured tau (reported in normalized units).
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np

from context_interpretability.adapters.base import (
    CapabilityError, InterpretabilityAdapter)
from context_interpretability.experiments.common import (
    CleanCache, ExperimentData)
from context_interpretability.metrics import prediction_distance as pdist
from context_interpretability.schema import ResultsWriter, cell_done

METHOD = "forecast_lens"


def run(adapter: InterpretabilityAdapter, data: ExperimentData, config: dict,
        out_dir: str, run_meta=None, seed: int = 0) -> List[str]:
    if not adapter.capabilities.supports_forecast_lens:
        raise CapabilityError(f"{adapter.name}: forecast lens unsupported")

    metric = config.get("primary_metric", "mae")
    tau = float((config.get("forecast_lens") or {}).get("stability_tau", 0.05))
    layers = adapter.get_layer_names()
    cache = CleanCache(adapter, data, metric,
                       cache_dir=os.path.join(out_dir, "clean_cache"))
    pairs = adapter.effective_context_lengths(config["context_lengths"])
    if run_meta:
        run_meta.note_samples("exp3_forecast_lens", data.n)
    W_max = max(e for _r, e in pairs)
    final_full_pred, _ = cache.get(W_max)     # yhat^(L, W_max), frozen reference
    cells: List[str] = []
    saturation: dict = {}

    for requested, W in pairs:
        cell = os.path.join(out_dir, data.name, f"w{W}")
        cells.append(cell)
        if cell_done(cell):
            try:                      # recover the summary entry on resume
                with open(os.path.join(cell, "done.json")) as f:
                    saturation[str(W)] = json.load(f).get("saturation_layer", -1)
            except Exception:  # noqa: BLE001
                saturation[str(W)] = -1
            continue
        window = data.window(W)
        final_same_pred, final_same_loss = cache.get(W)   # yhat^(L, W)
        writer = ResultsWriter(cell)
        sat_layer = None

        for li, layer in enumerate(layers):
            lens_pred = adapter.forecast_from_layer(window, layer)
            lens_loss = cache.loss(lens_pred)
            d_full = pdist.l1_distance(lens_pred, final_full_pred)
            d_full_n = pdist.normalized_distance(final_full_pred, lens_pred)
            d_same = pdist.l1_distance(lens_pred, final_same_pred)
            d_same_n = pdist.normalized_distance(final_same_pred, lens_pred)
            if sat_layer is None and np.nanmean(d_same_n) < tau:
                sat_layer = li
            for i, sid in enumerate(data.sample_ids):
                writer.add(
                    model=adapter.name, dataset=data.name, sample_id=sid,
                    context_length=W, requested_context_length=requested,
                    horizon=adapter.horizon, method=METHOD,
                    perturbation_type="identity_skip", layer=layer,
                    metric=metric, seed=seed,
                    clean_loss=final_same_loss[i],
                    intervened_loss=lens_loss[i],
                    loss_delta=lens_loss[i] - final_same_loss[i],
                    # heatmap B/C sources: distance to final forecasts
                    prediction_distance=d_full[i],
                    prediction_distance_norm=d_same_n[i],
                    # keep the full-context normalized distance too
                    attribution_score=d_full_n[i],
                )
        saturation[str(W)] = sat_layer if sat_layer is not None else -1
        writer.finalize({"context_length": W, "layers": layers, "tau": tau,
                         "saturation_layer": saturation[str(W)]})
        print(f"[exp3][{adapter.name}] {data.name} w{W}: "
              f"saturation layer = {saturation[str(W)]}")

    sat_path = os.path.join(out_dir, data.name, "saturation_layers.json")
    os.makedirs(os.path.dirname(sat_path), exist_ok=True)
    with open(sat_path, "w") as f:
        json.dump({"tau": tau, "layers": layers,
                   "saturation_layer_by_context": saturation}, f, indent=2)
    return cells
