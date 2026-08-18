"""Slicing/attention/normalization decomposition (Experiment 7).

For a fixed full context W and visible suffix L, compare:

``sliced``
    Feed only the last L values.  Attention span, normalization statistics,
    positions and physical width all change.

``attention_mask/full_history_stats``
    Feed the genuine W values and attention-mask everything older than L.
    Positions/width and full-history preprocessing statistics stay fixed.

``attention_mask/tail_matched_stats``
    Keep width W and the same attention mask, but replace the hidden prefix by
    a deterministic sequence with exactly the visible tail's mean and standard
    deviation.  For models using global mean/std instance normalization this
    makes the full input carry the tail's normalization statistics while the
    visible values and positions are unchanged.

``sliced/full_history_stats``
    Feed exactly the same physical slice as ``sliced``, but normalize and
    denormalize it with the complete W-step history's mean and scale. This is
    the direct deployment intervention: the context selector remains unchanged
    and only the forecasting model's RevIN reference window changes.

The paired differences estimate:

* full-history mask - tail-stat mask: normalization/preprocessing effect;
* tail-stat mask - slice: position/width/residual implementation effect;
* full-history mask - slice: total masking-vs-slicing gap.

The tail-stat intervention is an operational normalization control, not a
universal proof: robust/nonlinear scalers and architectures that read hidden
token states outside attention can retain additional paths.  These caveats are
recorded in every cell.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from context_interpretability.adapters.base import CapabilityError
from context_interpretability.experiments.common import CleanCache, ExperimentData
from context_interpretability.metrics import forecast_metrics as fm
from context_interpretability.metrics import prediction_distance as pdist
from context_interpretability.schema import ResultsWriter, cell_done

METHOD = "context_decomposition"
VARIANT_FULL = "attention_mask/full_history_stats"
VARIANT_TAIL = "attention_mask/tail_matched_stats"
VARIANT_TAIL_MEAN = "attention_mask/tail_mean_full_scale"
VARIANT_TAIL_SCALE = "attention_mask/full_mean_tail_scale"
VARIANT_TAIL_DIRECT = "attention_mask/tail_mean_tail_scale_direct"
VARIANT_SLICE_FULL_STATS = "sliced/full_history_stats"
LIMITATIONS = (
    "The tail-matched-prefix condition isolates mean/std normalization exactly "
    "only for global affine instance normalization. Robust/nonlinear scaling, "
    "padding rules, and direct readout from nominally hidden token states can "
    "contribute to the residual tail-stat-mask versus slice gap."
)


def tail_stat_matched_prefix(full_window: np.ndarray,
                             visible_timesteps: int) -> np.ndarray:
    """Replace the hidden prefix with values of the tail's mean/std.

    The visible suffix is bitwise unchanged.  The generated prefix is a tiled
    tail template, re-standardized per sample, so prefix and tail have the same
    population mean/variance and therefore their concatenation does too.
    """
    x = np.asarray(full_window, dtype=np.float32)
    W = x.shape[1]
    L = int(visible_timesteps)
    if not 0 < L <= W:
        raise ValueError(f"visible_timesteps must be in [1, {W}], got {L}")
    prefix_len = W - L
    if prefix_len == 0:
        return np.ascontiguousarray(x.copy())
    out = x.copy()
    tail = x[:, -L:]
    for i in range(len(x)):
        mu = float(tail[i].mean())
        sd = float(tail[i].std())
        template = np.resize(tail[i], prefix_len).astype(np.float32, copy=True)
        tmu, tsd = float(template.mean()), float(template.std())
        if sd <= 1e-8 or tsd <= 1e-8:
            template.fill(mu)
        else:
            template = (template - tmu) * (sd / tsd) + mu
            # One final affine correction limits float32 accumulation error.
            template = ((template - float(template.mean()))
                        * (sd / (float(template.std()) + 1e-12)) + mu)
        out[i, :prefix_len] = template
    return np.ascontiguousarray(out)


def _loss(metric: str, pred: np.ndarray, data: ExperimentData) -> np.ndarray:
    return fm.compute_loss(metric, pred, data.targets,
                           context=data.contexts,
                           season_length=data.season_length)


def run(adapter, data: ExperimentData, config: dict, out_dir: str,
        run_meta=None, seed: int = 0) -> List[str]:
    if not adapter.capabilities.supports_attention_masking:
        raise CapabilityError(
            f"{adapter.name}: context decomposition needs attention masking")
    cfg = config.get("context_decomposition") or {}
    metrics = list(dict.fromkeys(
        str(m).lower() for m in cfg.get("metrics", ["mae", "mse"])))
    bad = set(metrics) - {"mae", "mse", "smape", "mase"}
    if bad:
        raise ValueError(f"Unsupported decomposition metrics: {sorted(bad)}")
    pairs = adapter.effective_context_lengths(config["context_lengths"])
    effective = [e for _r, e in pairs]
    if len(effective) < 2:
        raise ValueError("context decomposition needs at least two lengths")
    configured_full = cfg.get("full_contexts")
    if configured_full:
        wanted = {int(v) for v in configured_full}
        full_contexts = [w for w in effective if w in wanted]
    elif cfg.get("all_full_contexts", False):
        full_contexts = effective[1:]
    else:
        full_contexts = [effective[-1]]
    visible_cfg = cfg.get("visible_lengths")
    visible_wanted = ({int(v) for v in visible_cfg}
                      if visible_cfg else None)
    run_stat_ablation = bool(cfg.get("statistic_ablation", False))
    run_slice_full_stats = bool(cfg.get("slice_full_history_stats", False))
    stat_supported = getattr(adapter, "family", None) in {
        "chronos_bolt", "patchtst_fm"}

    cache = CleanCache(
        adapter, data, metrics[0], cache_dir=os.path.join(out_dir, "clean_cache"))
    if run_meta:
        run_meta.note_samples("exp7_context_decomposition", data.n)
        run_meta.note("exp7_context_decomposition_full_contexts", full_contexts)
    cells: List[str] = []
    for W in full_contexts:
        cell = os.path.join(out_dir, data.name, f"w{W}")
        cells.append(cell)
        if cell_done(cell):
            continue
        full = data.window(W)
        visibles = [L for L in effective if L < W
                    and (visible_wanted is None or L in visible_wanted)]
        writer = ResultsWriter(cell)
        prediction_files = []
        for L in visibles:
            sliced_pred = cache.get(L)[0]
            masked_full_pred = adapter.forecast_attention_masked(full, L)
            matched = tail_stat_matched_prefix(full, L)
            masked_tail_pred = adapter.forecast_attention_masked(matched, L)
            stat_predictions = {}
            if run_slice_full_stats and stat_supported:
                from experiments.context_normalization_override import (
                    normalization_reference_override,
                )
                with normalization_reference_override(
                        adapter.family, adapter._base, full):
                    stat_predictions[VARIANT_SLICE_FULL_STATS] = \
                        adapter.forecast(full[:, -L:])
            if run_stat_ablation and stat_supported:
                from experiments.context_normalization_override import (
                    normalization_stat_override,
                )
                for name, use_mean, use_scale in (
                    (VARIANT_TAIL_MEAN, True, False),
                    (VARIANT_TAIL_SCALE, False, True),
                    (VARIANT_TAIL_DIRECT, True, True),
                ):
                    with normalization_stat_override(
                            adapter.family, adapter._base, L,
                            tail_mean=use_mean, tail_scale=use_scale):
                        stat_predictions[name] = \
                            adapter.forecast_attention_masked(full, L)
            pred_path = os.path.join(cell, f"predictions_L{L}.npz")
            np.savez_compressed(
                pred_path, sliced=sliced_pred,
                attention_mask_full_history_stats=masked_full_pred,
                attention_mask_tail_matched_stats=masked_tail_pred,
                **{name.replace("/", "__"): pred
                   for name, pred in stat_predictions.items()})
            prediction_files.append(os.path.basename(pred_path))

            variants = {
                VARIANT_FULL: masked_full_pred,
                VARIANT_TAIL: masked_tail_pred,
                **stat_predictions,
            }
            for metric in metrics:
                sliced_loss = _loss(metric, sliced_pred, data)
                for variant, pred in variants.items():
                    variant_loss = _loss(metric, pred, data)
                    distance = pdist.l1_distance(pred, sliced_pred)
                    distance_norm = pdist.normalized_distance(
                        sliced_pred, pred)
                    for i, sid in enumerate(data.sample_ids):
                        writer.add(
                            model=adapter.name, dataset=data.name,
                            sample_id=sid, context_length=W,
                            requested_context_length=W,
                            horizon=adapter.horizon, block_index=L,
                            lookback_start=L, lookback_end=W,
                            method=METHOD, perturbation_type=variant,
                            metric=metric, seed=seed,
                            clean_loss=sliced_loss[i],
                            intervened_loss=variant_loss[i],
                            loss_delta=variant_loss[i] - sliced_loss[i],
                            prediction_distance=distance[i],
                            prediction_distance_norm=distance_norm[i])
        writer.finalize({
            "full_context": W, "visible_lengths": visibles,
            "metrics": metrics, "prediction_files": prediction_files,
            "components": {
                "normalization_preprocessing":
                    f"{VARIANT_FULL} - {VARIANT_TAIL}",
                "position_width_residual": f"{VARIANT_TAIL} - sliced",
                "total_masking_vs_slicing": f"{VARIANT_FULL} - sliced",
            },
            "limitations": LIMITATIONS,
            "statistic_ablation": run_stat_ablation and stat_supported,
            "slice_full_history_stats": (
                run_slice_full_stats and stat_supported),
        })
        print(f"[exp7][{adapter.name}] {data.name} W={W}: "
              f"{len(visibles)} visible lengths x {len(metrics)} metrics done")
    return cells
