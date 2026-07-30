"""Predictor-curve contrast saliency.

This experiment explains the context-length *predictor*, rather than the
forecasting TSFM.  The predictor consumes one fixed-width canonical context and
emits every point of its predicted error-vs-context curve in one forward pass.
For two candidate windows the scalar attribution target is therefore

    S(x) = E_hat(L_long | x, h) - E_hat(L_short | x, h).

Positive signed attribution means that a timestep pushes the predictor toward
believing the long window has higher error; negative attribution favours the
long window.  There are no differently-sized predictor inputs in this
calculation: ``L_short`` and ``L_long`` select two output coordinates.

Both predictor architectures in ``experiments.predict_context_length`` are
supported (PatchTST and bidirectional Mamba), because both expose the same
differentiable forward contract.  A Mamba checkpoint still requires
``mamba-ssm`` in the active server environment.

Outputs per (dataset, short, long) cell:
  * ``results.csv`` — signed, per-block IG mass in the common schema;
  * ``attr_<baseline>.npz`` — raw timestep attribution, scalar contrast and
    completeness error;
  * ``done.json`` — checkpoint/config provenance and convergence metadata.

Integrated gradients are corroborative rather than causal evidence.  Validate
the maps with exp1 perturbations or paired prefix replacements.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from context_interpretability.adapters.base import blocks_for_context, thin_blocks
from context_interpretability.experiments.common import ExperimentData
from context_interpretability.schema import ResultsWriter, cell_done

METHOD = "predictor_contrast_saliency"
LIMITATIONS = (
    "Predictor contrast saliency explains the learned predictor, not direct "
    "TSFM causality. Integrated gradients are baseline-dependent and can be "
    "small under saturation; validate salient blocks with interventions."
)


def _predictor_dir(config: dict, adapter) -> str:
    """Resolve a direct best-checkpoint directory for this label model."""
    cfg = config.get("predictor_contrast_saliency") or {}
    explicit = cfg.get("predictor_dir")
    if isinstance(explicit, dict):
        explicit = explicit.get(adapter.name) or explicit.get(
            getattr(adapter, "family", ""))
    if explicit:
        path = os.path.expanduser(str(explicit))
    else:
        root = str(cfg.get(
            "predictor_root",
            os.environ.get("PREDICTCSL_PREDICTOR_ROOT",
                           "logs/experiments/context_length_predictor")))
        path = os.path.join(root, adapter.name)
    if not (os.path.isfile(os.path.join(path, "best_model.pt"))
            and os.path.isfile(os.path.join(path, "best_config.json"))):
        raise FileNotFoundError(
            f"No predictor best_model.pt + best_config.json under {path!r}. "
            "Set predictor_contrast_saliency.predictor_dir (a string or a "
            "model/family mapping) to the trained predictor directory.")
    return path


def _load_predictor(path: str, device: str):
    """Load the production predictor without importing TSFM inference code."""
    import torch
    from experiments.predict_context_length import build_predictor

    with open(os.path.join(path, "best_config.json")) as f:
        cfg = json.load(f)
    model = build_predictor(cfg, int(cfg["n_windows"]),
                            int(cfg["n_horizons"]))
    state = torch.load(os.path.join(path, "best_model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg


def prepare_predictor_inputs(contexts: np.ndarray,
                             context_length: int) -> np.ndarray:
    """Production-equivalent tail crop/zero-left-pad + instance z-scoring."""
    rows = []
    for context in contexts:
        ctx = np.nan_to_num(np.asarray(context, dtype=np.float32), nan=0.0)
        if len(ctx) >= context_length:
            x = ctx[-context_length:]
        else:
            x = np.concatenate([
                np.zeros(context_length - len(ctx), dtype=np.float32), ctx])
        rows.append((x - float(x.mean())) / (float(x.std()) + 1e-8))
    return np.ascontiguousarray(np.stack(rows), dtype=np.float32)


def _curve_scores(model, x, horizon_idx, training_objective: str):
    curve, _recon, _patches, _mask = model(x.unsqueeze(-1), horizon_idx)
    if training_objective == "classification":
        # Same conversion as stage-3 inference: argmin(-sigmoid(logits)) is the
        # classifier argmax.  The contrast is then a score contrast, not a
        # calibrated error difference; this distinction is written to metadata.
        curve = -curve.sigmoid()
    return curve


def contrast_target(model, x, horizon_idx, short_idx: int, long_idx: int,
                    training_objective: str = "curve"):
    curve = _curve_scores(model, x, horizon_idx, training_objective)
    return curve[:, int(long_idx)] - curve[:, int(short_idx)]


def integrated_gradients_contrast(
    model,
    x: np.ndarray,
    baseline: np.ndarray,
    horizon_idx: int,
    short_idx: int,
    long_idx: int,
    steps: int,
    device: str,
    training_objective: str = "curve",
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return timestep IG, completeness error and S(x), all per sample."""
    import torch

    attrs = np.empty_like(x, dtype=np.float64)
    completeness = np.empty(len(x), dtype=np.float64)
    contrasts = np.empty(len(x), dtype=np.float32)
    batch_size = max(1, int(batch_size))
    alphas = (np.arange(int(steps)) + 0.5) / int(steps)
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        xt = torch.from_numpy(np.ascontiguousarray(x[start:stop])).to(device)
        bt = torch.from_numpy(
            np.ascontiguousarray(baseline[start:stop])).to(device)
        diff = xt - bt
        hidx = torch.full((stop - start,), int(horizon_idx), dtype=torch.long,
                          device=device)
        grad_sum = torch.zeros_like(xt, dtype=torch.float64, device="cpu")
        for alpha in alphas:
            xi = (bt + float(alpha) * diff).detach().requires_grad_(True)
            scalar = contrast_target(
                model, xi, hidx, short_idx, long_idx,
                training_objective=training_objective).sum()
            (grad,) = torch.autograd.grad(scalar, xi)
            grad_sum += grad.detach().to("cpu", torch.float64)
        attr = (diff.detach().to("cpu", torch.float64)
                * grad_sum / int(steps)).numpy()
        with torch.no_grad():
            fx = contrast_target(model, xt, hidx, short_idx, long_idx,
                                 training_objective).cpu().numpy()
            fb = contrast_target(model, bt, hidx, short_idx, long_idx,
                                 training_objective).cpu().numpy()
        attrs[start:stop] = attr
        completeness[start:stop] = np.abs(attr.sum(axis=1) - (fx - fb))
        contrasts[start:stop] = fx
    return attrs, completeness, contrasts


def make_baselines(x: np.ndarray, names: Sequence[str], seed: int,
                   n_random: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    mean = x.mean(axis=1, keepdims=True)
    if "zero" in names:
        out["zero"] = np.zeros_like(x)
    if "context_mean" in names:
        out["context_mean"] = np.broadcast_to(mean, x.shape).copy()
    if "random_sample" in names and len(x) > 1:
        rng = np.random.default_rng(seed)
        for k in range(int(n_random)):
            perm = rng.permutation(len(x))
            perm = np.where(perm == np.arange(len(x)),
                            (perm + 1) % len(x), perm)
            out[f"random_sample_{k}"] = x[perm].copy()
    if not out:
        raise ValueError("predictor contrast saliency needs at least one baseline")
    return out


def _window_pairs(window_grid: Sequence[int], cfg: dict
                  ) -> List[Tuple[int, int]]:
    pairs = cfg.get("window_pairs")
    if pairs:
        resolved = [(int(a), int(b)) for a, b in pairs]
    else:
        short = int(cfg.get("short_window") or window_grid[0])
        long = int(cfg.get("long_window") or window_grid[-1])
        resolved = [(short, long)]
    known = {int(v) for v in window_grid}
    for short, long in resolved:
        if short not in known or long not in known:
            raise ValueError(
                f"window pair ({short}, {long}) not in predictor grid "
                f"{list(window_grid)}")
        if short >= long:
            raise ValueError(f"expected short < long, got ({short}, {long})")
    return resolved


def _signed_block_mass(attr: np.ndarray, blocks, W: int) -> np.ndarray:
    signed = np.stack(
        [attr[:, b.input_slice(W)].sum(axis=1) for b in blocks], axis=1)
    denom = np.abs(attr).sum(axis=1, keepdims=True) + 1e-12
    return signed / denom


def run(adapter, data: ExperimentData, config: dict, out_dir: str,
        run_meta=None, seed: int = 0) -> List[str]:
    cfg = config.get("predictor_contrast_saliency") or {}
    path = _predictor_dir(config, adapter)
    model, pcfg = _load_predictor(path, adapter.device)
    context_length = int(pcfg["context_length"])
    window_grid = [int(v) for v in pcfg["window_grid"]]
    horizon_grid = [int(v) for v in pcfg["horizon_grid"]]
    horizon_idx = int(np.argmin(np.abs(
        np.asarray(horizon_grid) - int(adapter.horizon))))
    selected_horizon = horizon_grid[horizon_idx]
    objective = str(pcfg.get("training_objective", "curve"))
    x = prepare_predictor_inputs(data.contexts, context_length)
    steps = int(cfg.get("steps", 64))
    batch_size = int(cfg.get("batch_size", 16))
    baselines = make_baselines(
        x, cfg.get("baselines", ["zero", "random_sample"]), seed,
        int(cfg.get("n_random_baselines", 3)))
    patch_length = int(getattr(model, "patch_length", 1) or 1)
    blocks = blocks_for_context(context_length, patch_length)
    blocks, thinned = thin_blocks(
        blocks, int(cfg.get("max_blocks", config.get("max_blocks_per_context", 64))))
    if run_meta:
        run_meta.note_samples("exp6_predictor_contrast_saliency", data.n)
        run_meta.note("predictor_contrast_saliency_checkpoint", path)
        if thinned:
            run_meta.note_subsampling(
                "exp6_predictor_contrast_saliency",
                f"{len(blocks)} of {context_length // patch_length} predictor patches")

    cells: List[str] = []
    for short, long in _window_pairs(window_grid, cfg):
        cell = os.path.join(out_dir, data.name,
                            f"h{selected_horizon}_L{short}_to_L{long}")
        cells.append(cell)
        if cell_done(cell):
            continue
        writer = ResultsWriter(cell)
        short_idx, long_idx = window_grid.index(short), window_grid.index(long)
        completeness_meta = {}
        contrast_meta = {}
        for baseline_name, baseline in baselines.items():
            attr, completeness, contrast = integrated_gradients_contrast(
                model, x, baseline, horizon_idx, short_idx, long_idx, steps,
                adapter.device, objective, batch_size=batch_size)
            block_attr = _signed_block_mass(attr, blocks, context_length)
            np.savez_compressed(
                os.path.join(cell, f"attr_{baseline_name}.npz"),
                attributions=attr.astype(np.float32),
                signed_block_attribution=block_attr.astype(np.float32),
                contrast=contrast.astype(np.float32),
                completeness_err=completeness.astype(np.float32),
                lookback_start=np.asarray(
                    [b.lookback_start for b in blocks], dtype=np.int32),
                lookback_end=np.asarray(
                    [b.lookback_end for b in blocks], dtype=np.int32))
            completeness_meta[baseline_name] = float(np.mean(completeness))
            contrast_meta[baseline_name] = float(np.mean(contrast))
            ptype = f"ig/{baseline_name}/L{short}_to_L{long}"
            for bi, block in enumerate(blocks):
                for i, sid in enumerate(data.sample_ids):
                    writer.add(
                        model=f"context_predictor:{adapter.name}",
                        dataset=data.name, sample_id=sid,
                        context_length=context_length,
                        requested_context_length=context_length,
                        horizon=selected_horizon, block_index=block.index,
                        lookback_start=block.lookback_start,
                        lookback_end=block.lookback_end,
                        method=METHOD, perturbation_type=ptype,
                        metric=("predicted_z_error_contrast"
                                if objective == "curve"
                                else "classification_score_contrast"),
                        attribution_score=block_attr[i, bi], seed=seed)
        writer.finalize({
            "predictor_dir": path,
            "predictor_arch": pcfg.get("arch", "patchtst"),
            "training_objective": objective,
            "contrast_semantics": (
                "predicted_z_error(long)-predicted_z_error(short)"
                if objective == "curve" else
                "-sigmoid(logit_long)-(-sigmoid(logit_short))"),
            "short_window": short, "long_window": long,
            "predictor_context_length": context_length,
            "requested_horizon": adapter.horizon,
            "selected_horizon": selected_horizon,
            "steps": steps, "batch_size": batch_size,
            "patch_length": patch_length,
            "thinned": thinned, "limitations": LIMITATIONS,
            "mean_completeness_error": completeness_meta,
            "mean_contrast": contrast_meta,
        })
        print(f"[exp6][{adapter.name}] {data.name}: "
              f"S=E({long})-E({short}), h={selected_horizon}, "
              f"{len(baselines)} baselines done")
    return cells
