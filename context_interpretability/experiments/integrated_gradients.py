"""
Experiment 5 — integrated-gradient temporal sensitivity (spec §8).

Corroborative, NOT the primary causal evidence. The implementation and every
report generated from it must carry the §8.7 limitations (see
``LIMITATIONS`` below, embedded into each cell's done-marker and the figures).

IG for baseline X': (X - X') * ∫_0^1 dF(X' + a(X - X'))/dX da, approximated
with a midpoint Riemann sum over ``steps`` alphas, batched by
``internal_batch_size``. Targets: mean forecast over the horizon (primary) and
loss (supplementary). Baselines: channel-wise context mean (== zero after
instance normalization for mean-centering models), and random samples from the
dataset; a seasonal baseline when the dataset declares a season. Never a single
baseline (spec §8.3).

Completeness check: |sum_j phi_j - (F(X) - F(X'))| reported per baseline, with
a convergence sweep over {16, 32, 64} steps.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from context_interpretability.adapters.base import (
    CapabilityError, InterpretabilityAdapter)
from context_interpretability.experiments.common import (
    CleanCache, ExperimentData, block_grid)
from context_interpretability.schema import ResultsWriter, cell_done

METHOD = "integrated_gradients"

LIMITATIONS = (
    "Low gradient attribution is not proof of causal irrelevance; gradients "
    "may be affected by model saturation; attribution depends on the baseline; "
    "the perturbation experiments remain the principal causal evidence."
)


# ------------------------------------------------------------------------------
#  Core IG computation (torch)
# ------------------------------------------------------------------------------

def _scalar_target(pred, target_kind: str, y_true=None):
    """Reduce (N, H) forecasts to per-sample scalars for attribution."""
    import torch
    if target_kind == "mean_forecast":
        return pred.mean(dim=-1)
    if target_kind.startswith("horizon_step:"):
        h = int(target_kind.split(":", 1)[1])
        return pred[:, h]
    if target_kind == "sum_normalized":
        return (pred / (pred.detach().abs().mean(dim=-1, keepdim=True)
                        + 1e-8)).sum(dim=-1)
    if target_kind == "loss":
        if y_true is None:
            raise ValueError("loss target needs ground truth")
        return (pred - y_true).abs().mean(dim=-1)
    raise ValueError(f"Unknown IG target {target_kind!r}")


def integrated_gradients(adapter: InterpretabilityAdapter, x: np.ndarray,
                         baseline: np.ndarray, steps: int,
                         internal_batch_size: int, target_kind: str,
                         y_true: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-timestep attributions (N, W) + completeness error (N,).

    Midpoint rule over alpha; gradients accumulated in float64 on CPU.
    """
    import torch
    device = adapter.device
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    bt = torch.from_numpy(np.ascontiguousarray(baseline, dtype=np.float32))
    yt = (torch.from_numpy(np.ascontiguousarray(y_true, dtype=np.float32))
          .to(device) if y_true is not None else None)
    diff = xt - bt
    alphas = (np.arange(steps) + 0.5) / steps
    grad_sum = torch.zeros_like(xt, dtype=torch.float64)

    for start in range(0, steps, max(1, internal_batch_size)):
        for a in alphas[start:start + max(1, internal_batch_size)]:
            xi = (bt + float(a) * diff).clone().to(device).requires_grad_(True)
            pred = adapter.forecast_differentiable(xi)
            scalar = _scalar_target(pred, target_kind, yt).sum()
            (grad,) = torch.autograd.grad(scalar, xi)
            grad_sum += grad.detach().to("cpu", torch.float64)

    attr = (diff.to(torch.float64) * grad_sum / steps).numpy()

    # completeness: F(X) - F(X') from two extra forwards (grad graph built —
    # the adapter's differentiable path checks grad_fn — but never backprop'd)
    f_x = _scalar_target(
        adapter.forecast_differentiable(
            xt.to(device).requires_grad_(True)).detach(),
        target_kind, yt)
    f_b = _scalar_target(
        adapter.forecast_differentiable(
            bt.to(device).requires_grad_(True)).detach(),
        target_kind, yt)
    total = (f_x - f_b).float().cpu().numpy()
    completeness_err = np.abs(attr.sum(axis=1) - total)
    return attr, completeness_err


# ------------------------------------------------------------------------------
#  Baselines (spec §8.3)
# ------------------------------------------------------------------------------

def make_baselines(window: np.ndarray, data: ExperimentData, cfg: dict,
                   seed: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    names = cfg.get("baselines",
                    ["context_mean", "zero_after_norm", "random_sample"])
    mean = np.nanmean(window, axis=1, keepdims=True)
    if "context_mean" in names:
        out["context_mean"] = np.broadcast_to(mean, window.shape).copy()
    if "zero_after_norm" in names:
        # instance-normalizing models map the channel-mean constant to zero
        # post-norm — the same array, kept as its own named baseline so the
        # report states what was actually used.
        out["zero_after_norm"] = np.broadcast_to(mean, window.shape).copy()
    if "seasonal" in names and data.season_length > 1:
        m = data.season_length
        shifted = np.roll(window, m, axis=1)
        shifted[:, :m] = mean
        out["seasonal"] = shifted
    if "random_sample" in names:
        rng = np.random.default_rng(seed)
        for k in range(int(cfg.get("n_random_baselines", 3))):
            perm = rng.permutation(window.shape[0])
            perm = np.where(perm == np.arange(len(perm)),
                            (perm + 1) % len(perm), perm)
            out[f"random_sample_{k}"] = window[perm].copy()
    return out


def aggregate_to_blocks(attr: np.ndarray, blocks, W: int) -> np.ndarray:
    """|attribution| summed per block, normalized per sample (spec §8.5).
    Returns (N, n_blocks)."""
    mags = np.stack(
        [np.abs(attr[:, b.input_slice(W)]).sum(axis=1) for b in blocks],
        axis=1)
    denom = np.abs(attr).sum(axis=1, keepdims=True) + 1e-12
    return mags / denom


# ------------------------------------------------------------------------------
#  Runner
# ------------------------------------------------------------------------------

def run(adapter: InterpretabilityAdapter, data: ExperimentData, config: dict,
        out_dir: str, run_meta=None, seed: int = 0) -> List[str]:
    if not adapter.capabilities.supports_integrated_gradients:
        raise CapabilityError(
            f"{adapter.name}: integrated gradients unsupported")

    icfg = config.get("integrated_gradients") or {}
    steps = int(icfg.get("steps", 64))
    conv_steps = sorted(set(int(s) for s in
                            icfg.get("convergence_steps", [16, 32, 64])))
    ibs = int(icfg.get("internal_batch_size", 8))
    target_kind = icfg.get("target", "mean_forecast")
    metric = config.get("primary_metric", "mae")

    cache = CleanCache(adapter, data, metric,
                       cache_dir=os.path.join(out_dir, "clean_cache"))
    pairs = adapter.effective_context_lengths(config["context_lengths"])
    if run_meta:
        run_meta.note_samples("exp5_integrated_gradients", data.n)
    cells: List[str] = []

    for requested, W in pairs:
        cell = os.path.join(out_dir, data.name, f"w{W}")
        cells.append(cell)
        if cell_done(cell):
            continue
        window = data.window(W)
        clean_loss = cache.get(W)[1]
        blocks, thinned, P = block_grid(adapter, W, config)
        baselines = make_baselines(window, data, icfg, seed)
        writer = ResultsWriter(cell)
        cell_meta: dict = {"context_length": W, "steps": steps,
                           "target": target_kind, "limitations": LIMITATIONS,
                           "completeness": {}, "convergence": {}}

        for bl_name, bl in baselines.items():
            # convergence sweep on the primary baseline order (spec §8.4)
            conv = {}
            for s in conv_steps:
                if s == steps:
                    continue
                a_s, ce_s = integrated_gradients(
                    adapter, window, bl, s, ibs, target_kind)
                conv[str(s)] = {
                    "mean_completeness_err": float(np.nanmean(ce_s)),
                    "mean_abs_attr": float(np.nanmean(np.abs(a_s)))}
            attr, comp_err = integrated_gradients(
                adapter, window, bl, steps, ibs, target_kind)
            conv[str(steps)] = {
                "mean_completeness_err": float(np.nanmean(comp_err)),
                "mean_abs_attr": float(np.nanmean(np.abs(attr)))}
            cell_meta["convergence"][bl_name] = conv
            cell_meta["completeness"][bl_name] = float(np.nanmean(comp_err))

            block_attr = aggregate_to_blocks(attr, blocks, W)
            np.savez(os.path.join(cell, f"attr_{bl_name}.npz"),
                     attributions=attr.astype(np.float32),
                     block_attr=block_attr.astype(np.float32),
                     completeness_err=comp_err,
                     block_starts=[b.lookback_start for b in blocks])
            for bi, blk in enumerate(blocks):
                for i, sid in enumerate(data.sample_ids):
                    writer.add(
                        model=adapter.name, dataset=data.name, sample_id=sid,
                        context_length=W, requested_context_length=requested,
                        horizon=adapter.horizon, block_index=blk.index,
                        lookback_start=blk.lookback_start,
                        lookback_end=blk.lookback_end,
                        method=METHOD, perturbation_type=f"ig/{bl_name}",
                        metric=metric, seed=seed,
                        clean_loss=clean_loss[i],
                        attribution_score=block_attr[i, bi],
                    )

        # supplementary loss-based attribution on the first baseline (§8.2)
        if icfg.get("supplementary_loss_target", True) and baselines:
            bl_name, bl = next(iter(baselines.items()))
            attr_l, _ = integrated_gradients(
                adapter, window, bl, steps, ibs, "loss", data.targets)
            np.savez(os.path.join(cell, "attr_loss_target.npz"),
                     attributions=attr_l.astype(np.float32),
                     baseline=bl_name)

        writer.finalize(cell_meta)
        print(f"[exp5][{adapter.name}] {data.name} w{W}: "
              f"{len(baselines)} baselines, completeness "
              f"{ {k: round(v, 5) for k, v in cell_meta['completeness'].items()} }")
    return cells
