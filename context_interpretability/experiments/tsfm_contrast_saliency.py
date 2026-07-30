"""TSFM loss-contrast saliency across two input context lengths.

For one forecasting model, future target y and nested tail windows, explain

    S(x, y) = loss(f(x[-L_long:]), y) - loss(f(x[-L_short:]), y).

Both forecasts see the same future target.  The differentiable input is the
long window; the short branch consumes its trailing ``L_short`` values.  Its
gradient therefore lands naturally in the shared ``L_long`` coordinate system:

* blocks older than L_short affect only the long branch;
* blocks inside the shared suffix contain the difference between how the long
  and short executions process the same observations.

Unlike predictor-contrast saliency (exp6), this experiment directly explains
the forecasting TSFM's realised MAE/MSE contrast. It requires the adapter's
``forecast_differentiable`` path. At present that is implemented only for
PatchTST-FM; unsupported families are skipped explicitly by the runner.
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np

from context_interpretability.adapters.base import CapabilityError
from context_interpretability.experiments.common import (
    ExperimentData, block_grid)
from context_interpretability.experiments.integrated_gradients import (
    make_baselines)
from context_interpretability.schema import ResultsWriter, cell_done

METHOD = "tsfm_loss_contrast_saliency"
LIMITATIONS = (
    "The map explains a realised loss contrast and therefore depends on the "
    "future target, loss (MAE/MSE), and IG baseline. Low gradients are not "
    "proof of irrelevance. Different-length branches also change model "
    "normalization, positions and padding; use exp7 to decompose those effects."
)


def differentiable_loss(pred, target, metric: str):
    if metric == "mae":
        return (pred - target).abs().mean(dim=-1)
    if metric == "mse":
        return (pred - target).square().mean(dim=-1)
    raise ValueError(f"TSFM contrast saliency supports mae|mse, got {metric!r}")


def loss_contrast(adapter, long_input, target, short_length: int,
                  metric: str):
    """Per-sample E_long-E_short on one common long-input tensor."""
    long_pred = adapter.forecast_differentiable(long_input)
    short_pred = adapter.forecast_differentiable(
        long_input[:, -int(short_length):])
    long_loss = differentiable_loss(long_pred, target, metric)
    short_loss = differentiable_loss(short_pred, target, metric)
    return long_loss - short_loss, long_loss, short_loss


def integrated_gradients_loss_contrast(
    adapter,
    x_long: np.ndarray,
    baseline_long: np.ndarray,
    targets: np.ndarray,
    short_length: int,
    metric: str,
    steps: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raw IG, completeness error, S(x), E_long and E_short per sample."""
    import torch

    n = len(x_long)
    attrs = np.empty_like(x_long, dtype=np.float64)
    completeness = np.empty(n, dtype=np.float64)
    contrasts = np.empty(n, dtype=np.float32)
    long_losses = np.empty(n, dtype=np.float32)
    short_losses = np.empty(n, dtype=np.float32)
    alphas = (np.arange(int(steps)) + 0.5) / int(steps)
    batch_size = max(1, int(batch_size))

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        xt = torch.from_numpy(np.ascontiguousarray(
            x_long[start:stop], dtype=np.float32)).to(adapter.device)
        bt = torch.from_numpy(np.ascontiguousarray(
            baseline_long[start:stop], dtype=np.float32)).to(adapter.device)
        yt = torch.from_numpy(np.ascontiguousarray(
            targets[start:stop], dtype=np.float32)).to(adapter.device)
        diff = xt - bt
        grad_sum = torch.zeros_like(xt, dtype=torch.float64, device="cpu")

        for alpha in alphas:
            xi = (bt + float(alpha) * diff).detach().requires_grad_(True)
            contrast, _el, _es = loss_contrast(
                adapter, xi, yt, short_length, metric)
            (grad,) = torch.autograd.grad(contrast.sum(), xi)
            grad_sum += grad.detach().to("cpu", torch.float64)

        attr = (diff.detach().to("cpu", torch.float64)
                * grad_sum / int(steps)).numpy()
        with torch.no_grad():
            sx, el, es = loss_contrast(
                adapter, xt, yt, short_length, metric)
            sb, _bel, _bes = loss_contrast(
                adapter, bt, yt, short_length, metric)
        attrs[start:stop] = attr
        completeness[start:stop] = np.abs(
            attr.sum(axis=1) - (sx - sb).cpu().numpy())
        contrasts[start:stop] = sx.cpu().numpy()
        long_losses[start:stop] = el.cpu().numpy()
        short_losses[start:stop] = es.cpu().numpy()
    return attrs, completeness, contrasts, long_losses, short_losses


def _window_pairs(effective: Sequence[int], cfg: dict
                  ) -> List[Tuple[int, int]]:
    explicit = cfg.get("window_pairs")
    if explicit:
        pairs = [(int(short), int(long)) for short, long in explicit]
    else:
        pairs = [(int(cfg.get("short_window") or effective[0]),
                  int(cfg.get("long_window") or effective[-1]))]
    known = set(int(v) for v in effective)
    for short, long in pairs:
        if short not in known or long not in known:
            raise ValueError(
                f"TSFM contrast pair ({short}, {long}) is not in the effective "
                f"grid {list(effective)}")
        if short >= long:
            raise ValueError(f"expected short < long, got ({short}, {long})")
    return pairs


def _signed_block_mass(attr: np.ndarray, blocks, W: int) -> np.ndarray:
    signed = np.stack(
        [attr[:, block.input_slice(W)].sum(axis=1) for block in blocks], axis=1)
    return signed / (np.abs(attr).sum(axis=1, keepdims=True) + 1e-12)


def run(adapter, data: ExperimentData, config: dict, out_dir: str,
        run_meta=None, seed: int = 0) -> List[str]:
    if not adapter.capabilities.supports_integrated_gradients:
        raise CapabilityError(
            f"{adapter.name}: TSFM loss-contrast saliency needs a "
            "differentiable forecast path")
    cfg = config.get("tsfm_contrast_saliency") or {}
    metrics = list(dict.fromkeys(
        str(m).lower() for m in cfg.get("metrics", ["mae", "mse"])))
    bad = set(metrics) - {"mae", "mse"}
    if bad:
        raise ValueError(f"TSFM contrast metrics must be mae|mse: {sorted(bad)}")
    steps = int(cfg.get("steps", 64))
    batch_size = int(cfg.get("batch_size", 8))
    pairs = adapter.effective_context_lengths(config["context_lengths"])
    effective = [length for _requested, length in pairs]
    cells: List[str] = []

    if run_meta:
        run_meta.note_samples("exp8_tsfm_contrast_saliency", data.n)
    for short, long in _window_pairs(effective, cfg):
        long_window = data.window(long)
        blocks, thinned, patch_length = block_grid(adapter, long, config)
        baselines = make_baselines(long_window, data, cfg, seed)
        for metric in metrics:
            cell = os.path.join(
                out_dir, data.name, f"{metric}_L{short}_to_L{long}")
            cells.append(cell)
            if cell_done(cell):
                continue
            writer = ResultsWriter(cell)
            completeness_meta = {}
            contrast_meta = {}
            for baseline_name, baseline in baselines.items():
                attr, comp, contrast, long_loss, short_loss = (
                    integrated_gradients_loss_contrast(
                        adapter, long_window, baseline, data.targets,
                        short, metric, steps, batch_size))
                block_attr = _signed_block_mass(attr, blocks, long)
                np.savez_compressed(
                    os.path.join(cell, f"attr_{baseline_name}.npz"),
                    attributions=attr.astype(np.float32),
                    signed_block_attribution=block_attr.astype(np.float32),
                    completeness_err=comp.astype(np.float32),
                    loss_contrast=contrast.astype(np.float32),
                    long_loss=long_loss.astype(np.float32),
                    short_loss=short_loss.astype(np.float32),
                    lookback_start=np.asarray(
                        [b.lookback_start for b in blocks], dtype=np.int32),
                    lookback_end=np.asarray(
                        [b.lookback_end for b in blocks], dtype=np.int32))
                completeness_meta[baseline_name] = float(np.mean(comp))
                contrast_meta[baseline_name] = float(np.mean(contrast))
                ptype = f"ig/{baseline_name}/L{short}_to_L{long}"
                for bi, block in enumerate(blocks):
                    for i, sid in enumerate(data.sample_ids):
                        writer.add(
                            model=adapter.name, dataset=data.name,
                            sample_id=sid, context_length=long,
                            requested_context_length=long,
                            horizon=adapter.horizon, block_index=block.index,
                            lookback_start=block.lookback_start,
                            lookback_end=block.lookback_end,
                            method=METHOD, perturbation_type=ptype,
                            metric=metric, seed=seed,
                            clean_loss=short_loss[i],
                            intervened_loss=long_loss[i],
                            loss_delta=contrast[i],
                            attribution_score=block_attr[i, bi])
            writer.finalize({
                "short_window": short, "long_window": long,
                "target": f"{metric}(forecast_long,y)-{metric}(forecast_short,y)",
                "steps": steps, "batch_size": batch_size,
                "block_length": patch_length, "thinned": thinned,
                "mean_loss_contrast": contrast_meta,
                "mean_completeness_error": completeness_meta,
                "limitations": LIMITATIONS,
            })
            print(f"[exp8][{adapter.name}] {data.name}: "
                  f"{metric.upper()}(L={long})-{metric.upper()}(L={short}), "
                  f"{len(baselines)} baselines done")
    return cells

