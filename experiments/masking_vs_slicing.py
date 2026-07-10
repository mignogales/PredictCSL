"""
Slicing vs. attention-masking comparison (proof-of-concept, transformer TSFMs).

For one TSFM, over the synthetic pool and the standard ``WINDOW_GRID``, compute
two error-vs-context curves and overlay them:

  * **sliced**  — the existing saturation curve: feed the last ``L`` genuine
    timesteps (``forecast_window``).
  * **masked**  — feed the SAME full-window input every time, but attention-mask
    everything older than the last ``L`` timesteps
    (``context_attention_mask``), so only the attention *span* shrinks while
    normalization + positions stay over the full window.

If the two curves coincide, the saturation effect is purely attention span; if
they diverge, the normalization / positional change that slicing also induces is
doing part of the work.

ENV: run Sundial / TimeMoE in the legacy ``transformers==4.40.1`` env
(``TSFM_sundial_patch``); the main env's newer transformers breaks their remote
mask build before any masking runs. Chronos-2 / PatchTST-FM use the main env.

Runs on the SERVER (needs the TSFM + GPU). Start small:

    python -m experiments.masking_vs_slicing --model PatchTST-FM-R1 --n-series 256
    python -m experiments.masking_vs_slicing --model Sundial-Base-128M --n-series 128 \
        --full-window 4096 --batch-size 8

Output (under ``logs/experiments/masking_vs_slicing/<model>/``):
    curves.npz  — grid, sliced/masked per-series MAE matrices + aggregates
    overlay_mae.png, overlay_mae_normalized.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.build_context_length_dataset import (
    WINDOW_GRID,
    _forecast_uniform,
    forecast_window,
    generate_dataset,
    setup_model,
)
from experiments.context_attention_mask import SUPPORTED_FAMILIES, context_attention_mask
from experiments.models_config import CATALOG


def _resolve_model(display: str):
    for m in CATALOG:
        if m.display == display:
            return m
    raise SystemExit(
        f"Unknown --model {display!r}. Choices: {[m.display for m in CATALOG]}")


def forecast_masked(
    family, base, model_id, contexts, full_window, real_lengths,
    horizon, last_timesteps, batch_size, device,
):
    """Full-window forecast with attention restricted to the last ``last_timesteps``.

    Mirrors ``forecast_window``'s per-width bucketing (so timesfm/moirai-style
    recompiles would still see one runner per width), but feeds ``full_window``
    genuine timesteps and installs the attention mask instead of slicing.
    """
    n = contexts.shape[0]
    eff = np.minimum(int(full_window), np.asarray(real_lengths))
    grid = np.asarray(sorted(set(WINDOW_GRID)))
    eff_buck = np.minimum(
        eff, grid[np.clip(np.searchsorted(grid, eff, side="right") - 1, 0, None)])

    out = torch.empty((n, horizon), device=device, dtype=torch.float32)
    for W in np.unique(eff_buck):
        idx = np.flatnonzero(eff_buck == W)
        x_grp = torch.from_numpy(
            np.ascontiguousarray(contexts[idx, -int(W):])).unsqueeze(-1)
        L_vis = min(int(last_timesteps), int(W))
        with context_attention_mask(family, base, L_vis, int(W)):
            med = _forecast_uniform(
                family, base, model_id, x_grp, int(W), horizon, batch_size, device)
        out[torch.as_tensor(idx, device=device, dtype=torch.long)] = med
    return out


def per_series_mae(pred: torch.Tensor, target: np.ndarray) -> np.ndarray:
    """Mean-over-horizon absolute error, per series (numpy, on CPU)."""
    tgt = torch.from_numpy(target).to(pred.device, dtype=torch.float32)
    return (pred - tgt).abs().mean(dim=1).detach().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="display name (models_config)")
    ap.add_argument("--n-series", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--full-window", type=int, default=max(WINDOW_GRID),
                    help="fixed input length fed on every grid point (masked run)")
    ap.add_argument("--windows", type=int, nargs="*", default=None,
                    help="subset of WINDOW_GRID to evaluate")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="logs/experiments/masking_vs_slicing")
    args = ap.parse_args()

    spec = _resolve_model(args.model)
    if spec.family not in SUPPORTED_FAMILIES:
        raise SystemExit(
            f"{spec.display} (family {spec.family}) has no maskable attention. "
            f"Supported: {sorted(SUPPORTED_FAMILIES)}.")

    grid = [w for w in WINDOW_GRID
            if w <= args.full_window and (args.windows is None or w in args.windows)]
    if not grid:
        raise SystemExit("Empty window grid after filtering by --full-window/--windows.")

    print(f"[{spec.display}] generating {args.n_series} synthetic series...")
    contexts, targets, _n_seg, real_lengths = generate_dataset(args.n_series, args.seed)
    targets = targets[:, :args.horizon]

    print(f"[{spec.display}] loading model on {args.device}...")
    base = setup_model(spec.family, spec.model_id, args.device)

    n = args.n_series
    sliced = np.full((n, len(grid)), np.nan, dtype=np.float32)
    masked = np.full((n, len(grid)), np.nan, dtype=np.float32)

    for j, L in enumerate(grid):
        with torch.no_grad():
            pred_s = forecast_window(
                spec.family, base, spec.model_id, contexts, L, real_lengths,
                args.horizon, args.batch_size, args.device)
            pred_m = forecast_masked(
                spec.family, base, spec.model_id, contexts, args.full_window,
                real_lengths, args.horizon, L, args.batch_size, args.device)
        sliced[:, j] = per_series_mae(pred_s, targets)
        masked[:, j] = per_series_mae(pred_m, targets)
        print(f"  L={L:5d}  sliced MAE={np.nanmean(sliced[:, j]):.4f}  "
              f"masked MAE={np.nanmean(masked[:, j]):.4f}")

    out_dir = Path(args.out) / spec.display
    out_dir.mkdir(parents=True, exist_ok=True)

    sliced_mean = np.nanmean(sliced, axis=0)
    masked_mean = np.nanmean(masked, axis=0)
    # Per-series normalization: divide each series' curve by its own value at the
    # largest evaluated window, so heterogeneous scales don't swamp the average.
    denom_s = sliced[:, [-1]]
    denom_m = masked[:, [-1]]
    sliced_norm = np.nanmean(sliced / np.where(denom_s == 0, np.nan, denom_s), axis=0)
    masked_norm = np.nanmean(masked / np.where(denom_m == 0, np.nan, denom_m), axis=0)

    np.savez(
        out_dir / "curves.npz",
        grid=np.array(grid), sliced=sliced, masked=masked,
        sliced_mean=sliced_mean, masked_mean=masked_mean,
        sliced_norm=sliced_norm, masked_norm=masked_norm,
        horizon=args.horizon, full_window=args.full_window)

    for fname, ys, ylabel in [
        ("overlay_mae.png", (sliced_mean, masked_mean), "mean MAE"),
        ("overlay_mae_normalized.png", (sliced_norm, masked_norm),
         "mean MAE (÷ own value at max window)"),
    ]:
        plt.figure(figsize=(7, 4.5))
        plt.plot(grid, ys[0], "o-", label="sliced (feed last L)")
        plt.plot(grid, ys[1], "s--", label="masked (full window, attend last L)")
        plt.xscale("log", base=2)
        plt.xlabel("effective context L (timesteps)")
        plt.ylabel(ylabel)
        plt.title(f"{spec.display} — h={args.horizon}, full={args.full_window}, "
                  f"n={n}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=130)
        plt.close()

    print(f"[{spec.display}] wrote {out_dir}")


if __name__ == "__main__":
    main()
