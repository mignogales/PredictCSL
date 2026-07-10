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
    MAX_HORIZON,
    MAX_WINDOW,
    PERIOD_MAX,
    PERIOD_MIN,
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


def generate_wave_dataset(n_series: int, seed: int, n_components: int,
                          noise_std: float):
    """Controlled multi-frequency wave pool — a clean diagnostic alternative to
    the full synthetic generator.

    Each series is a sum of ``n_components`` sinusoids: component k has a period
    log-uniform in its own band (the ``[PERIOD_MIN, PERIOD_MAX]`` range split into
    ``n_components`` log-bands, so the components sit at distinct scales — e.g. one
    short, one medium, one long), a random amplitude and phase, plus optional iid
    noise. No padding (``real_len == MAX_WINDOW``). Returns the same 4-tuple shape
    as ``generate_dataset`` so the forecast machinery is reused unchanged.
    """
    total_length = MAX_WINDOW + MAX_HORIZON
    t = np.arange(total_length, dtype=np.float64)
    contexts = np.empty((n_series, MAX_WINDOW), dtype=np.float32)
    targets = np.empty((n_series, MAX_HORIZON), dtype=np.float32)

    # Log-spaced period bands so the K components live at separated scales.
    edges = np.geomspace(PERIOD_MIN, PERIOD_MAX, n_components + 1)
    for i in range(n_series):
        rng = np.random.RandomState(seed + i)
        series = np.zeros(total_length, dtype=np.float64)
        for k in range(n_components):
            period = np.exp(rng.uniform(np.log(edges[k]), np.log(edges[k + 1])))
            amp = rng.uniform(0.5, 1.5)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            series += amp * np.sin(2.0 * np.pi * t / period + phase)
        if noise_std > 0:
            series += rng.normal(0.0, noise_std, size=total_length)
        contexts[i] = series[:MAX_WINDOW]
        targets[i] = series[MAX_WINDOW:MAX_WINDOW + MAX_HORIZON]

    n_segments = np.ones((n_series,), dtype=np.int32)
    real_lengths = np.full((n_series,), MAX_WINDOW, dtype=np.int32)
    return contexts, targets, n_segments, real_lengths


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


# Per-family full-window caps (mirror the v5 skip guards) so one sweep can feed
# each model its largest safe context without a manual --full-window per model.
FAMILY_MAX_CONTEXT = {"sundial": 2880, "toto": 4096, "patchtst_fm": 8192,
                      "chronos2": 8192}
TIMEMOE_MAX_TOTAL = 4096                      # context + horizon must not exceed this
MOIRAI_MAX_TOTAL = 8192                       # max_seq_len(512) x patch(16); ctx+horizon

# Env split: these two need the legacy transformers==4.40.1 env; the rest run in
# the main env. Used by the 'main' / 'legacy' group tokens.
MAIN_FAMILIES = {"chronos2", "chronos_bolt", "moirai", "toto"}
LEGACY_FAMILIES = {"sundial", "timemoe"}


def family_full_window(family: str, horizon: int, max_grid: int):
    """Largest grid window this family can be fed as the full input."""
    cap = max_grid
    if family == "timemoe":
        cap = min(cap, TIMEMOE_MAX_TOTAL - int(horizon))
    elif family == "moirai":
        cap = min(cap, MOIRAI_MAX_TOTAL - int(horizon))
    elif family in FAMILY_MAX_CONTEXT:
        cap = min(cap, FAMILY_MAX_CONTEXT[family])
    grid = [w for w in sorted(set(WINDOW_GRID)) if w <= cap]
    return grid[-1] if grid else None


def resolve_models(tokens):
    """Expand model/group tokens to ModelSpecs (maskable run-set only, deduped).

    Tokens: 'all' (every maskable run-set model except the leaky patchtst_fm),
    'main' (main-env families), 'legacy' (Sundial/TimeMoE), or a display name."""
    runset = [m for m in CATALOG if m.run and m.family in SUPPORTED_FAMILIES]
    out = []
    for tok in tokens:
        t = str(tok).lower()
        if t == "all":
            out += [m for m in runset if m.family != "patchtst_fm"]
        elif t == "main":
            out += [m for m in runset if m.family in MAIN_FAMILIES]
        elif t == "legacy":
            out += [m for m in runset if m.family in LEGACY_FAMILIES]
        else:
            match = [m for m in CATALOG if m.display == tok]
            if not match:
                raise SystemExit(
                    f"Unknown model/group {tok!r}. Groups: all/main/legacy; or a "
                    f"display name in {[m.display for m in runset]}.")
            out += match
    seen, res = set(), []
    for m in out:
        if m.display not in seen and m.family in SUPPORTED_FAMILIES:
            seen.add(m.display)
            res.append(m)
    return res


def _gen_signal(token, args):
    """Materialize one signal's pool. Targets are full MAX_HORIZON width (sliced
    to each horizon later), so the same data is reused across horizons."""
    tok = token.lower()
    if tok == "synthetic":
        c, t, _ns, rl = generate_dataset(args.n_series, args.seed)
        return c, t, rl, "synthetic"
    if tok == "sine":
        n_comp = 1
    elif tok.startswith("wave") and tok[4:].isdigit():
        n_comp = int(tok[4:])
    else:
        raise SystemExit(f"Unknown --signals token {token!r} "
                         "(use 'synthetic', 'sine', or 'wave<N>').")
    c, t, _ns, rl = generate_wave_dataset(
        args.n_series, args.seed, n_comp, args.wave_noise)
    return c, t, rl, f"wave{n_comp}"


def run_model(spec, args) -> None:
    """One model: every horizon x every signal -> per-signal overlays + a combined
    overview per horizon. Output tree: <Model>/h<H>/<signal>/."""
    print(f"[{spec.display}] loading on {args.device}...")
    base = setup_model(spec.family, spec.model_id, args.device)
    n = args.n_series
    # Generate each signal pool once (horizon-independent) and reuse.
    signal_data = {tok: _gen_signal(tok, args) for tok in args.signals}

    for H in args.horizons:
        full_window = family_full_window(spec.family, H, args.full_window)
        if full_window is None:
            print(f"[{spec.display}] h={H}: SKIP (no servable window under caps).")
            continue
        grid = [w for w in WINDOW_GRID
                if w <= full_window and (args.windows is None or w in args.windows)]
        if not grid:
            print(f"[{spec.display}] h={H}: SKIP (empty window grid).")
            continue
        print(f"[{spec.display}] h={H}  full_window={full_window}  grid={grid}")

        overview = []                 # (sig_tag, sliced_mean, masked_mean)
        for token in args.signals:
            contexts, targets_full, real_lengths, sig_tag = signal_data[token]
            targets = targets_full[:, :H]

            sliced = np.full((n, len(grid)), np.nan, dtype=np.float32)
            masked = np.full((n, len(grid)), np.nan, dtype=np.float32)
            for j, L in enumerate(grid):
                with torch.no_grad():
                    pred_s = forecast_window(
                        spec.family, base, spec.model_id, contexts, L, real_lengths,
                        H, args.batch_size, args.device)
                    pred_m = forecast_masked(
                        spec.family, base, spec.model_id, contexts, full_window,
                        real_lengths, H, L, args.batch_size, args.device)
                sliced[:, j] = per_series_mae(pred_s, targets)
                masked[:, j] = per_series_mae(pred_m, targets)
                print(f"  [h{H}/{sig_tag}] L={L:5d}  "
                      f"sliced={np.nanmean(sliced[:, j]):.4f}  "
                      f"masked={np.nanmean(masked[:, j]):.4f}")

            out_dir = Path(args.out) / spec.display / f"h{H}" / sig_tag
            out_dir.mkdir(parents=True, exist_ok=True)
            sliced_mean = np.nanmean(sliced, axis=0)
            masked_mean = np.nanmean(masked, axis=0)
            # Per-series normalization by each series' own value at the largest window.
            denom_s = sliced[:, [-1]]
            denom_m = masked[:, [-1]]
            sliced_norm = np.nanmean(sliced / np.where(denom_s == 0, np.nan, denom_s), axis=0)
            masked_norm = np.nanmean(masked / np.where(denom_m == 0, np.nan, denom_m), axis=0)
            np.savez(out_dir / "curves.npz",
                     grid=np.array(grid), sliced=sliced, masked=masked,
                     sliced_mean=sliced_mean, masked_mean=masked_mean,
                     sliced_norm=sliced_norm, masked_norm=masked_norm,
                     horizon=H, full_window=full_window)

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
                plt.title(f"{spec.display} [{sig_tag}] — h={H}, "
                          f"full={full_window}, n={n}")
                plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(out_dir / fname, dpi=130); plt.close()
            print(f"[{spec.display}] wrote {out_dir}")
            overview.append((sig_tag, sliced_mean, masked_mean))

        # Combined overview per horizon: one subplot per signal.
        if len(overview) > 1:
            k = len(overview)
            fig, axes = plt.subplots(1, k, figsize=(5 * k, 4.2), squeeze=False)
            for ax, (sig_tag, s_mean, m_mean) in zip(axes[0], overview):
                ax.plot(grid, s_mean, "o-", label="sliced")
                ax.plot(grid, m_mean, "s--", label="masked")
                ax.set_xscale("log", base=2)
                ax.set_xlabel("context L")
                ax.set_title(sig_tag)
                ax.grid(True, alpha=0.3)
            axes[0][0].set_ylabel("mean MAE")
            axes[0][0].legend(fontsize=8)
            fig.suptitle(f"{spec.display} — masked vs sliced by signal "
                         f"(h={H}, full={full_window}, n={n})", fontweight="bold")
            fig.tight_layout()
            ov_path = Path(args.out) / spec.display / f"h{H}" / "overview_by_signal.png"
            fig.savefig(ov_path, dpi=130)
            plt.close(fig)
            print(f"[{spec.display}] wrote {ov_path}")

    del base
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Masked-vs-sliced diagnostic across models x signals.")
    ap.add_argument("--models", nargs="+", default=["all"],
                    help="display names or group tokens: 'all' (every maskable "
                         "model except the leaky patchtst_fm), 'main' (main-env "
                         "families), 'legacy' (Sundial/TimeMoE, need the legacy "
                         "transformers==4.40.1 env).")
    ap.add_argument("--model", default=None,
                    help="convenience alias for a single --models entry.")
    ap.add_argument("--n-series", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizons", type=int, nargs="+", default=[64, 256, 1024],
                    help=f"forecast horizons; each gets its own h<H>/ folder "
                         f"(must be <= MAX_HORIZON={MAX_HORIZON}).")
    ap.add_argument("--full-window", type=int, default=max(WINDOW_GRID),
                    help="upper bound on the full input length (capped per family).")
    ap.add_argument("--windows", type=int, nargs="*", default=None,
                    help="subset of WINDOW_GRID to evaluate")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="logs/experiments/masking_vs_slicing")
    ap.add_argument("--signals", nargs="+", default=["synthetic", "wave1", "wave3"],
                    help="signal types to evaluate; EACH gets its own overlay graph, "
                         "plus a combined overview across signals. Tokens: 'synthetic' "
                         "(full generator), 'sine' / 'wave<N>' (N sinusoid components).")
    ap.add_argument("--wave-noise", type=float, default=0.0,
                    help="iid noise std added to wave signals.")
    args = ap.parse_args()

    bad = [h for h in args.horizons if h > MAX_HORIZON or h < 1]
    if bad:
        raise SystemExit(f"--horizons {bad} out of range (1..{MAX_HORIZON}).")

    specs = resolve_models([args.model] if args.model else args.models)
    if not specs:
        raise SystemExit("No maskable models selected.")
    print(f"Sweep: {len(specs)} model(s) x {len(args.signals)} signal(s): "
          f"{[m.display for m in specs]}")

    done, failed = [], []
    for spec in specs:
        try:
            run_model(spec, args)
            done.append(spec.display)
        except Exception as exc:                       # noqa: BLE001 - isolate models
            failed.append((spec.display, repr(exc)))
            print(f"[{spec.display}] FAILED: {exc}")
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    print(f"\nDone: {done}")
    if failed:
        print("Failed:")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
