#!/usr/bin/env python
"""
Summarize the CSL predictor's compute footprint as a percentage of each TSFM.

Core question: when we forecast with a foundation model, how expensive is it to
*also* run the context-length (CSL) predictor that recommends the window? The
predictor is the Patch-Transformer trained in ``predict_context_length.py``; it
runs **once per series** (a single CLS-token forward) and emits the whole
error-vs-context curve. The "full model" is each labeled TSFM running one
forecast at its largest context (the ``full_window`` strategy in stage 4).

This script reports, per TSFM:

    predictor_MACs / tsfm_full_window_MACs  x 100      (a percentage)

so you can quote a line like "the predictor costs ~2% of a Moirai2 forward
pass" (cf. the PREDICTCSL_CHEAP_PREDICTOR note in predict_context_length.py).

Both sides use the **same** MAC proxy as stage 4
(``theoretical_flops`` in compare_window_strategies_gifteval.py), so the ratio
is internally consistent with the FLOPs columns the pipeline already emits. It
is a proxy (ignores embeddings/norms/heads kernel constants), comparable as a
ratio rather than an exact instruction count.

Nothing here needs a GPU or torch — it reads ``best_config.json`` (the predictor
architecture chosen by the random search) and computes MACs analytically. If the
config file is absent (the real runs live on the server), pass the architecture
on the command line or accept the printed defaults.

Usage (run anywhere the source tree lives)::

    python -m experiments.summarize_predictor_overhead
    python -m experiments.summarize_predictor_overhead \
        --predictor-root logs/experiments/context_length_predictor
    python -m experiments.summarize_predictor_overhead --horizon 96 --csv overhead.csv

Each labeled model trains its OWN predictor, so the per-model best_config.json
(and hence the predictor GMAC) differs; the script loads each separately from
<predictor-root>/<display>/best_config.json. Pass --best-config to force a single
shared config for every model instead.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

# Reuse the EXACT FLOPs proxy + per-family architecture table that stage 4 uses,
# so the predictor/TSFM ratio matches the pipeline's flops_savings numbers.
from experiments.compare_window_strategies_gifteval import (
    DEFAULT_PATCH_SIZES,
    MODEL_ARCH,
    _layer_macs,
    _n_patches,
    theoretical_flops,
)

# Grid top from build_context_length_dataset.WINDOW_GRID (the largest context any
# TSFM is ever asked for); each family's full_window is capped by arch.max_window.
GRID_MAX_WINDOW = 8192

# Labeled models, mirroring build_context_length_dataset.MODELS as (display, family).
# Each labeling model trains its OWN predictor (independent random search), whose
# best_config.json lives at <predictor_root>/<display>/best_config.json — keyed on
# the *display* name (the dataset-dir basename used as run_label). So the predictor
# architecture (and hence its GMAC) differs per model; we load each one separately.
# The display string is also passed to theoretical_flops, whose infer_model_family
# keys on substrings of it.
MODELS: List[tuple] = [
    ("Chronos2-Small",    "chronos2"),
    ("Chronos2-Synth",    "chronos2"),
    ("ChronosBolt-Small", "chronos_bolt"),
    ("ChronosBolt-Base",  "chronos_bolt"),
    ("Moirai2-Small",     "moirai"),
    ("TimesFM2.5-200M",   "timesfm"),
    ("PatchTST-FM-R1",    "patchtst_fm"),
    ("Sundial-Base-128M", "sundial"),
    ("TimeMoE-200M",      "timemoe"),
    ("Toto-2.0-313m",     "toto"),
    ("FlowState-R1",      "flowstate"),
    ("TiRex",             "tirex"),
]

# Fallback predictor architecture if best_config.json is not reachable locally.
# Mirrors a mid-range pick from HP_SPACE (predict_context_length.py); override
# with --patch-length / --d-model / --num-layers or point --best-config at the
# real artifact.
DEFAULT_PREDICTOR = {
    "context_length":    8192,
    "patch_length":      64,
    "d_model":           128,
    "num_hidden_layers": 4,
    "n_windows":         18,   # len(WINDOW_GRID)
    "n_horizons":        6,    # len(HORIZON_GRID)
}


# ==============================================================================
#  PREDICTOR COST MODEL  (same MAC accounting style as theoretical_flops)
# ==============================================================================

def predictor_macs(cfg: Dict) -> float:
    """Per-series inference MACs for the CSL Patch-Transformer.

    Inference path (forward() in predict_context_length.py, eval mode, B=1):
        patchify -> patch_embed -> prepend CLS(+horizon embed) -> encoder
        -> LayerNorm -> curve_head on the CLS token.

    The recon head is auxiliary (training only) and is *excluded* — at deploy
    time you only read the curve from the CLS token. The encoder term uses the
    same ``_layer_macs`` helper as the TSFM proxy (unified stack, dense FFN with
    f = 4*d_model, no cross-attention), so the two sides are commensurable.
    """
    P = int(cfg["patch_length"])
    d = int(cfg["d_model"])
    n_layers = int(cfg["num_hidden_layers"])
    n_patches = int(cfg["context_length"]) // P
    n_windows = int(cfg["n_windows"])

    f = 4 * d                       # dim_feedforward = d_model * 4
    L = n_patches + 1               # patches + CLS token

    patch_embed = n_patches * P * d             # Linear(P -> d) over N patches
    encoder = n_layers * _layer_macs(L, d, f)   # dense transformer stack
    curve_head = d * d + d * n_windows          # MLP on the single CLS token
    return float(patch_embed + encoder + curve_head)


def predictor_params(cfg: Dict) -> int:
    """Analytic parameter count of the predictor (for context, not the headline).

    Matches the module layout in PatchTSTContextLength: patch embed, CLS token,
    positional + horizon embeddings, ``n_layers`` of nn.TransformerEncoderLayer
    (norm_first, dim_ff = 4*d), final LayerNorm, and the two heads.
    """
    P = int(cfg["patch_length"])
    d = int(cfg["d_model"])
    n_layers = int(cfg["num_hidden_layers"])
    n_patches = int(cfg["context_length"]) // P
    n_windows = int(cfg["n_windows"])
    n_horizons = int(cfg.get("n_horizons", DEFAULT_PREDICTOR["n_horizons"]))

    f = 4 * d
    patch_embed = P * d + d
    cls_token = d
    pos_embed = (n_patches + 1) * d
    horizon_embed = n_horizons * d
    # nn.TransformerEncoderLayer: MHA in/out proj + 2 FFN linears + 2 LayerNorms.
    per_layer = (4 * d * d + 4 * d) + (2 * (d * f) + f + d) + (2 * 2 * d)
    encoder = n_layers * per_layer
    final_norm = 2 * d
    curve_head = (d * d + d) + (d * n_windows + n_windows)
    recon_head = d * P + P
    return int(patch_embed + cls_token + pos_embed + horizon_embed
               + encoder + final_norm + curve_head + recon_head)


# ==============================================================================
#  CONFIG LOADING
# ==============================================================================

def load_predictor_config(path: Optional[str], overrides: Dict) -> Dict:
    """Read best_config.json if present; apply any CLI overrides on top."""
    cfg = dict(DEFAULT_PREDICTOR)
    source = "built-in defaults (no best_config.json found)"
    if path and os.path.isfile(path):
        with open(path) as fh:
            raw = json.load(fh)
        for k in ("context_length", "patch_length", "d_model",
                  "num_hidden_layers", "n_windows", "n_horizons"):
            if raw.get(k) is not None:
                cfg[k] = raw[k]
        # n_windows can also be derived from a stored window_grid.
        if raw.get("window_grid") is not None:
            cfg["n_windows"] = len(raw["window_grid"])
        source = path
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    if cfg["context_length"] % cfg["patch_length"] != 0:
        raise ValueError(
            f"context_length={cfg['context_length']} not divisible by "
            f"patch_length={cfg['patch_length']}.")
    return cfg, source


def _resolve_predictor_root(arg_root: Optional[str]) -> str:
    return (arg_root
            or os.environ.get("PREDICTCSL_PREDICTOR_ROOT",
                              "logs/experiments/context_length_predictor"))


# ==============================================================================
#  REPORT
# ==============================================================================

def build_rows(
    predictor_root: str,
    forced_config: Optional[str],
    overrides: Dict,
    horizon: int,
    models: List[tuple],
) -> List[Dict]:
    """One row per labeled model, each using ITS OWN predictor best_config.json.

    Each labeling model trains a separate predictor (independent random search),
    so the architecture — and therefore the predictor GMAC — differs per model.
    Config is resolved at <predictor_root>/<display>/best_config.json unless
    --best-config forces a single shared config for every row.
    """
    rows: List[Dict] = []
    for display, family in models:
        if family not in MODEL_ARCH:
            continue
        path = forced_config or os.path.join(
            predictor_root, display, "best_config.json")
        cfg, _src = load_predictor_config(path, overrides)
        found = bool(forced_config) or os.path.isfile(path)

        pred = predictor_macs(cfg)
        arch = MODEL_ARCH[family]
        full_window = min(arch.max_window, GRID_MAX_WINDOW)
        half_window = max(1, full_window // 2)
        quarter_window = max(1, full_window // 4)
        tsfm_full = theoretical_flops(display, full_window, horizon, DEFAULT_PATCH_SIZES)
        tsfm_half = theoretical_flops(display, half_window, horizon, DEFAULT_PATCH_SIZES)
        tsfm_quarter = theoretical_flops(display, quarter_window, horizon, DEFAULT_PATCH_SIZES)
        rows.append({
            "model": display,
            "family": family,
            "config_found": found,
            "pred_arch": f"{cfg['patch_length']}/{cfg['d_model']}/{cfg['num_hidden_layers']}",
            "pred_params_m": predictor_params(cfg) / 1e6,
            "full_window": full_window,
            "half_window": half_window,
            "quarter_window": quarter_window,
            "tsfm_gmac_full": tsfm_full / 1e9,
            "tsfm_gmac_half": tsfm_half / 1e9,
            "tsfm_gmac_quarter": tsfm_quarter / 1e9,
            "predictor_gmac": pred / 1e9,
            "pct_of_full": 100.0 * pred / tsfm_full if tsfm_full > 0 else float("nan"),
        })
    rows.sort(key=lambda r: r["pct_of_full"], reverse=True)
    return rows


def print_report(predictor_root: str, forced_config: Optional[str],
                 horizon: int, rows: List[Dict]) -> None:
    print("=" * 100)
    print("CSL PREDICTOR COMPUTE FOOTPRINT  (per-series, vs full-window TSFM)")
    print("=" * 100)
    src = forced_config or f"{predictor_root}/<model>/best_config.json"
    print(f"  predictor config source : {src}")
    print(f"  TSFM forecast horizon   : {horizon}  "
          f"(predictor cost is horizon-independent)")
    print(f"  predictor arch column   : patch/d_model/layers  (* = config not "
          f"found, using defaults)")
    print("-" * 100)
    print(f"  {'TSFM':<20}{'pred(p/d/L)':>13}{'full_w':>8}{'GMAC@full':>11}"
          f"{'GMAC@50%':>10}{'GMAC@25%':>10}{'predGMAC':>10}{'% of full':>11}")
    print("-" * 100)
    for r in rows:
        name = r["model"] + ("" if r["config_found"] else " *")
        print(f"  {name:<20}{r['pred_arch']:>13}{r['full_window']:>8}"
              f"{r['tsfm_gmac_full']:>11.2f}{r['tsfm_gmac_half']:>10.2f}"
              f"{r['tsfm_gmac_quarter']:>10.2f}{r['predictor_gmac']:>10.4f}"
              f"{r['pct_of_full']:>10.3f}%")
    print("-" * 100)
    pct_vals = [r["pct_of_full"] for r in rows]
    print(f"  range across models     : {min(pct_vals):.3f}%  ..  {max(pct_vals):.3f}%")
    print(f"  mean                    : {sum(pct_vals) / len(pct_vals):.3f}%")
    n_missing = sum(1 for r in rows if not r["config_found"])
    if n_missing:
        print(f"  ({n_missing}/{len(rows)} models used the fallback default config "
              f"— marked with *)")
    print("=" * 100)
    print("  Note: MAC proxy (same as stage-4 theoretical_flops); the predictor")
    print("  runs once per series and returns the whole curve, while the TSFM cost")
    print("  is one forecast. SSM/recurrent families (flowstate, tirex) are LINEAR")
    print("  in context, so their proxy GMAC is a loose upper bound -> the % shown")
    print("  for them understates the predictor's true relative cost.")


def write_csv(path: str, horizon: int, rows: List[Dict]) -> None:
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "family", "config_found", "pred_arch_p_d_L",
                    "pred_params_m", "full_window", "half_window",
                    "quarter_window", "horizon", "tsfm_gmac_full",
                    "tsfm_gmac_half", "tsfm_gmac_quarter", "predictor_gmac",
                    "pct_of_full"])
        for r in rows:
            w.writerow([r["model"], r["family"], r["config_found"],
                        r["pred_arch"], f"{r['pred_params_m']:.4f}",
                        r["full_window"], r["half_window"], r["quarter_window"],
                        horizon, f"{r['tsfm_gmac_full']:.6f}",
                        f"{r['tsfm_gmac_half']:.6f}", f"{r['tsfm_gmac_quarter']:.6f}",
                        f"{r['predictor_gmac']:.6f}", f"{r['pct_of_full']:.6f}"])
    print(f"\n  Wrote {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize CSL predictor cost as a % of each full TSFM.")
    p.add_argument("--predictor-root", type=str, default=None,
                   help="Root holding per-model predictors "
                        "(<root>/<display>/best_config.json). Default: "
                        "PREDICTCSL_PREDICTOR_ROOT or "
                        "logs/experiments/context_length_predictor.")
    p.add_argument("--best-config", type=str, default=None,
                   help="Force a SINGLE best_config.json for every model "
                        "(overrides per-model lookup under --predictor-root).")
    p.add_argument("--horizon", type=int, default=48,
                   help="Forecast horizon for the TSFM cost (default 48).")
    p.add_argument("--models", nargs="*", default=None,
                   help="Display names to include (default: all in MODELS).")
    p.add_argument("--patch-length", type=int, default=None,
                   help="Override predictor patch_length.")
    p.add_argument("--d-model", type=int, default=None,
                   help="Override predictor d_model.")
    p.add_argument("--num-layers", type=int, default=None,
                   help="Override predictor num_hidden_layers.")
    p.add_argument("--csv", type=str, default=None,
                   help="Optional path to write the per-model table as CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "patch_length":      args.patch_length,
        "d_model":           args.d_model,
        "num_hidden_layers": args.num_layers,
    }
    predictor_root = _resolve_predictor_root(args.predictor_root)

    models = MODELS
    if args.models:
        wanted = set(args.models)
        models = [(d, f) for (d, f) in MODELS if d in wanted]
        unknown = wanted - {d for (d, _f) in MODELS}
        if unknown:
            raise SystemExit(
                f"Unknown model display names: {sorted(unknown)}\n"
                f"Choose from: {[d for (d, _f) in MODELS]}")

    rows = build_rows(predictor_root, args.best_config, overrides,
                      args.horizon, models)
    print_report(predictor_root, args.best_config, args.horizon, rows)
    if args.csv:
        write_csv(args.csv, args.horizon, rows)


if __name__ == "__main__":
    main()
