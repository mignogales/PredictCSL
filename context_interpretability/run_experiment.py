"""
Unified entry point for the context-saturation interpretability framework.

Runs on the SERVER (TSFM deps + GPU); see the repo CLAUDE.md — nothing runs
locally. Examples:

    # everything the model supports, synthetic pool
    python -m context_interpretability.run_experiment --models PatchTST-FM-R1

    # attention masking (exp0) + perturbation (exp1) on GiftEval data
    python -m context_interpretability.run_experiment \
        --models Chronos2-Small --experiments exp0 exp1 --source gifteval

    # the mandatory synthetic distant-dependency control, all maskable models
    python -m context_interpretability.run_experiment \
        --models Sundial-Base-128M TimeMoE-200M --experiments exp4

    # regenerate figures + hypothesis report from saved results (no GPU)
    python -m context_interpretability.run_experiment \
        --models PatchTST-FM-R1 --analyze-only

Output tree: <output_root>/<model>/exp<k>_<name>/<dataset>/w<W>/results.csv
(+ done.json per cell — resumable), run_meta.json, figures/, hypothesis
reports. Unsupported (model, experiment) pairs are logged in
run_meta.skipped_capabilities.
"""

from __future__ import annotations

import argparse
import os
import random
import traceback
from typing import Dict, List

import numpy as np

from context_interpretability import RESULTS_ROOT
from context_interpretability.adapters.base import CapabilityError
from context_interpretability.runmeta import RunMeta

EXPERIMENTS = {
    "exp0": ("exp0_attention_masking", "attention_masking"),
    "exp1": ("exp1_perturbation", "perturbation"),
    "exp2": ("exp2_activation_patching", "activation_patching"),
    "exp3": ("exp3_forecast_lens", "forecast_lens"),
    "exp4": ("exp4_synthetic_controls", "synthetic_controls"),
    "exp5": ("exp5_integrated_gradients", "integrated_gradients"),
    "exp6": ("exp6_predictor_contrast_saliency",
             "predictor_contrast_saliency"),
    "exp7": ("exp7_context_decomposition", "context_decomposition"),
    "exp8": ("exp8_tsfm_contrast_saliency", "tsfm_contrast_saliency"),
}

# capability gate per experiment (None -> any forecaster; exp4 gates its
# sub-methods internally)
CAPABILITY_GATE = {
    "exp0": "attention_masking",
    "exp1": None,
    "exp2": "activation_patching",
    "exp3": "forecast_lens",
    "exp4": None,
    "exp5": "integrated_gradients",
    # exp6 differentiates the separate context-length predictor checkpoint;
    # it does not require a differentiable TSFM adapter.
    "exp6": None,
    "exp7": "attention_masking",
    "exp8": "integrated_gradients",
}


def load_config(path: str, overrides: Dict[str, object]) -> dict:
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        # determinism where practical (spec §10.2); warn_only keeps models with
        # non-deterministic kernels usable.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            pass
    except Exception:  # noqa: BLE001
        pass


def _load_datasets(config: dict, adapter) -> List:
    from context_interpretability.data import loaders
    dcfg = config.get("datasets") or {}
    source = dcfg.get("source", "synthetic")
    max_samples = int(config.get("max_samples", 256))
    if source == "synthetic":
        scfg = dcfg.get("synthetic") or {}
        data = loaders.load_synthetic(
            min(int(scfg.get("n_series", 256)), max_samples),
            int(scfg.get("seed", config.get("seed", 42))), adapter.horizon)
        return [data]
    if source == "gifteval":
        gcfg = dcfg.get("gifteval") or {}
        pairs = adapter.effective_context_lengths(config["context_lengths"])
        min_ctx = max(e for _r, e in pairs)
        return loaders.load_gifteval(
            gcfg.get("datasets") or [], gcfg.get("term", "short"),
            adapter.horizon, max_samples, min_ctx)
    raise SystemExit(f"Unknown datasets.source {source!r}")


def run_model(display: str, exps: List[str], config: dict, device: str,
              analyze_only: bool, force_selected: bool = False) -> None:
    from context_interpretability.adapters.registry import build_adapter
    from context_interpretability.analysis import figures, hypotheses
    from context_interpretability.analysis.significance import save_significance
    from context_interpretability.schema import load_results

    run_dir = os.path.join(config.get("output_root", RESULTS_ROOT), display)
    os.makedirs(run_dir, exist_ok=True)
    seed = int(config.get("seed", 42))
    _seed_everything(seed)

    if not analyze_only:
        bcfg = config.get("dynamic_batching") or {}
        adapter = build_adapter(
            display, horizon=int(config["horizon"]), device=device,
            batch_size=int(config.get("batch_size", 16)),
            dynamic_batching=bool(bcfg.get("enabled", True)),
            batch_reference_context=int(bcfg.get("reference_context", 8192)),
            max_batch_size=int(bcfg.get("max_batch_size", 256)))
        meta = RunMeta(run_dir, config, display, device, seed)
        try:
            adapter.load()
            meta.note("checkpoint", adapter.model_id)
            am = adapter.metadata()
            meta.note("precision", am.get("precision"))
            meta.note("adapter", am)
            meta.note("effective_context_lengths", {
                str(r): e for r, e in adapter.effective_context_lengths(
                    config["context_lengths"])})
            _run_experiments(adapter, exps, config, run_dir, meta, seed,
                             force_selected=force_selected)
        finally:
            meta.finalize()
            adapter.close()

    # -- analysis: always regenerable from saved files --------------------------
    tol = float((config.get("synthetic_controls") or {}).get(
        "sufficient_context_tolerance", 0.05))
    figures.generate_all(run_dir, tolerance=tol)
    df = load_results(run_dir)
    if not df.empty:
        acfg = config.get("analysis") or {}
        save_significance(
            df[df["method"].isin(["perturbation", "attention_masking"])],
            os.path.join(run_dir, "significance_blocks.csv"),
            n_boot=int(acfg.get("bootstrap_samples", 2000)),
            alpha=float(acfg.get("alpha", 0.05)), seed=seed)
    hypotheses.evaluate(run_dir, tolerance=tol)


def _run_experiments(adapter, exps: List[str], config: dict, run_dir: str,
                     meta: RunMeta, seed: int,
                     force_selected: bool = False) -> None:
    import importlib

    enabled = config.get("experiments") or {}
    datasets = None
    for key in exps:
        subdir, modname = EXPERIMENTS[key]
        # An explicit CLI selection is an opt-in and overrides a conservative
        # config default of false (used by expensive/new experiments). With no
        # --experiments argument, the config remains authoritative.
        if not force_selected and not enabled.get(subdir, True):
            meta.skip(subdir, "disabled in config")
            continue
        gate = CAPABILITY_GATE[key]
        if gate and not adapter.capabilities.supported(gate):
            meta.skip(subdir, f"{adapter.name}: capability "
                              f"supports_{gate} is false")
            continue
        mod = importlib.import_module(
            f"context_interpretability.experiments.{modname}")
        out_dir = os.path.join(run_dir, subdir)
        try:
            if key == "exp4":
                mod.run(adapter, config, out_dir, run_meta=meta, seed=seed)
            else:
                if datasets is None:
                    datasets = _load_datasets(config, adapter)
                    if not datasets:
                        raise SystemExit("No evaluation datasets loaded")
                for data in datasets:
                    mod.run(adapter, data, config, out_dir, run_meta=meta,
                            seed=seed)
        except CapabilityError as exc:
            meta.skip(subdir, str(exc))
        except Exception:  # noqa: BLE001 — isolate experiments like run_all does
            meta.skip(subdir, f"FAILED:\n{traceback.format_exc()}")
        meta.save()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Context-saturation interpretability experiments")
    default_cfg = os.path.join(os.path.dirname(__file__), "configs",
                               "experiments.yaml")
    ap.add_argument("--config", default=default_cfg)
    ap.add_argument("--models", nargs="+", required=True,
                    help="model display names (experiments/models_config.py)")
    ap.add_argument("--experiments", nargs="+", default=None,
                    choices=list(EXPERIMENTS), metavar="expK",
                    help=(f"subset of {list(EXPERIMENTS)}; an explicit subset "
                          "overrides per-experiment enabled:false settings"))
    ap.add_argument("--source", choices=["synthetic", "gifteval"], default=None,
                    help="override datasets.source")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="override output_root")
    ap.add_argument("--device", default=None)
    ap.add_argument("--analyze-only", action="store_true",
                    help="regenerate figures + reports from saved results")
    args = ap.parse_args()

    config = load_config(args.config, {
        "horizon": args.horizon, "batch_size": args.batch_size,
        "max_samples": args.max_samples, "seed": args.seed,
        "output_root": args.out,
    })
    if args.source:
        config.setdefault("datasets", {})["source"] = args.source

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"

    failed = []
    selected_experiments = args.experiments or list(EXPERIMENTS)
    explicitly_selected = args.experiments is not None
    for display in args.models:
        print(f"\n=== {display} ({', '.join(selected_experiments)}) ===")
        try:
            run_model(display, selected_experiments, config, device,
                      args.analyze_only, force_selected=explicitly_selected)
        except Exception as exc:  # noqa: BLE001 — isolate models
            failed.append((display, repr(exc)))
            traceback.print_exc()
    if failed:
        print("\nFAILED models:")
        for name, err in failed:
            print(f"  {name}: {err}")

    # Cross-model figures are pure post-processing and therefore work both
    # after a fresh run and with --analyze-only.
    from context_interpretability.analysis import figures
    figures.generate_all_models(
        config.get("output_root", RESULTS_ROOT), args.models)


if __name__ == "__main__":
    main()
