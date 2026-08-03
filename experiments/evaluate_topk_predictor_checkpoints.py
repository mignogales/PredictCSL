#!/usr/bin/env python3
"""Evaluate the top-k saved predictor trials on cached GiftEval forecasts.

This is a robustness check for Stage 2 hyperparameter selection. For every
requested predictor objective it:

1. ranks the durable ``trial_NNN_best.pt`` checkpoints by the same synthetic
   validation metric used by training (keeping the selected winner first);
2. packages each trial as a normal ``best_model.pt + best_config.json`` run;
3. runs only the predictor overlay against a symlinked canonical ``datasets/``
   cache, with ``--cached-only`` preventing accidental TSFM inference;
4. runs the normal Stage-4 comparison and writes a cross-trial consistency CSV
   and JSON summary.

No retraining or Chronos inference is performed. PatchTST predictors can run on
CPU, but Mamba's fused kernels require CUDA in the supported project environment,
so use ``--device cuda`` whenever ``mamba`` or ``mamba_cls`` is requested.

Example (run on the GPU server later)::

    python -m experiments.evaluate_topk_predictor_checkpoints \
      --model Chronos2-Small --variants mamba mamba_cls --top-k 3

Use ``--prepare-only`` to rank/package checkpoints and print the commands without
executing predictor inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class Variant:
    predictor_dir: str
    selection_metric: str
    needs_cuda: bool = False


VARIANTS: Dict[str, Variant] = {
    "cheap": Variant("context_length_predictor_v3", "val_regret"),
    "cheap_cls": Variant(
        "context_length_predictor_v3_classification", "val_regret"),
    "risk": Variant("context_length_predictor_v3_risk", "val_risk_score"),
    "mamba": Variant(
        "context_length_predictor_v4", "val_regret", needs_cuda=True),
    "mamba_cls": Variant(
        "context_length_predictor_v4_classification", "val_regret",
        needs_cuda=True),
}

METRIC_KEYS = (
    "val_curve_mse", "val_recon_mse", "val_combined", "val_regret",
    "val_harm_p90", "val_harmed_rate", "val_risk_score", "val_win_acc",
    "val_top3_acc", "auto_batch_size", "auto_lr",
)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def rank_trials(source_dir: Path, metric: str, top_k: int) -> List[Dict[str, Any]]:
    """Return the chosen winner followed by the next-best durable trials."""
    best_cfg = _read_json(source_dir / "best_config.json")
    selected_idx = int(best_cfg["trial_idx"])
    trials: List[Dict[str, Any]] = []
    for trial_json in sorted((source_dir / "trials").glob("trial_[0-9][0-9][0-9].json")):
        trial = _read_json(trial_json)
        idx = int(trial["trial_idx"])
        weights = source_dir / "trials" / f"trial_{idx:03d}_best.pt"
        if trial.get("failed", False) or not weights.is_file() or not _finite(trial.get(metric)):
            continue
        trial["_weights_path"] = str(weights)
        trials.append(trial)
    if not trials:
        raise RuntimeError(f"No valid trials with weights under {source_dir}")

    trials.sort(key=lambda row: (float(row[metric]), int(row["trial_idx"])))
    selected = next((row for row in trials
                     if int(row["trial_idx"]) == selected_idx), None)
    if selected is None:
        raise RuntimeError(
            f"Selected trial {selected_idx} has no usable checkpoint in {source_dir}")
    ordered = [selected] + [row for row in trials if row is not selected]
    return ordered[:top_k]


def package_trial(source_dir: Path, trial: Dict[str, Any], package_dir: Path,
                  metric: str, rank: int) -> Dict[str, Any]:
    """Expose a durable trial through the standard predictor-run interface."""
    package_dir.mkdir(parents=True, exist_ok=True)
    idx = int(trial["trial_idx"])
    weights = Path(trial["_weights_path"]).resolve()
    model_link = package_dir / "best_model.pt"
    if model_link.is_symlink():
        if model_link.resolve() != weights:
            raise RuntimeError(f"Refusing to replace mismatched symlink {model_link}")
    elif model_link.exists():
        raise RuntimeError(f"Refusing to overwrite existing file {model_link}")
    else:
        model_link.symlink_to(weights)

    cfg = _read_json(source_dir / "best_config.json")
    cfg.update(trial.get("cfg", {}))
    cfg["trial_idx"] = idx
    cfg["selection_metric"] = metric
    for key in METRIC_KEYS:
        if key in trial:
            cfg[key] = trial[key]
    _atomic_json(package_dir / "best_config.json", cfg)
    weight_stat = weights.stat()
    package_meta = {
        "rank": rank,
        "trial_idx": idx,
        "selection_metric": metric,
        "selection_score": float(trial[metric]),
        "source_dir": str(source_dir.resolve()),
        "weights": str(weights),
        "weights_size": weight_stat.st_size,
        "weights_mtime_ns": weight_stat.st_mtime_ns,
    }
    _atomic_json(package_dir / "topk_package.json", package_meta)
    return package_meta


def _ensure_dataset_link(run_dir: Path, canonical_cache: Path) -> None:
    source = (canonical_cache / "datasets").resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Canonical datasets cache not found: {source}")
    run_dir.mkdir(parents=True, exist_ok=True)
    link = run_dir / "datasets"
    if link.is_symlink():
        if link.resolve() != source:
            raise RuntimeError(f"Mismatched datasets symlink: {link} -> {link.resolve()}")
    elif link.exists():
        raise RuntimeError(f"Expected a datasets symlink, found real path: {link}")
    else:
        link.symlink_to(source, target_is_directory=True)


def _run(cmd: Sequence[str], dry_run: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(list(cmd), check=True)


def _read_result(result_dir: Path, variant: str, rank: int,
                 trial: Dict[str, Any]) -> Dict[str, Any]:
    summary = _read_json(result_dir / "summary_stats.json")
    with (result_dir / "flops_savings.csv").open(newline="") as handle:
        flops_rows = list(csv.DictReader(handle))
    flops = next((row for row in flops_rows if row["strategy"] == "pred"), None)
    if flops is None:
        raise RuntimeError(f"No primary 'pred' row in {result_dir / 'flops_savings.csv'}")
    full_norm = float(summary["full_mase"]["geomean_norm"])
    pred_norm = float(summary["pred_mase"]["geomean_norm"])
    return {
        "variant": variant,
        "rank": rank,
        "trial_idx": int(trial["trial_idx"]),
        "selection_metric": VARIANTS[variant].selection_metric,
        "selection_score": float(trial[VARIANTS[variant].selection_metric]),
        "normalized_mase_full": full_norm,
        "normalized_mase_pred": pred_norm,
        "relative_mase_gain_pct": 100.0 * (full_norm - pred_norm) / full_norm,
        "total_instances": int(float(flops["total_instances"])),
        "total_full_flops": float(flops["total_full_flops"]),
        "total_predictor_flops": float(flops["total_strategy_flops"]),
        "total_flops_saved": float(flops["flops_saved"]),
        "total_flops_saved_pct": 100.0 * float(flops["pct_flops_saved"]),
    }


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)
    os.replace(tmp, path)


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"variants": {}}
    for variant in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        block: Dict[str, Any] = {
            "n_checkpoints": len(group),
            "trial_indices": [row["trial_idx"] for row in group],
        }
        for key in ("normalized_mase_pred", "relative_mase_gain_pct",
                    "total_flops_saved_pct", "total_flops_saved"):
            values = [float(row[key]) for row in group]
            block[key] = {
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        result["variants"][variant] = block
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Chronos2-Small")
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS),
                        default=list(VARIANTS))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--master-root",
                        default="logs/experiments/master_recompute")
    parser.add_argument("--canonical-cache", default=None,
                        help="Complete Stage-3 run tree whose datasets/ is reused.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--predictor-batch-size", type=int, default=64)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Re-run overlays/comparisons even when result files exist.")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.device == "cpu":
        incompatible = [name for name in args.variants if VARIANTS[name].needs_cuda]
        if incompatible and not args.prepare_only:
            parser.error("Mamba variants require --device cuda in this project: "
                         + ", ".join(incompatible))
    return args


def main() -> None:
    args = parse_args()
    master_root = Path(args.master_root)
    canonical = Path(args.canonical_cache or
                     master_root / "window_ablation_gifteval" / "general")
    output = Path(args.output_root or
                  master_root / "topk_predictor_consistency" / args.model)
    rows: List[Dict[str, Any]] = []

    for variant_name in args.variants:
        variant = VARIANTS[variant_name]
        source = master_root / variant.predictor_dir / args.model
        trials = rank_trials(source, variant.selection_metric, args.top_k)
        print(f"\n{variant_name}: top {len(trials)} trial(s) by "
              f"{variant.selection_metric}")
        for rank, trial in enumerate(trials, start=1):
            idx = int(trial["trial_idx"])
            tag = f"rank_{rank:02d}_trial_{idx:03d}"
            package = output / "checkpoints" / variant_name / tag
            run_dir = output / "runs" / variant_name / tag
            result_dir = output / "results" / variant_name / tag
            package_meta = package_trial(
                source, trial, package, variant.selection_metric, rank)
            _ensure_dataset_link(run_dir, canonical)
            print(f"  rank {rank}: trial {idx:03d} "
                  f"{variant.selection_metric}={float(trial[variant.selection_metric]):.6f}")

            stage3 = [
                sys.executable, "-m", "experiments.test_window_ablation_gifteval_v5",
                "--models", args.model,
                "--predictor-dir", str(package),
                "--cache-root", str(run_dir),
                "--device", args.device,
                "--num-gpus", "1",
                "--predictor-batch-size", str(args.predictor_batch_size),
                "--no-plots", "--cached-only",
            ]
            stage4 = [
                sys.executable, "-m", "experiments.compare_window_strategies_gifteval",
                "--run-dir", str(run_dir),
                "--models", args.model,
                "--output-dir", str(result_dir),
                "--mase-metric", "mase_gluonts_real",
            ]
            if args.prepare_only or args.dry_run:
                _run(stage3, dry_run=True)
                _run(stage4, dry_run=True)
                continue
            evaluation_meta_path = result_dir / "topk_evaluation.json"
            evaluation_meta = (_read_json(evaluation_meta_path)
                               if evaluation_meta_path.is_file() else None)
            complete = (
                (result_dir / "summary_stats.json").is_file()
                and (result_dir / "flops_savings.csv").is_file()
                and evaluation_meta == package_meta
            )
            if args.force or not complete:
                _run(stage3, dry_run=False)
                _run(stage4, dry_run=False)
                _atomic_json(evaluation_meta_path, package_meta)
            else:
                print(f"  cached result: {result_dir}")
            rows.append(_read_result(result_dir, variant_name, rank, trial))
            _write_csv(output / "trial_results.csv", rows)

    if rows:
        summary = _aggregate(rows)
        summary.update({
            "model": args.model,
            "top_k": args.top_k,
            "canonical_cache": str(canonical.resolve()),
            "note": ("FLOPs are summed over n_instances; MASE is the standard "
                     "unweighted GiftEval geometric mean."),
        })
        _atomic_json(output / "consistency_summary.json", summary)
        print(f"\nConsistency results: {output / 'trial_results.csv'}")
        print(f"Consistency summary: {output / 'consistency_summary.json'}")
    else:
        print(f"\nPrepared checkpoints under {output}; no inference was run.")


if __name__ == "__main__":
    main()
