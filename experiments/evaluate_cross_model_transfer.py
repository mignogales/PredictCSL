"""Evaluate a context selector trained for one TSFM on another TSFM's curves.

The target TSFM is never loaded: this runner requires a completed canonical
GiftEval window cache and invokes the normal Stage-3 overlay in ``--cached-only``
mode.  It therefore isolates the scientific question: do the structural cues
learned from a source TSFM transfer to a target TSFM?

For each target, the script creates an isolated run tree that symlinks the
canonical ``datasets/`` cache, writes source-predictor curves under that tree,
and runs the ordinary strategy comparison. Existing target-specific predictor
results are left untouched.

Example (run on the server after target window grids are complete)::

    python -m experiments.evaluate_cross_model_transfer \\
      --canonical-run-dir logs/experiments/master_recompute/window_ablation_gifteval/general \\
      --source-predictor-dir logs/experiments/master_recompute/context_length_predictor/Chronos2-Small \\
      --targets Chronos2-Synth Chronos2-Base
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def transfer_tree(output_root: Path, source: Path, target: str) -> Path:
    """Stable isolated output path for one source->target transfer."""
    return output_root / source.name / f"to_{target}"


def ensure_datasets_link(tree: Path, canonical_run_dir: Path) -> None:
    """Link immutable target forecast cells without copying or re-running them."""
    source = canonical_run_dir / "datasets"
    if not source.is_dir():
        raise FileNotFoundError(f"Canonical datasets cache is missing: {source}")
    tree.mkdir(parents=True, exist_ok=True)
    link = tree / "datasets"
    if link.is_symlink():
        if link.resolve() != source.resolve():
            raise RuntimeError(f"{link} points to {link.resolve()}, expected {source}")
        return
    if link.exists():
        raise RuntimeError(
            f"{link} already exists and is not the expected symlink; refusing to "
            "mix transfer outputs with another cache."
        )
    link.symlink_to(os.path.relpath(source, start=tree), target_is_directory=True)


def overlay_command(args: argparse.Namespace, tree: Path, target: str) -> List[str]:
    command = [
        sys.executable, "-m", "experiments.test_window_ablation_gifteval_v5",
        "--models", target,
        "--predictor-dir", str(Path(args.source_predictor_dir).resolve()),
        "--cache-root", str(tree),
        "--cached-only",
        "--short-context-mode", args.short_context_mode,
        "--device", args.device,
    ]
    canonical_index = Path(args.canonical_run_dir) / "results.csv"
    if canonical_index.is_file():
        command.extend(["--preloaded-results-csv", str(canonical_index)])
    if args.no_plots:
        command.append("--no-plots")
    return command


def compare_command(tree: Path, target: str) -> List[str]:
    return [
        sys.executable, "-m", "experiments.compare_window_strategies_gifteval",
        "--run-dir", str(tree),
        "--models", target,
        "--output-dir", str(tree / "strategy_comparison"),
    ]


def run(args: argparse.Namespace) -> None:
    canonical = Path(args.canonical_run_dir).resolve()
    source = Path(args.source_predictor_dir).resolve()
    if not (source / "best_model.pt").is_file() or not (source / "best_config.json").is_file():
        raise FileNotFoundError(
            f"Source predictor must contain best_model.pt + best_config.json: {source}")
    output_root = Path(args.output_root).resolve()
    manifest = {
        "canonical_run_dir": str(canonical),
        "source_predictor_dir": str(source),
        "targets": list(args.targets),
        "short_context_mode": args.short_context_mode,
        "device": args.device,
        "runs": [],
    }

    for target in args.targets:
        tree = transfer_tree(output_root, source, target)
        ensure_datasets_link(tree, canonical)
        overlay = overlay_command(args, tree, target)
        compare = compare_command(tree, target)
        manifest["runs"].append({
            "target": target, "tree": str(tree),
            "overlay_command": overlay, "compare_command": compare,
        })
        print(f"\nSource {source.name} -> target {target}\n  output: {tree}")
        if args.dry_run:
            print("  " + " ".join(overlay))
            print("  " + " ".join(compare))
            continue
        subprocess.run(overlay, check=True)
        subprocess.run(compare, check=True)

    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / f"{source.name}_transfer_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\nTransfer manifest: {output_root / f'{source.name}_transfer_manifest.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-run-dir", required=True,
                        help="Completed canonical ablation tree containing datasets/.")
    parser.add_argument("--source-predictor-dir", required=True,
                        help="Predictor checkpoint trained with the source TSFM labels.")
    parser.add_argument("--targets", nargs="+", required=True,
                        help="Target TSFM display names with complete cached GiftEval grids.")
    parser.add_argument("--output-root", default=None,
                        help="Default: <canonical parent>/cross_model_transfer.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Only used for predictor inference; target TSFM inference is forbidden.")
    parser.add_argument("--short-context-mode", choices=["skip", "pad"], default="pad")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = str(
            Path(args.canonical_run_dir).resolve().parent / "cross_model_transfer")
    return args


if __name__ == "__main__":
    run(parse_args())
