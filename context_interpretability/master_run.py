"""
Cross-environment master launcher for all context-interpretability models.

The model implementations cannot all coexist in one Python environment. This
launcher uses the authoritative routing in ``experiments.master_run_all``:

* predictcsl-main   — modern/main-stack models
* predictcsl-legacy — Sundial (and TimeMoE if it returns to the run catalog)
* predictcsl-toto   — Toto-2.0-313m
* predictcsl-tirex  — TiRex2

Each group writes into the same resumable output tree. After all inference
groups finish, a main-environment ``--analyze-only`` pass regenerates every
per-model figure and the combined attention-masking/slicing figure.

Run this launcher from the main environment on the GPU server:

    conda run -n predictcsl-main \
      python -m context_interpretability.master_run

Examples:

    # Only the dedicated-environment models
    python -m context_interpretability.master_run \
      --models Sundial-Base-128M Toto-2.0-313m TiRex2

    # Exp0 attention masking + exp1 perturbation on GiftEval
    python -m context_interpretability.master_run \
      --experiments exp0 exp1 --source gifteval

    # Rebuild plots from every selected model's saved rows, without inference
    python -m context_interpretability.master_run --analyze-only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

from experiments import master_run_all as env_master


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Run-set model displays to execute (default: every run-set model).")
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        choices=["exp0", "exp1", "exp2", "exp3", "exp4", "exp5",
                 "exp6", "exp7", "exp8"],
        help="Experiment subset forwarded to run_experiment.")
    parser.add_argument(
        "--source", choices=["synthetic", "harmonic", "kernelsynth", "gifteval"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Skip every model environment and only regenerate saved analyses.")
    return parser.parse_args(argv)


def _forwarded_args(args: argparse.Namespace) -> List[str]:
    """Arguments shared by inference-group and final analysis commands."""
    forwarded: List[str] = []
    if args.experiments:
        forwarded += ["--experiments", *args.experiments]
    for flag, value in [
        ("--source", args.source),
        ("--config", args.config),
        ("--out", args.out),
        ("--horizon", args.horizon),
        ("--batch-size", args.batch_size),
        ("--max-samples", args.max_samples),
        ("--seed", args.seed),
        ("--device", args.device),
    ]:
        if value is not None:
            forwarded += [flag, str(value)]
    return forwarded


def _group_cmd(env: Optional[str], displays: List[str],
               forwarded: List[str]) -> List[str]:
    return env_master._py(
        env,
        "context_interpretability.run_experiment",
        "--models", *displays,
        *forwarded,
    )


def _analysis_cmd(displays: List[str], forwarded: List[str]) -> List[str]:
    return env_master._py(
        None,
        "context_interpretability.run_experiment",
        "--models", *displays,
        *forwarded,
        "--analyze-only",
    )


def _require_main_environment() -> None:
    """Fail early when the launcher was accidentally started in another env."""
    active = os.environ.get("CONDA_DEFAULT_ENV")
    accepted = {
        env_master.MAIN_ENV,
        "TSFM_moirai",  # legacy server alias documented in envs/README.md
    }
    if active and active not in accepted:
        raise SystemExit(
            "Launch the interpretability master from predictcsl-main, e.g.\n"
            "  conda run -n predictcsl-main "
            "python -m context_interpretability.master_run\n"
            f"Current CONDA_DEFAULT_ENV is {active!r}.")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    _require_main_environment()
    groups = env_master._resolve_groups(args.models)
    displays = [model for models in groups.values() for model in models]
    forwarded = _forwarded_args(args)

    print("Interpretability master routing:")
    for env, models in groups.items():
        print(f"  {env_master._env_label(env)}: {models}")

    started = time.perf_counter()
    if not args.analyze_only:
        for env, models in groups.items():
            env_master._preflight_env(env)
            env_master._run(
                _group_cmd(env, models, forwarded),
                "Interpretability "
                f"[{env_master._env_label(env)}]: {models}",
            )

    # Analysis never imports a model implementation. Run it once in main over
    # all selected directories so the final shared figure contains all models,
    # rather than only the last environment group.
    env_master._run(
        _analysis_cmd(displays, forwarded),
        "Interpretability analysis — all selected models",
    )
    elapsed = (time.perf_counter() - started) / 60
    print(f"\nInterpretability master complete in {elapsed:.1f} min.")


if __name__ == "__main__":
    main(sys.argv[1:])
