"""
PredictCSL recomputation master: label -> train -> GiftEval -> compare.

The requested predictor matrix is deliberately small:

  * constrained/cheap PatchTST, curve regression
  * constrained/cheap PatchTST, soft top-3 window classification
  * bidirectional Mamba, curve regression
  * bidirectional Mamba, soft top-3 window classification

Stage 1 runs once. All four predictor variants share one canonical GiftEval cell
cache, so the expensive TSFM forecasts are also computed once; only predictor
inference and post-processing repeat. The full/base predictor, period strategy,
and robust-timing-only v5 pass are not part of this master.

The default output root is a new self-contained tree
``logs/experiments/master_recompute`` so incompatible old 8k pools and cached
checkpoints cannot be mixed into this recomputation.

Cross-env routing
-----------------
A few model families can't share the main env (hard transformers/torch conflicts;
see ``envs/README.md``), so their TSFM-loading work must run in a dedicated conda
env. Master therefore splits every per-model subprocess by env group and prefixes
the non-main groups with ``conda run -n <env>``:

  * ``predictcsl-main``   — every family except the dedicated envs below (the env master is
    itself launched in; the main group uses the current interpreter).
  * ``predictcsl-legacy`` — Sundial + TimeMoE (transformers==4.40.1).
  * ``predictcsl-toto``   — Toto-2.0-313m (Python 3.12 + toto-2/toto-models).
  * ``predictcsl-tirex``  — TiRex2 (torch>=2.8 / numpy 2.x + tirex2).

This is correct because (a) Stage 1 only loads a model when its shards are pending
and (b) the v5 ablation lazy-imports each TSFM loader only for that model's cells —
so a ``--models <one family>`` run in its env never touches the packages it lacks.
``predictcsl-toto`` has no ``mamba-ssm``, so Toto is skipped for the Mamba variant
(v4); see ``ENVS_WITHOUT_MAMBA``.

  !! Launch master IN predictcsl-main:  conda run -n predictcsl-main python -m experiments.master_run_all

-------------------------------------------------------------------------------
MAINTENANCE RULE: when you add a new ``run_all_*`` orchestrator, add it to the
``VARIANTS`` registry below with the stages it should SKIP (everything already
produced by an earlier entry). When you add a model family that needs its own
env, add it to ``FAMILY_ENV`` (and ``ENVS_WITHOUT_MAMBA`` if that env lacks
mamba). That keeps master_run_all the single fuse-everything entry point.
-------------------------------------------------------------------------------

Usage
-----
    python -m experiments.master_run_all                       # everything, all models
    python -m experiments.master_run_all --models Chronos2-Small
    python -m experiments.master_run_all --models TiRex2 --stage1-batch-size 8 --stage1-shard-size 50
    python -m experiments.master_run_all --only-variants cheap mamba
    python -m experiments.master_run_all --test               # all selected variants, reduced
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from colorama import Fore

from experiments import models_config


@dataclass
class Variant:
    name: str                       # short id used by --only/--skip-variants
    module: str                     # python -m target
    skip_stages: List[str]          # base stages already produced upstream
    extra: List[str] = field(default_factory=list)   # variant-specific flags
    needs_mamba: bool = False        # predictor needs mamba-ssm (skip mamba-less envs)
    label: str = ""                 # human description for the banner


# Authoritative variant registry (see MAINTENANCE RULE in the module docstring).
# Order matters: each entry's skip set assumes everything above it has run.
VARIANTS: List[Variant] = [
    Variant("cheap", "experiments.run_all_v3", ["1"],
            label="cheap PatchTST · curve regression"),
    Variant("cheap_cls", "experiments.run_all_v3", ["1"],
            extra=["--training-objective", "classification"],
            label="cheap PatchTST · soft top-3 classification"),
    Variant("mamba", "experiments.run_all_v4", ["1"], needs_mamba=True,
            label="Mamba · curve regression"),
    Variant("mamba_cls", "experiments.run_all_v4", ["1"], needs_mamba=True,
            extra=["--training-objective", "classification"],
            label="Mamba · soft top-3 classification"),
]


# ── Cross-env routing (see the docstring + envs/README.md) ───────────────────
# Master must be launched IN this env; main-group models use the current
# interpreter, env-specific groups are dispatched through `conda activate`.
MAIN_ENV = "predictcsl-main"
# Model family -> dedicated conda env. Families absent here run in the main group.
FAMILY_ENV: Dict[str, str] = {
    "sundial": "predictcsl-legacy",   # transformers==4.40.1
    "timemoe": "predictcsl-legacy",
    "toto":    "predictcsl-toto",      # Python 3.12 + toto-2/toto-models
    "tirex":   "predictcsl-tirex",     # torch>=2.8 + tirex2 package
}
# Envs without mamba-ssm -> can't run the Mamba predictor variant (v4).
ENVS_WITHOUT_MAMBA = {"predictcsl-toto", "predictcsl-tirex"}
ENV_ALIASES = {
    "predictcsl-legacy": ("predictcsl-legacy", "TSFM_sundial_patch"),
    "predictcsl-toto": ("predictcsl-toto", "TSFM_toto"),
    "predictcsl-tirex": ("predictcsl-tirex", "predictcsl-test", "TSFM_tirex2"),
}
_CONDA_ENV_NAMES: Optional[set] = None


def _banner(text: str) -> None:
    bar = "═" * 78
    print(Fore.CYAN + f"\n{bar}\n  {text}\n{bar}" + Fore.RESET)


def _env_label(env: Optional[str]) -> str:
    return env or MAIN_ENV


def _resolve_conda_env(env: str) -> str:
    """Resolve canonical env names against the legacy names on the GPU server."""
    global _CONDA_ENV_NAMES
    if _CONDA_ENV_NAMES is None:
        try:
            raw = subprocess.run(
                ["conda", "env", "list", "--json"], check=True,
                capture_output=True, text=True).stdout
            _CONDA_ENV_NAMES = {
                os.path.basename(os.path.normpath(p))
                for p in json.loads(raw).get("envs", [])
            }
        except Exception as exc:
            raise SystemExit(
                f"Could not inspect conda environments: {exc}. Run the master "
                "from a shell with conda available.") from exc
    for candidate in ENV_ALIASES.get(env, (env,)):
        if candidate in _CONDA_ENV_NAMES:
            return candidate
    raise SystemExit(
        f"Required conda env {env!r} not found. Tried "
        f"{ENV_ALIASES.get(env, (env,))}; run envs/setup-all.sh first.")


def _py(env: Optional[str], *module_and_args: str) -> List[str]:
    """Build a ``python -m <module> [args]`` command, dispatched into ``env``.

    The main group (``env`` is None or equals MAIN_ENV) runs on the current
    interpreter; any other env is launched through a real ``conda activate`` so
    compiler/binutils activation scripts match an interactive shell.
    """
    if env is None or env == MAIN_ENV:
        return [sys.executable, "-m", *module_and_args]
    resolved = _resolve_conda_env(env)
    py_args = " ".join(shlex.quote(arg) for arg in ("-m", *module_and_args))
    shell_cmd = (
        'source "$(conda info --base)/etc/profile.d/conda.sh"; '
        f"conda activate {shlex.quote(resolved)}; "
        f"exec python {py_args}"
    )
    return ["bash", "-lc", shell_cmd]


def _python_code(env: Optional[str], code: str) -> List[str]:
    """Build a ``python -c <code>`` command in the selected env."""
    if env is None or env == MAIN_ENV:
        return [sys.executable, "-c", code]
    resolved = _resolve_conda_env(env)
    shell_cmd = (
        'source "$(conda info --base)/etc/profile.d/conda.sh"; '
        f"conda activate {shlex.quote(resolved)}; "
        f"exec python -c {shlex.quote(code)}"
    )
    return ["bash", "-lc", shell_cmd]


ENV_PREFLIGHTS: Dict[str, str] = {
    "predictcsl-toto": r"""
import sys
print("python", sys.executable)
import torch
print("torch", torch.__version__, "cuda", getattr(torch.version, "cuda", None))
if not torch.__version__.startswith("2.5.1"):
    raise SystemExit("predictcsl-toto expects torch==2.5.1 for the pinned Toto stack")
import toto2
print("toto2", toto2.__file__)
""",
    "predictcsl-tirex": r"""
import sys
print("python", sys.executable)
import torch
print("torch", torch.__version__, "cuda", getattr(torch.version, "cuda", None))
if not torch.__version__.startswith("2.8.0"):
    raise SystemExit("predictcsl-tirex expects torch==2.8.0 for the pinned TiRex2 stack")
from tirex2 import TimeseriesType, load_model
print("tirex2 import OK")
""",
}


def _preflight_env(env: Optional[str]) -> None:
    """Check dedicated model envs before expensive stage-1 work starts."""
    label = _env_label(env)
    probe = ENV_PREFLIGHTS.get(label)
    if probe is None:
        return
    proc = subprocess.run(
        _python_code(env, probe),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode == 0:
        print(Fore.GREEN + f"  preflight {label}: OK" + Fore.RESET)
        return

    details = "\n".join(
        line for line in (proc.stdout + proc.stderr).strip().splitlines()
        if line.strip()
    )
    repair = {
        "predictcsl-toto": "bash envs/repair-toto.sh predictcsl-toto",
        "predictcsl-tirex": "bash envs/repair-tirex.sh predictcsl-tirex",
    }[label]
    raise SystemExit(
        Fore.RED
        + f"Preflight failed for {label} before launching stage 1.\n"
        + details
        + "\n\nRepair this env with:\n"
        + f"  {repair}"
        + Fore.RESET
    )


def _resolve_groups(models: Optional[List[str]]) -> "OrderedDict[Optional[str], List[str]]":
    """Group the selected run-set displays by the env they must run in.

    Returns an ordered map ``{env_or_None: [display, ...]}`` with the main group
    (key None) first, then each dedicated env — so the cheap main work starts
    before the env-switching subprocesses.
    """
    pairs = models_config.run_pairs()                # [(display, family)], catalog order
    if models:
        known = {d for d, _ in pairs}
        bad = [m for m in models if m not in known]
        if bad:
            raise SystemExit(
                f"Unknown --models: {bad}. Known: {sorted(known)}")
        pairs = [(d, f) for d, f in pairs if d in models]

    groups: "OrderedDict[Optional[str], List[str]]" = OrderedDict()
    groups[None] = []                                # main group first, always present
    for display, family in pairs:
        groups.setdefault(FAMILY_ENV.get(family), []).append(display)
    if not groups[None]:
        del groups[None]
    return groups


def _run(cmd: List[str], title: str) -> None:
    """Stream a subprocess to the terminal (so each variant's own bars show),
    timing it and raising on failure."""
    _banner(title)
    print(Fore.MAGENTA + f"  $ {' '.join(cmd)}" + Fore.RESET)
    t0 = time.perf_counter()
    rc = subprocess.run(cmd, check=False).returncode
    dt = time.perf_counter() - t0
    if rc != 0:
        raise SystemExit(
            Fore.RED + f"  ✗ {title} failed (exit {rc}) after {dt/60:.1f} min." + Fore.RESET)
    print(Fore.GREEN + f"  ✓ {title} done in {dt/60:.1f} min." + Fore.RESET)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    names = [v.name for v in VARIANTS]
    p.add_argument("--models", nargs="+", default=None,
                   help="Subset of display names from run_all.MODELS_TO_RUN to run.")
    p.add_argument("--only-variants", nargs="+", default=None, choices=names,
                   help=f"Run only these variants (default: all of {names}).")
    p.add_argument("--skip-variants", nargs="+", default=[], choices=names,
                   help="Variants to skip.")
    p.add_argument("--force", nargs="*", default=None,
                   metavar="STAGE",
                   help="Forwarded verbatim to every variant (and the stage-1 "
                        "pre-run) as run_all's --force: pass stage numbers "
                        "(e.g. --force 3 4) to re-run those stages even when their "
                        "done-marker is present, or --force with no argument to "
                        "force all active stages. Needed after adding datasets/"
                        "models to a catalog, since the stage-level done-markers "
                        "are coarse (per-model) and won't otherwise notice new "
                        "cells — the per-cell caches inside each stage still make "
                        "the re-run cheap (only the new cells compute).")
    p.add_argument("--no-rollup", action="store_true",
                   help="Skip the final cross-predictor rollup_all_predictors pass.")
    p.add_argument("--stage1-batch-size", type=int, default=None,
                   help="Forwarded to build_context_length_dataset --batch-size. "
                        "Useful for slow/heavy labelers such as TiRex.")
    p.add_argument("--stage1-shard-size", type=int, default=None,
                   help="Forwarded to build_context_length_dataset --shard-size "
                        "so long label jobs checkpoint more frequently.")
    p.add_argument("--stage1-windows", type=int, nargs="+", default=None,
                   help="Forwarded to build_context_length_dataset --windows.")
    p.add_argument("--stage1-n-series", type=int, default=None,
                   help="Forwarded to build_context_length_dataset --n-series "
                        "when creating a fresh synthetic pool.")
    p.add_argument("--output-root", default="logs/experiments/master_recompute",
                   help="Fresh self-contained root for all datasets, predictors, "
                        "GiftEval cells, and comparisons (default: "
                        "logs/experiments/master_recompute).")
    p.add_argument("--test", action="store_true",
                   help="Smoke-test all selected cheap/Mamba objectives end-to-end "
                        "in <output-root>/_smoke_test (kept for inspection).")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Pass -v to each variant (stream its subprocess output).")
    return p.parse_args()


def _stage1_build_args(args: argparse.Namespace) -> List[str]:
    extra: List[str] = []
    if args.stage1_batch_size is not None:
        extra += ["--batch-size", str(args.stage1_batch_size)]
    if args.stage1_shard_size is not None:
        extra += ["--shard-size", str(args.stage1_shard_size)]
    if args.stage1_windows is not None:
        extra += ["--windows", *[str(w) for w in args.stage1_windows]]
    if args.stage1_n_series is not None:
        extra += ["--n-series", str(args.stage1_n_series)]
    return extra


def _stage1_cmd(
    env: Optional[str],
    displays: List[str],
    vflag: List[str],
    fflag: List[str],
    build_args: List[str],
) -> List[str]:
    args = [
        "experiments.run_all",
        "--only-stages", "1",
        "--models", *displays,
        *vflag,
        *fflag,
    ]
    if build_args:
        args += ["--build-args", *build_args]
    return _py(env, *args)


def _variant_cmd(
    env: Optional[str],
    variant: Variant,
    displays: List[str],
    vflag: List[str],
    fflag: List[str],
) -> List[str]:
    args = [variant.module]
    if variant.skip_stages:
        args += ["--skip-stages", *variant.skip_stages]
    args += variant.extra + ["--models", *displays] + vflag + fflag
    return _py(env, *args)


def main() -> None:
    args = parse_args()

    run_root = os.path.normpath(args.output_root)
    os.environ["PREDICTCSL_MASTER_ROOT"] = run_root
    os.environ["PREDICTCSL_DATASET_ROOT"] = os.path.join(
        run_root, "context_length_dataset")
    os.environ["PREDICTCSL_ABLATION_ROOT"] = os.path.join(
        run_root, "window_ablation_gifteval")
    os.environ["PREDICTCSL_RUN_LOG_ROOT"] = os.path.join(run_root, "run_all_logs")

    wanted = set(args.only_variants) if args.only_variants else {v.name for v in VARIANTS}
    wanted -= set(args.skip_variants)
    variants = [v for v in VARIANTS if v.name in wanted]
    if not variants:
        raise SystemExit(f"No variants selected. Known: {[v.name for v in VARIANTS]}")

    groups = _resolve_groups(args.models)
    vflag = ["-v"] if args.verbose else []
    stage1_build_args = _stage1_build_args(args)
    # --force forwarding: None -> no flag; [] -> bare --force (all active stages);
    # [...] -> --force <stages>. Passed to every subprocess verbatim; run_all only
    # acts on it for stages that are active (not in that variant's --skip-stages),
    # so forcing e.g. stage 3 is a no-op on variants that reuse it.
    if args.force is None:
        fflag: List[str] = []
    elif args.force == []:
        fflag = ["--force"]
    else:
        fflag = ["--force", *args.force]
    print(Fore.CYAN + f"Master run — variants: {[v.name for v in variants]}" + Fore.RESET)
    for env, displays in groups.items():
        print(Fore.CYAN + f"  env {_env_label(env)}: {displays}" + Fore.RESET)
    for env in groups:
        _preflight_env(env)

    t_start = time.perf_counter()

    if args.test:
        smoke_root = os.path.join(run_root, "_smoke_test")
        os.environ["PREDICTCSL_MASTER_ROOT"] = smoke_root
        os.environ["PREDICTCSL_TEST"] = "1"
        os.environ["PREDICTCSL_DATASET_ROOT"] = os.path.join(
            smoke_root, "context_length_dataset")
        os.environ["PREDICTCSL_ABLATION_ROOT"] = os.path.join(
            smoke_root, "window_ablation_gifteval")
        os.environ["PREDICTCSL_RUN_LOG_ROOT"] = os.path.join(
            smoke_root, "run_all_logs")

    # ---- Stage 1 once, per env group (each labels only its own families). ----
    for env, displays in groups.items():
        build_args = list(stage1_build_args)
        if args.test:
            build_args += ["--n-series", str(200)]
        stage1 = _stage1_cmd(env, displays, vflag, fflag, build_args)
        _run(stage1,
             f"Stage 1 — labeling [{_env_label(env)}]: {displays}")

    # ---- Each variant × env group, with its shared stages skipped. -----------
    for v in variants:
        for env, displays in groups.items():
            if v.needs_mamba and _env_label(env) in ENVS_WITHOUT_MAMBA:
                print(Fore.YELLOW
                      + f"  ⤷ skip {v.name} for {displays} "
                        f"({_env_label(env)} has no mamba-ssm)." + Fore.RESET)
                continue
            cmd = _variant_cmd(env, v, displays, vflag, fflag)
            _run(cmd, f"{v.name} [{_env_label(env)}] — {v.label}  "
                      f"(skip stages {v.skip_stages or 'none'})")

    # ---- Final combined cross-predictor overview (pure post-processing). -----
    # Reads on-disk outputs only (no TSFM load) -> runs once in the main env.
    if not args.no_rollup:
        rollup = _py(None, "experiments.rollup_all_predictors")
        ablation_root = os.environ["PREDICTCSL_ABLATION_ROOT"]
        rollup += ["--run-dir", os.path.join(ablation_root, "general_v3"),
                   "--output-dir", os.path.join(ablation_root, "general_all")]
        if args.models:
            rollup += ["--models", *args.models]
        _run(rollup, "rollup_all_predictors — combined cross-predictor overview")

    total = time.perf_counter() - t_start
    print(Fore.GREEN + f"\nMaster run complete in {total/60:.1f} min." + Fore.RESET)


if __name__ == "__main__":
    main()
