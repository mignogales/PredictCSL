"""
PredictCSL recomputation master: GIFT-Eval forecast/parity -> label -> train ->
predictor overlay -> compare -> per-instance window evaluation.

The requested predictor matrix is deliberately small:

  * constrained/cheap PatchTST, curve regression
  * constrained/cheap bidirectional Mamba, curve regression

The first pass computes full-native and window-ablation forecasts before any
synthetic labeling or predictor training, and immediately reports leaderboard
parity. All predictor variants share that canonical GiftEval cell cache, so
only predictor inference and post-processing repeat. The default invocation
stops after every model completes this first pass, providing an explicit review
gate. ``--pipeline-only`` then runs the remaining work model-by-model: one model
finishes labels, predictor training, overlay, and comparison before the next
model starts. The full/base predictor and robust-timing-only v5 pass are not
part of this master.

Phase 6 evaluates both predictors using the historical dataset-shared choice and
a genuine per-instance W_i choice. It retrieves each selected instance's cached
leaderboard MASE and compares against full-native context, fixed windows,
non-period heuristics, and dataset/per-instance oracles.

The default output root is a new self-contained tree
``logs/experiments/master_recompute`` so incompatible old 8k pools and cached
checkpoints cannot be mixed into this recomputation. Re-running Phase 1 skips
models already marked complete in ``leaderboard_parity_summary.json``; pass
``--force 3`` (or bare ``--force``) to launch their ablation again.

Cross-env routing
-----------------
A few model families can't share the main env (hard transformers/torch conflicts;
see ``envs/README.md``), so their TSFM-loading work must run in a dedicated conda
env. Master therefore splits every per-model subprocess by env group and prefixes
the non-main groups with ``conda run -n <env>``:

  * ``predictcsl-main``   — every family except the dedicated envs below (the env master is
    itself launched in; the main group uses the current interpreter).
  * ``predictcsl-patchtst`` — PatchTST-FM-R1, using the exact leaderboard-era
    Granite API and torch SDPA behavior.
  * ``predictcsl-legacy`` — Sundial + TimeMoE (transformers==4.40.1).
  * ``predictcsl-toto``   — Toto-2.0-313m (Python 3.12 + toto-2/toto-models).
  * ``predictcsl-tirex``  — TiRex2 (torch>=2.8 / numpy 2.1 + tirex2).

This is correct because (a) Stage 1 only loads a model when its shards are pending
and (b) the v5 ablation lazy-imports each TSFM loader only for that model's cells —
so a ``--models <one family>`` run in its env never touches the packages it lacks.
The Toto and TiRex envs have no compatible ``mamba-ssm``. Their Mamba
predictor-only stages are therefore routed back through ``predictcsl-main`` and
consume the already-complete native forecast cache in ``--cached-only`` mode.
The PatchTST env uses the official torch-2.8 Mamba wheels and runs the complete
predictor matrix directly.

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
    python -m experiments.master_run_all                       # all-model ablation, then stop
    python -m experiments.master_run_all --pipeline-only       # reviewed: model-by-model pipeline
    python -m experiments.master_run_all --resume-phase6       # restart at Phase 6, then roll up
    python -m experiments.master_run_all --continue-after-ablation  # both without review stop
    python -m experiments.master_run_all --models Chronos2-Small
    python -m experiments.master_run_all --force 3              # repeat cached ablation
    python -m experiments.master_run_all --models TiRex2 --stage1-batch-size 8 --stage1-shard-size 50
    python -m experiments.master_run_all --test               # both predictors, reduced
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

from experiments import datasets_config, models_config
from experiments.gifteval_inference_recipes import inference_recipe


@dataclass
class Variant:
    name: str                       # short id used by --only/--skip-variants
    module: str                     # python -m target
    skip_stages: List[str]          # base stages already produced upstream
    extra: List[str] = field(default_factory=list)   # variant-specific flags
    needs_mamba: bool = False        # predictor needs mamba-ssm (skip mamba-less envs)
    label: str = ""                 # human description for the banner
    ablation_tree: str = ""         # subdir under PREDICTCSL_ABLATION_ROOT


# Authoritative variant registry (see MAINTENANCE RULE in the module docstring).
# Order matters: each entry's skip set assumes everything above it has run.
VARIANTS: List[Variant] = [
    Variant("cheap", "experiments.run_all_v3", ["1"],
            label="cheap PatchTST · curve regression",
            ablation_tree="general_v3"),
    Variant("mamba", "experiments.run_all_v4", ["1"],
            extra=["--cheap"], needs_mamba=True,
            label="cheap Mamba · curve regression",
            ablation_tree="general_v4"),
]


# ── Cross-env routing (see the docstring + envs/README.md) ───────────────────
# Master must be launched IN this env; main-group models use the current
# interpreter, env-specific groups are dispatched through `conda activate`.
MAIN_ENV = "predictcsl-main"
# Model family -> dedicated conda env. Families absent here run in the main group.
FAMILY_ENV: Dict[str, str] = {
    "patchtst_fm": "predictcsl-patchtst",  # published Granite API + torch 2.8 SDPA
    "sundial": "predictcsl-legacy",   # transformers==4.40.1
    "timemoe": "predictcsl-legacy",
    "toto":    "predictcsl-toto",      # Python 3.12 + toto-2/toto-models
    "tirex":   "predictcsl-tirex",     # torch>=2.8 + tirex2 package
}
# Native model envs without mamba-ssm. Entries may still have a predictor-only
# fallback in MAMBA_PREDICTOR_ENV_OVERRIDES below.
ENVS_WITHOUT_MAMBA = {
    "predictcsl-toto", "predictcsl-tirex",
}
# Predictor-only work can run outside the model's native environment once Phase
# 1 has filled the canonical forecast cache. Keep this explicit per environment:
# the v4 subprocess is guarded by --cached-only-ablation, so a missing/stale cell
# fails instead of attempting to import the foundation model in the fallback env.
MAMBA_PREDICTOR_ENV_OVERRIDES: Dict[str, str] = {
    "predictcsl-toto": MAIN_ENV,
    "predictcsl-tirex": MAIN_ENV,
}
ENV_ALIASES = {
    "predictcsl-patchtst": ("predictcsl-patchtst", "TSFM_PATCH"),
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
    "predictcsl-patchtst": r"""
import inspect
import sys
print("python", sys.executable)
import torch
print("torch", torch.__version__, "cuda", getattr(torch.version, "cuda", None))
if not torch.__version__.startswith("2.8.0"):
    raise SystemExit("predictcsl-patchtst expects torch==2.8.0 for leaderboard parity")
import transformers
print("transformers", transformers.__version__)
if not transformers.__version__.startswith("4.56.0"):
    raise SystemExit("predictcsl-patchtst expects transformers==4.56.0")
import tsfm_public
from tsfm_public import PatchTSTFMForPrediction
version = getattr(tsfm_public, "__version__", "")
print("granite-tsfm", version)
signature = inspect.signature(PatchTSTFMForPrediction.forward)
if "inputs" not in signature.parameters or "past_values" in signature.parameters:
    raise SystemExit("predictcsl-patchtst expects the leaderboard-era inputs= Granite API")
if "e4d488689" not in version:
    raise SystemExit("predictcsl-patchtst expects granite-tsfm commit e4d488689")
from gift_eval.data import Dataset
print("gift_eval import OK")
import causal_conv1d
import mamba_ssm
from mamba_ssm import Mamba
print("causal-conv1d", causal_conv1d.__version__)
print("mamba-ssm", mamba_ssm.__version__)
if not causal_conv1d.__version__.startswith("1.5.4"):
    raise SystemExit("predictcsl-patchtst expects causal-conv1d==1.5.4")
if not mamba_ssm.__version__.startswith("2.2.5"):
    raise SystemExit("predictcsl-patchtst expects mamba-ssm==2.2.5")
""",
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
from dotenv import load_dotenv
load_dotenv()
import torch
print("torch", torch.__version__, "cuda", getattr(torch.version, "cuda", None))
if not torch.__version__.startswith("2.8.0"):
    raise SystemExit("predictcsl-tirex expects torch==2.8.0 for the pinned TiRex2 stack")
import numpy as np
print("numpy", np.__version__)
if not np.__version__.startswith("2.1.3"):
    raise SystemExit("predictcsl-tirex expects numpy==2.1.3 for tirex-2")
from importlib.metadata import version
datasets_version = version("datasets")
print("datasets", datasets_version)
if datasets_version != "2.21.0":
    raise SystemExit(
        "predictcsl-tirex expects datasets==2.21.0 for NumPy-2-compatible GIFT-Eval loading"
    )
import pyarrow as pa
from datasets.formatting.formatting import NumpyArrowExtractor
targets = NumpyArrowExtractor().extract_column(
    pa.table({"target": [[1.0, 2.0], [3.0, 4.0]]})
)
if targets.shape != (2, 2):
    raise SystemExit("predictcsl-tirex datasets NumPy formatter smoke test failed")
from tirex2 import TimeseriesType, load_model
print("tirex2 import OK")
from gift_eval.data import Dataset
print("gift_eval import OK")
from experiments.tirex_compat import (
    TirexCheckpointAccessError,
    require_tirex_checkpoint_access,
)
try:
    require_tirex_checkpoint_access()
except TirexCheckpointAccessError as exc:
    raise SystemExit(str(exc)) from exc
print("TiRex GIFT-Eval checkpoint access OK")
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
    if label == "predictcsl-tirex" and "TiREX checkpoint access denied" in details:
        raise SystemExit(
            Fore.RED
            + f"Preflight failed for {label} before launching GPU workers.\n"
            + details
            + Fore.RESET
        )
    repair = {
        "predictcsl-patchtst": "bash envs/setup-patchtst.sh",
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
    phase = p.add_mutually_exclusive_group()
    phase.add_argument(
        "--pipeline-only", action="store_true",
        help=("Skip the all-model GIFT-Eval ablation pass and run the remaining "
              "pipeline model-by-model. Use this after reviewing "
              "leaderboard_parity_summary.json from the default run."),
    )
    phase.add_argument(
        "--continue-after-ablation", action="store_true",
        help=("Run the all-model ablation and immediately continue through the "
              "model-by-model pipeline instead of stopping for review."),
    )
    phase.add_argument(
        "--resume-phase6", action="store_true",
        help=("Skip forecast generation, labeling, and predictor training; resume "
              "at Phase 6 per-instance evaluation, then run the Phase 7 rollup. "
              "This is the direct restart path after a Phase 6 failure."),
    )
    p.add_argument("--force", nargs="*", default=None,
                   metavar="STAGE",
                   help="Forwarded verbatim to every variant (and the stage-1 "
                        "pre-run) as run_all's --force: pass stage numbers "
                        "(e.g. --force 3 4 or --force 5 for instance evaluation) "
                        "to re-run those stages even when their "
                        "done-marker is present, or --force with no argument to "
                        "force all active stages. Needed after adding datasets/"
                        "models to a catalog, since the stage-level done-markers "
                        "are coarse (per-model) and won't otherwise notice new "
                        "cells — the per-cell caches inside each stage still make "
                        "the re-run cheap (only the new cells compute).")
    p.add_argument("--no-rollup", action="store_true",
                   help="Skip the final cross-predictor rollup_all_predictors pass.")
    p.add_argument("--no-instance-eval", action="store_true",
                   help="Skip Phase 6 genuine per-instance window evaluation.")
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
    p.add_argument("--stage1-device", choices=["cuda", "cpu"], default="cuda",
                   help=("Device required for synthetic TSFM labeling. Defaults "
                         "to cuda and fails fast when CUDA is unavailable; pass "
                         "cpu only for an intentional CPU run."))
    p.add_argument("--output-root", default="logs/experiments/master_recompute",
                   help="Fresh self-contained root for all datasets, predictors, "
                        "GiftEval cells, and comparisons (default: "
                        "logs/experiments/master_recompute).")
    p.add_argument("--test", action="store_true",
                   help="Smoke-test all selected cheap/Mamba objectives end-to-end "
                        "in <output-root>/_smoke_test (kept for inspection).")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Pass -v to each variant (stream its subprocess output).")
    p.add_argument("--verbose-ablation", action="store_true",
                   help="Stream only stage 3 output live in each variant.")
    return p.parse_args()


def _stage1_build_args(args: argparse.Namespace) -> List[str]:
    # Master recomputations are production-sized. Make their device requirement
    # explicit so a broken CUDA environment cannot silently turn Stage 1 into a
    # multi-hour CPU job. The standalone builder retains its auto-device mode.
    extra: List[str] = ["--device", args.stage1_device]
    if args.stage1_batch_size is not None:
        extra += ["--batch-size", str(args.stage1_batch_size)]
    if args.stage1_shard_size is not None:
        extra += ["--shard-size", str(args.stage1_shard_size)]
    if args.stage1_windows is not None:
        extra += ["--windows", *[str(w) for w in args.stage1_windows]]
    if args.stage1_n_series is not None:
        extra += ["--n-series", str(args.stage1_n_series)]
    return extra


def _stage_forced(force: Optional[List[str]], stage: str) -> bool:
    """Whether master ``--force`` explicitly covers ``stage``."""
    return force == [] or (force is not None and stage in force)


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


def _forecast_precompute_cmd(
    env: Optional[str],
    displays: List[str],
    test: bool = False,
) -> List[str]:
    """Build predictor-independent GIFT-Eval caches for one model family."""
    cache_root = os.path.join(
        os.environ["PREDICTCSL_ABLATION_ROOT"], "general")
    args = [
        "experiments.test_window_ablation_gifteval_v5",
        "--models", *displays,
        "--cache-root", cache_root,
        "--forecast-only",
        "--short-context-mode", "skip",
        "--no-plots",
    ]
    if test:
        # Match run_all_v3/v4's smoke subset so phase 4 is cache-only too.
        args += ["--test-datasets", "3", "--test-datasets-seed", "42"]
    return _py(env, *args)


def _forecast_precompute_done(
    cache_root: str,
    display: str,
    *,
    test: bool = False,
) -> tuple[bool, str]:
    """Return whether Phase 1 already completed for one model.

    The forecast-only ablation writes ``leaderboard_parity_summary.json`` only
    after a model has traversed its complete dataset cohort.  Reusing that
    durable checkpoint avoids starting the heavyweight stage-3 process merely
    for it to discover that every individual cell is cached.
    """
    marker = os.path.join(cache_root, "leaderboard_parity_summary.json")
    try:
        with open(marker) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, "no readable leaderboard parity checkpoint"

    record = payload.get("models", {}).get(display)
    if not isinstance(record, dict):
        return False, "model absent from leaderboard parity checkpoint"

    expected_cells = min(3, len(datasets_config.datasets_to_run())) \
        if test else len(datasets_config.datasets_to_run())
    cells = record.get("cells")
    recorded_expected = record.get("expected_cells")
    if (record.get("complete") is not True
            or cells != expected_cells
            or recorded_expected != expected_cells):
        return False, f"incomplete/stale cohort ({cells}/{expected_cells} cells)"

    family_by_display = dict(models_config.run_pairs())
    family = family_by_display.get(display)
    if family is None:
        return False, "model absent from active catalog"
    expected_recipe = inference_recipe(family)
    if record.get("inference_recipe") != expected_recipe:
        return False, "stale model inference recipe"

    return True, f"leaderboard parity checkpoint ({cells}/{expected_cells} cells)"


def _precompute_model_groups(displays: List[str]) -> List[List[str]]:
    """Group selected models by family while preserving first-family order.

    Models in one family share the exact window grid. Running them in one v5
    process lets its GiftEvalCache survive across checkpoints. This primarily
    avoids loading/preprocessing all 97 datasets three times for Chronos2
    Small/Base/Synth while keeping unlike grids in separate invocations.
    """
    family_by_display = dict(models_config.run_pairs())
    grouped: "OrderedDict[str, List[str]]" = OrderedDict()
    for display in displays:
        grouped.setdefault(family_by_display[display], []).append(display)
    return list(grouped.values())


def _variant_cmd(
    env: Optional[str],
    variant: Variant,
    displays: List[str],
    vflag: List[str],
    fflag: List[str],
    only_stage: Optional[str] = None,
) -> List[str]:
    args = [variant.module]
    if only_stage is not None:
        args += ["--only-stages", only_stage]
    elif variant.skip_stages:
        args += ["--skip-stages", *variant.skip_stages]
    args += variant.extra + ["--models", *displays] + vflag + fflag
    return _py(env, *args)


def _variant_route(
    model_env: Optional[str], variant: Variant,
) -> Optional[tuple[Optional[str], List[str]]]:
    """Choose the env and safety flags for one model/predictor pairing.

    ``None`` means the variant is unsupported. A tuple may itself contain a
    ``None`` env, which is the normal in-process/main-environment route.
    """
    if not variant.needs_mamba:
        return model_env, []
    model_env_label = _env_label(model_env)
    if model_env_label in MAMBA_PREDICTOR_ENV_OVERRIDES:
        return (
            MAMBA_PREDICTOR_ENV_OVERRIDES[model_env_label],
            ["--cached-only-ablation"],
        )
    if model_env_label in ENVS_WITHOUT_MAMBA:
        return None
    return model_env, []


def main() -> None:
    args = parse_args()
    if args.resume_phase6 and args.test:
        raise SystemExit(
            "--resume-phase6 cannot be combined with --test: resume must read "
            "the production output tree, not the isolated smoke-test tree.")

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
    if args.verbose_ablation:
        vflag.append("--verbose-ablation")
    stage1_build_args = _stage1_build_args(args)
    if _stage_forced(args.force, "1"):
        stage1_build_args.append("--regenerate-pool")
    # --force forwarding: None -> no flag; [] -> bare --force (all active stages);
    # [...] -> --force <stages>. Passed to every subprocess verbatim; run_all only
    # acts on it for stages that are active (not in that variant's --skip-stages),
    # so forcing e.g. stage 3 is a no-op on variants that reuse it.
    force_ablation = _stage_forced(args.force, "3")
    if args.force is None:
        fflag: List[str] = []
    elif args.force == []:
        fflag = ["--force"]
    else:
        base_forced = [
            stage for stage in args.force if stage in {"1", "2", "3", "4"}]
        fflag = ["--force", *base_forced] if base_forced else []
    print(Fore.CYAN + f"Master run — variants: {[v.name for v in variants]}" + Fore.RESET)
    for env, displays in groups.items():
        print(Fore.CYAN + f"  env {_env_label(env)}: {displays}" + Fore.RESET)
    # Phase 6/7 are inference-free and run in the current main environment;
    # restarting there must not require every foundation-model environment.
    if not args.resume_phase6:
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

    # ---- Phase 1: predictor-independent GIFT-Eval forecasts + parity. --------
    # One model per invocation keeps its native window grid exact (no union of
    # unrelated family grids). Every model completes before the review gate.
    if not args.pipeline_only and not args.resume_phase6:
        cache_root = os.path.join(
            os.environ["PREDICTCSL_ABLATION_ROOT"], "general")
        for env, displays in groups.items():
            for model_group in _precompute_model_groups(displays):
                pending: List[str] = []
                for display in model_group:
                    done, summary = _forecast_precompute_done(
                        cache_root, display, test=args.test)
                    if done and not force_ablation:
                        print(Fore.WHITE
                              + f"Phase 1 · {display} · cached ({summary}) — skipping"
                              + Fore.RESET)
                    else:
                        if done:
                            print(Fore.YELLOW
                                  + f"Phase 1 · {display} · cached ({summary}) "
                                    "but --force 3 given — re-running"
                                  + Fore.RESET)
                        pending.append(display)
                if not pending:
                    continue
                precompute = _forecast_precompute_cmd(
                    env, pending, test=args.test)
                _run(
                    precompute,
                    f"Phase 1 — GIFT-Eval window ablation/parity "
                    f"[{_env_label(env)}]: {pending}",
                )

        # A real run deliberately stops here so leaderboard discrepancies are
        # caught before spending time on synthetic labels/predictor sweeps.
        # Smoke tests remain end-to-end; --continue-after-ablation is the
        # explicit unattended/full-run escape hatch.
        if not args.continue_after_ablation and not args.test:
            parity_path = os.path.join(
                os.environ["PREDICTCSL_ABLATION_ROOT"], "general",
                "leaderboard_parity_summary.json",
            )
            total = time.perf_counter() - t_start
            print(Fore.GREEN
                  + f"\nAll-model ablation complete in {total/60:.1f} min."
                  + Fore.RESET)
            print(Fore.CYAN
                  + f"Review: {parity_path}\n"
                  + "Then run: python -m experiments.master_run_all "
                    "--pipeline-only"
                  + Fore.RESET)
            return

    # ---- Remaining work is deliberately MODEL-major. -----------------------
    # For one model: build labels once, then run every applicable predictor
    # variant through train -> cached ablation overlay -> comparison. Only after
    # that complete model pipeline succeeds do we advance to the next model.
    if not args.resume_phase6:
        for env, displays in groups.items():
            for display in displays:
                build_args = list(stage1_build_args)
                if args.test:
                    build_args += ["--n-series", str(200)]
                stage1 = _stage1_cmd(env, [display], vflag, fflag, build_args)
                _run(
                    stage1,
                    f"Model pipeline · {display} · synthetic labeling "
                    f"[{_env_label(env)}]",
                )

                for v in variants:
                    model_env_label = _env_label(env)
                    route = _variant_route(env, v)
                    if route is None:
                        print(Fore.YELLOW
                              + f"  ⤷ skip {v.name} for {display} "
                                f"({model_env_label} has no mamba-ssm and no "
                                "predictor-env override)." + Fore.RESET)
                        continue
                    predictor_env, route_extra = route
                    if predictor_env != env:
                        print(Fore.CYAN
                              + f"  ↳ route {v.name} for {display} through "
                                f"{_env_label(predictor_env)} (native forecasts: "
                                "cached-only)." + Fore.RESET)
                    cmd = _variant_cmd(
                        predictor_env, v, [display], vflag, fflag + route_extra)
                    _run(
                        cmd,
                        f"Model pipeline · {display} · {v.name} · stages 2→4 "
                        f"[{_env_label(predictor_env)}]",
                    )

    # ---- Phase 6: genuine per-instance context choice. -----------------------
    # This master intentionally excludes the period policy. The evaluator reads
    # only the fixed-grid and predictor caches produced above.
    if not args.no_instance_eval:
        ablation_root = os.environ["PREDICTCSL_ABLATION_ROOT"]
        instance_out = os.path.join(
            os.environ["PREDICTCSL_MASTER_ROOT"],
            "instance_window_evaluation")
        instance_cmd = _py(
            None, "experiments.evaluate_instance_windows",
            "--ablation-root", ablation_root,
            "--output-dir", instance_out,
        )
        if args.models:
            instance_cmd += ["--models", *args.models]
        _run(
            instance_cmd,
            "Phase 6a — per-instance W_i evaluation (cheap PatchTST + Mamba)",
        )

        failure_cmd = _py(
            None, "experiments.analyze_instance_failures",
            "--instance-dir", instance_out,
            "--output-dir", os.path.join(
                os.environ["PREDICTCSL_MASTER_ROOT"],
                "instance_failure_analysis"),
            "--top-cells", "50",
        )
        if args.models:
            failure_cmd += ["--models", *args.models]
        _run(
            failure_cmd,
            "Phase 6b — per-instance predictor failure diagnostics",
        )

    # ---- Final combined cross-predictor overview (pure post-processing). -----
    # Reads on-disk outputs only (no TSFM load) -> runs once in the main env.
    if not args.no_rollup:
        rollup = _py(None, "experiments.rollup_all_predictors")
        ablation_root = os.environ["PREDICTCSL_ABLATION_ROOT"]
        rollup += ["--run-dir", os.path.join(ablation_root, "general_v3"),
                   "--output-dir", os.path.join(ablation_root, "general_all"),
                   "--plot-strategies", "pred_cheap", "pred_mamba",
                   "pred_cheap_instance", "pred_mamba_instance"]
        if args.models:
            rollup += ["--models", *args.models]
        _run(rollup, "Phase 7 — combined cross-predictor overview")

    total = time.perf_counter() - t_start
    print(Fore.GREEN + f"\nMaster run complete in {total/60:.1f} min." + Fore.RESET)


if __name__ == "__main__":
    main()
