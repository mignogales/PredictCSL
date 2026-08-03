"""
PredictCSL end-to-end pipeline orchestrator.

For every entry in MODELS_TO_RUN, runs the four pipeline stages with
deterministic per-family output paths:

    Stage 1 — build_context_length_dataset.py --model-idx <i>
              -> logs/experiments/context_length_dataset/<family>/

    Stage 2 — predict_context_length.py --dataset-dir <ds_root>/<family>
              -> logs/experiments/context_length_predictor/<family>/

    Stage 3 — test_window_ablation_gifteval_v5.py
                  --models <display> --predictor-dir <pred_root>/<family>
                  --cache-root <win_root>/general
              -> logs/experiments/window_ablation_gifteval/general/
                 (shared gifteval ablation results for ALL models — raw metrics
                  under datasets/, real-vs-predicted curves under
                  models/<display>/, combined results.csv)

    Stage 4 — compare_window_strategies_gifteval.py
                  --run-dir <win_root>/general --models <display>
                  --output-dir <win_root>/<display>/strategy_comparison
              -> logs/experiments/window_ablation_gifteval/<display>/strategy_comparison/
                 (per-model comparison between window-selection methods:
                  full window vs oracle-best vs predictor)

Comment out entries in MODELS_TO_RUN to skip families. Re-running is safe:
each stage caches its own work (per-shard for stage 1, per-trial for stage 2,
per-(dataset, model, term, window) cell for stage 3, derived artefacts for
stage 4) and overwrites in place.

The orchestrator additionally short-circuits a stage when its done-marker is
already on disk, so a re-run only spawns subprocesses for the stages that
still need work. Markers checked:

    stage 1 — meta.json with shards_done == shards_total
    stage 2 — best_model.pt + best_config.json
    stage 3 — predictor_meta.json (last file written by v5)
    stage 4 — strategy_comparison/summary_stats.json

Pass --force to re-run cached stages anyway (--force 2 3 for specific ones,
or --force with no arg to force everything active).

Usage
-----
    python -m experiments.run_all                       # all stages, all listed models
    python -m experiments.run_all --skip-stages 1 2     # only stages 3 and 4
    python -m experiments.run_all --models Chronos2-Small Moirai2-Small
    python -m experiments.run_all --force 4             # re-run stage 4 even if cached
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from typing import Callable, List, Sequence, Tuple

from colorama import Fore
from tqdm.auto import tqdm

# Authoritative model catalog. --model-idx for stage 1 indexes into THIS list,
# so we resolve indices from it directly rather than from a hand-copied mirror.
from experiments.build_context_length_dataset import (
    MODELS as BUILD_MODELS,
    window_grid_for_family,
)
from experiments import models_config
from experiments import datasets_config
from experiments.gifteval_reference import NORMALIZATION_REFERENCE
from experiments.gifteval_inference_recipes import inference_recipe
from experiments.gifteval_metric_version import METRIC_SUITE_VER


# Master run set: (model_id, family, display), sourced from the single config in
# experiments.models_config (run=True rows). Stage 1 resolves --model-idx by
# matching model_id against build_context_length_dataset.MODELS (see _model_idx),
# so order here is free. To add/drop a family end-to-end, flip its `run` flag in
# models_config — every consumer (this orchestrator + v2/v3/v4, the v5 ablation,
# the predictor-overhead summary) picks up the change.
MODELS_TO_RUN: List[Tuple[str, str, str]] = models_config.models_to_run()

DATASET_ROOT   = os.environ.get(
    "PREDICTCSL_DATASET_ROOT", "logs/experiments/context_length_dataset")
PREDICTOR_ROOT = os.environ.get(
    "PREDICTCSL_PREDICTOR_ROOT", "logs/experiments/context_length_predictor")
ABLATION_ROOT  = os.environ.get(
    "PREDICTCSL_ABLATION_ROOT", "logs/experiments/window_ablation_gifteval")
# Per-stage subprocess output is captured here in quiet mode (the default), so
# the terminal shows only the top-level tqdm bar. One log per (model, stage);
# overwritten on each run. Tail is surfaced if a stage fails.
RUN_LOG_ROOT   = os.environ.get(
    "PREDICTCSL_RUN_LOG_ROOT", "logs/experiments/run_all_logs")
# Shared, cross-model gifteval ablation output (stage 3). One folder for every
# model — v5 keys its artefacts by (dataset, model_short), so models coexist
# here without collision and share the dataset cache. Normalized MASE always
# reads its denominator from the shipped GIFT-Eval leaderboard CSV.
ABLATION_GENERAL = os.path.join(ABLATION_ROOT, "general")

# Per-model strategy-comparison subfolder (stage 4). Made mode-specific in main()
# so that --short-context-mode pad never collides with / reuses skip-mode caches.
STRATEGY_SUBDIR = "strategy_comparison"

# --test (smoke) mode: the whole pipeline runs end-to-end for every model and
# dataset but is shrunk dramatically (smallest+largest window only, a handful of
# synthetic series, a 2-trial / 3-epoch predictor sweep) and redirected into this
# throwaway tree, which is deleted once the run finishes. Lets you confirm every
# stage wires up and every model/dataset loads, without touching real results.
TEST_ROOT       = "logs/experiments/_smoke_test"
TEST_N_SERIES   = 200   # synthetic series built in stage 1 under --test
TEST_N_DATASETS = 3     # randomly sampled (dataset, term) entries in stage 3 under --test


def _model_idx(model_id: str, display: str) -> int:
    # Resolve --model-idx against the build script's OWN MODELS list (imported
    # as BUILD_MODELS), keyed by the unique checkpoint id. This is the only safe
    # key: --model-idx is positional into build's MODELS, so matching by id
    # guarantees the index always points at the same checkpoint the build script
    # will load — no hand-maintained mirror to drift out of sync. (Matching by
    # display would break for variants whose display differs from build's, e.g.
    # a synthetic-data variant reusing the same checkpoint.)
    for i, (mid, _, _) in enumerate(BUILD_MODELS):
        if mid == model_id:
            return i
    available = "\n".join(
        f"    [{i}] {mid}  ({disp})" for i, (mid, _, disp) in enumerate(BUILD_MODELS))
    raise ValueError(
        f"Model id {model_id!r} (display {display!r}) is not in "
        f"experiments/build_context_length_dataset.py MODELS, so stage 1 cannot "
        f"build it. Add it there first. Available checkpoints:\n{available}"
    )


# Overall progress bar (set in main). Orchestrator-level messages are routed
# through _emit so they print *above* the bar instead of clobbering it.
#
# Quiet mode (the default) captures each subprocess's stdout/stderr to a log
# file under RUN_LOG_ROOT and suppresses informational orchestrator chatter, so
# the terminal shows only the top-level tqdm bar (model · stage). Pass
# --verbose to restore the old behaviour: subprocess output streams raw to the
# terminal and every banner / command echo is printed.
_BAR: "tqdm | None" = None
_QUIET: bool = True
# Stream stage 3 only while leaving the other stages on their compact progress
# bars. Set by --verbose-ablation in run_all and its variant wrappers.
_VERBOSE_ABLATION: bool = False


def _emit(msg: str, level: str = "info") -> None:
    # In quiet mode only warnings/errors reach the terminal; "info" chatter is
    # swallowed so the bar stands alone.
    if _QUIET and level == "info":
        return
    if _BAR is not None:
        _BAR.write(msg)
    else:
        print(msg)


def _banner(text: str) -> None:
    bar = "═" * 78
    _emit(Fore.CYAN + f"\n{bar}\n  {text}\n{bar}" + Fore.RESET)


def _tail(path: str, n: int = 40) -> str:
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(log unavailable)"


# --- Per-unit duration cache (powers the ETA) --------------------------------
# A single subprocess exposes no internal progress, so a remaining-time estimate
# is impossible on a unit's first run. Instead we persist each (model, stage)
# wall-clock and reuse it as the bar's `total` next time, turning the timer into
# a real progress bar with an ETA. Stored as {label: seconds} in RUN_LOG_ROOT.

def _durations_path() -> str:
    return os.path.join(RUN_LOG_ROOT, "durations.json")


def _load_durations() -> dict:
    try:
        with open(_durations_path()) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_duration(label: str, dt: float) -> None:
    os.makedirs(RUN_LOG_ROOT, exist_ok=True)
    data = _load_durations()
    prev = data.get(label)
    # EMA: track the current machine without being whipsawed by one slow run.
    data[label] = round(0.5 * prev + 0.5 * dt, 2) if prev else round(dt, 2)
    tmp = _durations_path() + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, _durations_path())


def _read_progress(path: str):
    """Read {done, total} from a sweep_progress.json; returns (None, None) on any error."""
    try:
        with open(path) as f:
            d = json.load(f)
        return int(d["done"]), int(d["total"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None, None


def _run(cmd: Sequence[str], stage: str, display: str,
         progress_file: str = None) -> float:
    """Run one (model, stage) unit, raise on non-zero. Returns wall time.

    Quiet mode (default): each unit gets its own tqdm line that stays on screen
    after finishing.  Two flavours:

      progress_file set  — trial-count bar (n / total trials [elapsed<ETA]).
                           The child writes sweep_progress.json after every trial;
                           we poll it and call bar.update() so tqdm accumulates a
                           real rate.  ETA appears after the first trial completes.

      progress_file None — wall-clock bar using the cached duration from the
                           previous run (seconds as the unit).  First ever run of
                           a unit shows elapsed-only.

    Verbose mode: raw subprocess output streams to the terminal (no bar).
    """
    label = f"{display} · {stage}"
    t0 = time.perf_counter()

    stream_output = not _QUIET or (_VERBOSE_ABLATION and stage == "3/ablation")
    if stream_output:
        _emit(Fore.MAGENTA + f"  $ {' '.join(cmd)}" + Fore.RESET)
        res = subprocess.run(cmd, check=False)
        dt = time.perf_counter() - t0
        if res.returncode != 0:
            raise RuntimeError(
                f"Stage {stage} failed (exit {res.returncode}) after {dt:.1f}s.")
        _emit(Fore.GREEN + f"  ✓ {label} done in {dt:.1f}s" + Fore.RESET)
        return dt

    os.makedirs(RUN_LOG_ROOT, exist_ok=True)
    safe = f"{display}_{stage.replace('/', '-')}".replace(" ", "_")
    log_path = os.path.join(RUN_LOG_ROOT, f"{safe}.log")

    # ---- choose bar style -----------------------------------------------
    if progress_file is not None:
        # Trial-count bar.  Total is read from the progress file once it
        # appears; until then the bar is indeterminate.
        done0, total0 = _read_progress(progress_file)
        bar = tqdm(
            total=total0, initial=done0 or 0,
            desc=label, leave=True, unit="trial", dynamic_ncols=True,
            bar_format=("{desc}  {n}/{total} trials"
                        " |{bar}|  [{elapsed}<{remaining}]{postfix}"),
        )
    else:
        # Wall-clock bar backed by the duration cache.
        expected = _load_durations().get(label)
        if expected and expected > 0:
            bar = tqdm(
                total=round(expected), desc=label, leave=True, unit="s",
                dynamic_ncols=True,
                bar_format=("{desc} {percentage:3.0f}%|{bar}|"
                            " {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]{postfix}"),
            )
        else:
            bar = tqdm(
                total=None, desc=label, leave=True, dynamic_ncols=True,
                bar_format="{desc}  {elapsed} (first run — no ETA){postfix}",
            )

    global _BAR
    _BAR = bar
    try:
        bar.set_postfix_str("running")
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)

            while proc.poll() is None:
                if progress_file is not None:
                    done, total = _read_progress(progress_file)
                    if done is not None:
                        # Give bar a total on first successful read.
                        if bar.total is None and total:
                            bar.total = total
                            bar.refresh()
                        # Advance by the delta — this is what feeds tqdm's rate
                        # estimator and produces an honest ETA.
                        delta = done - bar.n
                        if delta > 0:
                            bar.update(delta)
                elif hasattr(bar, "total") and bar.total:
                    # Wall-clock bar: nudge n toward total so the fill moves.
                    bar.n = min(time.perf_counter() - t0, bar.total)
                bar.refresh()
                time.sleep(0.5)

        dt = time.perf_counter() - t0
        ok = proc.returncode == 0
        # Snap to 100 % on success so the bar looks clean when it freezes.
        if ok and bar.total:
            bar.n = bar.total
        bar.set_postfix_str(f"✓ {dt:.1f}s" if ok else f"✗ exit {proc.returncode}")
        bar.refresh()
    finally:
        bar.close()
        _BAR = None

    if proc.returncode != 0:
        _emit(
            Fore.RED
            + f"  ✗ {label} failed (exit {proc.returncode}) after {dt:.1f}s — "
              f"last lines of {log_path}:\n" + _tail(log_path) + Fore.RESET,
            level="error",
        )
        raise RuntimeError(
            f"Stage {stage} failed (exit {proc.returncode}) after {dt:.1f}s. "
            f"See {log_path}.")
    _save_duration(label, dt)   # record wall-clock for non-progress-file stages
    return dt


def stage_1_build(model_id: str, display: str, family: str,
                  extra: Sequence[str]) -> float:
    idx = _model_idx(model_id, display)
    return _run(
        [sys.executable, "-m", "experiments.build_context_length_dataset",
         "--model-idx", str(idx), *extra],
        stage="1/build", display=display,
    )


def stage_2_predictor(display: str, family: str, extra: Sequence[str]) -> float:
    dataset_dir = os.path.join(DATASET_ROOT, display)
    # The predictor writes sweep_progress.json to its run_dir after each trial.
    # Passing the path here lets _run drive a "N/60 trials [ETA]" bar instead of
    # the generic elapsed-only timer.
    progress_file = os.path.join(PREDICTOR_ROOT, display, "sweep_progress.json")
    return _run(
        [sys.executable, "-m", "experiments.predict_context_length",
         "--dataset-dir", dataset_dir, *extra],
        stage="2/predictor", display=display,
        progress_file=progress_file,
    )


def stage_3_ablation(display: str, family: str, extra: Sequence[str]) -> float:
    predictor_dir = os.path.join(PREDICTOR_ROOT, display)
    return _run(
        [sys.executable, "-m", "experiments.test_window_ablation_gifteval_v5",
         "--models", display,
         "--predictor-dir", predictor_dir,
         "--cache-root", ABLATION_GENERAL, *extra],
        stage="3/ablation", display=display,
    )


def stage_4_compare(display: str, family: str, extra: Sequence[str]) -> float:
    out_dir = os.path.join(ABLATION_ROOT, display, STRATEGY_SUBDIR)
    return _run(
        [sys.executable, "-m", "experiments.compare_window_strategies_gifteval",
         "--run-dir", ABLATION_GENERAL,
         "--models", display,
         "--output-dir", out_dir, *extra],
        stage="4/compare", display=display,
    )


def stage_4_rollup(extra: Sequence[str], out_dir: str = None,
                   models: Sequence[str] = None) -> float:
    """Cross-model final pass: one overview figure + flops_savings_all_models.csv
    spanning the requested models (or every model present when ``models`` is
    omitted).

    ``out_dir`` defaults to the run dir itself (owned by the default
    `mase_gluonts_real`); a `mase_gluonts` run routes it to ``rollup_gluonts/`` so
    the two metrics' run-level artefacts never overwrite each other. Restricting
    a subset run is important: unrelated stale model trees must not make a
    successful ``--models ...`` pipeline fail during its final rollup."""
    model_args = ["--models", *models] if models else []
    return _run(
        [sys.executable, "-m", "experiments.compare_window_strategies_gifteval",
         "--run-dir", ABLATION_GENERAL,
         "--output-dir", out_dir or ABLATION_GENERAL,
         "--rollup-only", *model_args, *extra],
        stage="4/rollup", display="ALL",
    )


STAGES = {
    "1": ("build",     stage_1_build),
    "2": ("predictor", stage_2_predictor),
    "3": ("ablation",  stage_3_ablation),
    "4": ("compare",   stage_4_compare),
}


# ==============================================================================
#  DONE CHECKS  (stage-level skip)
# ==============================================================================
#  Each returns (done, summary). `done` short-circuits the subprocess call;
#  `summary` is shown to the user so they know what was cached. These checks
#  are intentionally cheap: a few stat() calls and one JSON read per stage.

def _done_stage_1(family: str, display: str = "") -> Tuple[bool, str]:
    """Stage 1 is done when every labeling shard for this model is on disk.
    meta.json carries shards_done / shards_total — written each invocation."""
    meta_path = os.path.join(DATASET_ROOT, display, "meta.json")
    if not os.path.isfile(meta_path):
        return False, "no meta.json"
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, "meta.json unreadable"
    done = int(meta.get("shards_done", 0))
    total = int(meta.get("shards_total", 0))
    expected_grid = window_grid_for_family(family)
    if meta.get("window_grid") != expected_grid:
        return False, (
            f"stale window grid {meta.get('window_grid')} != {expected_grid}")
    expected_recipe = inference_recipe(family)
    if meta.get("inference_recipe") != expected_recipe:
        return False, (
            f"stale inference recipe {meta.get('inference_recipe')!r} != "
            f"{expected_recipe!r}")
    pct = 100.0 * done / total if total > 0 else 0.0
    if total > 0 and done >= total:
        return True, f"{done}/{total} shards ({pct:.1f}%)"
    return False, f"{done}/{total} shards ({pct:.1f}%)"


def _done_stage_2(family: str, display: str = "") -> Tuple[bool, str]:
    """Stage 2 is done when the predictor's final artefacts are persisted.
    Trials are sweep-internal cache; what downstream stages need is
    best_model.pt + best_config.json."""
    fdir = os.path.join(PREDICTOR_ROOT, display)
    have_model = os.path.isfile(os.path.join(fdir, "best_model.pt"))
    have_cfg   = os.path.isfile(os.path.join(fdir, "best_config.json"))
    trial_dir  = os.path.join(fdir, "trials")
    n_trials = (sum(1 for n in os.listdir(trial_dir)
                    if n.startswith("trial_") and n.endswith(".json"))
                if os.path.isdir(trial_dir) else 0)
    if have_model and have_cfg:
        try:
            with open(os.path.join(fdir, "best_config.json")) as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False, "best_config.json unreadable"
        expected_recipe = inference_recipe(family)
        if cfg.get("label_inference_recipe") != expected_recipe:
            return False, "predictor was trained on stale inference labels"
        expected_arch = os.environ.get(
            "PREDICTCSL_PREDICTOR_ARCH", "patchtst").lower()
        if cfg.get("arch", "patchtst") != expected_arch:
            return False, "predictor architecture is stale"
        expected_objective = os.environ.get(
            "PREDICTCSL_TRAINING_OBJECTIVE", "curve").lower()
        if cfg.get("training_objective", "curve") != expected_objective:
            return False, "predictor training objective is stale"
        # Old v4 artifacts predate this flag and came from the unconstrained
        # search. Do not let them make a new master --cheap run skip stage 2.
        expected_cheap_mamba = (
            expected_arch == "mamba"
            and os.environ.get("PREDICTCSL_CHEAP_PREDICTOR") == "1"
        )
        if expected_cheap_mamba and cfg.get("cheap_search") is not True:
            return False, "predictor came from the old unconstrained Mamba search"
        expected_grid = window_grid_for_family(family)
        if cfg.get("window_grid") != expected_grid:
            return False, "predictor output grid is stale"
        if (expected_objective == "risk"
                and cfg.get("risk_selection_version") != 1):
            return False, "risk predictor predates tail-aware model selection"
        if expected_objective == "risk":
            expected_policy = float(os.environ.get(
                "PREDICTCSL_RISK_POLICY_WEIGHT", "1.0"))
            expected_harm = float(os.environ.get(
                "PREDICTCSL_RISK_FULL_HARM_WEIGHT", "2.0"))
            expected_temperature = float(os.environ.get(
                "PREDICTCSL_RISK_SOFTMAX_TEMPERATURE", "0.25"))
            expected_p90 = float(os.environ.get(
                "PREDICTCSL_RISK_SELECTION_P90_WEIGHT", "0.5"))
            expected_rate = float(os.environ.get(
                "PREDICTCSL_RISK_SELECTION_HARM_RATE_WEIGHT", "0.25"))
            if (cfg.get("risk_policy_weight") != expected_policy
                    or cfg.get("risk_full_harm_weight") != expected_harm
                    or cfg.get("risk_softmax_temperature") != expected_temperature
                    or cfg.get("risk_selection_p90_weight") != expected_p90
                    or cfg.get("risk_selection_harm_rate_weight") != expected_rate):
                return False, "risk predictor uses stale risk/selection weights"
        return True, f"best_model.pt + {n_trials} trials"
    return False, (
        "missing best_model.pt/best_config.json "
        f"({n_trials} trials cached so far)"
    )


def _done_stage_3(family: str, display: str = "") -> Tuple[bool, str]:
    """Stage 3 (for one model) is done when v5 wrote that model's
    compare_real_vs_predicted/compare_summary.csv into the shared general
    folder — the terminal per-model artefact of v5's main()."""
    compare_dir = os.path.join(
        ABLATION_GENERAL, "models", display, "compare_real_vs_predicted")
    marker = os.path.join(compare_dir, "compare_summary.csv")
    if not os.path.isfile(marker):
        return False, "no compare_summary.csv"
    try:
        with open(marker, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return False, "compare_summary.csv unreadable"
    if not rows:
        return False, "empty compare_summary.csv"
    expected_cells = {
        (display_name, term)
        for _ge_name, term, display_name, _to_univariate
        in datasets_config.datasets_to_run()
    }
    actual_cells = {
        (str(row.get("dataset_display")), str(row.get("term"))) for row in rows
    }
    if actual_cells != expected_cells:
        return False, (
            f"compare cohort is stale ({len(actual_cells)}/{len(expected_cells)} cells)")
    expected_recipe = inference_recipe(family)
    current = 0
    for row in rows:
        if expected_recipe is not None:
            if row.get("inference_recipe") != expected_recipe:
                return False, "compare artifacts use stale model forecasts"
        term = str(row["term"])
        dataset_display = str(row["dataset_display"])
        metrics_path = os.path.join(
            ABLATION_GENERAL, "datasets", dataset_display, display,
            f"t{term}", "wfull_native", "metrics.json")
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False, f"current full-native caches {current}/{len(rows)}"
        if metrics.get("_metric_suite_ver", 0) < METRIC_SUITE_VER:
            return False, (
                "stale metric cache "
                f"({dataset_display}/t{term}; rerun stage 3)")
        if expected_recipe is not None:
            if metrics.get("_inference_recipe") != expected_recipe:
                return False, (
                    "stale model inference cache "
                    f"({dataset_display}/t{term})")
        current += 1
    n_npz = sum(1 for n in os.listdir(compare_dir) if n.endswith(".npz"))
    return True, f"{n_npz} comparison .npz files"


def _done_stage_4(family: str, display: str = "") -> Tuple[bool, str]:
    """Stage 4 is done when its summary uses the current normalization source."""
    out = os.path.join(ABLATION_ROOT, display, STRATEGY_SUBDIR,
                       "summary_stats.json")
    if os.path.isfile(out):
        try:
            with open(out) as f:
                stats = json.load(f)
            reference = stats.get("headline_aggregation", {}).get("reference")
            cohort_size = stats.get("headline_aggregation", {}).get("cohort_size")
        except (OSError, json.JSONDecodeError):
            return False, "summary_stats.json unreadable"
        if reference == NORMALIZATION_REFERENCE:
            expected_size = len(datasets_config.datasets_to_run())
            if cohort_size != expected_size:
                return False, (
                    f"summary cohort is stale ({cohort_size}/{expected_size} cells)")
            expected_recipe = inference_recipe(family)
            if expected_recipe is not None:
                recipe = stats.get("inference_recipes", {}).get(family)
                if recipe != expected_recipe:
                    return False, "summary_stats.json uses stale model forecasts"
            return True, "summary_stats.json uses published naive CSV"
        return False, "summary_stats.json uses stale/local naive denominator"
    return False, "summary_stats.json missing"


DONE_CHECKS: dict[str, Callable[..., Tuple[bool, str]]] = {
    "1": _done_stage_1,
    "2": _done_stage_2,
    "3": _done_stage_3,
    "4": _done_stage_4,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--models", nargs="+", default=None,
        help="Subset of display names from MODELS_TO_RUN to actually run.",
    )
    p.add_argument(
        "--skip-stages", nargs="+", default=[], choices=list(STAGES),
        help="Stage numbers to skip (1=build, 2=predictor, 3=ablation, 4=compare).",
    )
    p.add_argument(
        "--only-stages", nargs="+", default=None, choices=list(STAGES),
        help="If set, run only these stages (mutually exclusive with --skip-stages).",
    )
    p.add_argument(
        "--force", nargs="*", default=None, choices=list(STAGES),
        help=("Re-run stages even when their done-marker is present. "
              "Pass stage numbers (e.g. --force 2 3) or use --force with no "
              "argument to force every active stage."),
    )
    p.add_argument(
        "--mase-metric", choices=["mase_gluonts", "mase_gluonts_real"],
        default="mase_gluonts_real",
        help=("Which MASE drives stage 4 (compare). Exactly two exist: the ported "
              "leaderboard `mase_gluonts` and the gluonts-machinery "
              "`mase_gluonts_real` (default — the leaderboard-faithful one). Each "
              "writes to its own strategy_comparison_gluonts[_real]/ subdir so "
              "they never collide, and stage 4 always emits the other metric's "
              "twin files alongside. The ablation computes ALL columns, so this "
              "only affects stage 4."),
    )
    p.add_argument(
        "--short-context-mode", choices=["skip", "pad"], default="pad",
        help=("Stage-3 handling of instances shorter than the ablation window. "
              "'skip': exclude them at that window (original behaviour). "
              "'pad' (default): never skip — feed each instance its available context "
              "(PatchTST-FM NaN-padded to native context). Threaded to stage 3."),
    )
    p.add_argument(
        "--test", action="store_true",
        help=("Smoke test: run the full pipeline for every selected model and "
              "dataset, but dramatically reduced (smallest+largest window only, "
              f"{TEST_N_SERIES} synthetic series, a 2-trial/3-epoch predictor "
              "sweep, no plots, no per-cell cache). All output is redirected into "
              f"{TEST_ROOT}/ and deleted afterwards (see --keep-test-output), so "
              "real datasets/results are never touched."),
    )
    p.add_argument(
        "--keep-test-output", action="store_true",
        help=f"With --test, do not delete {TEST_ROOT}/ at the end (for debugging).",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help=("Stream each stage's subprocess output to the terminal (old "
              "behaviour). Default is quiet: only the top-level tqdm bar shows, "
              f"and per-stage output is captured under {RUN_LOG_ROOT}/."),
    )
    p.add_argument(
        "--verbose-ablation", action="store_true",
        help=("Stream stage 3's subprocess output live while keeping all other "
              "stages on their compact progress bars. Useful for monitoring "
              "dataset loading, cache hits, GPU inference, and per-cell timing."),
    )
    # Pass-through extras per stage.
    for sid, (name, _) in STAGES.items():
        p.add_argument(
            f"--{name}-args", nargs=argparse.REMAINDER, default=[],
            help=f"Extra CLI args appended to stage {sid} ({name}).",
        )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    global _QUIET, _VERBOSE_ABLATION
    _QUIET = not args.verbose
    _VERBOSE_ABLATION = args.verbose_ablation

    # ---- Smoke-test mode --------------------------------------------------
    # Redirect every stage's output into a throwaway tree and switch the stage
    # scripts into their reduced "test" configuration. Stage scripts pick up the
    # reductions/roots from these env vars at *import* time, which is the only
    # spawn-safe channel: build/predictor fan out to per-GPU `mp spawn` workers
    # that re-import the module, so a main()-level override would not reach them
    # but an inherited env var does. Stages 3/4 take their roots via CLI (set on
    # the globals below). Everything lands under TEST_ROOT and is deleted in the
    # finally-block unless --keep-test-output is given.
    global DATASET_ROOT, PREDICTOR_ROOT, ABLATION_ROOT, RUN_LOG_ROOT
    if args.test:
        DATASET_ROOT   = os.path.join(TEST_ROOT, "context_length_dataset")
        PREDICTOR_ROOT = os.path.join(TEST_ROOT, "context_length_predictor")
        ABLATION_ROOT  = os.path.join(TEST_ROOT, "window_ablation_gifteval")
        RUN_LOG_ROOT   = os.path.join(TEST_ROOT, "run_all_logs")
        os.environ["PREDICTCSL_TEST"]            = "1"
        os.environ["PREDICTCSL_DATASET_ROOT"]    = DATASET_ROOT
        os.environ["PREDICTCSL_PREDICTOR_ROOT"]  = PREDICTOR_ROOT
        print(Fore.YELLOW
              + f"SMOKE TEST: reduced full pipeline -> {TEST_ROOT}/ "
              + ("(kept)" if args.keep_test_output else "(deleted afterwards)")
              + Fore.RESET)

    # Route pad-mode artefacts to a separate run dir + strategy subfolder so the
    # two short-context strategies never share (and silently reuse) caches. Stage
    # 1/2 are unaffected — the predictor is trained on stage-1 labels regardless.
    # Recompute from the (possibly test-overridden) ABLATION_ROOT — the module
    # default was derived from the real root at import.
    global ABLATION_GENERAL, STRATEGY_SUBDIR
    if args.short_context_mode == "pad":
        ABLATION_GENERAL = os.path.join(ABLATION_ROOT, "general_pad")
        STRATEGY_SUBDIR = "strategy_comparison_pad"
        print(Fore.CYAN + f"Short-context mode: pad  ->  run dir {ABLATION_GENERAL}"
              + Fore.RESET)
    else:
        ABLATION_GENERAL = os.path.join(ABLATION_ROOT, "general")

    # Both metrics share the SAME ablation cells (just different metric columns).
    # The default `mase_gluonts_real` (leaderboard-faithful) OWNS the plain
    # strategy_comparison/ subdir — it replaces the legacy custom-`mase` outputs
    # there, and downstream consumers that hardcode the plain path (the robust
    # timing stage reads <display>/strategy_comparison/comparison.csv) keep
    # working. Only the non-default port gets a suffixed subdir.
    _MASE_SUBDIR_SUFFIX = {"mase_gluonts": "_gluonts"}
    if args.mase_metric in _MASE_SUBDIR_SUFFIX:
        STRATEGY_SUBDIR = STRATEGY_SUBDIR + _MASE_SUBDIR_SUFFIX[args.mase_metric]
    print(Fore.CYAN + f"MASE metric: {args.mase_metric}  ->  compare subdir {STRATEGY_SUBDIR}"
          + Fore.RESET)

    if args.only_stages and args.skip_stages:
        raise SystemExit("Use either --skip-stages or --only-stages, not both.")
    active = (set(args.only_stages) if args.only_stages
              else set(STAGES) - set(args.skip_stages))

    selected = [m for m in MODELS_TO_RUN
                if args.models is None or m[2] in args.models]
    if not selected:
        raise SystemExit(
            f"No models selected. MODELS_TO_RUN={[m[2] for m in MODELS_TO_RUN]} "
            f"--models={args.models}")

    extras_by_stage = {
        "1": list(args.build_args),
        "2": list(args.predictor_args),
        "3": list(args.ablation_args) + ["--short-context-mode", args.short_context_mode],
        "4": list(args.compare_args) + ["--mase-metric", args.mase_metric],
    }
    if args.test:
        # Stage 1: few synthetic series (windows are collapsed via PREDICTCSL_TEST).
        # Stage 2: trial/epoch counts collapse via PREDICTCSL_TEST (no CLI needed).
        # Stage 3: smallest+largest window come from the predictor grid; sample a
        #          few datasets (fresh seed each run so coverage rotates, but the
        #          same seed reaches every sharded worker via the coordinator's
        #          forwarded argv); skip plots and the per-cell cache.
        test_ds_seed = random.randrange(1_000_000)
        extras_by_stage["1"] += ["--n-series", str(TEST_N_SERIES)]
        extras_by_stage["3"] += [
            "--no-plots", "--no-cell-cache",
            "--test-datasets", str(TEST_N_DATASETS),
            "--test-datasets-seed", str(test_ds_seed),
        ]
        print(Fore.YELLOW
              + f"SMOKE TEST: sampling {TEST_N_DATASETS} datasets (seed={test_ds_seed})"
              + Fore.RESET)

    # --force semantics: None = no override, [] = force all active, [...] = force listed
    if args.force is None:
        forced = set()
    elif args.force == []:
        forced = set(STAGES)
    else:
        forced = set(args.force)

    print(Fore.CYAN + f"Models: {[m[2] for m in selected]}" + Fore.RESET)
    print(Fore.CYAN + f"Active stages: {sorted(active)}" + Fore.RESET)
    if forced:
        print(Fore.YELLOW + f"Forced re-runs: {sorted(forced)}" + Fore.RESET)

    def _maybe_run(sid: str, model_id: str, family: str, display: str) -> None:
        name = STAGES[sid][0]
        label = f"{display} · {sid}/{name}"
        done, summary = DONE_CHECKS[sid](family, display)
        if done and sid not in forced:
            # Cached units finish instantly — one line, not a live bar.
            tqdm.write(Fore.WHITE
                       + f"{label} … · cached ({summary}) — skipping"
                       + Fore.RESET)
            return
        if done and sid in forced:
            _emit(Fore.YELLOW
                  + f"  ! {label} cached ({summary}) but --force given — re-running"
                  + Fore.RESET)
        extras = extras_by_stage[sid]
        if sid == "1":
            stage_1_build(model_id, display, family, extras)
        elif sid == "2":
            stage_2_predictor(display, family, extras)
        elif sid == "3":
            stage_3_ablation(display, family, extras)
        else:
            stage_4_compare(display, family, extras)

    # Each (model × active stage) unit gets its own tqdm bar (created in _run),
    # so a full run leaves one persistent line per stage+model combination.
    ordered_stages = [s for s in ("1", "2", "3", "4") if s in active]

    t_start = time.perf_counter()
    try:
        for model_id, family, display in selected:
            if not _QUIET:
                _banner(f"{display}  ({family})  —  {model_id}")
            for sid in ordered_stages:
                _maybe_run(sid, model_id, family, display)

        # Cross-model final pass: the single overview figure + grand-total CSV
        # spanning every model in the shared run dir. Always runs when stage 4 is
        # active (it's cheap post-processing and the per-model passes never emit
        # the cross-model artefacts).
        if "4" in active:
            if not _QUIET:
                _banner("ALL MODELS  —  cross-model overview")
            # Default `mase_gluonts_real` owns the plain general/ rollup (the
            # canonical, leaderboard-faithful overview); the port goes to its own
            # subdir so the two never overwrite each other.
            _rollup_subdir = {"mase_gluonts": "rollup_gluonts"}
            rollup_dir = (os.path.join(ABLATION_GENERAL, _rollup_subdir[args.mase_metric])
                          if args.mase_metric in _rollup_subdir else ABLATION_GENERAL)
            stage_4_rollup(
                extras_by_stage["4"], out_dir=rollup_dir,
                models=[display for _model_id, _family, display in selected],
            )

        total = time.perf_counter() - t_start
        print(Fore.GREEN + f"\nAll done in {total/60:.1f} min." + Fore.RESET)
    finally:
        # Remove the throwaway smoke-test tree whether the run succeeded or a
        # stage raised — the point of --test is to leave nothing behind.
        if args.test and not args.keep_test_output:
            shutil.rmtree(TEST_ROOT, ignore_errors=True)
            print(Fore.YELLOW + f"SMOKE TEST: removed {TEST_ROOT}/" + Fore.RESET)
        elif args.test:
            print(Fore.YELLOW + f"SMOKE TEST: output kept at {TEST_ROOT}/"
                  + Fore.RESET)


if __name__ == "__main__":
    main()
