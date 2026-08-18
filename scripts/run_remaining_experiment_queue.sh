#!/usr/bin/env bash
# Run the remaining paper experiments sequentially on the GPU server.
#
# Order:
#   1. wait for the active master recomputation (unless WAIT_FOR_MASTER=0)
#   2. inference-free oracle-distribution analysis
#   3. all 12 synthetic single-factor sweeps, routed through the pinned envs
#   4. the main interpretability set (Exp1 + Exp4 + Exp7), all active models
#
# Every underlying experiment is resumable. Re-running this launcher fills only
# missing cells and regenerates aggregate plots. Physical GPU 0 is the default;
# CUDA remaps it to cuda:0 inside every child process.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MAIN_ENV="${MAIN_ENV:-predictcsl-main}"
PATCHTST_ENV="${PATCHTST_ENV:-TSFM_PATCH}"
LEGACY_ENV="${LEGACY_ENV:-predictcsl-legacy}"
TOTO_ENV="${TOTO_ENV:-predictcsl-toto}"
TIREX_ENV="${TIREX_ENV:-predictcsl-tirex}"

# Background SSH shells on ando do not necessarily source Conda's shell hook.
# Resolve the executable once instead of relying on the interactive PATH.
CONDA_BIN="${CONDA_BIN:-}"
if [[ -z "$CONDA_BIN" ]]; then
  CONDA_BIN="$(command -v conda 2>/dev/null || true)"
fi
if [[ -z "$CONDA_BIN" && -x "$REPO_ROOT/../../miniconda3/bin/conda" ]]; then
  CONDA_BIN="$REPO_ROOT/../../miniconda3/bin/conda"
fi
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "Could not find a Conda executable; set CONDA_BIN explicitly." >&2
  exit 2
fi

MASTER_ROOT="${MASTER_ROOT:-logs/experiments/master_recompute}"
CANONICAL_CACHE="${CANONICAL_CACHE:-$MASTER_ROOT/window_ablation_gifteval/general}"
SWEEP_ROOT="${PREDICTCSL_SWEEP_ROOT:-logs/experiments/synth_param_sweeps}"
INTERPRETABILITY_OUT="${INTERPRETABILITY_OUT:-logs/experiments/context_interpretability}"
GPU_ID="${GPU_ID:-0}"
WAIT_FOR_MASTER="${WAIT_FOR_MASTER:-1}"
MASTER_PATTERN="${MASTER_PATTERN:-python -m experiments.master_run_all --pipeline-only}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PREDICTCSL_SWEEP_ROOT="$SWEEP_ROOT"

LOG_DIR="$MASTER_ROOT/launch_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/remaining_experiment_queue_${STAMP}.log"
LOCK_DIR="$LOG_DIR/.remaining_experiment_queue.lock"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another remaining-experiment queue appears active: $LOCK_DIR" >&2
  exit 2
fi
cleanup() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

log() {
  printf '%s\n' "$*" | tee -a "$LOG_FILE"
}

run_conda() {
  local env_name="$1"
  shift
  log "+ conda run --no-capture-output -n $env_name python $*"
  "$CONDA_BIN" run --no-capture-output -n "$env_name" python "$@" 2>&1 \
    | tee -a "$LOG_FILE"
}

failures=()
run_phase() {
  local label="$1"
  shift
  log ""
  log "=============================================================================="
  log "$label"
  log "=============================================================================="
  if "$@"; then
    log "DONE: $label"
  else
    local status=$?
    failures+=("$label (exit $status)")
    log "FAILED: $label (exit $status); continuing with the queue"
  fi
}

wait_for_master() {
  if [[ "$WAIT_FOR_MASTER" != "1" ]]; then
    log "WAIT_FOR_MASTER=$WAIT_FOR_MASTER; not waiting for the master pipeline."
    return
  fi
  local polls=0
  while pgrep -f "$MASTER_PATTERN" >/dev/null; do
    if (( polls % 10 == 0 )); then
      log "Master recomputation is still active; checking again in 60 seconds."
    fi
    sleep 60
    ((polls += 1))
  done
  log "Master recomputation is no longer active; starting queued experiments."
}

do_oracle() {
  [[ -d "$CANONICAL_CACHE/datasets" ]] || {
    log "Missing canonical cache: $CANONICAL_CACHE/datasets"
    return 2
  }
  run_conda "$MAIN_ENV" -m experiments.analyze_oracle_distributions \
    --run-dir "$CANONICAL_CACHE" \
    --output-dir "$CANONICAL_CACHE/oracle_distribution_analysis" \
    --split-repeats 200
}

do_sweep_group() {
  local env_name="$1"
  shift
  run_conda "$env_name" -m experiments.synth_param_sweeps \
    --models "$@" --device cuda:0
}

do_sweep_plots() {
  run_conda "$MAIN_ENV" -m experiments.synth_param_sweeps \
    --plot-only --models \
    Chronos2-Small Moirai2-Small TimesFM2.5-200M PatchTST-FM-R1 \
    Sundial-Base-128M Chronos2-Synth Chronos2-Base ChronosBolt-Base \
    Toto-2.0-313m FlowState-R1 TiRex2
  run_conda "$MAIN_ENV" -m experiments.plot_synth_sweep_results \
    --root "$SWEEP_ROOT"
  run_conda "$MAIN_ENV" -m experiments.plot_synth_sweep_alignment \
    --root "$SWEEP_ROOT" --per-cell
}

report_sweep_coverage() {
  local completed
  completed="$(find "$SWEEP_ROOT" -mindepth 3 -maxdepth 3 -name done.json \
    -type f 2>/dev/null | wc -l | tr -d ' ')"
  log "Synthetic sweep coverage: $completed/132 model-experiment cells have done.json."
  if [[ "$completed" != "132" ]]; then
    log "WARNING: the sweep matrix is incomplete; rerun this queue to retry missing cells."
  fi
}

do_interpretability() {
  # Include Chronos2-Small so the final analysis contains the existing reference.
  # Its completed Exp1/Exp4/Exp7 cells are skipped by the underlying runner.
  run_conda "$MAIN_ENV" -m context_interpretability.master_run \
    --models \
    Chronos2-Small Moirai2-Small TimesFM2.5-200M PatchTST-FM-R1 \
    Sundial-Base-128M Chronos2-Synth Chronos2-Base ChronosBolt-Base \
    Toto-2.0-313m FlowState-R1 TiRex2 \
    --experiments exp1 exp4 exp7 \
    --source synthetic \
    --out "$INTERPRETABILITY_OUT" \
    --device cuda:0
}

log "Remaining experiment queue"
log "repository: $REPO_ROOT"
log "conda: $CONDA_BIN"
log "physical GPU: $GPU_ID (visible as cuda:0)"
log "log: $LOG_FILE"
wait_for_master

run_phase "1/8 Oracle-distribution analysis (CPU/inference-free)" do_oracle

# synth_param_sweeps imports each TSFM implementation, so keep the same
# cross-environment routing as master_run_all.py.
run_phase "2/8 Synthetic sweeps — main models" do_sweep_group "$MAIN_ENV" \
  Chronos2-Small Moirai2-Small TimesFM2.5-200M Chronos2-Synth \
  Chronos2-Base ChronosBolt-Base FlowState-R1
run_phase "3/8 Synthetic sweeps — PatchTST-FM" do_sweep_group "$PATCHTST_ENV" \
  PatchTST-FM-R1
run_phase "4/8 Synthetic sweeps — Sundial" do_sweep_group "$LEGACY_ENV" \
  Sundial-Base-128M
run_phase "5/8 Synthetic sweeps — Toto" do_sweep_group "$TOTO_ENV" \
  Toto-2.0-313m
run_phase "6/8 Synthetic sweeps — TiRex2" do_sweep_group "$TIREX_ENV" \
  TiRex2
run_phase "7/8 Synthetic sweep aggregate plots" do_sweep_plots
report_sweep_coverage

run_phase "8/8 Interpretability Exp1 + Exp4 + Exp7 — all active models" \
  do_interpretability

log ""
if (( ${#failures[@]} > 0 )); then
  log "Queue finished with ${#failures[@]} failed phase(s):"
  printf '  - %s\n' "${failures[@]}" | tee -a "$LOG_FILE"
  log "Re-run the same script after fixing the reported error; completed cells resume."
  exit 1
fi

log "Queue finished successfully. Consolidated log: $LOG_FILE"
