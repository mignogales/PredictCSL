#!/usr/bin/env bash
# Recompute only the long-tail synthetic sweeps on their extended ratio grid.
# This intentionally overwrites Delay, Horizon, and SNR cells: their result
# arrays gain r={24,32,48,64}, while every other experiment is left untouched.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_BIN="${CONDA_BIN:-$REPO_ROOT/../../miniconda3/bin/conda}"
MAIN_ENV="${MAIN_ENV:-predictcsl-main}"
PATCHTST_ENV="${PATCHTST_ENV:-TSFM_PATCH}"
LEGACY_ENV="${LEGACY_ENV:-predictcsl-legacy}"
TOTO_ENV="${TOTO_ENV:-predictcsl-toto}"
TIREX_ENV="${TIREX_ENV:-predictcsl-tirex}"
GPU_ID="${GPU_ID:-0}"
REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-1}"
WAIT_FOR_IDLE_GPU="${WAIT_FOR_IDLE_GPU:-1}"
SWEEP_ROOT="${PREDICTCSL_SWEEP_ROOT:-logs/experiments/synth_param_sweeps}"
LOG_DIR="${LOG_DIR:-logs/experiments/run_logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/extended_synth_tail_sweeps_$(date +%Y%m%d_%H%M%S).log"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PREDICTCSL_SWEEP_ROOT="$SWEEP_ROOT"

if [[ "$REQUIRE_IDLE_GPU" == "1" ]] && command -v nvidia-smi >/dev/null; then
  GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
    | awk -F, -v gpu_idx="$GPU_ID" '$1 + 0 == gpu_idx {gsub(/ /, "", $2); print $2}')"
  if [[ -z "$GPU_UUID" ]]; then
    echo "Could not resolve physical GPU $GPU_ID." >&2
    exit 2
  fi
  while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
      | tr -d ' ' | grep -Fxq "$GPU_UUID"; do
    if [[ "$WAIT_FOR_IDLE_GPU" != "1" ]]; then
      echo "Physical GPU $GPU_ID already has a compute process; refusing to overlap." >&2
      exit 2
    fi
    echo "Physical GPU $GPU_ID is busy; waiting 60 seconds before retrying."
    sleep 60
  done
fi

run_group() {
  local env_name="$1"
  shift
  "$CONDA_BIN" run --no-capture-output -n "$env_name" \
    python -m experiments.synth_param_sweeps \
    --force --experiments delay horizon snr --models "$@" --device cuda:0
}

{
  echo "Extended synthetic tail sweeps"
  echo "physical GPU: $GPU_ID (visible as cuda:0)"
  echo "output root: $SWEEP_ROOT"
  run_group "$MAIN_ENV" \
    Chronos2-Small Moirai2-Small TimesFM2.5-200M Chronos2-Synth \
    Chronos2-Base ChronosBolt-Base FlowState-R1
  run_group "$PATCHTST_ENV" PatchTST-FM-R1
  run_group "$LEGACY_ENV" Sundial-Base-128M
  run_group "$TOTO_ENV" Toto-2.0-313m
  run_group "$TIREX_ENV" TiRex2

  "$CONDA_BIN" run --no-capture-output -n "$MAIN_ENV" \
    python -m experiments.synth_param_sweeps --plot-only
  "$CONDA_BIN" run --no-capture-output -n "$MAIN_ENV" \
    python -m experiments.plot_synth_sweep_results --root "$SWEEP_ROOT"
  "$CONDA_BIN" run --no-capture-output -n "$MAIN_ENV" \
    python -m experiments.plot_synth_sweep_alignment --root "$SWEEP_ROOT"
  "$CONDA_BIN" run --no-capture-output -n "$MAIN_ENV" \
    python -m experiments.plot_synth_forecast_examples --device cuda:0 \
    --output-dir "$SWEEP_ROOT/summary_plots"
} 2>&1 | tee "$LOG_FILE"

echo "Complete. Log: $LOG_FILE"
