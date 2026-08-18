#!/usr/bin/env bash
# Fill the 12 missing PatchTST sweep cells, then regenerate all-model plots.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_BIN="${CONDA_BIN:-$REPO_ROOT/../../miniconda3/bin/conda}"
PATCHTST_ENV="${PATCHTST_ENV:-TSFM_PATCH}"
GPU_ID="${GPU_ID:-0}"
REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-1}"
SWEEP_ROOT="${PREDICTCSL_SWEEP_ROOT:-logs/experiments/synth_param_sweeps}"
LOG_DIR="${LOG_DIR:-logs/experiments/run_logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/patchtst_synth_sweeps_$(date +%Y%m%d_%H%M%S).log"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PREDICTCSL_SWEEP_ROOT="$SWEEP_ROOT"

if [[ "$REQUIRE_IDLE_GPU" == "1" ]] && command -v nvidia-smi >/dev/null; then
  GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
    | awk -F, -v gpu_idx="$GPU_ID" '$1 + 0 == gpu_idx {gsub(/ /, "", $2); print $2}')"
  if [[ -z "$GPU_UUID" ]]; then
    echo "Could not resolve physical GPU $GPU_ID." >&2
    exit 2
  fi
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
      | tr -d ' ' | grep -Fxq "$GPU_UUID"; then
    echo "Physical GPU $GPU_ID already has a compute process; refusing to overlap." >&2
    echo "Wait for it to become idle or set REQUIRE_IDLE_GPU=0 explicitly." >&2
    exit 2
  fi
fi

"$CONDA_BIN" run -n "$PATCHTST_ENV" python -c \
  'import torch; import experiments.synth_param_sweeps' >/dev/null

{
  echo "PatchTST synthetic sweeps"
  echo "physical GPU: $GPU_ID (visible as cuda:0)"
  echo "environment: $PATCHTST_ENV"
  echo "output root: $SWEEP_ROOT"
  "$CONDA_BIN" run --no-capture-output -n "$PATCHTST_ENV" \
    python -m experiments.synth_param_sweeps \
    --models PatchTST-FM-R1 --device cuda:0

  STRICT=1 PREDICTCSL_SWEEP_ROOT="$SWEEP_ROOT" \
    bash scripts/plot_synth_sweep_results.sh
} 2>&1 | tee "$LOG_FILE"

echo "Complete. Log: $LOG_FILE"
