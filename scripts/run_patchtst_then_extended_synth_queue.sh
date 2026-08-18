#!/usr/bin/env bash
# CUDA-0 queue: complete PatchTST first, then extend Delay/Horizon/SNR and
# regenerate every synthetic-sweep report.  It never overlaps another job.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU_ID="${GPU_ID:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"

gpu_busy() {
  local uuid
  uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
    | awk -F, -v gpu_idx="$GPU_ID" '$1 + 0 == gpu_idx {gsub(/ /, "", $2); print $2}')"
  [[ -n "$uuid" ]] && nvidia-smi --query-compute-apps=gpu_uuid \
    --format=csv,noheader,nounits | tr -d ' ' | grep -Fxq "$uuid"
}

while gpu_busy; do
  echo "CUDA $GPU_ID is busy; PatchTST queue will retry in ${POLL_SECONDS}s."
  sleep "$POLL_SECONDS"
done

GPU_ID="$GPU_ID" REQUIRE_IDLE_GPU=1 bash scripts/run_patchtst_synth_sweeps_and_plot.sh
GPU_ID="$GPU_ID" REQUIRE_IDLE_GPU=1 WAIT_FOR_IDLE_GPU=1 \
  bash scripts/run_extended_synth_tail_sweeps.sh
