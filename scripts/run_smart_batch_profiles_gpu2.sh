#!/usr/bin/env bash
# Measure the exact execution buckets needed by the precomputed GiftEval policy.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
GPU_ID="${GPU_ID:-2}"
CONDA_ROOT="${CONDA_ROOT:-/home/mnogales/miniconda3/envs}"
ROOT="logs/experiments/master_recompute/context_selection_reviewer_followups"
OUT="$ROOT/smart_batch_profile"
LOG="$ROOT/launch_logs/smart_batch_profile_gpu${GPU_ID}.log"
STATUS="$ROOT/launch_logs/smart_batch_profile_gpu${GPU_ID}_status.tsv"
mkdir -p "$OUT" "$(dirname "$LOG")"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

run_model() {
  local env_name="$1" model="$2" slug="$3"
  local histogram="$ROOT/fronts/$slug/full_score/selected_window_histograms.csv"
  printf '%s START %s\n' "$(date --iso-8601=seconds)" "$model" | tee -a "$LOG"
  if PYTHONPATH=. "$CONDA_ROOT/$env_name/bin/python" \
      -m experiments.profile_all_models_forward \
      --models "$model" --bucket-histogram "$histogram" \
      --warmup 1 --repeats 3 --batch-size 32 --output-dir "$OUT" \
      >>"$LOG" 2>&1; then
    printf '%s\tcomplete\n' "$model" >>"$STATUS"
  else
    local code=$?
    printf '%s\tfailed:%s\n' "$model" "$code" >>"$STATUS"
  fi
}

run_model predictcsl-main Chronos2-Base chronos2_base
run_model predictcsl-main Chronos2-Small chronos2_small
run_model predictcsl-main Chronos2-Synth chronos2_synth
run_model predictcsl-main ChronosBolt-Base chronos_bolt_base
run_model predictcsl-main FlowState-R1 flowstate_r1
run_model predictcsl-main Moirai2-Small moirai2_small
run_model TSFM_PATCH PatchTST-FM-R1 patchtst_fm_r1
run_model predictcsl-legacy Sundial-Base-128M sundial_base_128m
run_model predictcsl-main TimesFM2.5-200M timesfm2p5
run_model predictcsl-toto Toto-2.0-313m toto_2_0_313m
run_model predictcsl-tirex TiRex2 tirex2

PYTHONPATH=. "$CONDA_ROOT/predictcsl-main/bin/python" \
  -m experiments.profile_all_models_forward --combine-only --output-dir "$OUT" \
  >>"$LOG" 2>&1
printf '%s ALL COMPLETE\n' "$(date --iso-8601=seconds)" | tee -a "$LOG"
