#!/usr/bin/env bash
# Add TiRex2 to the exact-bucket smart-batching profile and refresh the rollup.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
GPU_ID="${GPU_ID:-3}"
CONDA_ROOT="${CONDA_ROOT:-/home/mnogales/miniconda3/envs}"
ROOT="logs/experiments/master_recompute/context_selection_reviewer_followups"
OUT="$ROOT/smart_batch_profile"
HISTOGRAM="$ROOT/fronts/tirex2/full_score/selected_window_histograms.csv"
LOG="$ROOT/launch_logs/smart_batch_tirex_gpu${GPU_ID}.log"

mkdir -p "$OUT" "$(dirname "$LOG")"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

printf '%s START TiRex2 smart-batch profile on GPU %s\n' \
  "$(date --iso-8601=seconds)" "$GPU_ID" | tee -a "$LOG"
PYTHONPATH=. "$CONDA_ROOT/predictcsl-tirex/bin/python" \
  -m experiments.profile_all_models_forward \
  --models TiRex2 --bucket-histogram "$HISTOGRAM" \
  --warmup 1 --repeats 3 --batch-size 32 --output-dir "$OUT" \
  >>"$LOG" 2>&1

PYTHONPATH=. "$CONDA_ROOT/predictcsl-main/bin/python" \
  -m experiments.profile_all_models_forward --combine-only --output-dir "$OUT" \
  >>"$LOG" 2>&1
PYTHONPATH=. "$CONDA_ROOT/predictcsl-main/bin/python" \
  -m experiments.summarize_smart_batch_profiles \
  --profile-dir "$OUT" --histogram-root "$ROOT/fronts" \
  --output "$OUT/smart_batch_policy_speedups.csv" \
  >>"$LOG" 2>&1
printf '%s COMPLETE TiRex2 smart-batch profile\n' \
  "$(date --iso-8601=seconds)" | tee -a "$LOG"
