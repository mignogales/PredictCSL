#!/usr/bin/env bash
# Run one or both counterfactual direct-backbone timing profiles.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
GPU_ID="${GPU_ID:-3}"
MODEL="${MODEL:-all}"
CONDA_ROOT="${CONDA_ROOT:-/home/mnogales/miniconda3/envs}"
ROOT="logs/experiments/master_recompute/context_selection_reviewer_followups"
OUT="$ROOT/unpadded_backbone_counterfactual"
LOG="$ROOT/launch_logs/unpadded_backbone_${MODEL}_gpu${GPU_ID}.log"
mkdir -p "$OUT" "$(dirname "$LOG")"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

run_one() {
  local env_name="$1" model="$2" slug="$3"
  printf '%s START %s on GPU %s\n' "$(date --iso-8601=seconds)" "$model" "$GPU_ID" | tee -a "$LOG"
  PYTHONPATH=. "$CONDA_ROOT/$env_name/bin/python" \
    -m experiments.benchmark_unpadded_backbone_counterfactual \
    --model "$model" \
    --histogram "$ROOT/fronts/$slug/full_score/selected_window_histograms.csv" \
    --output-dir "$OUT" --batch-size 32 --warmup 2 --repeats 5 \
    >>"$LOG" 2>&1
  printf '%s COMPLETE %s\n' "$(date --iso-8601=seconds)" "$model" | tee -a "$LOG"
}

case "$MODEL" in
  PatchTST-FM-R1) run_one TSFM_PATCH PatchTST-FM-R1 patchtst_fm_r1 ;;
  TiRex2) run_one predictcsl-tirex TiRex2 tirex2 ;;
  all)
    run_one TSFM_PATCH PatchTST-FM-R1 patchtst_fm_r1
    run_one predictcsl-tirex TiRex2 tirex2
    ;;
  *) echo "Unsupported MODEL=$MODEL" >&2; exit 2 ;;
esac
