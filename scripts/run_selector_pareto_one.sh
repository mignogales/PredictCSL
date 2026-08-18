#!/usr/bin/env bash
# Run one matched-resolution compact-tree Pareto input independently.
set -Eeuo pipefail

if (( $# != 2 )); then
  echo "Usage: $0 MODEL_DISPLAY MODEL_SLUG" >&2
  exit 2
fi
MODEL="$1"
SLUG="$2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/mnogales/miniconda3/envs/TSFM_PATCH/bin/python}"
MASTER_ROOT="${MASTER_ROOT:-logs/experiments/master_recompute}"
SYNTH_ROOT="$MASTER_ROOT/context_length_dataset"
GRID_V1="$MASTER_ROOT/window_ablation_gifteval/general"
GRID_V2="$MASTER_ROOT/window_ablation_gifteval_grid_v2/general"
POLICY="$MASTER_ROOT/compact_context_risk_all_models/$SLUG/distill_tree_depth8/policy.joblib"
OUTPUT_ROOT="$MASTER_ROOT/context_selection_reviewer_followups/selector_pareto"
PROFILES_ROOT="$OUTPUT_ROOT/compact_profiles/$SLUG"
PROFILES="$PROFILES_ROOT/full_score/synthetic_calibration.json"
EVALUATION="$OUTPUT_ROOT/compact_dense/$SLUG/full_score"
REPORT="$EVALUATION/real_evaluation.json"
LOG="$OUTPUT_ROOT/logs/$SLUG.log"

mkdir -p "$OUTPUT_ROOT/logs"
echo "START independent $MODEL $(date --iso-8601=seconds)" >"$LOG"
if [[ ! -s "$PROFILES" || "$PROFILES" -ot "$POLICY" ]]; then
  PYTHONPATH=. "$PYTHON_BIN" \
    -m experiments.prepare_context_risk_ablation_profiles \
    --policy "$POLICY" --model-short "$MODEL" \
    --synthetic-dir "$SYNTH_ROOT/$MODEL" \
    --output-root "$PROFILES_ROOT" --dense-points 101 >>"$LOG" 2>&1
fi
if [[ ! -s "$REPORT" || "$REPORT" -ot "$PROFILES" ]]; then
  PYTHONPATH=. "$PYTHON_BIN" \
    -m experiments.evaluate_context_risk_profile_override \
    --policy "$POLICY" --profiles-json "$PROFILES" \
    --output-dir "$EVALUATION" --model-short "$MODEL" \
    --synthetic-dir "$SYNTH_ROOT/$MODEL" \
    --cache-roots "$GRID_V1" "$GRID_V2" --n-jobs 1 >>"$LOG" 2>&1
fi
echo "DONE $MODEL $(date --iso-8601=seconds)" >>"$LOG"
