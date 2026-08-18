#!/usr/bin/env bash
# Frozen all-model rank-distilled single-tree Pareto experiment (CPU-only).
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/mnogales/miniconda3/envs/TSFM_PATCH/bin/python}"
MASTER_ROOT="${MASTER_ROOT:-logs/experiments/master_recompute}"
SYNTH_ROOT="$MASTER_ROOT/context_length_dataset"
GRID_V1="$MASTER_ROOT/window_ablation_gifteval/general"
GRID_V2="$MASTER_ROOT/window_ablation_gifteval_grid_v2/general"
TEACHER_ROOT="$MASTER_ROOT/calibrated_context_risk_extreme"
EXTRA_ROOT="$MASTER_ROOT/context_selection_reviewer_followups/fronts"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MASTER_ROOT/rank_tree_context_risk_all_models}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$MASTER_ROOT/context_selection_reviewer_followups/rank_tree_selector_pareto}"
PILOT_ROOT="$MASTER_ROOT/improved_tree_distillation_pilot/chronos2_base"
JOBS="${RANK_TREE_JOBS:-3}"
LOG_DIR="$OUTPUT_ROOT/logs"
LOCK_DIR="$OUTPUT_ROOT/.run.lock"

MODEL_SPECS=(
  "Chronos2-Base|chronos2_base"
  "Chronos2-Small|chronos2_small"
  "Chronos2-Synth|chronos2_synth"
  "ChronosBolt-Base|chronos_bolt_base"
  "FlowState-R1|flowstate_r1"
  "Moirai2-Small|moirai2_small"
  "PatchTST-FM-R1|patchtst_fm_r1"
  "Sundial-Base-128M|sundial_base_128m"
  "TiRex2|tirex2"
  "TimesFM2.5-200M|timesfm2p5"
  "Toto-2.0-313m|toto_2_0_313m"
)

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "RANK_TREE_JOBS must be a positive integer, got: $JOBS" >&2
  exit 2
fi
mkdir -p "$LOG_DIR" "$SUMMARY_ROOT"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Rank-tree suite already active: $LOCK_DIR" >&2
  exit 2
fi
cleanup() { rmdir "$LOCK_DIR" 2>/dev/null || true; }
trap cleanup EXIT

run_one() {
  local model="$1" slug="$2"
  local root="$OUTPUT_ROOT/$slug"
  local candidate="$root/tree_rank_d8_leaf16"
  local policy="$candidate/policy.joblib"
  local profiles="$candidate/synthetic_calibration.json"
  local evaluation="$root/full_score"
  local report="$evaluation/real_evaluation.json"
  local teacher="$TEACHER_ROOT/$slug/policy.joblib"
  local log="$LOG_DIR/$slug.log"
  echo "START $model $(date --iso-8601=seconds)" >"$log"
  if [[ "$slug" == "chronos2_base" && ! -s "$policy" \
        && -s "$PILOT_ROOT/tree_rank_d8_leaf16/policy.joblib" ]]; then
    mkdir -p "$root"
    cp -a "$PILOT_ROOT/tree_rank_d8_leaf16" "$candidate"
  fi
  if [[ "$slug" == "chronos2_base" && ! -s "$report" \
        && -s "$PILOT_ROOT/tree_rank_d8_leaf16/gifteval_frozen/real_evaluation.json" ]]; then
    mkdir -p "$evaluation"
    cp -a "$PILOT_ROOT/tree_rank_d8_leaf16/gifteval_frozen/." "$evaluation/"
  fi
  if [[ ! -s "$policy" || ! -s "$profiles" ]]; then
    PYTHONPATH=. "$PYTHON_BIN" -m experiments.improve_tree_distillation \
      --model-short "$model" --synthetic-dir "$SYNTH_ROOT/$model" \
      --teacher-policy "$teacher" --output-root "$root" \
      --modes rank --tree-depths 8 --tree-min-samples-leaves 16 \
      --dense-points 101 >>"$log" 2>&1
  fi
  if [[ ! -s "$report" || "$report" -ot "$profiles" ]]; then
    PYTHONPATH=. "$PYTHON_BIN" \
      -m experiments.evaluate_context_risk_profile_override \
      --policy "$policy" --profiles-json "$profiles" \
      --output-dir "$evaluation" --model-short "$model" \
      --synthetic-dir "$SYNTH_ROOT/$model" \
      --cache-roots "$GRID_V1" "$GRID_V2" --n-jobs 1 >>"$log" 2>&1
  fi
  echo "DONE $model $(date --iso-8601=seconds)" >>"$log"
}

pids=()
active=0
failed=0
for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r model slug <<<"$spec"
  run_one "$model" "$slug" &
  pids+=("$!")
  ((active += 1))
  if ((active >= JOBS)); then
    if ! wait "${pids[0]}"; then failed=1; fi
    pids=("${pids[@]:1}")
    ((active -= 1))
  fi
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if ((failed)); then
  echo "At least one rank-tree worker failed; inspect $LOG_DIR" >&2
  exit 1
fi

summary_args=()
for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r model slug <<<"$spec"
  summary_args+=(--model "$model=$slug")
done
PYTHONPATH=. "$PYTHON_BIN" -m experiments.summarize_selector_pareto \
  --extra-root "$EXTRA_ROOT" --compact-root "$OUTPUT_ROOT" \
  --compact-label "Rank-distilled depth-8 tree" \
  --output-dir "$SUMMARY_ROOT/combined" "${summary_args[@]}" \
  >"$SUMMARY_ROOT/combined.log" 2>&1

echo "RANK-TREE PARETO COMPLETE: $SUMMARY_ROOT"
