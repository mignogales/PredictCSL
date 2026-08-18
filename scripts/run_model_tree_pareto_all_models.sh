#!/usr/bin/env bash
# Frozen validation-selected model-tree GiftEval evaluation (CPU-only caches).
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/mnogales/miniconda3/envs/TSFM_PATCH/bin/python}"
MASTER_ROOT="${MASTER_ROOT:-logs/experiments/master_recompute}"
SYNTH_ROOT="$MASTER_ROOT/context_length_dataset"
GRID_V1="$MASTER_ROOT/window_ablation_gifteval/general"
GRID_V2="$MASTER_ROOT/window_ablation_gifteval_grid_v2/general"
EXTRA_ROOT="$MASTER_ROOT/context_selection_reviewer_followups/fronts"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MASTER_ROOT/model_tree_context_risk_all_models}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$MASTER_ROOT/context_selection_reviewer_followups/model_tree_selector_pareto}"
JOBS="${MODEL_TREE_EVAL_JOBS:-3}"
LOG_DIR="$OUTPUT_ROOT/evaluation_logs"
LOCK_DIR="$OUTPUT_ROOT/.evaluation.lock"

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

mkdir -p "$LOG_DIR" "$SUMMARY_ROOT"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Model-tree evaluation already active: $LOCK_DIR" >&2
  exit 2
fi
cleanup() { rmdir "$LOCK_DIR" 2>/dev/null || true; }
trap cleanup EXIT

run_one() {
  local model="$1" slug="$2"
  local root="$OUTPUT_ROOT/$slug"
  local summary="$root/explainable_alternatives_screen.json"
  local evaluation="$root/full_score"
  local report="$evaluation/real_evaluation.json"
  local log="$LOG_DIR/$slug.log"
  if [[ ! -s "$summary" ]]; then
    echo "Missing frozen validation selection: $summary" >&2
    return 2
  fi
  local candidate
  candidate="$($PYTHON_BIN -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["best_candidate"]["candidate"])' \
    "$summary")"
  local policy="$root/$candidate/policy.joblib"
  local profiles="$root/$candidate/synthetic_calibration.json"
  if [[ ! -s "$policy" || ! -s "$profiles" ]]; then
    echo "Missing selected candidate artifacts: $root/$candidate" >&2
    return 2
  fi
  echo "START $model candidate=$candidate $(date --iso-8601=seconds)" >"$log"
  if [[ ! -s "$report" || "$report" -ot "$profiles" ]]; then
    PYTHONPATH=. "$PYTHON_BIN" \
      -m experiments.evaluate_context_risk_profile_override \
      --policy "$policy" --profiles-json "$profiles" \
      --output-dir "$evaluation" --model-short "$model" \
      --synthetic-dir "$SYNTH_ROOT/$model" \
      --cache-roots "$GRID_V1" "$GRID_V2" --n-jobs 1 >>"$log" 2>&1
  fi
  echo "DONE $model candidate=$candidate $(date --iso-8601=seconds)" >>"$log"
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
  echo "At least one model-tree evaluation failed; inspect $LOG_DIR" >&2
  exit 1
fi

summary_args=()
for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r model slug <<<"$spec"
  summary_args+=(--model "$model=$slug")
done
PYTHONPATH=. "$PYTHON_BIN" -m experiments.summarize_selector_pareto \
  --extra-root "$EXTRA_ROOT" --compact-root "$OUTPUT_ROOT" \
  --compact-label "Explainable model tree" \
  --output-dir "$SUMMARY_ROOT/combined" "${summary_args[@]}" \
  >"$SUMMARY_ROOT/combined.log" 2>&1

echo "MODEL-TREE PARETO COMPLETE: $SUMMARY_ROOT"
