#!/usr/bin/env bash
# Validation-only model-tree screen for all 11 TSFMs (CPU-only).
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/mnogales/miniconda3/envs/TSFM_PATCH/bin/python}"
MASTER_ROOT="${MASTER_ROOT:-logs/experiments/master_recompute}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MASTER_ROOT/model_tree_context_risk_all_models}"
PILOT_ROOT="$MASTER_ROOT/explainable_selector_alternatives_pilot"
JOBS="${MODEL_TREE_JOBS:-3}"
LOG_DIR="$OUTPUT_ROOT/logs"
LOCK_DIR="$OUTPUT_ROOT/.screen.lock"

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

mkdir -p "$LOG_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Model-tree screen already active: $LOCK_DIR" >&2
  exit 2
fi
cleanup() { rmdir "$LOCK_DIR" 2>/dev/null || true; }
trap cleanup EXIT

run_one() {
  local model="$1" slug="$2"
  local root="$OUTPUT_ROOT/$slug"
  local summary="$root/explainable_alternatives_screen.json"
  local pilot="$PILOT_ROOT/$slug/explainable_alternatives_screen.json"
  local log="$LOG_DIR/$slug.log"
  if [[ -s "$summary" ]]; then
    echo "SKIP $model (complete)" >"$log"
    return
  fi
  if [[ -s "$pilot" ]]; then
    mkdir -p "$root"
    cp -a "$PILOT_ROOT/$slug/." "$root/"
    echo "DONE $model (reused discovery screen)" >"$log"
    return
  fi
  echo "START $model $(date --iso-8601=seconds)" >"$log"
  PYTHONPATH=. "$PYTHON_BIN" -m experiments.explainable_selector_alternatives \
    --model-short "$model" \
    --synthetic-dir "$MASTER_ROOT/context_length_dataset/$model" \
    --teacher-policy "$MASTER_ROOT/calibrated_context_risk_extreme/$slug/policy.joblib" \
    --output-root "$root" --families model_tree \
    --model-tree-depths 4 5 --model-tree-alphas 1 100 \
    --dense-points 101 >>"$log" 2>&1
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
  echo "At least one model-tree screen failed; inspect $LOG_DIR" >&2
  exit 1
fi

PYTHONPATH=. "$PYTHON_BIN" -m experiments.summarize_explainable_alternatives \
  --root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/validation_summary.csv" \
  >"$OUTPUT_ROOT/validation_summary.log" 2>&1
echo "MODEL-TREE SCREEN COMPLETE: $OUTPUT_ROOT"
