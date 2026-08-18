#!/usr/bin/env bash
# Build exact cached GiftEval Pareto fronts for every model and combine them.
# CPU-only: this reads existing Stage-3/4 forecast caches and performs no TSFM
# inference. Each model runs both cell-balanced and series-weighted FLOP axes.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/mnogales/miniconda3/envs/TSFM_PATCH/bin/python}"
MASTER_ROOT="${MASTER_ROOT:-logs/experiments/master_recompute}"
RUN_DIR="${RUN_DIR:-$MASTER_ROOT/window_ablation_gifteval/general_v4}"
COMPARISON_ROOT="$MASTER_ROOT/window_ablation_gifteval"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MASTER_ROOT/oracle_pareto_frontier_all_models}"
JOBS="${PARETO_JOBS:-2}"
LOG_DIR="$OUTPUT_ROOT/logs"
LOCK_DIR="$OUTPUT_ROOT/.run.lock"

MODELS=(
  Chronos2-Base
  Chronos2-Small
  Chronos2-Synth
  ChronosBolt-Base
  FlowState-R1
  Moirai2-Small
  PatchTST-FM-R1
  Sundial-Base-128M
  TiRex2
  TimesFM2.5-200M
  Toto-2.0-313m
)

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARETO_JOBS must be a positive integer, got: $JOBS" >&2
  exit 2
fi
mkdir -p "$LOG_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "All-model Pareto suite already active: $LOCK_DIR" >&2
  exit 2
fi
cleanup() { rmdir "$LOCK_DIR" 2>/dev/null || true; }
trap cleanup EXIT

run_model() {
  local model="$1"
  local comparison="$COMPARISON_ROOT/$model/strategy_comparison_v4/comparison.csv"
  local log="$LOG_DIR/$model.log"
  if [[ ! -f "$comparison" ]]; then
    echo "Missing comparison table: $comparison" >&2
    return 1
  fi

  echo "START $model $(date --iso-8601=seconds)" >"$log"
  for weighting in cell instances; do
    local suffix=""
    if [[ "$weighting" == instances ]]; then
      suffix="_instance_weighted"
    fi
    local output="$OUTPUT_ROOT/$model$suffix"
    if [[ -s "$output/report.json" \
          && -s "$output/oracle_supported_frontier.csv" \
          && "$output/report.json" -nt "$comparison" \
          && "$output/oracle_supported_frontier.csv" -nt "$comparison" ]]; then
      echo "SKIP complete $model $weighting" >>"$log"
      continue
    fi
    PYTHONPATH=. "$PYTHON_BIN" -m experiments.oracle_pareto_frontier_gifteval \
      --run-dir "$RUN_DIR" \
      --comparison-csv "$comparison" \
      --model "$model" \
      --selector "Mamba curve=$comparison" \
      --flops-weighting "$weighting" \
      --output-dir "$output" >>"$log" 2>&1
  done
  echo "DONE $model $(date --iso-8601=seconds)" >>"$log"
}

echo "Launching ${#MODELS[@]} cached Pareto jobs with concurrency=$JOBS"
pids=()
active=0
failed=0
for model in "${MODELS[@]}"; do
  run_model "$model" &
  pids+=("$!")
  ((active += 1))
  if (( active >= JOBS )); then
    if ! wait "${pids[0]}"; then
      failed=1
    fi
    pids=("${pids[@]:1}")
    ((active -= 1))
  fi
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed )); then
  echo "At least one per-model Pareto job failed; inspect $LOG_DIR" >&2
  exit 1
fi

summary_args=()
for model in "${MODELS[@]}"; do
  summary_args+=(--model "$model")
done
PYTHONPATH=. "$PYTHON_BIN" -m experiments.summarize_oracle_pareto_all_models \
  --input-root "$OUTPUT_ROOT" \
  --output-dir "$OUTPUT_ROOT/combined" \
  "${summary_args[@]}" >"$LOG_DIR/combined.log" 2>&1

echo "ALL-MODEL PARETO COMPLETE: $OUTPUT_ROOT"
