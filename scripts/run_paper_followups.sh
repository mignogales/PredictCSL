#!/usr/bin/env bash
# Launch the remaining paper experiments on the GPU server.
#
# Run from the repository root after this working tree (including the new
# analysis scripts) has been copied/pulled to the server:
#
#   bash scripts/run_paper_followups.sh all
#
# Select only independent parts if a prerequisite is not yet available:
#
#   bash scripts/run_paper_followups.sh oracle interpretability
#   bash scripts/run_paper_followups.sh transfer timing
#
# Every Python command writes resumable output in LOG_ROOT.  Re-running a task
# is safe: the underlying stages skip their completed cache cells/checkpoints.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MAIN_ENV="${MAIN_ENV:-predictcsl-main}"
LOG_ROOT="${LOG_ROOT:-logs/experiments/master_recompute}"
CANONICAL_CACHE="${CANONICAL_CACHE:-$LOG_ROOT/window_ablation_gifteval/general}"
GPU_ID="${GPU_ID:-}"

TRANSFER_SOURCE="${TRANSFER_SOURCE:-$LOG_ROOT/context_length_predictor_v3/Chronos2-Small}"
TRANSFER_TARGETS="${TRANSFER_TARGETS:-Chronos2-Synth Chronos2-Base}"
TRANSFER_SHORT_CONTEXT_MODE="${TRANSFER_SHORT_CONTEXT_MODE:-skip}"
TIMING_MODELS="${TIMING_MODELS:-Chronos2-Small}"
TIMING_STRATEGY_SUBDIRS="${TIMING_STRATEGY_SUBDIRS:-strategy_comparison_v3 strategy_comparison_v4}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda}"

# Pin the whole launcher to one physical GPU when requested.  CUDA remaps that
# device to logical cuda:0, which is what the Python jobs below receive.  If
# GPU_ID is empty, the server's existing CUDA_VISIBLE_DEVICES is respected.
if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

mkdir -p "$LOG_ROOT/launch_logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/launch_logs/paper_followups_${RUN_ID}.log"
echo "GPU selection: ${GPU_ID:-existing CUDA_VISIBLE_DEVICES (first visible GPU)}" | tee -a "$LOG_FILE"

run_main() {
  echo "+ conda run -n $MAIN_ENV python $*" | tee -a "$LOG_FILE"
  conda run -n "$MAIN_ENV" python "$@" 2>&1 | tee -a "$LOG_FILE"
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 2
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 2
  fi
}

do_oracle() {
  echo "== Oracle-distribution analysis ==" | tee -a "$LOG_FILE"
  require_dir "$CANONICAL_CACHE/datasets"
  # Inference-free; safe to run while another GPU experiment is active.
  run_main -m experiments.analyze_oracle_distributions \
    --run-dir "$CANONICAL_CACHE" --split-repeats 200
}

do_transfer() {
  echo "== Cross-model transfer: Chronos2-Small -> $TRANSFER_TARGETS ==" | tee -a "$LOG_FILE"
  require_dir "$CANONICAL_CACHE/datasets"
  require_file "$TRANSFER_SOURCE/best_model.pt"
  require_file "$TRANSFER_SOURCE/best_config.json"
  # Uses --cached-only internally: it will never load or rerun a target TSFM.
  run_main -m experiments.evaluate_cross_model_transfer \
    --canonical-run-dir "$CANONICAL_CACHE" \
    --source-predictor-dir "$TRANSFER_SOURCE" \
    --targets $TRANSFER_TARGETS --short-context-mode "$TRANSFER_SHORT_CONTEXT_MODE" \
    --device "$CUDA_DEVICE"
}

do_timing() {
  echo "== Robust wall-clock + peak-memory benchmark: $TIMING_MODELS ==" | tee -a "$LOG_FILE"
  require_dir "$CANONICAL_CACHE/datasets"
  # 3 warmups + 10 synchronized forwards per selected strategy window.  The
  # resulting timing.json includes CUDA baseline/peak allocated/reserved GB.
  run_main -m experiments.benchmark_window_timing_gifteval \
    --run-dir "$CANONICAL_CACHE" --models $TIMING_MODELS \
    --strategy-subdirs $TIMING_STRATEGY_SUBDIRS \
    --device "$CUDA_DEVICE" --warmup 3 --repeats 10 --num-gpus 1
}

do_interpretability() {
  echo "== Interpretability: perturbation, controlled lag, slicing/masking ==" | tee -a "$LOG_FILE"
  # exp4 is the controlled synthetic distant-lag test. exp7 is explicitly
  # selected because it is intentionally opt-in in experiments.yaml.
  run_main -m context_interpretability.run_experiment \
    --models Chronos2-Small --experiments exp1 exp4 exp7 \
    --source synthetic --device "${INTERPRETABILITY_DEVICE:-cuda:0}"
}

do_interpretability_multimodel() {
  echo "== Interpretability: multi-model long-lag follow-up ==" | tee -a "$LOG_FILE"
  bash scripts/run_interpretability_multimodel.sh 2>&1 | tee -a "$LOG_FILE"
}

usage() {
  cat <<'EOF'
Usage: bash scripts/run_paper_followups.sh {all|oracle|transfer|timing|interpretability|interpretability-multimodel} [...]

Environment overrides: MAIN_ENV, LOG_ROOT, CANONICAL_CACHE, GPU_ID,
TRANSFER_SOURCE, TRANSFER_TARGETS, TRANSFER_SHORT_CONTEXT_MODE, TIMING_MODELS,
TIMING_STRATEGY_SUBDIRS, CUDA_DEVICE.
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

for task in "$@"; do
  case "$task" in
    all)
      do_oracle
      do_transfer
      do_timing
      do_interpretability
      ;;
    oracle) do_oracle ;;
    transfer) do_transfer ;;
    timing) do_timing ;;
    interpretability) do_interpretability ;;
    interpretability-multimodel) do_interpretability_multimodel ;;
    -h|--help|help) usage ;;
    *) echo "Unknown task: $task" >&2; usage; exit 1 ;;
  esac
done

echo "Finished requested tasks. Consolidated log: $LOG_FILE"
