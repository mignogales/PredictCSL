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
#   bash scripts/run_paper_followups.sh backfill transfer timing
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
TIMING_MODELS="${TIMING_MODELS:-Chronos2-Small}"
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

do_backfill() {
  echo "== Inference-free legacy cache alignment backfill ==" | tee -a "$LOG_FILE"
  require_dir "$CANONICAL_CACHE/datasets"

  local cached_models=()
  while IFS= read -r model; do
    [[ -n "$model" ]] && cached_models+=("$model")
  done < <(
    find "$CANONICAL_CACHE/datasets" -mindepth 2 -maxdepth 2 -type d \
      -exec basename {} \; | sort -u
  )
  if [[ ${#cached_models[@]} -eq 0 ]]; then
    echo "No cached models found under $CANONICAL_CACHE/datasets" >&2
    exit 2
  fi

  echo "Cached models: ${cached_models[*]}" | tee -a "$LOG_FILE"
  # This visits cached Stage-3 cells and adds served_index where possible.
  # --cached-only is the safety fuse: a missing/stale cell aborts instead of
  # loading a foundation model. --forecast-only avoids predictor inference.
  # Run one model at a time so each family uses its own canonical context grid;
  # a multi-family forecast-only call would construct their grid union.
  local model
  for model in "${cached_models[@]}"; do
    run_main -m experiments.test_window_ablation_gifteval_v5 \
      --models "$model" --cache-root "$CANONICAL_CACHE" \
      --cached-only --forecast-only --short-context-mode skip \
      --device "$CUDA_DEVICE" --num-gpus 1 --no-plots
  done
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
    --targets $TRANSFER_TARGETS --device "$CUDA_DEVICE"
}

do_timing() {
  echo "== Robust wall-clock + peak-memory benchmark: $TIMING_MODELS ==" | tee -a "$LOG_FILE"
  require_dir "$CANONICAL_CACHE/datasets"
  # 3 warmups + 10 synchronized forwards per selected strategy window.  The
  # resulting timing.json includes CUDA baseline/peak allocated/reserved GB.
  run_main -m experiments.benchmark_window_timing_gifteval \
    --run-dir "$CANONICAL_CACHE" --models $TIMING_MODELS \
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

usage() {
  cat <<'EOF'
Usage: bash scripts/run_paper_followups.sh {all|backfill|oracle|transfer|timing|interpretability} [...]

Environment overrides: MAIN_ENV, LOG_ROOT, CANONICAL_CACHE, GPU_ID,
TRANSFER_SOURCE, TRANSFER_TARGETS, TIMING_MODELS, CUDA_DEVICE.
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

for task in "$@"; do
  case "$task" in
    all)
      do_backfill
      do_oracle
      do_transfer
      do_timing
      do_interpretability
      ;;
    backfill) do_backfill ;;
    oracle) do_oracle ;;
    transfer) do_transfer ;;
    timing) do_timing ;;
    interpretability) do_interpretability ;;
    -h|--help|help) usage ;;
    *) echo "Unknown task: $task" >&2; usage; exit 1 ;;
  esac
done

echo "Finished requested tasks. Consolidated log: $LOG_FILE"
