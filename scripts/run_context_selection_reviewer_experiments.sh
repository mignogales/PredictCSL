#!/usr/bin/env bash
# Reviewer-proofing bundle for the calibrated context selector.
#
# Prepared phases (all resumable at the artifact level):
#   fronts          synthetic-only score-component profiles, fixed/horizon
#                   baselines, and real dense-front evaluation (CPU)
#   seeds           five independent ExtraTrees fits/evaluations + clustered
#                   bootstrap intervals (CPU; cached TSFM forecasts only)
#   selector-timing feature extraction + ExtraTrees/compact-tree overhead (CPU)
#   forecast-timing robust TSFM timing for every selected risk-policy window
#                   (two-GPU dataset sharding; final compute-heavy phase)
#   end-to-end      post-benchmark selector + TSFM rollup (CPU summary only)
#
# The calibrated-risk implementation is maintained in
# .codex_remote_staging/experiments/. Sync that directory into the server's
# experiments/ package before launching this queue; the ordinary experiment and
# summary files in this checkout must be synced as well.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MASTER_ROOT="${MASTER_ROOT:-logs/experiments/master_recompute}"
PYTHON_BIN="${PYTHON_BIN:-/home/mnogales/miniconda3/envs/TSFM_PATCH/bin/python}"
CONDA_ROOT="${CONDA_ROOT:-/home/mnogales/miniconda3/envs}"
GRID_V1="${GRID_V1:-$MASTER_ROOT/window_ablation_gifteval/general}"
GRID_V2="${GRID_V2:-$MASTER_ROOT/window_ablation_gifteval_grid_v2/general}"
SYNTH_ROOT="${SYNTH_ROOT:-$MASTER_ROOT/context_length_dataset}"
TEACHER_ROOT="${TEACHER_ROOT:-$MASTER_ROOT/calibrated_context_risk_extreme}"
COMPACT_ROOT="${COMPACT_ROOT:-$MASTER_ROOT/compact_context_risk_all_models}"
OUT_ROOT="${OUT_ROOT:-$MASTER_ROOT/context_selection_reviewer_followups}"
TIMING_GPU_IDS_TEXT="${TIMING_GPU_IDS:-${GPU_ID:-2 3}}"
GPU_FREE_THRESHOLD_MIB="${GPU_FREE_THRESHOLD_MIB:-1024}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
TIMING_MAX_SERIES="${TIMING_MAX_SERIES:-64}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
SEEDS_TEXT="${SEEDS:-17 29 42 71 101}"

MODEL_SPECS=(
  "Chronos2-Base|chronos2_base|predictcsl-main"
  "Chronos2-Small|chronos2_small|predictcsl-main"
  "Chronos2-Synth|chronos2_synth|predictcsl-main"
  "ChronosBolt-Base|chronos_bolt_base|predictcsl-main"
  "FlowState-R1|flowstate_r1|predictcsl-main"
  "Moirai2-Small|moirai2_small|predictcsl-main"
  "PatchTST-FM-R1|patchtst_fm_r1|TSFM_PATCH"
  "Sundial-Base-128M|sundial_base_128m|predictcsl-legacy"
  "TimesFM2.5-200M|timesfm2p5|predictcsl-main"
  "Toto-2.0-313m|toto_2_0_313m|predictcsl-toto"
)
if [[ -n "${MODEL_SPECS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a MODEL_SPECS <<<"$MODEL_SPECS_OVERRIDE"
fi
read -r -a SEED_VALUES <<<"$SEEDS_TEXT"
read -r -a TIMING_GPU_ID_VALUES <<<"$TIMING_GPU_IDS_TEXT"
(( ${#TIMING_GPU_ID_VALUES[@]} > 0 )) || {
  echo "TIMING_GPU_IDS must contain at least one physical GPU index." >&2
  exit 2
}
for gpu_id in "${TIMING_GPU_ID_VALUES[@]}"; do
  [[ "$gpu_id" =~ ^[0-9]+$ ]] || {
    echo "Invalid physical GPU index in TIMING_GPU_IDS: $gpu_id" >&2
    exit 2
  }
done
TIMING_GPU_CSV="${TIMING_GPU_ID_VALUES[*]}"
TIMING_GPU_CSV="${TIMING_GPU_CSV// /,}"

mkdir -p "$OUT_ROOT/launch_logs"
LOG_FILE="$OUT_ROOT/launch_logs/reviewer_experiments_$(date +%Y%m%d_%H%M%S).log"
log() { printf '%s\n' "$*" | tee -a "$LOG_FILE"; }
run_cpu() {
  log "+ $PYTHON_BIN $*"
  PYTHONPATH=. "$PYTHON_BIN" "$@" 2>&1 | tee -a "$LOG_FILE"
}
require_file() { [[ -s "$1" ]] || { log "Missing required file: $1"; return 2; }; }

wait_for_timing_gpus() {
  local busy=() gpu_id used_mib
  while true; do
    busy=()
    for gpu_id in "${TIMING_GPU_ID_VALUES[@]}"; do
      used_mib="$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used \
        --format=csv,noheader,nounits | tr -d ' ')"
      if (( used_mib > GPU_FREE_THRESHOLD_MIB )); then
        busy+=("gpu${gpu_id}=${used_mib}MiB")
      fi
    done
    if (( ${#busy[@]} == 0 )); then
      log "Timing GPUs are free: $TIMING_GPU_CSV"
      return
    fi
    log "Waiting for timing GPUs (${busy[*]}); retrying in ${GPU_POLL_SECONDS}s."
    sleep "$GPU_POLL_SECONDS"
  done
}

front_one() {
  local model="$1" slug="$2"
  local source_policy="$TEACHER_ROOT/$slug/policy.joblib"
  local root="$OUT_ROOT/fronts/$slug"
  require_file "$source_policy"
  mkdir -p "$root"
  run_cpu -m experiments.prepare_context_risk_ablation_profiles \
    --policy "$source_policy" --model-short "$model" \
    --synthetic-dir "$SYNTH_ROOT/$model" --output-root "$root"
  local variants=(mean_only mean_plus_uncertainty mean_plus_harm full_score)
  for variant in "${variants[@]}"; do
    local extra=()
    [[ "$variant" == "full_score" ]] && extra+=(--fixed-baselines)
    run_cpu -m experiments.evaluate_context_risk_profile_override \
      --policy "$source_policy" \
      --profiles-json "$root/$variant/synthetic_calibration.json" \
      --output-dir "$root/$variant" --model-short "$model" \
      --synthetic-dir "$SYNTH_ROOT/$model" \
      --cache-roots "$GRID_V1" "$GRID_V2" "${extra[@]}"
  done
  run_cpu -m experiments.summarize_context_risk_ablations \
    --run "mean_only=$root/mean_only/real_evaluation.json" \
    --run "mean_plus_uncertainty=$root/mean_plus_uncertainty/real_evaluation.json" \
    --run "mean_plus_harm=$root/mean_plus_harm/real_evaluation.json" \
    --run "full_score=$root/full_score/real_evaluation.json" \
    --output-dir "$root/summary"
}

seed_one() {
  local model="$1" slug="$2" seed="$3"
  local out="$OUT_ROOT/multiseed/$slug/seed_$seed"
  mkdir -p "$out"
  if [[ -s "$out/real_cells.csv" && -s "$out/real_evaluation.json" ]]; then
    log "CACHED multiseed $model seed=$seed"
    return
  fi
  run_cpu -m experiments.calibrated_context_risk --stage all \
    --model-short "$model" --synthetic-dir "$SYNTH_ROOT/$model" \
    --cache-roots "$GRID_V1" "$GRID_V2" --output-dir "$out" \
    --seed "$seed" --n-jobs 12
}

summarize_seeds() {
  local model="$1" slug="$2"
  local arguments=()
  for seed in "${SEED_VALUES[@]}"; do
    arguments+=(--run "$seed=$OUT_ROOT/multiseed/$slug/seed_$seed/real_cells.csv")
  done
  run_cpu -m experiments.summarize_context_risk_multiseed \
    "${arguments[@]}" --bootstrap-repeats 10000 \
    --output-dir "$OUT_ROOT/multiseed/$slug/summary"
}

ensure_compact_policy() {
  local model="$1" slug="$2"
  local teacher="$TEACHER_ROOT/$slug/policy.joblib"
  local compact="$COMPACT_ROOT/$slug/distill_tree_depth8/policy.joblib"
  require_file "$teacher"
  if [[ ! -s "$compact" ]]; then
    log "Compact depth-8 policy missing for $model; preparing it synthetically."
    run_cpu -m experiments.distill_calibrated_context_risk \
      --model-short "$model" --synthetic-dir "$SYNTH_ROOT/$model" \
      --teacher-policy "$teacher" --output-root "$COMPACT_ROOT/$slug" \
      --families tree --tree-depths 8 --dense-points 101
  fi
  require_file "$compact"
}

selector_timing_one() {
  local model="$1" slug="$2"
  local teacher="$TEACHER_ROOT/$slug/policy.joblib"
  local compact="$COMPACT_ROOT/$slug/distill_tree_depth8/policy.joblib"
  ensure_compact_policy "$model" "$slug"
  run_cpu -m experiments.benchmark_context_selector_overhead \
    --synthetic-dir "$SYNTH_ROOT/$model" \
    --policy "extra_trees=$teacher" --policy "compact_tree=$compact" \
    --batch-sizes 1 32 256 --warmups 3 --repeats 15 \
    --output "$OUT_ROOT/timing/$slug/selector_overhead.json"
}

ensure_compact_histogram() {
  local model="$1" slug="$2"
  local compact_dir="$COMPACT_ROOT/$slug/distill_tree_depth8"
  ensure_compact_policy "$model" "$slug"
  if [[ -s "$compact_dir/selected_window_histograms.csv" ]]; then
    return
  fi
  run_cpu -m experiments.calibrated_context_risk --stage evaluate \
    --model-short "$model" --synthetic-dir "$SYNTH_ROOT/$model" \
    --cache-roots "$GRID_V1" "$GRID_V2" --output-dir "$compact_dir" --n-jobs 1
}

forecast_timing_one() {
  local model="$1" slug="$2" env_name="$3"
  local teacher_hist="$OUT_ROOT/fronts/$slug/full_score/selected_window_histograms.csv"
  local compact_hist="$COMPACT_ROOT/$slug/distill_tree_depth8/selected_window_histograms.csv"
  require_file "$teacher_hist"
  ensure_compact_histogram "$model" "$slug"
  export CUDA_VISIBLE_DEVICES="$TIMING_GPU_CSV"
  log "+ timing $model ($env_name) on physical GPUs $TIMING_GPU_CSV "\
      "(${#TIMING_GPU_ID_VALUES[@]} dataset shards)"
  PYTHONPATH=. "$CONDA_ROOT/$env_name/bin/python" \
    -m experiments.benchmark_window_timing_gifteval \
    --run-dir "$GRID_V1" --models "$model" --device cuda \
    --num-gpus "${#TIMING_GPU_ID_VALUES[@]}" \
    --warmup 3 --repeats 10 --max-series "$TIMING_MAX_SERIES" \
    --selection-csvs "$teacher_hist" "$compact_hist" \
    --selection-methods full_native balanced efficiency max_efficiency \
    2>&1 | tee -a "$LOG_FILE"
}

rebuild_timing_summary() {
  local histograms=()
  local models=()
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r model slug _env_name <<<"$spec"
    models+=("$model")
    histograms+=(
      "$OUT_ROOT/fronts/$slug/full_score/selected_window_histograms.csv"
      "$COMPACT_ROOT/$slug/distill_tree_depth8/selected_window_histograms.csv"
    )
  done
  run_cpu -m experiments.benchmark_window_timing_gifteval \
    --run-dir "$GRID_V1" --summary-only --device cpu --num-gpus 1 \
    --models "${models[@]}" --max-series "$TIMING_MAX_SERIES" \
    --selection-csvs "${histograms[@]}" \
    --selection-methods full_native balanced efficiency max_efficiency
}

require_timing_prerequisites() {
  local spec model slug env_name
  # First prove the CPU queue has finished. Do not generate compact histograms
  # while the multi-seed/selector jobs are still contending for CPU resources.
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r model slug env_name <<<"$spec"
    require_file "$OUT_ROOT/fronts/$slug/full_score/selected_window_histograms.csv"
    require_file "$OUT_ROOT/timing/$slug/selector_overhead.json"
  done
  # Histogram evaluation is CPU-only and resumable. Materialize any missing
  # compact-policy assignments now, before the GPU wait/timing boundary.
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r model slug env_name <<<"$spec"
    ensure_compact_histogram "$model" "$slug"
    require_file "$COMPACT_ROOT/$slug/distill_tree_depth8/selected_window_histograms.csv"
  done
  log "All CPU prerequisites exist for ${#MODEL_SPECS[@]} non-TiRex models."
}

end_to_end_one() {
  local model="$1" slug="$2"
  local teacher_hist="$OUT_ROOT/fronts/$slug/full_score/selected_window_histograms.csv"
  local compact_hist="$COMPACT_ROOT/$slug/distill_tree_depth8/selected_window_histograms.csv"
  run_cpu -m experiments.summarize_context_selection_end_to_end \
    --model "$model" --timing-summary "$GRID_V1/timing_summary.csv" \
    --selector-overhead "$OUT_ROOT/timing/$slug/selector_overhead.json" \
    --native-histogram "$teacher_hist" \
    --policy "ExtraTrees balanced" "$teacher_hist" balanced extra_trees \
    --policy "Compact tree balanced" "$compact_hist" balanced compact_tree \
    --selector-batch-size 256 \
    --output "$OUT_ROOT/timing/$slug/end_to_end.json"
}

run_models() {
  local function_name="$1"
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r model slug env_name <<<"$spec"
    "$function_name" "$model" "$slug" "$env_name"
  done
}

run_seed_matrix() {
  local active=0
  local failures=0
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r model slug _env_name <<<"$spec"
    for seed in "${SEED_VALUES[@]}"; do
      seed_one "$model" "$slug" "$seed" &
      active=$((active + 1))
      if (( active >= MAX_PARALLEL )); then
        wait -n || failures=1
        active=$((active - 1))
      fi
    done
  done
  while (( active > 0 )); do
    wait -n || failures=1
    active=$((active - 1))
  done
  (( failures == 0 )) || return 1
  run_models summarize_seeds
}

usage() {
  echo "Usage: $0 {fronts|seeds|selector-timing|forecast-timing|end-to-end|all} [...]"
}

[[ $# -gt 0 ]] || { usage; exit 2; }
for phase in "$@"; do
  case "$phase" in
    fronts) run_models front_one ;;
    seeds) run_seed_matrix ;;
    selector-timing) run_models selector_timing_one ;;
    forecast-timing)
      require_timing_prerequisites
      wait_for_timing_gpus
      run_models forecast_timing_one
      rebuild_timing_summary
      ;;
    end-to-end) run_models end_to_end_one ;;
    all)
      run_models front_one
      run_seed_matrix
      run_models selector_timing_one
      # Keep GPU speed measurement last. The end-to-end step below only reads
      # its artifacts and writes the final CPU-side summary.
      require_timing_prerequisites
      wait_for_timing_gpus
      run_models forecast_timing_one
      rebuild_timing_summary
      run_models end_to_end_one
      ;;
    -h|--help|help) usage; exit 0 ;;
    *) log "Unknown phase: $phase"; usage; exit 2 ;;
  esac
done
log "Completed requested reviewer-experiment phases. Log: $LOG_FILE"
