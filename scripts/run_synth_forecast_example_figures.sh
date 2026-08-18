#!/usr/bin/env bash
# Produce cached real forecasts for every synthetic DGP, then render compact
# insufficient-vs-sufficient context panels containing all four models.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_BIN="${CONDA_BIN:-$REPO_ROOT/../../miniconda3/bin/conda}"
MAIN_ENV="${MAIN_ENV:-predictcsl-main}"
TOTO_ENV="${TOTO_ENV:-predictcsl-toto}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_DIR="${PREDICTCSL_SWEEP_ROOT:-logs/experiments/synth_param_sweeps}/summary_plots"
LOG_DIR="${LOG_DIR:-logs/experiments/run_logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/synth_forecast_examples_$(date +%Y%m%d_%H%M%S).log"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

run_main() {
  "$CONDA_BIN" run --no-capture-output -n "$MAIN_ENV" python -m \
    experiments.plot_synth_forecast_examples --device cuda:0 --output-dir "$OUTPUT_DIR" \
    --compute-only --models Chronos2-Small Moirai2-Small FlowState-R1
}

run_toto() {
  "$CONDA_BIN" run --no-capture-output -n "$TOTO_ENV" python -m \
    experiments.plot_synth_forecast_examples --device cuda:0 --output-dir "$OUTPUT_DIR" \
    --compute-only --models Toto-2.0-313m
}

render() {
  "$CONDA_BIN" run --no-capture-output -n "$MAIN_ENV" python -m \
    experiments.plot_synth_forecast_examples --output-dir "$OUTPUT_DIR" --plot-only
}

{
  echo "Synthetic forecast example figures"
  echo "physical GPU: $GPU_ID (visible as cuda:0)"
  echo "output: $OUTPUT_DIR/09_forecast_examples"
  run_main
  run_toto
  render
} 2>&1 | tee "$LOG_FILE"

echo "Complete. Log: $LOG_FILE"
