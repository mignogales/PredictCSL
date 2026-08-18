#!/usr/bin/env bash
# Rebuild the Chronos2-Small oracle Pareto analyses with audited FLOPs and all
# calibrated-risk operating points. CPU-only; no TSFM inference is performed.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/mnogales/miniconda3/envs/TSFM_PATCH/bin/python}"
MASTER_ROOT="${MASTER_ROOT:-logs/experiments/master_recompute}"
START_HOUR="${START_HOUR:-22}"
LOG_DIR="$MASTER_ROOT/launch_logs"
LOG_FILE="$LOG_DIR/overnight_pareto.log"
LOCK_DIR="$LOG_DIR/.overnight_pareto.lock"
# The v4 comparison NPZs live under general_v4; its datasets/ entry is a
# symlink to the canonical general/datasets cache, so both action curves and
# per-instance metrics resolve from this single run root.
RUN_DIR="$MASTER_ROOT/window_ablation_gifteval/general_v4"
COMPARISON_ROOT="$MASTER_ROOT/window_ablation_gifteval/Chronos2-Small"
COMPARISON="$COMPARISON_ROOT/strategy_comparison_v4/comparison.csv"
OUTPUT_ROOT="$MASTER_ROOT/oracle_pareto_frontier_audited"

mkdir -p "$LOG_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Pareto queue already active: $LOCK_DIR" >&2
  exit 2
fi
cleanup() { rmdir "$LOCK_DIR" 2>/dev/null || true; }
trap cleanup EXIT
log() { printf '%s\n' "$*" | tee -a "$LOG_FILE"; }

while (( 10#$(date +%H) < START_HOUR )); do
  log "Waiting until ${START_HOUR}:00 for the overnight Pareto run."
  sleep 60
done
while pgrep -f 'experiments.calibrated_context_risk --stage evaluate' >/dev/null; do
  log "Waiting for calibrated-risk FLOPs refreshes to finish."
  sleep 60
done

COMMON=(
  --run-dir "$RUN_DIR"
  --comparison-csv "$COMPARISON"
  --model Chronos2-Small
)
SELECTORS=(
  --selector "Mamba curve=$COMPARISON"
  --selector "Soft top-k classification=$COMPARISON_ROOT/strategy_comparison_v4_classification/comparison.csv"
  --selector "Adjacent pairwise=$COMPARISON_ROOT/strategy_comparison_v4_pairwise/comparison.csv"
  --selector "3% acceptable set=$COMPARISON_ROOT/strategy_comparison_v4_acceptable/comparison.csv"
)

log "START dataset-wise supported Pareto frontier"
PYTHONPATH=. "$PYTHON_BIN" -m experiments.oracle_pareto_frontier_gifteval \
  "${COMMON[@]}" "${SELECTORS[@]}" \
  --flops-weighting cell \
  --output-dir "$OUTPUT_ROOT/Chronos2-Small" >>"$LOG_FILE" 2>&1

log "START instance-weighted supported Pareto frontier"
PYTHONPATH=. "$PYTHON_BIN" -m experiments.oracle_pareto_frontier_gifteval \
  "${COMMON[@]}" "${SELECTORS[@]}" \
  --flops-weighting instances \
  --output-dir "$OUTPUT_ROOT/Chronos2-Small_instance_weighted" \
  >>"$LOG_FILE" 2>&1

RISK_ROOT="$MASTER_ROOT"
RISK_REFS=(
  --risk-reference "Balanced risk=$RISK_ROOT/calibrated_context_risk/chronos2_small/real_evaluation.json#balanced"
  --risk-reference "Aggressive risk=$RISK_ROOT/calibrated_context_risk/chronos2_small/real_evaluation.json#aggressive"
  --risk-reference "Very aggressive risk=$RISK_ROOT/calibrated_context_risk_very_aggressive/chronos2_small/real_evaluation.json#very_aggressive"
  --risk-reference "Very-very aggressive risk=$RISK_ROOT/calibrated_context_risk_very_very_aggressive/chronos2_small/real_evaluation.json#very_very_aggressive"
  --risk-reference "Extreme risk=$RISK_ROOT/calibrated_context_risk_extreme/chronos2_small/real_evaluation.json#extreme"
  --risk-reference "Very extreme risk=$RISK_ROOT/calibrated_context_risk_extreme/chronos2_small/real_evaluation.json#very_extreme"
)

log "START series-wise feasible Pareto envelope with calibrated-risk overlays"
PYTHONPATH=. "$PYTHON_BIN" -m experiments.serieswise_oracle_pareto_gifteval \
  --instance-dir "$MASTER_ROOT/instance_window_evaluation_chronos2_small_v3" \
  --comparison-csv "$COMPARISON" \
  --model-short Chronos2-Small \
  --output-dir "$OUTPUT_ROOT/Chronos2-Small_serieswise" \
  "${RISK_REFS[@]}" >>"$LOG_FILE" 2>&1

log "OVERNIGHT PARETO COMPLETE: $OUTPUT_ROOT"
