#!/usr/bin/env bash
# Source-disjoint real-data screen of explainable selectors (CPU only).
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/mnogales/miniconda3/envs/TSFM_PATCH/bin/python}"
DATA_ROOT="${DATA_ROOT:-logs/experiments/gifteval_pretrain_bounded_wide_v5}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/experiments/master_recompute/gifteval_pretrain_explainable_validation_v1}"
mkdir -p "$OUTPUT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

PYTHONPATH=. "$PYTHON_BIN" -m experiments.validate_explainable_gifteval_pretrain \
  --root "$DATA_ROOT" --output-dir "$OUTPUT_DIR" --n-jobs 12 \
  >"$OUTPUT_DIR/run.log" 2>&1

echo "GIFT-EVAL PRETRAIN EXPLAINABLE VALIDATION COMPLETE: $OUTPUT_DIR"
