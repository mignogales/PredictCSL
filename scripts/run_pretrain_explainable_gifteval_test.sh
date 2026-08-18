#!/usr/bin/env bash
# Frozen official GiftEval test of the pretraining-selected CART and comparator.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/mnogales/miniconda3/envs/TSFM_PATCH/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/experiments/master_recompute/gifteval_pretrain_explainable_official_test_v1}"
mkdir -p "$OUTPUT_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

PYTHONPATH=. "$PYTHON_BIN" -m experiments.evaluate_pretrain_explainable_gifteval_test \
  --output-dir "$OUTPUT_DIR" >"$OUTPUT_DIR/run.log" 2>&1

echo "PRETRAIN-SELECTED OFFICIAL GIFT-EVAL TEST COMPLETE: $OUTPUT_DIR"
