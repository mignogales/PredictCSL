#!/usr/bin/env bash
# =============================================================================
# predictcsl-legacy — Sundial (idx 5) + TimeMoE (idx 6) only.
#
# These two families ship trust_remote_code modeling files written for
# transformers 4.40.1 (legacy DynamicCache API). The main env's transformers 4.56
# rewrote that cache, and the old code can't be monkey-patched back — so they need
# their own pinned env. This is a CLONE of predictcsl-main with 3 packages re-pinned
# (torch and everything else stay identical, so loaders + GPU + shared cache match).
#
# Prereq: predictcsl-main must already exist (run envs/setup-main.sh first).
# Run:    bash envs/setup-legacy.sh
#
# Use (stage-1 labeling only; stages 2-5 run in main off the shared cache):
#   conda activate predictcsl-legacy
#   python -m experiments.build_context_length_dataset --model-idx 5   # Sundial
#   python -m experiments.build_context_length_dataset --model-idx 6   # TimeMoE
#
# Do NOT "fix" with use_cache=False: safe for TimeMoE but silently CORRUPTS Sundial.
# Verified working 2026-06-10.
# =============================================================================
set -euo pipefail

ENV=predictcsl-legacy

conda create --name "$ENV" --clone predictcsl-main -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

# pip "conflict" warnings from granite-tsfm / chronos-forecasting are harmless
# here — those families never load in this env.
pip install "transformers==4.40.1" "tokenizers==0.19.1" "huggingface-hub==0.23.4"

echo
python -c "import transformers; print('transformers', transformers.__version__)"
echo "predictcsl-legacy ready (Sundial idx 5 + TimeMoE idx 6, stage-1 only)."
