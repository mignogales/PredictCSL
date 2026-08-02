#!/usr/bin/env bash
# =============================================================================
# predictcsl-tirex — TiRex2 env.
#
# TiRex2's Python package is `tirex-2` and its import namespace is `tirex2`.
# It currently requires torch>=2.8 and numpy~=2.1.3, so it cannot live safely in
# predictcsl-main (torch 2.4.1 + numpy 1.26). GiftEval is installed with
# --no-deps below because its declared numpy/matplotlib pins are incompatible
# with TiRex2's runtime stack. Its historical datasets==2.17.x range is also
# incompatible with NumPy 2, so this env uses the API-compatible 2.21.0 release
# containing Hugging Face's NumPy-2 formatter fix.
#
# Run:  bash envs/setup-tirex.sh
# =============================================================================
set -euo pipefail

ENV=predictcsl-tirex
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRAINTS="$SCRIPT_DIR/predictcsl-tirex-constraints.txt"

conda create -n "$ENV" python=3.11 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

# Match the oldest CUDA-enabled stack published by TiRex2's Pixi config.
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

pip install -c "$CONSTRAINTS" \
    numpy==2.1.3 scipy pandas==2.2.3 matplotlib==3.9.4 \
    tqdm==4.67.3 colorama==0.4.6 python-dotenv==1.0.0 \
    datasets==2.21.0

pip install -c "$CONSTRAINTS" "tirex-2[gluonts]==0.1.1"
pip install --no-deps "git+https://github.com/SalesforceAIResearch/gift-eval.git@d8184bb51079bb5021332f8e5d7486c378a52202"

echo
python "$SCRIPT_DIR/check-tirex-env.py"
echo "predictcsl-tirex ready, including access to the gated NX-AI/TiRex-2-gifteval-zs checkpoint."
