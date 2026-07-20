#!/usr/bin/env bash
# =============================================================================
# predictcsl-tirex — TiRex2 env.
#
# TiRex2's Python package is `tirex-2` and its import namespace is `tirex2`.
# It currently requires torch>=2.8, so it cannot live safely in predictcsl-main
# (torch 2.4.1). GiftEval keeps numpy on the 1.26 line.
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
    numpy==1.26.4 scipy pandas==2.2.3 matplotlib==3.9.4 \
    tqdm==4.67.3 colorama==0.4.6 python-dotenv==1.0.0

pip install -c "$CONSTRAINTS" "tirex-2[gluonts]==0.1.1"
pip install -c "$CONSTRAINTS" "git+https://github.com/SalesforceAIResearch/gift-eval.git@d8184bb51079bb5021332f8e5d7486c378a52202"

echo
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
python -c "from tirex2 import TimeseriesType, load_model; print('tirex2 import OK')"
echo "predictcsl-tirex ready. Make sure Hugging Face access to NX-AI/TiRex-2 is accepted and HF_TOKEN is available."
