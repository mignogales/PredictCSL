#!/usr/bin/env bash
# =============================================================================
# predictcsl-main — the workhorse env.
#
# Runs: stage-1 labeling for every model EXCEPT Sundial (idx 5) & TimeMoE (idx 6),
#       stage-2 PatchTST predictor, stages 3-5 (GiftEval ablation / compare /
#       timing), period eval, embedding saturation.
#
# Run:  bash envs/setup-main.sh
#
# NOTE ON PINS: only the hard constraints are pinned. Harden the rest from a
# working server env:  pip freeze > /tmp/main-freeze.txt  and copy the versions.
# =============================================================================
set -euo pipefail

ENV=predictcsl-main

conda create -n "$ENV" python=3.11 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

# --- core scientific stack ---
#  numpy<2 : scipy in this project is built against numpy 1.x — DO NOT bump to 2.x.
pip install "numpy<2" scipy pandas matplotlib tqdm colorama python-dotenv

# --- deep learning ---
#  torch : install a GPU build that actually runs on the server driver (CUDA <= 12.2).
#          granite-tsfm pins torch>=2.10,<2.11, so this env uses a torch ~2.10 GPU
#          wheel. Verify torch.cuda.is_available() is True after install (see
#          README re: the CUDA 12.2 driver ceiling).
#  transformers : modern (~4.56). The legacy 4.40.1 pin lives in predictcsl-legacy.
pip install torch transformers

# --- TSFM model packages  (PyPI distribution -> import name) ---
pip install \
    chronos-forecasting \
    uni2ts \
    timesfm \
    granite-tsfm \
    gluonts \
    toto-models \
    "tirex-ts[gluonts,hfdataset]"   # tirex; use [all] for every extra

# --- installed FROM SOURCE (git) ---
#  CONFIRM the ref and pin a commit/tag once verified on the server.
pip install "git+https://github.com/SalesforceAIResearch/gift-eval.git"  # import: gift_eval

echo
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
echo "predictcsl-main ready."
