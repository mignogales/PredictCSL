#!/usr/bin/env bash
# =============================================================================
# predictcsl-toto — Toto-2.0-313m only (catalog idx 9).
#  (Mirror of the working server env TSFM_toto, captured 2026-06-24.)
#
# Toto needs its OWN env because toto-models 1.0.0 requires Python >=3.12 (it has
# no 3.11 wheel) and ships its own torch 2.5.1+cu121 stack. The import name
# `toto2` is provided by the `toto-2` distribution; `toto-models` carries the
# model code/weights. Both are required.
#
# Run:  bash envs/setup-toto.sh
#
# Use (Toto stage-1 labeling, and stage-3 ablation which also needs gift_eval):
#   conda activate predictcsl-toto
#   python -m experiments.build_context_length_dataset --model-idx 9   # Toto
# Stages 2/4/5 run in predictcsl-main off the shared cache.
#
# PINS: Python/torch/toto/gluonts confirmed from TSFM_toto's pip freeze. The
# generic stack + transformers are best-effort — harden from a full
# `pip freeze` of TSFM_toto if a resolver conflict shows up.
# =============================================================================
set -euo pipefail

ENV=predictcsl-toto

conda create -n "$ENV" python=3.12 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

# 1) torch FIRST, cu121 (<= the server's CUDA 12.2 driver ceiling).
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 2) Core scientific stack used by build_context_length_dataset.
pip install "numpy<2" scipy pandas matplotlib tqdm colorama python-dotenv

# 3) Toto model packages (confirmed pins). toto-2 provides `import toto2`.
#    toto-2 declares gluonts>=0.16, but the working TSFM_toto env runs gluonts
#    0.15.1 — so install toto first, then force-downgrade gluonts in a SEPARATE
#    invocation (pip then only warns about the unmet pin instead of erroring).
pip install toto-2==2.0.0 toto-models==1.0.0
pip install gluonts==0.15.1

# 4) gift_eval for Toto's stage-3 ablation (same pinned commit as main).
pip install "git+https://github.com/SalesforceAIResearch/gift-eval.git@d8184bb51079bb5021332f8e5d7486c378a52202"

echo
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
python -c "import toto2; print('toto2 ->', toto2.__file__)"
echo "predictcsl-toto ready (Toto idx 9)."
