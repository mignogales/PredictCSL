#!/usr/bin/env bash
# Repair an existing TiRex env back to the project-pinned TiRex2 stack.
set -euo pipefail

ENV="${1:-predictcsl-tirex}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRAINTS="$SCRIPT_DIR/predictcsl-tirex-constraints.txt"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

echo "Before repair:"
python - <<'PY'
import sys
print("python", sys.executable)
for name in ("torch", "numpy"):
    try:
        mod = __import__(name)
        print(name, getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        print(name, "failed:", repr(exc))
PY

pip install --force-reinstall \
    torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install --force-reinstall -c "$CONSTRAINTS" \
    numpy==1.26.4 scipy pandas==2.2.3 matplotlib==3.9.4 \
    tqdm==4.67.3 colorama==0.4.6 python-dotenv==1.0.0
pip install --force-reinstall -c "$CONSTRAINTS" "tirex-2[gluonts]==0.1.1"
pip install --force-reinstall -c "$CONSTRAINTS" \
    "git+https://github.com/SalesforceAIResearch/gift-eval.git@d8184bb51079bb5021332f8e5d7486c378a52202"

echo
echo "After repair:"
python - <<'PY'
import torch
from tirex2 import TimeseriesType, load_model
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda?", torch.cuda.is_available())
print("tirex2 import OK")
PY
