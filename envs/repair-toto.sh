#!/usr/bin/env bash
# Repair an existing Toto env back to the project-pinned Toto stack.
set -euo pipefail

ENV="${1:-predictcsl-toto}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRAINTS="$SCRIPT_DIR/predictcsl-toto-constraints.txt"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

echo "Before repair:"
python - <<'PY'
import sys
print("python", sys.executable)
for name in ("torch", "numpy", "gluonts"):
    try:
        mod = __import__(name)
        print(name, getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        print(name, "failed:", repr(exc))
PY

pip install --force-reinstall \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install --force-reinstall -c "$CONSTRAINTS" \
    "numpy<2" scipy pandas matplotlib tqdm colorama python-dotenv \
    toto-2==2.0.0 toto-models==1.0.0
pip install --force-reinstall -c "$CONSTRAINTS" gluonts==0.15.1
pip install --force-reinstall -c "$CONSTRAINTS" \
    "git+https://github.com/SalesforceAIResearch/gift-eval.git@d8184bb51079bb5021332f8e5d7486c378a52202"

echo
echo "After repair:"
python - <<'PY'
import torch
import toto2
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda?", torch.cuda.is_available())
print("toto2", toto2.__file__)
PY
