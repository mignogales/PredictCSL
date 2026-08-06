#!/usr/bin/env bash
# Read-only audit of legacy per-instance cache alignment.
# Run from the repository root on the GPU server:
#   bash scripts/audit_cache_alignment.sh
set -euo pipefail

RUN_DIR="${RUN_DIR:-logs/experiments/master_recompute/window_ablation_gifteval/general}"
export RUN_DIR

conda run -n "${MAIN_ENV:-predictcsl-main}" python - <<'PY'
from collections import Counter
from pathlib import Path
import os
import numpy as np

root = Path(os.environ["RUN_DIR"]) / "datasets"
files = sorted(root.glob("*/*/t*/w*/per_sample_metrics.npz"))
if not root.is_dir():
    raise SystemExit(f"Cache directory not found: {root}")

total = Counter()
missing = Counter()
bad = []

for path in files:
    model = path.parts[-4]
    total[model] += 1
    try:
        with np.load(path) as data:
            aligned = "served_index" in data.files
    except Exception as exc:
        aligned = False
        bad.append((path, f"unreadable: {exc}"))
    if not aligned:
        missing[model] += 1
        if len(bad) < 30:
            bad.append((path, "missing served_index"))

print(f"Cache root: {root}")
print(f"Total per-sample NPZ files: {len(files)}")
print("\nPer model:")
for model in sorted(total):
    print(f"  {model:<24} total={total[model]:4d}  missing_served_index={missing[model]:4d}")

if bad:
    print("\nFirst affected paths:")
    for path, reason in bad[:30]:
        print(f"  {reason}: {path}")
else:
    print("\nAll discovered NPZ files contain served_index.")
PY
