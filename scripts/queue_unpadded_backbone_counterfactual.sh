#!/usr/bin/env bash
# Wait for an existing queue/worker and a free GPU, then run the direct benchmark.
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 GPU_ID MODEL WAIT_PID" >&2
  exit 2
fi

GPU_ID="$1"
MODEL="$2"
WAIT_PID="$3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done
while true; do
  used="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "$used" -le 1024 ]]; then
    break
  fi
  sleep 60
done

GPU_ID="$GPU_ID" MODEL="$MODEL" \
  exec bash scripts/run_unpadded_backbone_counterfactual.sh
