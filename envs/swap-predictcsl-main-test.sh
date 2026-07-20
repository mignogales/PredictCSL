#!/usr/bin/env bash
# =============================================================================
# Swap two conda environment names:
#   predictcsl-main <-> predictcsl-test
#
# Conda does not provide a reliable in-place rename, so this uses clone/remove
# through a temporary env:
#   predictcsl-main -> __predictcsl_swap_tmp__
#   predictcsl-test -> predictcsl-main
#   __predictcsl_swap_tmp__ -> predictcsl-test
#
# Dry run:
#   bash envs/swap-predictcsl-main-test.sh
#
# Execute:
#   bash envs/swap-predictcsl-main-test.sh --yes
# =============================================================================
set -euo pipefail

FROM_A="predictcsl-main"
FROM_B="predictcsl-test"
TMP_ENV="__predictcsl_swap_tmp__"
DO_IT=0

if [[ "${1:-}" == "--yes" ]]; then
  DO_IT=1
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,24p' "$0"
  exit 0
elif [[ $# -gt 0 ]]; then
  echo "ERROR: unknown argument: $1" >&2
  echo "Use --yes to execute, or no args for a dry run." >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not on PATH. Open a conda-enabled shell first." >&2
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "$FROM_A" || "${CONDA_DEFAULT_ENV:-}" == "$FROM_B" || "${CONDA_DEFAULT_ENV:-}" == "$TMP_ENV" ]]; then
  echo "ERROR: activate a neutral env first, for example: conda activate base" >&2
  echo "Current env is: ${CONDA_DEFAULT_ENV}" >&2
  exit 1
fi

env_exists() {
  local wanted="$1"
  conda env list --json 2>/dev/null | python -c '
import json
import os
import sys

wanted = sys.argv[1]
data = json.load(sys.stdin)
names = {os.path.basename(os.path.normpath(path)) for path in data.get("envs", [])}
raise SystemExit(0 if wanted in names else 1)
' "$wanted"
}

run_or_print() {
  if [[ "$DO_IT" -eq 1 ]]; then
    printf '+ %q ' "$@"
    echo
    "$@"
  else
    printf 'DRY RUN: '
    printf '%q ' "$@"
    echo
  fi
}

for env in "$FROM_A" "$FROM_B"; do
  if ! env_exists "$env"; then
    echo "ERROR: required env does not exist: $env" >&2
    exit 1
  fi
done

if env_exists "$TMP_ENV"; then
  echo "ERROR: temporary env already exists: $TMP_ENV" >&2
  echo "Remove or rename it before running this swap." >&2
  exit 1
fi

cat <<EOF
About to swap conda env names:
  $FROM_A -> $FROM_B
  $FROM_B -> $FROM_A

Temporary env:
  $TMP_ENV
EOF

if [[ "$DO_IT" -ne 1 ]]; then
  echo
  echo "No changes made. Re-run with --yes to execute."
  echo
fi

run_or_print conda create --name "$TMP_ENV" --clone "$FROM_A" -y
run_or_print conda remove --name "$FROM_A" --all -y
run_or_print conda create --name "$FROM_A" --clone "$FROM_B" -y
run_or_print conda remove --name "$FROM_B" --all -y
run_or_print conda create --name "$FROM_B" --clone "$TMP_ENV" -y
run_or_print conda remove --name "$TMP_ENV" --all -y

echo
if [[ "$DO_IT" -eq 1 ]]; then
  echo "Swap complete."
else
  echo "Dry run complete."
fi
