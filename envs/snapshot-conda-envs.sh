#!/usr/bin/env bash
# =============================================================================
# Export or restore the conda envs used by experiments/master_run_all.py.
#
# Export from the machine that has the working envs:
#   bash envs/snapshot-conda-envs.sh export
#   bash envs/snapshot-conda-envs.sh export /path/to/predictcsl-env-snapshot
#
# Restore on another machine:
#   bash envs/snapshot-conda-envs.sh restore envs/conda-snapshots/YYYYmmdd-HHMMSS
#   bash envs/snapshot-conda-envs.sh restore --update-existing envs/conda-snapshots/YYYYmmdd-HHMMSS
#
# The canonical env names come from experiments/master_run_all.py:
#   predictcsl-main, predictcsl-patchtst, predictcsl-legacy,
#   predictcsl-toto, predictcsl-tirex
#
# Older server aliases are accepted during export and saved under canonical names.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

CANONICAL_ENVS=(
  predictcsl-main
  predictcsl-patchtst
  predictcsl-legacy
  predictcsl-toto
  predictcsl-tirex
)
CONDA_ENV_NAMES_CACHE=""
RESTORE_TMP_DIR=""

usage() {
  cat <<'EOF'
Usage:
  bash envs/snapshot-conda-envs.sh export [OUT_DIR]
  bash envs/snapshot-conda-envs.sh restore [--update-existing] SNAPSHOT_DIR
  bash envs/snapshot-conda-envs.sh list

Examples:
  bash envs/snapshot-conda-envs.sh export
  bash envs/snapshot-conda-envs.sh restore envs/conda-snapshots/20260720-153000
  bash envs/snapshot-conda-envs.sh restore --update-existing envs/conda-snapshots/20260720-153000
EOF
}

require_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not on PATH. Open a conda-enabled shell first." >&2
    exit 1
  fi
}

load_conda_env_names() {
  if [[ -n "$CONDA_ENV_NAMES_CACHE" ]]; then
    return 0
  fi

  CONDA_ENV_NAMES_CACHE="$(conda env list --json 2>/dev/null | python -c '
import json
import os
import sys

data = json.load(sys.stdin)
for path in data.get("envs", []):
    print(os.path.basename(os.path.normpath(path)))
'
)"
}

aliases_for() {
  case "$1" in
    predictcsl-main)
      echo "predictcsl-main TSFM_moirai"
      ;;
    predictcsl-patchtst)
      echo "predictcsl-patchtst TSFM_PATCH"
      ;;
    predictcsl-legacy)
      echo "predictcsl-legacy TSFM_sundial_patch"
      ;;
    predictcsl-toto)
      echo "predictcsl-toto TSFM_toto"
      ;;
    predictcsl-tirex)
      echo "predictcsl-tirex predictcsl-test TSFM_tirex2"
      ;;
    *)
      echo "$1"
      ;;
  esac
}

env_exists() {
  local wanted="$1"
  local name
  load_conda_env_names
  while IFS= read -r name; do
    if [[ "$name" == "$wanted" ]]; then
      return 0
    fi
  done <<< "$CONDA_ENV_NAMES_CACHE"
  return 1
}

resolve_env() {
  local canonical="$1"
  local candidate
  for candidate in $(aliases_for "$canonical"); do
    if env_exists "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

snapshot_stamp() {
  date +%Y%m%d-%H%M%S
}

normalize_export_file() {
  local canonical="$1"
  local yml="$2"
  local tmp="$yml.tmp"

  awk -v canonical="$canonical" '
    BEGIN { wrote_name = 0 }
    /^name:/ && wrote_name == 0 {
      print "name: " canonical
      wrote_name = 1
      next
    }
    /^prefix:/ { next }
    { print }
    END {
      if (wrote_name == 0) {
        print "name: " canonical
      }
    }
  ' "$yml" > "$tmp"
  mv "$tmp" "$yml"
}

write_manifest() {
  local out_dir="$1"
  local manifest="$out_dir/manifest.tsv"

  {
    echo "canonical_env	exported_from	file	pip_freeze"
    for canonical in "${CANONICAL_ENVS[@]}"; do
      if [[ -f "$out_dir/$canonical.yml" ]]; then
        local resolved
        resolved="$(cat "$out_dir/$canonical.source-env")"
        echo "$canonical	$resolved	$canonical.yml	$canonical.pip-freeze.txt"
      fi
    done
  } > "$manifest"
}

list_required() {
  echo "Conda envs used by experiments/master_run_all.py:"
  for canonical in "${CANONICAL_ENVS[@]}"; do
    if resolved="$(resolve_env "$canonical")"; then
      echo "  $canonical -> found as $resolved"
    else
      echo "  $canonical -> missing; aliases: $(aliases_for "$canonical")"
    fi
  done
}

export_envs() {
  local out_dir="${1:-$HERE/conda-snapshots/$(snapshot_stamp)}"
  mkdir -p "$out_dir"

  echo "Writing conda env snapshot to: $out_dir"
  echo

  local missing=0
  local canonical
  for canonical in "${CANONICAL_ENVS[@]}"; do
    local resolved
    if ! resolved="$(resolve_env "$canonical")"; then
      echo "MISSING: $canonical (tried: $(aliases_for "$canonical"))" >&2
      missing=1
      continue
    fi

    echo "Exporting $canonical from installed env $resolved ..."
    conda env export -n "$resolved" --no-builds > "$out_dir/$canonical.yml"
    normalize_export_file "$canonical" "$out_dir/$canonical.yml"
    conda run -n "$resolved" python -m pip freeze --all > "$out_dir/$canonical.pip-freeze.txt"
    printf "%s\n" "$resolved" > "$out_dir/$canonical.source-env"
  done

  write_manifest "$out_dir"
  cat > "$out_dir/README.txt" <<EOF
PredictCSL conda env snapshot
Created: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Repo: $REPO_ROOT

Restore:
  bash envs/snapshot-conda-envs.sh restore "$out_dir"

Restore into existing env names:
  bash envs/snapshot-conda-envs.sh restore --update-existing "$out_dir"

Notes:
  - YAML files are exported with conda env export --no-builds.
  - pip-freeze files are included for inspection/debugging.
  - CUDA/Linux-only wheels in these envs may not restore on macOS.
EOF

  echo
  if [[ "$missing" -ne 0 ]]; then
    echo "Snapshot finished with missing envs. See messages above." >&2
    exit 1
  fi
  echo "Done. Copy this directory to the target machine:"
  echo "  $out_dir"
}

restore_envs() {
  local update_existing=0
  if [[ "${1:-}" == "--update-existing" ]]; then
    update_existing=1
    shift
  fi

  local in_dir="${1:-}"
  if [[ -z "$in_dir" ]]; then
    usage
    exit 2
  fi
  if [[ ! -d "$in_dir" ]]; then
    echo "ERROR: snapshot dir does not exist: $in_dir" >&2
    exit 1
  fi

  local conda_platform
  conda_platform="$(conda info --json | python -c '
import json
import sys
print(json.load(sys.stdin).get("platform", ""))
')"
  if [[ "$conda_platform" != linux-* ]]; then
    RESTORE_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/predictcsl-conda-restore.XXXXXX")"
    trap 'rm -rf "$RESTORE_TMP_DIR"' EXIT
    echo "Target platform is $conda_platform; filtering Linux/CUDA-only snapshot entries."
  fi

  local canonical
  for canonical in "${CANONICAL_ENVS[@]}"; do
    local yml="$in_dir/$canonical.yml"
    local portable_requirements=""
    if [[ ! -f "$yml" ]]; then
      echo "Skipping $canonical: missing $yml"
      continue
    fi

    if [[ -n "$RESTORE_TMP_DIR" ]]; then
      local portable_yml="$RESTORE_TMP_DIR/$canonical.yml"
      portable_requirements="$RESTORE_TMP_DIR/$canonical.requirements.txt"
      awk '
        # A cross-platform restore only needs Conda to establish the Python
        # runtime. Native system libraries are resolved for the target host.
        /^dependencies:$/ { in_deps = 1; print; next }
        /^  - pip:$/ { exit }
        in_deps && /^  - / {
          if ($0 ~ /^  - (python|pip|setuptools|wheel)=/) print
          next
        }
        { print }
      ' "$yml" > "$portable_yml"
      awk '
        /^  - pip:$/ { in_pip = 1; next }
        !in_pip { next }
        in_pip && /^      - (nvidia-|cuda-|triton==|causal-conv1d==|mamba-ssm==)/ {
          next
        }
        in_pip && /^      - torch==/ {
          sub(/\+cu[0-9]+$/, "")
        }
        in_pip && /^      - salesforce-gift-eval==0\.0\.0a0$/ {
          print "git+https://github.com/SalesforceAIResearch/gift-eval.git@d8184bb51079bb5021332f8e5d7486c378a52202"
          next
        }
        in_pip && /^      - granite-tsfm==0\.3\.4\.dev9\+ge4d488689$/ {
          print "git+https://github.com/ibm-granite/granite-tsfm.git@e4d48868969281f2f4cbc520bd8354c9f9ea3d48"
          next
        }
        { sub(/^      - /, ""); print }
      ' "$yml" > "$portable_requirements"
      yml="$portable_yml"
    fi

    if env_exists "$canonical"; then
      if [[ "$update_existing" -eq 1 ]]; then
        echo "Updating existing env $canonical from $yml ..."
        conda env update -n "$canonical" -f "$yml" --prune
      else
        echo "Skipping existing env $canonical (pass --update-existing to update it)."
      fi
    else
      echo "Creating env $canonical from $yml ..."
      conda env create -n "$canonical" -f "$yml"
    fi

    if [[ -n "$portable_requirements" ]]; then
      echo "Installing portable pip snapshot into $canonical ..."
      conda run -n "$canonical" python -m pip install --no-deps -r "$portable_requirements"
    fi
  done

  echo
  echo "Restore pass complete. Run this smoke check next:"
  echo "  conda run -n predictcsl-main python -m experiments.master_run_all --test"
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    export)
      require_conda
      shift
      export_envs "$@"
      ;;
    restore)
      require_conda
      shift
      restore_envs "$@"
      ;;
    list)
      require_conda
      list_required
      ;;
    ""|-h|--help|help)
      usage
      ;;
    *)
      echo "ERROR: unknown command: $cmd" >&2
      usage
      exit 2
      ;;
  esac
}

main "$@"
