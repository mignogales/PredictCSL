PredictCSL conda env snapshot
Created: 2026-07-20T14:05:44Z
Repo: /home/mnogales/Projects/PredictCSL

Restore:
  bash envs/snapshot-conda-envs.sh restore "/home/mnogales/Projects/PredictCSL/envs/conda-snapshots/20260720-160527"

Restore into existing env names:
  bash envs/snapshot-conda-envs.sh restore --update-existing "/home/mnogales/Projects/PredictCSL/envs/conda-snapshots/20260720-160527"

Notes:
  - YAML files are exported with conda env export --no-builds.
  - pip-freeze files are included for inspection/debugging.
  - CUDA/Linux-only wheels in these envs may not restore on macOS.
