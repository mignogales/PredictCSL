PredictCSL conda env snapshot
Created: 2026-07-20T13:59:30Z
Repo: /home/mnogales/Projects/PredictCSL

Restore:
  bash envs/snapshot-conda-envs.sh restore "/home/mnogales/Projects/PredictCSL/envs/conda-snapshots/20260720-155913"

Restore into existing env names:
  bash envs/snapshot-conda-envs.sh restore --update-existing "/home/mnogales/Projects/PredictCSL/envs/conda-snapshots/20260720-155913"

Notes:
  - YAML files are exported with conda env export --no-builds.
  - pip-freeze files are included for inspection/debugging.
  - CUDA/Linux-only wheels in these envs may not restore on macOS.
