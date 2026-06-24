# PredictCSL environments (server-side)

> All real runs happen on the **server** (`ando`), not locally. These files are
> the **single source of truth** for the conda envs the pipeline needs. If you
> change a model loader or a dependency, update the matching file here.

## Why three envs (and not one)

The code forces **exactly three** environments because of two hard,
irreconcilable dependency conflicts. You cannot collapse these into one env.

| Conflict | Group A | Group B |
|----------|---------|---------|
| `transformers` | `4.40.1` — Sundial + TimeMoE ship `trust_remote_code` modeling files written for the legacy `DynamicCache` API | `~4.56` — every other TSFM + `granite-tsfm` need the modern API |
| `torch` / CUDA | `2.5.1+cu121` — required to compile `mamba-ssm` / `causal-conv1d` (server driver caps at CUDA 12.2) | `>=2.10` — pinned by `granite-tsfm` (PatchTST-FM, FlowState) |

So:

1. **`predictcsl-main`** — the workhorse. Modern `transformers`, full TSFM stack,
   GiftEval. Runs every stage for every model **except** Sundial & TimeMoE.
2. **`predictcsl-legacy`** — a *clone of main* with `transformers==4.40.1`. Used
   **only** for stage-1 labeling of Sundial (idx 5) and TimeMoE (idx 6).
3. **`predictcsl-mamba`** — a *minimal, from-scratch* env with `torch 2.5.1+cu121`
   + `mamba-ssm`. Used **only** for `run_all_v4`'s Mamba predictor (stage 2) and
   the cheap GiftEval stages 3–4 (TSFM loaders are lazy + cells are symlinked, so
   no TSFM stack needed here).

### Mapping to the old (messy) env names

These canonical names replace the drifted set in the notes. Rename on the server
to match, or keep your names and just treat these files as the spec:

| Canonical | Old name(s) seen on server |
|-----------|----------------------------|
| `predictcsl-main`   | `TSFM_moirai`, `TSFM_PATCH`, `TSFM_flowstate` |
| `predictcsl-legacy` | `TSFM_sundial_patch` |
| `predictcsl-mamba`  | (the `run_all_v4` clone) |

## Which env runs what

Stage-1 `--model-idx` order (from `experiments/models_config.py`, APPEND-ONLY):

| idx | model | family | env |
|-----|-------|--------|-----|
| 0 | Chronos2-Small | chronos2 | main |
| 1 | ChronosBolt-Small *(catalog only, run=False)* | chronos_bolt | main |
| 2 | Moirai2-Small | moirai (uni2ts) | main |
| 3 | TimesFM2.5-200M | timesfm | main |
| 4 | PatchTST-FM-R1 | patchtst_fm (granite-tsfm) | main |
| **5** | **Sundial-Base-128M** | **sundial** | **legacy** |
| **6** | **TimeMoE-200M** | **timemoe** | **legacy** |
| 7 | Chronos2-Synth | chronos2 | main |
| 8 | ChronosBolt-Base | chronos_bolt | main |
| 9 | Toto-2.0-313m | toto (toto2) | main |
| 10 | FlowState-R1 | flowstate (granite-tsfm) | main |
| 11 | TiRex | tirex | main |

- **Stages 2–5** (predictor train, GiftEval ablation, compare, timing) run in
  **main** for the default PatchTST predictor; in **mamba** only for the v4
  Mamba predictor.
- The on-disk per-cell cache is **shared across envs**, so each env just fills
  in its own families. Run stage 1 for the modern models in `main` and for
  Sundial/TimeMoE in `legacy`; everything downstream reads the merged cache.

### Typical server workflow

```bash
# --- main env: modern models, all stages ---
conda activate predictcsl-main
python -m experiments.run_all --models \
    Chronos2-Small Moirai2-Small TimesFM2.5-200M PatchTST-FM-R1 \
    Chronos2-Synth ChronosBolt-Base Toto-2.0-313m FlowState-R1 TiRex

# --- legacy env: fill the two trust_remote_code families (stage 1 only) ---
conda activate predictcsl-legacy
python -m experiments.build_context_length_dataset --model-idx 5   # Sundial
python -m experiments.build_context_length_dataset --model-idx 6   # TimeMoE

# --- back to main: now stages 2-5 see the full merged cache ---
conda activate predictcsl-main
python -m experiments.run_all --skip-stages 1     # predictor + ablation + compare

# --- mamba env: the v4 (Mamba) predictor variant ---
conda activate predictcsl-mamba
PREDICTCSL_PREDICTOR_ARCH=mamba python -m experiments.run_all_v4
```

## Setup

One `setup-*.sh` script per env, all in this directory. Run from the repo root.

```bash
bash envs/setup-all.sh        # builds all three, in the right order
```

Or one at a time (main must come before legacy — legacy clones it):

```bash
bash envs/setup-main.sh       # predictcsl-main   (workhorse)
bash envs/setup-legacy.sh     # predictcsl-legacy (clone of main + transformers 4.40.1)
bash envs/setup-mamba.sh      # predictcsl-mamba  (from-scratch, run_all_v4 Mamba predictor)
```

| Script | Env | Notes |
|--------|-----|-------|
| `setup-main.sh`   | `predictcsl-main`   | core stack + modern TSFMs (PyPI) + GiftEval (git) |
| `setup-legacy.sh` | `predictcsl-legacy` | clones main, re-pins `transformers`/`tokenizers`/`huggingface-hub` |
| `setup-mamba.sh`  | `predictcsl-mamba`  | ordered, from-scratch build (order is load-bearing) |
| `setup-all.sh`    | all three           | runs the above in order |

> **Pins:** the hard constraints (the conflicts above) are pinned exactly. Loose
> transitive versions are intentionally left open — **harden them from the
> currently-working server envs** with `pip freeze`/`conda env export` and paste
> the exact versions back into these files. A few model packages
> (`toto2`, `tirex`, `gift_eval`) are installed from source (git) and are noted
> as such rather than guessed.
