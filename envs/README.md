# PredictCSL environments (server-side)

> All real runs happen on the **server** (`ando`), not locally. These files are
> the **single source of truth** for the conda envs the pipeline needs, captured
> from the working server envs via `pip freeze` on 2026-06-24. If you change a
> model loader or a dependency, update the matching `setup-*.sh` here.

## Why three envs (and not one)

Three environments are forced by hard, irreconcilable dependency conflicts — you
cannot collapse them into one:

| | `predictcsl-main` | `predictcsl-legacy` | `predictcsl-toto` |
|---|---|---|---|
| Python | 3.11 | 3.11 | **3.12** (toto-models 1.0.0 has no 3.11 wheel) |
| torch | 2.4.1+cu121 | 2.4.1+cu121 (inherited) | **2.5.1+cu121** |
| transformers | 4.56.0 | **4.40.1** (Sundial/TimeMoE legacy `DynamicCache`) | (toto's own) |

- **`predictcsl-main`** — the workhorse. Modern TSFM stack + GiftEval **+ the
  Mamba predictor** (mamba-ssm ships here as a prebuilt wheel). Runs every model
  except Toto, Sundial, TimeMoE, and both stage-2 predictors (PatchTST + Mamba).
- **`predictcsl-legacy`** — a *clone of main* re-pinned to `transformers==4.40.1`.
  Only stage-1 labeling of Sundial (idx 5) + TimeMoE (idx 6).
- **`predictcsl-toto`** — standalone Python-3.12 env. Only Toto-2.0-313m (idx 9).

The per-cell on-disk cache is **shared across all envs**, so each just fills in
its own families; stages 2/4/5 run in `main` over the merged cache.

### Key trick: TSFM packages are git installs, not PyPI

In `main`, `chronos` / `granite-tsfm` / `timesfm` / `gift_eval` are installed
from **git at pinned commits**, not PyPI. PyPI `granite-tsfm` pins `torch>=2.10`
(and PyPI gluonts/chronos pull newer, conflicting deps); the pinned commits
resolve cleanly against `torch 2.4.1`. Don't "simplify" them to plain
`pip install <name>` — that's what reintroduces the resolver hell.

### Mapping to the env names on the server

| Canonical (these files) | Server env |
|-------------------------|------------|
| `predictcsl-main`   | `TSFM_moirai` |
| `predictcsl-legacy` | `TSFM_sundial_patch` |
| `predictcsl-toto`   | `TSFM_toto` |

You can keep the server names and just treat these scripts as the reproducible
spec; the canonical names are what new setups should use.

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
| **9** | **Toto-2.0-313m** | **toto (toto2)** | **toto** |
| 10 | FlowState-R1 | flowstate (granite-tsfm) | main |
| 11 | TiRex | tirex | main |

- **Stage 2** (predictor training): `main` for both the default PatchTST predictor
  and the Mamba variant (`PREDICTCSL_PREDICTOR_ARCH=mamba`, `run_all_v4`).
- **Stages 3–5** (GiftEval ablation / compare / timing): `main`. Toto's stage-3
  cells need `toto2` + `gift_eval`, so run those in the `toto` env.

### Typical server workflow

```bash
# --- main: modern models + both predictors + all stages ---
conda activate predictcsl-main
python -m experiments.run_all --models \
    Chronos2-Small Moirai2-Small TimesFM2.5-200M PatchTST-FM-R1 \
    Chronos2-Synth ChronosBolt-Base FlowState-R1 TiRex

# --- legacy: the two trust_remote_code families (stage 1 only) ---
conda activate predictcsl-legacy
python -m experiments.build_context_length_dataset --model-idx 5   # Sundial
python -m experiments.build_context_length_dataset --model-idx 6   # TimeMoE

# --- toto: Toto-2.0-313m (stage 1; stage 3 ablation also runs here) ---
conda activate predictcsl-toto
python -m experiments.build_context_length_dataset --model-idx 9   # Toto

# --- back to main: stages 2-5 now see the full merged cache ---
conda activate predictcsl-main
python -m experiments.run_all --skip-stages 1                       # predictor + ablation + compare
PREDICTCSL_PREDICTOR_ARCH=mamba python -m experiments.run_all_v4    # Mamba predictor variant
```

## Setup

One `setup-*.sh` script per env, all in this directory. Run from the repo root.

```bash
bash envs/setup-all.sh        # builds all three, in the right order
```

Or one at a time (main must come before legacy — legacy clones it):

```bash
bash envs/setup-main.sh       # predictcsl-main   (workhorse + Mamba predictor)
bash envs/setup-legacy.sh     # predictcsl-legacy (clone of main + transformers 4.40.1)
bash envs/setup-toto.sh       # predictcsl-toto   (Python 3.12, Toto only)
```

| Script | Env | Notes |
|--------|-----|-------|
| `setup-main.sh`   | `predictcsl-main`   | torch 2.4.1+cu121; modern TSFMs (git-pinned) + GiftEval + mamba |
| `setup-legacy.sh` | `predictcsl-legacy` | clones main, re-pins `transformers`/`tokenizers`/`huggingface-hub` |
| `setup-toto.sh`   | `predictcsl-toto`   | Python 3.12, torch 2.5.1+cu121, `toto-2` + `toto-models` |
| `setup-all.sh`    | all three           | runs the above in order |

> **Pins:** `main` is captured exactly from `TSFM_moirai`. `legacy` and `toto`
> have their hard constraints pinned exactly; a few generic deps in `toto` are
> best-effort — harden them from a full `pip freeze` of `TSFM_toto` if a resolver
> conflict shows up. Each script ends with a `torch.cuda.is_available()` + import
> sanity check.
