# PredictCSL environments (server-side)

> All real runs happen on the **server** (`ando`), not locally. These files are
> the **single source of truth** for the conda envs the pipeline needs, captured
> from the working server envs via `pip freeze` on 2026-06-24. If you change a
> model loader or a dependency, update the matching `setup-*.sh` here.

## Why four envs (and not one)

Four environments are forced by hard, irreconcilable dependency conflicts — you
cannot collapse them into one:

| | `predictcsl-main` | `predictcsl-legacy` | `predictcsl-toto` | `predictcsl-tirex` |
|---|---|---|---|---|
| Python | 3.11 | 3.11 | **3.12** (toto-models 1.0.0 has no 3.11 wheel) | 3.11 |
| torch | 2.4.1+cu121 | 2.4.1+cu121 (inherited) | **2.5.1+cu121** | **2.8.0+cu126** |
| numpy | 1.26.4 | 1.26.4 (inherited) | 1.x/2.x per resolver | **2.1.3** |
| transformers | 4.56.0 | **4.40.1** (Sundial/TimeMoE legacy `DynamicCache`) | (toto's own) | package resolver |

- **`predictcsl-main`** — the workhorse. Modern TSFM stack + GiftEval **+ the
  Mamba predictor** (mamba-ssm ships here as a prebuilt wheel). Runs every model
  except Toto, Sundial, TimeMoE, and TiRex2; runs both stage-2 predictors
  (PatchTST + Mamba) for compatible env groups.
- **`predictcsl-legacy`** — a *clone of main* re-pinned to `transformers==4.40.1`.
  Only stage-1 labeling of Sundial (idx 5) + TimeMoE (idx 6).
- **`predictcsl-toto`** — standalone Python-3.12 env. Only Toto-2.0-313m (idx 10).
- **`predictcsl-tirex`** — standalone TiRex2 env. The `tirex-2` package imports as
  `tirex2` and currently requires torch>=2.8 plus numpy 2.x, so it is kept out of
  `main`.

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
| `predictcsl-tirex`  | `predictcsl-test` / `TSFM_tirex2` |

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
| 8 | Chronos2-Base | chronos2 | main |
| 9 | ChronosBolt-Base | chronos_bolt | main |
| **10** | **Toto-2.0-313m** | **toto (toto2)** | **toto** |
| 11 | FlowState-R1 | flowstate (granite-tsfm) | main |
| **12** | **TiRex2** | **tirex (tirex2 package)** | **tirex** |

- **Stage 2** (predictor training): the default PatchTST predictor can run in
  each model's env group. The Mamba variant requires an env with `mamba-ssm`; the
  master skips mamba variants for `predictcsl-toto` and `predictcsl-tirex`.
- **Stages 3–5** (GiftEval ablation / compare / timing): run in the env that can
  import that model's TSFM package. The master handles this routing.

### Typical server workflow

```bash
# --- main: modern models + both predictors + all stages ---
conda activate predictcsl-main
python -m experiments.run_all --models \
    Chronos2-Small Moirai2-Small TimesFM2.5-200M PatchTST-FM-R1 \
    Chronos2-Synth Chronos2-Base ChronosBolt-Base FlowState-R1

# --- legacy: the two trust_remote_code families (stage 1 only) ---
conda activate predictcsl-legacy
python -m experiments.build_context_length_dataset --model-idx 5   # Sundial
python -m experiments.build_context_length_dataset --model-idx 6   # TimeMoE

# --- toto: Toto-2.0-313m (stage 1; stage 3 ablation also runs here) ---
conda activate predictcsl-toto
python -m experiments.build_context_length_dataset --model-idx 10  # Toto

# --- tirex: TiRex2 (stage 1 and TiRex2 GiftEval cells) ---
conda activate predictcsl-tirex
python -m experiments.build_context_length_dataset --model-idx 12  # TiRex2

# --- back to main: stages 2-5 now see the full merged cache ---
conda activate predictcsl-main
python -m experiments.run_all --skip-stages 1                       # predictor + ablation + compare
PREDICTCSL_PREDICTOR_ARCH=mamba python -m experiments.run_all_v4    # Mamba predictor variant
```

## Setup

One `setup-*.sh` script per env, all in this directory. Run from the repo root.

```bash
bash envs/setup-all.sh        # builds all four, in the right order
```

Or one at a time (main must come before legacy — legacy clones it):

```bash
bash envs/setup-main.sh       # predictcsl-main   (workhorse + Mamba predictor)
bash envs/setup-legacy.sh     # predictcsl-legacy (clone of main + transformers 4.40.1)
bash envs/setup-toto.sh       # predictcsl-toto   (Python 3.12, Toto only)
bash envs/setup-tirex.sh      # predictcsl-tirex  (TiRex2 only)
```

| Script | Env | Notes |
|--------|-----|-------|
| `setup-main.sh`   | `predictcsl-main`   | torch 2.4.1+cu121; modern TSFMs (git-pinned) + GiftEval + mamba |
| `setup-legacy.sh` | `predictcsl-legacy` | clones main, re-pins `transformers`/`tokenizers`/`huggingface-hub` |
| `setup-toto.sh`   | `predictcsl-toto`   | Python 3.12, torch 2.5.1+cu121, `toto-2` + `toto-models` |
| `setup-tirex.sh`  | `predictcsl-tirex`  | Python 3.11, torch 2.8.0+cu126, `tirex-2` / `tirex2` |
| `setup-all.sh`    | all four            | runs the above in order |

> **Pins:** `main` is captured exactly from `TSFM_moirai`. `legacy` and `toto`
> have their hard constraints pinned exactly; a few generic deps in `toto` are
> best-effort — harden them from a full `pip freeze` of `TSFM_toto` if a resolver
> conflict shows up. Each script ends with a `torch.cuda.is_available()` + import
> sanity check.
