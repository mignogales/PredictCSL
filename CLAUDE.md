# PredictCSL — read this first

> This file is the orientation map for the repo. Read it before doing anything
> else so you know what you're working with. (Filename is `CLAUDE.md` on purpose
> — Claude Code auto-loads it every session.)

## ⚠️ Execution environment — READ THIS

- **Nothing runs locally.** This machine (macOS) only holds the source code. All
  real runs — dataset building, predictor training, GiftEval ablation — happen on
  a **remote GPU server** after the code is pushed/synced there.
- **Do NOT look for local outputs.** There are no `.npy` curves, no trained
  checkpoints, no plots/images, no `logs/` tree here. The `.gitignore` excludes
  `/logs` and `/model saved` precisely because those live on the server. If you
  need to inspect results, they are on the server, not in this checkout.
- The local conda env (`taming-env`) is only partial. TSFM dependencies, GPUs,
  and the GiftEval data (`GIFT_EVAL` path in `.env`) exist on the server.
- So: edit code, reason about logic, write scripts — but don't try to *run* the
  pipeline or open result artifacts here. They won't exist.

## What this project is

A **zero-shot "useful context length" predictor for time-series foundation
models (TSFMs)**. Core idea: when forecasting with a TSFM, *more context is not
always better* — stale history from an old regime can hurt. We want a model that,
given a series, predicts how much of the tail context is actually useful (the
input window that minimizes that TSFM's forecast error).

"Useful context length" has no generative parameter — it's an *operational*
quantity, so we **measure** it on synthetic data and train a predictor to
reproduce the measurement zero-shot.

## The pipeline (4 stages + extras), all under `experiments/`

Orchestrated by **`run_all.py`** (and **`run_all_v2.py`**, which adds a period
strategy). Each stage caches its work and has a done-marker so re-runs only do
what's missing. Per-family deterministic output paths under
`logs/experiments/` (on the server).

Two more orchestrator variants reuse `run_all.py`'s machinery verbatim and only
swap the **stage-2 predictor** (architecture/search), redirecting it to its own
`*_v3`/`*_v4` roots while symlinking the expensive stage-3 GiftEval cells from
the shared `general/` tree (the window grid is dataset-derived, not
predictor-derived, so no TSFM re-inference):
- **`run_all_v3.py`** — same PatchTST-Transformer predictor but a *constrained*
  HP search (low-FLOP corner: big patches, narrow `d_model`, shallow, ~20
  trials), so the predictor's own cost is negligible vs the labeled TSFM.
- **`run_all_v4.py`** — swaps the Transformer encoder for a **bidirectional
  Mamba** (selective state-space) stack — O(N) in the patch-token count instead
  of O(N²) — so the predictor is even cheaper and the search can afford smaller
  patches. Selected via `PREDICTCSL_PREDICTOR_ARCH=mamba` (resolved at import in
  `predict_context_length.py`, alongside `MambaContextLength` + the
  `build_predictor` factory). Requires `mamba-ssm` + `causal-conv1d` on the
  server (imported lazily, so the patchtst path needs neither).

1. **`build_context_length_dataset.py`** — *labeling*. Generates ~50k synthetic
   series (length 8192 + horizon) with injected non-stationarity (1–4 regimes,
   level/variance shifts, per-segment trend/AR/seasonality, wave-type variety,
   spikes, gradual transitions, ~10% short/left-padded series). For each series
   and each TSFM, forecasts at every window in `WINDOW_GRID` and records the full
   error-vs-(context, horizon) surface (MAE & MSE). Multi-GPU: one worker per
   CUDA device drains a shard queue; shards resume on re-run. Output:
   `contexts.npy`, `targets.npy`, `curves_{mae,mse}.npy`, etc.

2. **`predict_context_length.py`** — *predictor training*. A Patch-Transformer
   (sibling of `predict_period.py`) trained on the synthetic labels with a
   dual objective: `L = λ_curve·MSE_curve + λ_recon·MSE_reconstruction`. Predicts
   the per-series error-vs-context curve (z-scored — it learns curve *shape*);
   argmin of the predicted curve = recommended context length. Horizon is a
   first-class conditioning input. Multi-GPU random search (`N_TRIALS=60`).
   Output: `best_model.pt` + `best_config.json`.

3. **`test_window_ablation_gifteval_v5.py`** — *real-data ablation*. Runs each
   TSFM at every grid window on **GiftEval** datasets, and overlays the trained
   predictor's curve against the real one per (dataset, term). (`v4` is the prior
   version without the predictor overlay.) Shared output tree
   (`.../general/`) across all models.

4. **`compare_window_strategies_gifteval.py`** — *strategy comparison*. Pure
   post-processing (no inference): compares MASE + wall-clock + theoretical
   FLOPs across strategies: `full_window` (max context), `best_window` (oracle
   argmin), `pred_window` (predictor zero-shot). Emits CSVs, summary stats, and
   many plots.

**Extras / v2 path:**
- **`period_window_eval.py`** — a 4th strategy: pick `L_i = max(2×strongest_period,
  horizon)` per instance (period via FFT+autocorrelation). Evaluated directly
  (off-grid), written as sidecars that stage 4 folds in as a `period` column.
- **`run_all_v2.py`** — reuses stages 1–3 from `run_all.py`, inserts the period
  stage, re-runs the comparison.
- **`predict_period.py`** — the original period-regression Patch-Transformer that
  `predict_context_length.py` is modeled on.

## Models labeled (the TSFMs under study)

Chronos2-Small, Chronos2-Synth, ChronosBolt-Small, Moirai2-Small,
TimesFM2.5-200M, PatchTST-FM-R1, Sundial-Base-128M, TimeMoE-200M.
Each has a `load_*` + `predict_*` wrapper in `build_context_length_dataset.py`.

### Model-specific gotchas baked into the code
- **PatchTST-FM** has a fixed 8192 context and no mask input → genuine samples are
  **NaN-padded** to native length (NaN is its missing-value indicator).
- Variable-length models (chronos, timesfm, moirai, sundial, timemoe) get only
  the genuine (un-padded) suffix; short series flatten the label curve past
  `real_len`.
- **Sundial & TimeMoE** are legacy `trust_remote_code` models that need a separate
  env (`transformers==4.40.1`); the main env's 4.56 breaks them. There's a
  `_patch_dynamic_cache_seen_tokens()` shim restoring removed `DynamicCache` APIs.
- **Sundial** caps context at 2880; **TimeMoE** caps context+horizon at 4096.

## Top-level files (not part of the pipeline)
- `AR_DIVERSITY_CHANGES.md`, `diagnose_ar_periodicity.md` — design notes on the
  synthetic AR/periodicity composition.
- `ar_sampler_demo.py`, `analyze_synth_composition.py` — standalone analysis demos.
- `.env` — holds `GIFT_EVAL` data path (server-side).

## Handy commands (run on the SERVER, not here)
```
python -m experiments.run_all                    # full pipeline, all models
python -m experiments.run_all --models Chronos2-Small
python -m experiments.run_all --skip-stages 1 2  # only ablation + compare
python -m experiments.run_all --test             # tiny end-to-end smoke run
python -m experiments.run_all_v2                 # + the 2×period strategy
python -m experiments.run_all_v3                 # constrained (cheap) PatchTST predictor
python -m experiments.run_all_v4                 # Mamba predictor (needs mamba-ssm)
python -m experiments.run_all_v4 --cheap         # + pin the cheap Mamba corner
```
Stage 1 alone: `python -m experiments.build_context_length_dataset --model-idx <i>`.
