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
   many plots. The wall-clock ("clock-wise") path (Stage 6 / `*_time` outputs)
   defaults to the **robust** forward-pass timing (`timing.json` mean ± std from
   the timing stage below), falling back per-cell to the single-shot
   `elapsed_seconds` and drawing std error bars when available
   (`--use-robust-timing` / `--no-use-robust-timing`).

**Extras / v2 path:**
- **`period_window_eval.py`** — a 4th strategy: pick `L_i = max(2×strongest_period,
  horizon)` per instance (period via FFT+autocorrelation). Evaluated directly
  (off-grid), written as sidecars that stage 4 folds in as a `period` column.
- **`run_all_v2.py`** — reuses stages 1–3 from `run_all.py`, inserts the period
  stage, re-runs the comparison.
- **`predict_period.py`** — the original period-regression Patch-Transformer that
  `predict_context_length.py` is modeled on.

**Robust timing stage (run_all_5 path):**
- **`benchmark_window_timing_gifteval.py`** — *robust wall-clock timing*. Reads
  each model's `comparison.csv`, collects the on-grid windows the strategies
  actually use (`full`/`best`/`pred`/variant `*_window`; period is off-grid so it
  keeps its single-shot timing), and times each one with `--warmup` discarded +
  `--repeats` timed forward passes, every pass `cuda.synchronize()`-bracketed and
  batch-build excluded. Writes `timing.json` (mean/std/min/median/cv) into the
  same per-cell dir as `metrics.json` — and because v3/v4 symlink their
  `datasets/`, the predictor-independent TSFM timing is measured once on
  `general/` and reused everywhere. Multi-GPU dataset-sharding like the ablation;
  resumes per cell.
- **`run_all_5.py`** — reuses stages 1–4 from `run_all.py`, adds the timing stage,
  then re-runs the comparison with `--use-robust-timing` so the wall-clock figures
  use mean ± std.

**Leaderboard-faithful MASE (`mase_gluonts`), computed alongside `mase`:**
- **`gifteval_mase.py`** — leaderboard-faithful MASE primitives: gluonts'
  `DEFAULT_SEASONALITIES` map (`D→1, W→1, H→24, S→3600, T→1440, M→12, Q→4`, with
  `divmod` for multipliers) + per-instance `seasonal_error` (`mean(|x[m:]−x[:−m]|)`
  over each series' own context). One auditable place for the definition.
- **Stage 3** (`test_window_ablation_gifteval_v5.py`) now computes `mase_gluonts`
  as a **first-class metric column** beside `mase` — it's the average of
  per-instance ratios `mean_h(|y−ŷ|) / seasonal_error_instance`, vs `mase` =
  `global_MAE / one pooled training-set seasonal-naive MAE` with the project's
  custom `D→7, W→52` map. `GiftEvalCache` precomputes the per-instance seasonal
  errors once; metrics/per-sample/`results.csv`/the ablation summary plots all
  carry it. Pre-existing cells are **backfilled with no TSFM re-inference**
  (`_backfill_mase_gluonts` divides the cached per-instance MAE by the
  data-derived seasonal error), so a plain ablation re-run populates it cheaply.
  Each `compare_*.npz` carries a parallel `real_curve_gluonts`.
- **Stage 4** (`compare_window_strategies_gifteval.py`) takes
  `--mase-metric {mase,mase_gluonts}` (default `mase`); with `mase_gluonts` it
  drives the whole strategy comparison + **all flops/time-savings outputs** off
  the gluonts curve. `run_all.py` exposes the same `--mase-metric`, routing its
  per-model outputs to `strategy_comparison_gluonts/` and the rollup to
  `general/rollup_gluonts/` so the two metrics never overwrite each other.
- Even in a **default `mase` run**, when the gluonts curve is present stage 4
  additionally scores every strategy on `mase_gluonts` (its own oracle-best
  window) to: draw the general `bar_aggregate_mase.png` on `mase_gluonts`, and
  emit a parallel `model_strategy_overview_gluonts.png` (+
  `flops_savings_all_models_gluonts.csv`) beside the default-`mase` overview. No
  extra inference — a second pass over the cached `compare_*.npz`
  (`run_has_gluonts_curve` gates it).

**Master orchestrator:**
- **`master_run_all.py`** — fuses *every* `run_all*` variant into one run while
  running each shared stage **exactly once**: Stage 1 up front, then each variant
  as a subprocess with an explicit `--skip-stages` set, then
  `rollup_all_predictors`. A `VARIANTS` registry is the single source of truth —
  **add any new `run_all_*` there (with the stages it should skip) so master
  stays the fuse-everything entry point and never recomputes a shared stage.**

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
python -m experiments.run_all_5                   # + robust wall-clock timing stage
python -m experiments.run_all_5 --repeats 10 --warmup 3
# Leaderboard-faithful MASE (mase_gluonts). Step 1 backfills the metric into the
# ablation cells WITHOUT TSFM re-inference (cells are cached); step 2 runs the
# strategy comparison + flops-savings off the gluonts curve into *_gluonts dirs.
python -m experiments.run_all --skip-stages 1 2 --force 3       # backfill mase_gluonts + default compare
python -m experiments.run_all --skip-stages 1 2 3 --mase-metric mase_gluonts  # gluonts compare + flops savings
python -m experiments.master_run_all              # fuse ALL variants, no repeated stages
python -m experiments.master_run_all --only-variants v1 v5
```
Stage 1 alone: `python -m experiments.build_context_length_dataset --model-idx <i>`.
Timing stage alone: `python -m experiments.benchmark_window_timing_gifteval --run-dir logs/experiments/window_ablation_gifteval/general`.
