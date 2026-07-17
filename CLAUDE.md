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

**Synthetic single-factor sweeps (standalone, NOT part of run_all\*):**
- **`synth_param_sweeps.py`** — per-TSFM plots of MAE vs **normalized context**
  (context/parameter). Twelve experiments, each sweeping ONE generative factor
  in minimal isolated series (factor + noise only); full designs + hypotheses
  in **`SYNTH_SWEEPS.md`**. Core seven: `period` (context/T), `seasonality`
  (composite repeating profile, context/S), `ar_order` (AR(p) with dominant
  pole matched to a target ACF timescale τ; curves per order, context/τ),
  `memory` (AR(1), τ = 1/(1−φ), context/τ), `delay` (trigger→response Hann
  bumps at lag d — patch-wide, not spikes; one response planted in the
  horizon, context/d), `regime` (fixed regime duration D, last boundary
  exactly D before forecast time, context/D), `horizon` (fixed signal,
  context/h). Extension five: `break_age` (ONE change point at age A —
  deconfounds regime age from duration; error should rise past context/A = 1;
  the most on-thesis probe), `snr` (fixed T=256, noise σ swept — does the
  saturation knee shift right ∝ σ²?), `multiscale` (inner T=64 wave with a
  k-periodic amplitude envelope → knee at context/(k·T) ≈ 1?), `period_drift`
  (OU log-period with timescale M — frequency staleness, context/M),
  `missing_gap` (mean-filled gap of length G at [end−1.5G, end−0.5G),
  context/G; appendix — tests constant spans, not native NaN handling).
  Contexts sit on a RATIO grid (context = r × parameter, r ∈ 0.25…16) so
  curves from different parameter values align on the same x-axis. Series are
  seeded per (experiment, bin) — identical across models. Reuses stage-1
  loaders/`_forecast_uniform` + context caps; (model, experiment) cell queue
  across GPUs with done-markers; `--plot-only` regenerates plots. Output:
  `logs/experiments/synth_param_sweeps/`.

**MASE metrics — exactly TWO drive everything (leaderboard parity is priority #1):**
- **`mase_gluonts_real`** (the DEFAULT) — gluonts' **own** `evaluate_forecasts` +
  `ev.MASE` on real Forecast objects: the exact HF-GiftEval leaderboard path.
- **`mase_gluonts`** — a validated *numpy port* of the same definition (average of
  per-instance ratios `mean_h(|y−ŷ|) / seasonal_error_instance`, gluonts season map
  `D→1, W→1, H→24, …`), backfillable into cached cells with no TSFM re-inference.
- The legacy project `mase` (pooled training-naive, custom `D→7, W→52` map) is
  **no longer consumed by stage 4 / the rollups** — cells lacking the gluonts
  curves are skipped loudly, never silently mixed.
- **`gifteval_mase.py`** — the one auditable home for both definitions:
  seasonality map + per-instance `seasonal_error` (port) and
  `gluonts_leaderboard_mase(...)` / `_build_gluonts_forecasts` (machinery; lazy
  gluonts import). Shared by stage 3, `period_window_eval.py` and
  `compare_mase_variants.py`.
- **Stage 3** (`test_window_ablation_gifteval_v5.py`) — computes both columns per
  cell. Seasonal errors come from **raw contexts (NaNs preserved)** — `GiftEvalCache`
  keeps `contexts_raw` for the metric paths and the NaN→0-filled `contexts` only
  for model input (leaderboard exactness; `MASE_GLUONTS_VER = 3`). On **fresh**
  inference `cell_mase_gluonts_real` runs the machinery; **cached cells** have no
  stored forecasts so `_real` **stands in with the port** (marked
  `_mase_gluonts_real_standin: true`; `--force 3` recomputes truly — needed for
  exactness only on NaN-label `*_with_missing` cells, elsewhere port == machinery).
  The naive baseline's `mase_gluonts_real` **runs the actual machinery** (it's the
  normalisation denominator). Each `compare_*.npz` carries `real_curve_gluonts`
  and `real_curve_gluonts_real`.
- **Stage 4 / `run_all.py`** — `--mase-metric {mase_gluonts, mase_gluonts_real}`
  (default `mase_gluonts_real`). The default **owns the plain**
  `strategy_comparison/` subdir + `general/` rollup (the timing stage reads the
  plain `comparison.csv`); `mase_gluonts` routes to `strategy_comparison_gluonts/`
  + `general/rollup_gluonts/`. **Figures are minimal by default**: exactly two per
  model, both on the primary metric —
  `bar_aggregate_mase_gluonts.png` (absolute) and
  `bar_aggregate_mase_gluonts_normalized.png` (÷ same-definition Seasonal-Naive,
  =1.0 line — the leaderboard-faithful headline). The minimal bars hold the core
  full/best/pred trio ONLY: appending period/v3/v4 bars dropna's the rows to that
  strategy's coverage, silently shrinking n (95→80) — extra-strategy bars exist
  solely under `--all-figures`. Every CSV /
  `summary_stats.json` still emits. `--all-figures` restores the historical set
  (metric twins — in that mode `_gluonts` in a filename means the PORT —
  scatters, histograms, per-dataset bars, rollup overview/table figures). The
  headline aggregate is the **leaderboard aggregation**: geomean over cells of
  `MASE / seasonal-naive MASE` (same definition) — the normalized bar figure, the
  `geomean_norm` stat, and the NORM columns in `flops_savings_all_models*.csv`.

**Interpretability framework (`context_interpretability/`, top-level package):**
Tests whether observations beyond the sufficient context are causally inert
(predictive redundancy) vs architecturally unreachable. Six experiments behind
one config (`configs/experiments.yaml`), one tabular schema (`schema.py`) and
one figure/hypothesis pipeline (`analysis/`): **exp0** = the EXISTING attention
masking (`experiments/context_attention_mask.py`, mechanism untouched — only
rescored into the common schema), **exp1** temporal-block perturbation (the
principal causal evidence: mean/permute/matched/noise), **exp2** activation
patching (recovery scores per layer×block), **exp3** frozen-head forecast lens
(identity-skip of deeper residual blocks), **exp4** synthetic distant-dependency
controls (MANDATORY; ridge oracles verify lag-d is genuinely predictive before
any model conclusion), **exp5** integrated gradients (corroborative only).
Model access goes through `adapters/` (capability flags in
`configs/models/capabilities.yaml`; unsupported methods are skipped + logged in
`run_meta.json`, never approximated); the TSFM adapter reuses `setup_model` /
`_forecast_uniform` / `context_attention_mask` verbatim. Entry point (SERVER):
`python -m context_interpretability.run_experiment --models <display> \
[--experiments exp0..exp5] [--source synthetic|gifteval] [--analyze-only]`.
Output under `logs/experiments/context_interpretability/<model>/`, per-cell
done-markers (resumable), figures + `hypotheses_report.{json,md}` (H1–H5)
regenerable offline. Tests run LOCALLY (no TSFM needed — dummy adapter):
`python -m unittest discover -s context_interpretability/tests -t .`.
Token-mapping / hook assumptions per family carry "verify on server" notes in
`adapters/tsfm.py`, mirroring `context_attention_mask.py`'s convention.

**Leaderboard sanity check (standalone, NOT part of run_all\*):**
- **`sanity_gifteval_leaderboard.py`** — clean-room replication of the OFFICIAL
  GiftEval leaderboard evaluation for one model, a line-for-line port of the
  official submission notebooks (SalesforceAIResearch/gift-eval
  `notebooks/timesfm2p5.ipynb` + `notebooks/chronos-2.ipynb`). Runs all 97
  configs with each model's official recipe, then diffs per config against the
  leaderboard's published `all_results.csv` and prints the PUBLISHED
  aggregation (geomean of MASE[0.5] ÷ published seasonal_naive MASE[0.5];
  reference CSVs in `experiments/leaderboard_reference/`). Targets:
  **timesfm-2.5 0.7050 (DEFAULT)**, chronos-2 0.6978, chronos-2-synth 0.7203.
  TimesFM-2.5 is the default replication gate because its official recipe has
  no extras (univariate flattening, independent series, full context capped
  15360, per-batch compile, 0.5-quantile head scored) — the cleanest
  apples-to-apples vs our pipeline. Chronos-2's official recipe DOES use
  native multivariate + `predict_batches_jointly` in-context cross-learning,
  which our pipeline deliberately lacks; knobs (`--univariate`,
  `--independent`, `--max-context 8192`) move the recipe stepwise toward the
  pipeline's to attribute the gap (knobs are part of the output tag).
  Known pipeline-vs-official TimesFM divergences to attribute: grid cap 8192
  vs 15360, and stage 3 scores TimesFM's POINT forecast as `median` while the
  official scores the 0.5-quantile head. Per-config JSON cache under
  `logs/experiments/sanity_leaderboard/<tag>/`, resumable.

**Master orchestrator:**
- **`master_run_all.py`** — fuses *every* `run_all*` variant into one run while
  running each shared stage **exactly once**: Stage 1 up front, then each variant
  as a subprocess with an explicit `--skip-stages` set, then
  `rollup_all_predictors`. A `VARIANTS` registry is the single source of truth —
  **add any new `run_all_*` there (with the stages it should skip) so master
  stays the fuse-everything entry point and never recomputes a shared stage.**

## Models labeled (the TSFMs under study)

Chronos2-Small/Synth/Base, ChronosBolt-Small/Base, Moirai2-Small,
TimesFM2.5-200M, PatchTST-FM-R1, Sundial-Base-128M, TimeMoE-200M, Toto-2.0-313m,
FlowState-R1, TiRex. Each has a `load_*` + `predict_*` wrapper in
`build_context_length_dataset.py` (catalog in `models_config.py`, append-only).

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
- **FlowState** needs `granite-tsfm>=0.3.6` (also serves PatchTST-FM): a git dev
  snapshot (0.3.4.dev9) silently random-initialized every `encoder.layers.*.out.*`
  weight of the r1.1 checkpoint → all-NaN forecasts → stage 3 writes nothing →
  opaque stage-4 "No records" crash. If FlowState metrics go NaN, check the
  from_pretrained load warnings first.
- **FlowState** is cadence-conditioned: stage 3 / every GiftEval consumer passes a
  per-dataset `scale_factor` (`flowstate_scale_factor()` in the v5 ablation +
  `cache.flowstate_scale` — exact port of IBM's leaderboard recipe: 24 /
  samples-per-cycle, domain decides daily→weekly-vs-yearly, `l2c` gets /7).
  Stage 1 keeps the neutral 1.0 (synthetic data has no cadence). The median MUST
  be read from `quantile_outputs` — on current tsfm_public the r1.1 config's
  `prediction_type='quantile'` is coerced to `'mean'`, making
  `prediction_outputs` a 3-D quantile-weighted mean (old tsfm_public 4-D
  quantile output kept as fallback).
- **TiRex**'s `forecast()` ALWAYS returns CPU tensors → wrappers move the median
  back to `device`. Internally it truncates AND NaN-left-pads every context to
  exactly its 2048 train length, so accuracy varies with the window but compute
  does not: wall-clock is flat across windows and the FLOPs column is a loose
  upper bound — don't sell context-length compute savings for TiRex.

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
# Refresh the gluonts metrics into cached ablation cells (cheap backfill of the
# port + corrected seasonal errors + machinery naive baseline, NO TSFM
# re-inference), then the default compare runs on mase_gluonts_real:
python -m experiments.run_all --skip-stages 1 2
# --force 3 re-infers stage 3, replacing port stand-ins with TRUE machinery
# values (only matters for exactness on NaN-label *_with_missing cells):
python -m experiments.run_all --skip-stages 1 2 --force 3
python -m experiments.run_all --skip-stages 1 2 3 --mase-metric mase_gluonts  # port-scored compare -> *_gluonts dirs
python -m experiments.master_run_all              # fuse ALL variants, no repeated stages
python -m experiments.master_run_all --only-variants v1 v5
python -m experiments.synth_param_sweeps          # single-factor sweeps, run set
python -m experiments.synth_param_sweeps --models Sundial-Base-128M TimeMoE-200M  # legacy env
python -m experiments.synth_param_sweeps --experiments period delay --plot-only
python -m experiments.sanity_gifteval_leaderboard                  # official-recipe TimesFM-2.5, target 0.7050
python -m experiments.sanity_gifteval_leaderboard --model chronos-2 # target 0.6978 (multivariate+joint recipe)
python -m experiments.sanity_gifteval_leaderboard --max-context 8192  # pipeline-style context cap
```
Stage 1 alone: `python -m experiments.build_context_length_dataset --model-idx <i>`.
Timing stage alone: `python -m experiments.benchmark_window_timing_gifteval --run-dir logs/experiments/window_ablation_gifteval/general`.
