# Alternative ideas for improving CSL prediction

Updated: 2026-08-10

## Scope and status caveat

The goal is to predict the context window that gives a time-series foundation
model (TSFM) the best forecast, while avoiding harmful shortening and retaining
useful compute savings.

This inventory is based on the code and launch scripts in this checkout. The
repository explicitly says that experiment outputs live on the remote server,
so **"implemented" below means that the method is represented in code; it does
not mean that it improved the final metric**. Runs inspected on the remote server
are recorded separately in the experiment log. Likewise, the uncommitted/new
scripts are marked as current work rather than completed evidence.

Status key:

- **Core**: part of the established pipeline.
- **Implemented**: code exists, but the local checkout has no result artifacts.
- **Current/queued**: recent or uncommitted work that appears to be in progress.
- **Diagnostic**: helps explain CSL but is not itself a deployed selector.
- **Screened**: tested end to end on the named configuration; result is recorded.
- **Running**: launched and not yet interpreted.
- **Retired**: deliberately removed from the active pipeline.
- **New**: no equivalent implementation was found in this checkout.

## Ideas already tried or currently being tried

### Selector targets and architectures

| Approach | Status | What is already covered | Main evidence in the repo |
|---|---|---|---|
| Predict the complete error-vs-window curve, then take its argmin | **Core** | Per-series, per-horizon z-scored curve regression with masked-patch reconstruction as an auxiliary loss | `experiments/predict_context_length.py` |
| Patch-Transformer predictor | **Core** | Original and constrained/cheap PatchTST-style predictors | `experiments/predict_context_length.py`, `experiments/run_all_v3.py` |
| Bidirectional Mamba predictor | **Implemented/core variant** | Linear-cost backbone, including a cheap search corner | `experiments/run_all_v4.py`, `experiments/master_run_all.py` |
| Soft top-k window classification | **Implemented** | Best/second/third windows receive weights 1, 1/2, and 1/4 | `experiments/predict_context_length.py` |
| Risk-aware curve prediction | **Implemented** | Calibrated log error relative to full context, differentiable expected regret, extra penalty for harm versus full context, and tail-aware checkpoint selection | `experiments/predict_context_length.py` |
| Adjacent cost-weighted pairwise ranking | **Screened: Chronos2-Small** | Ignores <=1% adjacent ties and weights decisive comparisons by absolute log error ratio; improved over native but did not beat curve/classification | `experiments/predict_context_length.py`; experiment log below |
| Ambiguity-aware acceptable action set | **Screened: Chronos2-Small** | Labels every window within 3% of the row oracle as acceptable; better than pairwise/curve on the official aggregate, but still behind soft top-k classification | `experiments/predict_context_length.py`; experiment log below |
| Continuous `log2(window)` regression | **Current/queued sanity work** | Capacity, memorization, and learning-curve tests on per-instance real oracle targets | `experiments/train_continuous_window_sanity.py`, `experiments/micro_overfit_window_regressor.py`, `experiments/window_regressor_learning_curve.py` |

### Policies layered on top of the predictor

| Approach | Status | What is already covered | Main evidence in the repo |
|---|---|---|---|
| Dataset-shared choice | **Core** | Average predicted curves within a dataset/term, then choose one window | `experiments/compare_window_strategies_gifteval.py` |
| Per-instance choice | **Implemented** | Choose an action for every row and retrieve that exact cached MASE | `experiments/evaluate_instance_windows.py` |
| Hierarchical shrinkage toward dataset consensus | **Screened: Chronos2-Small** | 10% instance score is the diagnostic compute-oriented point: 0.724371 MASE at 66.03% saving; accuracy delta vs 25% is not bootstrap-secure | `experiments/evaluate_instance_windows.py`, `SERIESWISE_PREDICTOR_EXPERIMENT.md` |
| Near-native bounded choice | **Implemented/current** | Prevent a dataset-shared selector from backing off more than a fixed number of grid steps | `experiments/evaluate_instance_windows.py` |
| Architecture score ensemble | **Implemented/current** | Average cheap-PatchTST and Mamba score curves before applying the guarded policy | `experiments/evaluate_instance_windows.py` |
| Top-k checkpoint robustness | **Implemented** | Evaluate several durable checkpoints against the same cached GiftEval cells | `experiments/evaluate_topk_predictor_checkpoints.py` |
| Learned "worth shortening" gate | **Current/queued** | Freeze the selector, use series/prediction features to predict realized improvement, calibrate conservative/balanced/aggressive thresholds, otherwise use native context | `experiments/train_shortening_worth_gate.py`, `experiments/evaluate_shortening_worth_gate.py` |
| Clean real-data worth gate | **Current/queued** | Train from GiftEval train/validation rolling origins without using official test labels; multi-model launcher exists | `experiments/train_gifteval_clean_worth_gate.py`, `scripts/run_multimodel_clean_worth_gate.sh` |
| Chronological real-data gates and deliberate test-contaminated capacity ceilings | **Current/queued; sanity only where marked** | Earlier origins to later origins, temporal hardness sweeps, and explicit contaminated upper bounds | `experiments/train_real_temporal_worth_gate.py`, `experiments/sweep_temporal_gate_hardness.py`, `experiments/train_real_test_worth_gate_sanity.py`, `experiments/train_on_gifteval_test_sanity.py` |

### Real-data, temporal, and feature-based alternatives

| Approach | Status | What is already covered | Main evidence in the repo |
|---|---|---|---|
| Conservative real-data bounded selector | **Current/queued** | Label GiftEvalPretrain rolling origins and select only among 4096/6144/8192 using source-disjoint splits | `experiments/pretrain_bounded_selector.py`, `experiments/evaluate_pretrained_bounded_gifteval.py` |
| Handcrafted-feature tree selector | **Current/queued** | Multi-scale statistics, autocorrelations, spectral features, ExtraTrees/RandomForest | `experiments/train_feature_bounded_selector.py` |
| Same-series temporal prediction | **Screened: negative** | Later-origin Mamba is 5.2% worse than native; held-out-series Mamba is 65.6% worse in the eight-cell diagnostic | `experiments/series_aware_window_splits.py`, `SERIESWISE_PREDICTOR_EXPERIMENT.md` |
| Previous-origin, series-median, and cell-median baselines | **Screened: negative** | Error/native is 1.465, 1.135, and 1.145 respectively on held-out last origins | `experiments/previous_origin_window_baselines.py`, `SERIESWISE_PREDICTOR_EXPERIMENT.md` |
| Stability-gated selector | **Promising narrow follow-up** | Stability >=0.6 covers 19.7% and reaches error/native 0.991 in the legacy-cache diagnostic; rerun on current cache and multiple seeds | `experiments/consecutive_origin_curve_stability.py`, `experiments/stability_gated_window_selector.py`, `SERIESWISE_PREDICTOR_EXPERIMENT.md` |

### Data design, transfer, and diagnostics

| Approach | Status | What is already covered | Main evidence in the repo |
|---|---|---|---|
| Large synthetic labeled pool | **Core** | Regimes, shifts, trends, AR, seasonality, spikes, gradual transitions, varying real lengths, and multiple horizons | `experiments/build_context_length_dataset.py` |
| More diverse AR processes | **Implemented/design update** | Root-sampled stable AR(1-3), stronger persistence, negative correlation, and oscillatory dynamics | `AR_DIVERSITY_CHANGES.md`, `diagnose_ar_periodicity.md` |
| Controlled single-factor synthetic sweeps | **Implemented/queued** | Period, structured seasonality, AR order, memory, delay, regime, horizon, break age, SNR, multiscale seasonality, period drift, and missing gaps | `SYNTH_SWEEPS.md`, `experiments/synth_param_sweeps.py` |
| Cross-TSFM transfer | **Implemented** | Apply a selector trained for one TSFM to another TSFM without rerunning target forecasts | `experiments/evaluate_cross_model_transfer.py` |
| Oracle distribution/agreement analysis | **Diagnostic/queued** | Per-instance oracle distributions, cross-model agreement, and split stability | `experiments/analyze_oracle_distributions.py` |
| Failure-tail analysis | **Diagnostic/current** | Rank harmful cells and inspect margins, score gaps, chosen windows, and per-row regret | `experiments/analyze_instance_failures.py` |
| Representation/state saturation | **Diagnostic** | Test whether embeddings or recurrent states plateau at the error-optimal context | `experiments/embedding_saturation.py`, `experiments/state_saturation.py` |
| Slicing, masking, normalization, and positional effects | **Diagnostic** | Separate true reachable-context effects from normalization and position changes | `experiments/context_attention_mask.py`, `experiments/masking_vs_slicing.py`, `context_interpretability/` |
| Period-detection heuristic | **Retired** | Period-based window selection was explored and removed from the active paper/pipeline | `experiments/archive/heuristics/` |
| Full/native, one global fixed window, dataset oracle, instance oracle | **Core baselines** | Already available; these should remain in every comparison | `experiments/evaluate_instance_windows.py` |

## Experiment log

### 2026-08-10 — series-wise Oracle predictability and Mamba shrinkage

**Status: unrestricted per-series prediction rejected; conservative stability
gate retained as a narrow follow-up.**

The official 97-cell shrinkage sweep found a useful compute point at 10% row
score plus 90% cell consensus: normalized MASE 0.724371 at 66.03% workload FLOPs
saving. Its -0.058% relative MASE change versus the previous 25% mixture has a
dataset-cluster bootstrap interval of [-0.248%, +0.073%], so this is not an
accuracy claim. It is practically tied while saving another 1.55 percentage
points of full workload FLOPs.

Repeated-origin controls explain why the 0.621624 series Oracle has not turned
into a strong learned policy. Across 455 consecutive-origin pairs, the exact
argmin repeats only 8.79% of the time and the previous argmin remains within 5%
of the next optimum only 25.93% of the time. Previous-origin and historical
median baselines all lose to native context. A small Mamba trained on earlier
origins is 5.2% worse than native on the held-out last origin; held-out-series
generalization is worse still.

A train-origin stability threshold of 0.6 is the one positive diagnostic: it
acts on 19.7% of series, obtains error/native 0.991, and limits >5% harm to 5.63%
on the older default cache. This needs a current-cache, multi-seed reproduction
before promotion. Full details are in `SERIESWISE_PREDICTOR_EXPERIMENT.md`.

### 2026-08-10 — adjacent pairwise Mamba, Chronos2-Small

**Status: screened end to end; keep as a possible ensemble/disagreement signal,
but do not widen the standalone sweep yet.**

Configuration:

- physical GPU 1 only (`CUDA_VISIBLE_DEVICES=1`);
- Chronos2-Small synthetic cache: 45,000 train / 5,000 validation series, 18
  candidate windows, and horizons 16/32/64/128/512/1024;
- cheap Mamba search, 4 trials;
- adjacent-pair BCE, <=1% relative-error ties ignored, decisive pairs weighted
  by clipped absolute log error ratio;
- cached-only GiftEval overlay: no Chronos2 inference was rerun;
- official 97-cell cohort using `mase_gluonts_real`.

Results (lower MASE is better):

| Selector | Synthetic val regret | Official normalized MASE | Gain vs full | Pooled FLOPs saved | Pooled time saved |
|---|---:|---:|---:|---:|---:|
| Full/native | — | 0.726708 | — | — | — |
| Existing Mamba curve (30 trials) | 0.19564 | 0.724344 | 0.325% | 56.49% | 52.62% |
| Existing Mamba classification (30 trials) | 0.19633 | **0.722412** | **0.591%** | 27.89% | 39.61% |
| New adjacent pairwise (4 trials) | 0.21321 | 0.725725 | 0.135% | 54.24% | 39.62% |

The pairwise policy beat full/native in 58 of the 95 selection-eligible cells,
but its mean relative cell gain was -0.12%, showing that a few larger harms
outweighed many small wins. The per-series evaluation reached geomean MASE
1.01425 versus 1.01317 for Mamba curve. The bounded dataset policy improved it
to 1.01033, still behind Mamba curve at 1.00875 and the best classification
control at 1.00853. The 4-versus-30 trial comparison is not perfectly matched,
but the synthetic and real rankings agree strongly enough that a larger
standalone pairwise sweep is not the next priority.

Remote artifacts:

- `logs/experiments/master_recompute/context_length_predictor_v4_pairwise/Chronos2-Small`
- `logs/experiments/master_recompute/window_ablation_gifteval/general_v4_pairwise`
- `logs/experiments/master_recompute/window_ablation_gifteval/Chronos2-Small/strategy_comparison_v4_pairwise`
- `logs/experiments/master_recompute/instance_window_evaluation_pairwise`

### 2026-08-10 — acceptable-set Mamba, Chronos2-Small

**Status: screened end to end; promising target-design evidence, but not the new
best selector. Do not widen the neural sweep before testing the decision rule.**

The first screen used a 3% acceptable set, a 0.5 credibility threshold, the same
cheap 4-trial Mamba budget, and the same cached-only 97-cell GiftEval evaluation.
This isolates target design from data, backbone, and forecast-cache changes.

The selected trial had synthetic validation regret 0.21834, p90 harm 9.04%, and
a 28.25% harmed rate. A different trial demonstrated the safety ceiling—20.93%
harmed and 5.09% p90 harm—but with regret 0.22199, so it was not selected.

Aggregate results:

| Selector | Official normalized MASE | Gain vs full | Pooled FLOPs saved | Pooled time saved |
|---|---:|---:|---:|---:|
| Full/native | 0.726708 | — | — | — |
| Existing Mamba curve | 0.724344 | 0.325% | 56.49% | 52.62% |
| Existing Mamba classification | **0.722412** | **0.591%** | 27.89% | 39.61% |
| Adjacent pairwise | 0.725725 | 0.135% | 54.24% | 39.62% |
| New 3% acceptable set | 0.723630 | 0.423% | 26.39% | 32.24% |

The acceptable set is the strongest of the newly tested objectives and improves
on ordinary curve regression in the leaderboard-faithful aggregate, but it does
not beat the existing soft top-k classifier and saves slightly less compute.
Its per-series policy has a narrowly better geomean than Mamba curve (1.01286
versus 1.01317), but a worse macro mean (1.48761 versus 1.48585), a worse
instance-weighted MASE, and only a 35.05% cell win rate. Its bounded dataset
policy reaches geomean 1.00931, behind curve at 1.00875 and the best
classification control at 1.00853.

Interpretation: ambiguity-aware labels contain useful signal, but the hard
`probability >= 0.5 -> choose largest` rule is probably too blunt. The cheapest
next ablation is deployment calibration on clean validation data: probability
threshold, largest-versus-argmax choice, and 1%/3%/5% label tolerance. Reuse this
checkpoint where possible before paying for a larger sweep.

Remote artifacts:

- `logs/experiments/master_recompute/context_length_predictor_v4_acceptable/Chronos2-Small`
- `logs/experiments/master_recompute/window_ablation_gifteval/general_v4_acceptable`
- `logs/experiments/master_recompute/window_ablation_gifteval/Chronos2-Small/strategy_comparison_v4_acceptable`
- `logs/experiments/master_recompute/instance_window_evaluation_acceptable`

## New alternatives worth trying

The first six ideas can be developed or at least screened without a free GPU.
They reuse cached labels, raw contexts, or CPU forecasts. They are ordered by my
expected information gained per unit of work, not by novelty.

### 1. Adjacent pairwise marginal-benefit selector

**Status: first neural screen completed; not competitive as a standalone
selector. Next use, if any, is as an ensemble/disagreement feature.**

Instead of estimating 18 absolute curve values, learn the local comparisons

```text
P(error(w_k) < error(w_{k+1}) - epsilon | series, horizon, model)
```

for adjacent grid windows. Weight each training pair by the magnitude of its
consequence, for example

```text
weight_k = clip(abs(log(error(w_k) / error(w_{k+1}))), 0, c).
```

This asks the exact decision question and makes nearly tied actions cheap to get
wrong. Absolute curve level, calibration, and arbitrary per-series z-scoring no
longer need to be learned. At inference, either scan from native toward shorter
windows until the predicted benefit is no longer credible, or aggregate adjacent
comparisons into a globally consistent score with isotonic/dynamic programming.

Start with the existing handcrafted features plus horizon using logistic
regression, HistGradientBoosting, or ExtraTrees. If it works, move the same loss
onto Mamba later. This is distinct from the current multi-class objective: the
training unit is a *cost-weighted comparison*, not one absolute class label.
Pairwise ranking losses are a standard way to avoid learning unnecessary
absolute scores; see [RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/).

**Key ablations:** adjacent pairs only versus all pairs; `epsilon` = 0%, 1%, 3%;
unweighted versus regret-weighted; independent pair votes versus consistent
aggregation.

### 2. Cheap rolling-origin surrogate curve

**Priority: P0. CPU-only. No future-label leakage and no TSFM calls.**

At prediction time, the observed history itself contains old forecast tasks.
Create several pseudo-origins inside the available context. At each origin and
candidate window, fit very cheap CPU forecasters (seasonal naive, ridge AR,
exponentially weighted linear regression, possibly Theta/ETS) and score the next
`h` already-observed values. Use the resulting *surrogate* error-vs-window curves
as features for the CSL selector:

- median winning window across pseudo-origins;
- frequency with which shortening beats full context;
- curve slope and adjacent differences;
- winner stability across origins;
- recent-origin-weighted winner;
- disagreement among cheap forecasting families.

The surrogate is not expected to match the TSFM exactly. It only needs to reveal
whether the current series is locally stationary, which scales repeat, and
whether old history hurts. Rolling-origin evaluation is a standard way to select
window length under structural instability; see [Inoue, Jin, and Rossi
(2017)](https://doi.org/10.1016/j.jeconom.2016.03.006).

Evaluate this both as a standalone selector and as extra features for idea 1.
This is likely more instance-specific than static summary features and cleaner
than using a previous TSFM oracle, which would not be available for a new series.

### 3. Change-point/run-length hybrid

**Priority: P0. CPU-only. No new TSFM inference.**

The synthetic thesis is that stale regimes can hurt, but no active selector in
the repo explicitly estimates the age of the current regime. Add multiscale
change features:

- age and confidence of the most recent level change;
- age and confidence of the most recent variance change;
- mean/variance/ACF distance between recent and older blocks;
- number of credible breaks in each candidate window;
- Bayesian posterior over current run length, if practical.

Use these in three progressively less restrictive tests:

1. direct baseline: snap the estimated run length to the grid;
2. feature augmentation for the pairwise selector;
3. candidate prior: penalize windows that cross a high-confidence recent break,
   but allow the learned selector to override it when older seasonal cycles help.

PELT is an exact multiple-change algorithm with linear cost under stated
conditions ([Killick, Fearnhead, and Eckley,
2012](https://arxiv.org/abs/1101.1438)); Bayesian online change-point detection
directly estimates the distribution of time since the last change ([Adams and
MacKay, 2007](https://arxiv.org/abs/0710.3742)). Try PELT first because it is
simple and fast. Detect on a downsampled series and refine locally near the last
break to keep 8192-point contexts cheap.

### 4. Ambiguity-aware, set-valued CSL labels

**Status: first 3%-tolerance Mamba screen completed. Better than pairwise and
curve on the official aggregate, but behind soft top-k classification; calibrate
the deployment rule before another training sweep.**

The exact argmin is unstable when many windows are essentially tied. Replace a
single target with an acceptable action set

```text
A_tau = {w: error(w) <= (1 + tau) * min_j error(w_j)}.
```

Train the model to place probability inside `A_tau`, then make the deployed
choice according to the real objective:

- **accuracy-first:** choose the largest credible member, which is safer against
  harmful shortening;
- **compute-aware:** choose the smallest member whose upper confidence bound is
  still within the tolerated regret;
- **balanced:** minimize predicted error plus `lambda * normalized_compute`.

Use `tau` in {1%, 3%, 5%}, and weight examples by their *decision sharpness*:
the regret between acceptable and clearly bad actions. This builds label
uncertainty into training instead of asking later gates to repair a noisy argmin.
The current soft top-k objective is rank-based and always assigns the same
weights; it does not account for whether first and second place differ by 0.01%
or 50%.

### 5. Synthetic-to-real feature matching and importance weighting

**Priority: P1, with a P0 CPU diagnostic. No new real TSFM labels needed.**

Before adding more synthetic mechanisms, measure where the synthetic and real
contexts differ. Compute the same scale-invariant feature vector for:

- the labeled synthetic pool;
- unlabeled GiftEvalPretrain or GiftEval training contexts, grouped by source;
- official test contexts only for a final, label-free shift audit.

Train a cross-validated domain classifier (`synthetic` versus `real`). If it can
separate them easily, inspect which features dominate. Estimate clipped density
ratios from the classifier and either:

1. importance-weight synthetic examples during CPU pairwise/tree training;
2. rejection/resample the existing labeled pool; or
3. retune generator parameter distributions toward under-covered real regions
   before spending GPU time on new labels.

Direct importance estimation is a classical response to covariate shift
([Sugiyama et al., 2007](https://proceedings.neurips.cc/paper/2007/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract.html)).
This is especially relevant here because feature-based forecast meta-learning
has found that reference-set representativeness matters and has used targeted
simulation to fill feature-space gaps ([FFORMPP](https://arxiv.org/abs/1908.11500)).

Do not trust unbounded weights. Clip them, report effective sample size, and use
source-disjoint validation. If the domain classifier remains highly accurate
after weighting, generator repair is more promising than another predictor
backbone.

### 6. Predictor-disagreement abstention

**Priority: P1. CPU/cached post-processing once predictions exist.**

The code already averages PatchTST and Mamba scores and evaluates top-k
checkpoints, but it does not appear to use *disagreement* as uncertainty. Build
an abstention rule from:

- entropy of predicted actions across top-k checkpoints;
- spread of predicted `log2(window)`;
- PatchTST-versus-Mamba action distance;
- fraction of members that predict harm relative to native;
- margin between the best two aggregate actions.

Calibrate one threshold on clean validation origins. Below confidence, choose
native or the bounded dataset consensus. Above it, use the instance policy.
Report a risk-coverage curve rather than one cherry-picked threshold. This is a
more direct use of the expensive checkpoint/architecture diversity already
being measured. If a formal guarantee becomes necessary, conformal risk control
provides a framework for controlling a monotone expected loss on a calibration
set ([Angelopoulos et al., 2022](https://arxiv.org/abs/2208.02814)); ordinary
validation-calibrated abstention is enough for the first screen.

### 7. ROCKET-style CPU representation baseline

**Priority: P1. CPU-only, moderate RAM.**

The handcrafted tree baseline may miss localized motifs, phase changes, and
multi-scale edges, while the neural predictor may be data-hungry. Transform the
raw series and first differences with MiniRocket or MultiRocket, then fit a
regularized linear adjacent-pair classifier or set-valued action model. Append
the current handcrafted features and horizon.

[MiniRocket](https://arxiv.org/abs/2012.08791) was designed as a very fast,
almost deterministic time-series transform, and
[MultiRocket](https://arxiv.org/abs/2102.00457) adds first differences and more
pooling operators. Subsample/dilate to cap memory on 8192-point inputs. This is a
useful CPU baseline even if it does not become the final method: it tests whether
the missing signal is representation quality or target/loss design.

### 8. One joint selector across TSFMs

**Priority: P2, when GPU is available. Labels can be reused.**

Current training is per labeling-model family and cross-model transfer applies
one source model to another after the fact. Instead, train one shared trunk on
all cached synthetic labels and condition the head on:

- TSFM identity embedding;
- family type (attention, SSM, xLSTM);
- context cap and patch/token size;
- whether native context is a distinct action;
- optional cheap model-level summary of the mean oracle distribution.

Use leave-one-TSFM-out validation as well as held-out-series validation. The
shared model gets many more supervised curve comparisons and can learn common
signals such as regime age while retaining model-specific responses. A
low-rank factorization is a good first version:

```text
predicted_error(series, window, model)
  = shared(series, window) + model_bias(window) + interaction(series, model).
```

Only pursue this after the CPU experiments identify a target/feature design
that beats the current one; otherwise it risks spending GPU time to scale the
same failure mode.

## Recommended order while GPUs are busy

1. **Freeze a leakage-safe evaluation manifest.** Use GiftEvalPretrain
   source-disjoint splits or GiftEval train/validation origins. Keep official
   test labels untouched except for final reporting. Deliberately contaminated
   scripts remain capacity checks only.
2. **Build one reusable CPU table** containing current handcrafted features,
   horizon, valid-window mask, all error curves, ambiguity sets, adjacent-pair
   labels/weights, item/source IDs, and chronological split IDs.
3. **Run the four cheapest selector experiments:** adjacent pairwise model,
   ambiguity-aware labels, PELT/run-length features, and their combination.
4. **Add rolling-origin surrogate features.** This is the best candidate for
   true per-instance adaptation without target leakage.
5. **Run the synthetic-versus-real domain audit.** Reweight the CPU selector
   before generating any new labels.
6. **Use checkpoint disagreement as an abstention signal** after the existing
   predictor curves are available.
7. Only then decide whether another neural training run is justified: pairwise
   Mamba, domain-weighted Mamba, or the joint multi-TSFM selector.

## Evaluation rules

Do not select a method using grid accuracy alone. A one-step miss on a flat
curve is harmless, whereas an exact-looking short-window choice can be very
harmful. Every experiment should report:

- leaderboard-faithful normalized `mase_gluonts_real`;
- selected MASE / native-context MASE;
- mean and p90 regret versus the per-instance oracle;
- fraction harmed by more than 1%, 3%, and 5% versus native;
- improvement rate and abstention/native-action rate;
- instance-weighted FLOPs and measured wall-clock savings where meaningful;
- results by dataset, horizon/term, TSFM, native series length, and recent-break
  age;
- at least three split seeds or a source/series bootstrap confidence interval.

For a CPU idea to earn a GPU follow-up, it should improve error versus native
and oracle regret on clean validation data **without increasing the 5%-harm
rate**, or give a clearly better risk/coverage/compute frontier than the current
worth gate. A result that only predicts the discrete oracle window more often,
without improving forecast loss, is not enough.

## Ideas not worth prioritizing yet

- **Another generic neural backbone:** PatchTST and Mamba already test the main
  quadratic-versus-linear backbone distinction. Target design and domain shift
  are more plausible bottlenecks.
- **A larger hyperparameter sweep:** expensive and unlikely to repair noisy
  argmins or synthetic-to-real mismatch.
- **More period-only rules:** these were explicitly retired, and CSL is not just
  a season-length problem.
- **A finer context grid before stabilizing the label:** it increases label cost
  and argmin variance. Pairwise/set-valued targets should be tested first.
- **More interpretability as a substitute for selection:** the mechanistic work
  is scientifically valuable, but it should become a feature or a validated
  selector proxy before receiving performance-oriented compute.

## Bottom line

The most promising departure is not a third backbone. It is to change the
learning problem from **"reconstruct every absolute curve value"** to
**"decide whether the next shortening step is worth it, with a cost proportional
to the harm of being wrong."** Combine that pairwise target with cheap
rolling-origin evidence, explicit recent-regime features, ambiguity-aware
labels, and abstention. All of those can be screened from cached data or on CPU
before asking for another GPU run.
