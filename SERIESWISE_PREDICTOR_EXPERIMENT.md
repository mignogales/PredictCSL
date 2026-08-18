# Series-wise CSL predictor experiment

Updated: 2026-08-10

## Question

The Chronos2-Small outcome Oracle reaches normalized GiftEval MASE 0.621624 at
73.55% theoretical FLOPs saving when it chooses a different context for every
forecast instance. This experiment asks how much of that hindsight advantage is
temporally stable and learnable by a series-level Mamba policy.

## 1. Official GiftEval Mamba shrinkage sweep

The existing Mamba score curve for every one of the 371,330 forecast instances
was mixed with its unlabeled dataset/term consensus:

```text
regularized_score = alpha * row_score + (1 - alpha) * cell_mean_score
```

All points use the official 97-cell normalized `mase_gluonts_real` aggregation
and workload-summed theoretical FLOPs.

| Row weight `alpha` | Normalized MASE | FLOPs saved |
|---:|---:|---:|
| 0.00 | 0.724498 | 65.97% |
| 0.05 | 0.724458 | **66.18%** |
| **0.10** | **0.724371** | 66.03% |
| 0.15 | 0.725035 | 65.68% |
| 0.25 (previous default) | 0.724793 | 64.47% |
| 0.40 | 0.724557 | 62.77% |
| 0.60 | 0.724507 | 61.08% |
| 1.00 (raw row) | 0.726465 | 57.81% |

The diagnostic best is `alpha=0.10`. Relative to the previous `alpha=0.25`, it
changes normalized MASE by -0.058% and saves another 1.55 percentage points of
full workload FLOPs. Relative to dataset-shared Mamba (0.724344 at 58.03%
saving), it gives essentially the same MASE (+0.0037% relative) with about 8.0
additional percentage points of saving.

This is not yet a new official result because the sweep inspected GiftEval test
outcomes. A paired 10,000-sample dataset-cluster bootstrap for `alpha=0.10`
versus `alpha=0.25` gives a 95% interval of [-0.248%, +0.073%] for relative
MASE change; the accuracy advantage is therefore not statistically secure. It
is best interpreted as a better compute operating point at practically tied
MASE.

A leave-one-dataset-display-out adaptive alpha selector chose 0.10 in 49 of 55
folds, 0.60 in five, and 0.00 in one. Its aggregated result was 0.726622 MASE at
65.70% saving, substantially worse than fixed 0.10. With this calibration set,
learning a separate shrinkage strength by dataset is too unstable.

## 2. Is the series Oracle temporally predictable?

Eight cells with repeated forecast origins were selected from the current
master caches: 526 examples from 71 series, with earlier origins used for
training and the last origin held out.

### Consecutive-origin stability

Across 455 consecutive-origin pairs:

| Diagnostic | Result |
|---|---:|
| Exact Oracle argmin repeated | 8.79% |
| Median curve-shape Pearson correlation | 0.313 |
| Median curve-rank Spearman correlation | 0.294 |
| Previous argmin within 1% at next origin | 12.53% |
| Previous argmin within 5% | 25.93% |
| Previous argmin within 10% | 35.82% |
| Median next-origin regret ratio of previous argmin | 1.211 |

The large per-instance Oracle gap is therefore dominated by unstable
origin-specific outcomes, not a persistent series identity signal.

### Simple temporal baselines

On the 71 held-out last origins:

| Predictor | Error / native | Improvement rate |
|---|---:|---:|
| Previous-origin Oracle window | 1.465 | 32.39% |
| Median Oracle window of earlier same-series origins | 1.135 | 40.85% |
| Median Oracle window of earlier same-cell origins | 1.145 | 36.62% |

None beats native context. The most recent outcome is especially unreliable.

## 3. Small series-aware Mamba test

A two-layer bidirectional Mamba regressor was trained for 1,200 updates on CUDA
1. This is a small temporal learnability test using MAE curves, not the official
97-cell MASE score.

| Protocol | Train/test size | Test grid accuracy | Test error / native | Test improvement rate |
|---|---:|---:|---:|---:|
| Earlier origins -> last origin of same series | 455 / 71 | 4.23% | 1.052 | 49.30% |
| Other series -> held-out series | 386 / 140 | 28.57% | 1.656 | 52.14% |

The model fits its training labels but fails on later origins and held-out
series. Increasing model capacity would scale the same target-instability
problem rather than solve it.

## 4. Stability-gated Mamba

The existing gate measures error-curve correlation using earlier origins only
and permits the series predictor to act only above a stability threshold.
This check ran on the older default cache because the server copy of the script
had a hard-coded cache root; the local script now accepts `--ablation-root`, so
the figures below are directional rather than directly comparable with the
official master run.

| Stability threshold | Coverage | Error / native | Harmed >5% |
|---:|---:|---:|---:|
| no gate | 100.0% | 1.0429 | 33.80% |
| 0.2 | 54.9% | 1.0210 | 16.90% |
| 0.4 | 33.8% | 1.0186 | 9.86% |
| **0.6** | **19.7%** | **0.9910** | **5.63%** |
| 0.8 | 5.6% | 0.9988 | 2.82% |

This is the only cleanly positive predictor result: a conservative stability
gate finds a small subset where acting improves mean error by about 0.9%.

## 5. Cross-model alpha=0 fallback rule

The consensus-only (`alpha=0`) Mamba selector was evaluated on all 11 TSFMs
under two row-level fallback rules. `feasible_backoff` chooses the best-ranked
available grid/native action. `native_fallback` first takes the unconstrained
consensus argmin and retains the row's native context whenever that preferred
action is unavailable.

Native fallback improves normalized MASE for 7/11 models, with the largest gain
on PatchTST-FM-R1 (-1.02% relative), but loses 1.1--4.1 percentage points of
FLOPs savings on every model. It hurts Chronos2-Small (+0.51%) and
Moirai2-Small (+0.77%). This is a model-dependent accuracy/compute tradeoff,
not a universally safer replacement for feasible backoff.

| TSFM | Full/native MASE | Full saving | Alpha=0 backoff MASE | Backoff saving | Alpha=0 native-fallback MASE | Native-fallback saving |
|---|---:|---:|---:|---:|---:|---:|
| Chronos2-Base | **0.704738** | 0.00% | 0.706731 | 48.31% | 0.707689 | 46.44% |
| Chronos2-Small | 0.726708 | 0.00% | **0.724498** | 65.24% | 0.728206 | 63.93% |
| Chronos2-Synth | **0.735228** | 0.00% | 0.738722 | 35.73% | 0.737089 | 34.37% |
| ChronosBolt-Base | **0.809586** | 0.00% | 0.822113 | 34.01% | 0.819605 | 29.87% |
| FlowState-R1 | **0.709001** | 0.00% | 0.713808 | 50.89% | 0.712068 | 48.87% |
| Moirai2-Small | 0.737778 | 0.00% | **0.731673** | 65.50% | 0.737343 | 62.40% |
| PatchTST-FM-R1 | **0.704527** | 0.00% | 0.718110 | 49.35% | 0.710779 | 47.92% |
| Sundial-Base-128M | **0.750750** | 0.00% | 0.760866 | 23.15% | 0.759634 | 19.46% |
| TiRex2 | **0.696689** | 0.00% | 0.705405 | 23.95% | 0.703324 | 21.15% |
| TimesFM2.5-200M | **0.705040** | 0.00% | 0.712975 | 49.64% | 0.710420 | 48.51% |
| Toto-2.0-313m | **0.703999** | 0.00% | 0.707185 | 21.65% | 0.707257 | 18.90% |

Against full/native rather than against the other alpha=0 rule, only the
Chronos2-Small backoff and both Moirai2-Small alpha=0 policies improve MASE.
The other models trade some accuracy for substantial compute reduction.

### Dataset consistency of alpha=0 feasible backoff

No dataset beats full/native in every one of the 11 TSFMs. BitbrainsFS-H is
closest (10/11 model cells, -0.37% aggregate MASE, 21.53% FLOPs saved), followed
by Hospital (8/11, -0.81%, 23.72%). Other aggregate-positive datasets with a
majority of model wins include ETTm1-W, USBirths-M, M4-Daily, and M4-Weekly.

| Canonical GiftEval family | Full/native normalized MASE | Alpha=0 backoff normalized MASE | Relative MASE change | Model-term win rate | FLOPs saved |
|---|---:|---:|---:|---:|---:|
| M4 | 0.803 | 0.801 | -0.19% | 43.9% | 27.1% |
| Electricity | 0.715 | 0.716 | +0.21% | 46.6% | 35.6% |
| ETT | 0.881 | 0.880 | -0.11% | 35.8% | 34.3% |
| Solar | 0.862 | 0.869 | +0.80% | 27.3% | 35.0% |
| Jena Weather | 0.718 | 0.736 | +2.52% | 26.0% | 40.5% |

## Decision

Do not train a larger unrestricted per-series Mamba against exact Oracle
windows. The Oracle is a useful heterogeneity/hindsight bound, but its argmin is
not persistent enough to serve as a hard supervised target.

The justified follow-up is narrower:

1. retain the dataset/term Mamba curve as the default policy;
2. use a small row residual (`alpha` near 0.05-0.10) only as a compute-oriented
   operating point, calibrated outside official GiftEval test;
3. permit a larger row residual only when earlier-origin curve stability exceeds
   a conservative threshold;
4. predict acceptable/regret-bounded action sets rather than exact argmins; and
5. rerun the stability gate on the current master cache with at least three
   seeds before promoting it.

## Artifacts

- `logs/experiments/master_recompute/serieswise_predictor/shrinkage_sweep/shrinkage_sweep_summary.csv`
- `logs/experiments/master_recompute/serieswise_predictor/shrinkage_sweep/alpha01_vs_alpha025_bootstrap.json`
- `logs/experiments/master_recompute/serieswise_predictor/shrinkage_sweep/lodo_report.json`
- `logs/experiments/master_recompute/serieswise_predictor/all_models_shrinkage/alpha0_native_vs_backoff.csv`
- `logs/experiments/master_recompute/serieswise_predictor/all_models_shrinkage/alpha0_native_vs_backoff_deltas.csv`
- `logs/experiments/master_recompute/serieswise_predictor/all_models_shrinkage/alpha0_backoff_dataset_consistency.csv`
- `logs/experiments/master_recompute/serieswise_predictor/all_models_shrinkage/alpha0_backoff_popular_datasets.csv`
- `logs/experiments/master_recompute/serieswise_predictor/series_aware_window_splits.json`
- `logs/experiments/master_recompute/serieswise_predictor/previous_origin_window_baselines.json`
- `logs/experiments/master_recompute/serieswise_predictor/consecutive_origin_curve_stability.json`
- `logs/experiments/master_recompute/serieswise_predictor/stability_gated_window_selector_legacy_cache.json`
