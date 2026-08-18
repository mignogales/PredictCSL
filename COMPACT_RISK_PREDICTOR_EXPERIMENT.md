# Compact and interpretable calibrated-risk predictors

Date: 2026-08-13

## Question

Can the 192-tree ExtraTrees calibrated-risk selector be replaced by a much
smaller or more interpretable model without materially changing its zero-shot
GIFT-Eval behavior?

## Controlled design

The Chronos2-Base ExtraTrees policy was frozen as the teacher. All students use
the same 94 engineered features, 12,000 synthetic training series, 2,000
disjoint synthetic calibration series, 600,000 sampled
series--window--horizon pairs, five synthetic harm profiles, and 97-cell
GIFT-Eval evaluation cohort as the original policy.

No GIFT-Eval outcomes were used to choose the students. Candidate selection
used only held-out synthetic calibration results. Two target designs were
screened:

1. Direct training from clipped synthetic log-risk plus a smooth approximation
   to the more-than-5% harm event.
2. Distillation of the frozen ExtraTrees composite score,
   `predicted_log_risk + 0.5 * predicted_harm_probability`.

The screen covered Ridge models, single trees of depth 3/4/6/8/10, single MLPs
with 32 and 64x32 hidden units, and a three-member 64x32 MLP ensemble. The
synthetically selected interpretable and compact students were the distilled
depth-8 tree and distilled single 64x32 MLP. The 5 KB distilled Ridge model was
also evaluated as a fully linear glass-box endpoint.

## Main real-data results

Positive MASE change means worse than native context. Harm is the fraction of
instances whose MASE is more than 5% worse than native.

| Model | Artifact | Model-only latency, one 18-window series | Balanced FLOPs saved | Balanced harm | Balanced MASE change | Max-eff. FLOPs saved | Max-eff. harm | Max-eff. MASE change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ExtraTrees teacher | 1.268 GB | 35.993 ms | 18.05% | 0.96% | -0.040% | 77.43% | 12.45% | +1.368% |
| Distilled depth-8 tree | 22.2 KB | 0.126 ms | 11.49% | 0.79% | +0.011% | 72.70% | 9.19% | +0.886% |
| Distilled 64x32 MLP | 66.3 KB | 0.241 ms | 25.46% | 3.10% | +0.081% | 75.98% | 20.55% | +2.231% |
| Distilled Ridge | 4.9 KB | 0.252 ms | 15.91% | 1.62% | +0.099% | 66.81% | 15.04% | +1.196% |

The depth-8 tree is the only compact student that retains a credible low-risk
frontier. At the balanced profile it gives up 6.55 percentage points of FLOPs
saving relative to ExtraTrees, while slightly reducing the observed harm rate
and remaining essentially tied with native MASE. At Max efficiency it saves
72.70% of FLOPs with lower harm and lower aggregate MASE degradation than the
teacher, although it also saves 4.73 percentage points less compute.

The small MLP more closely matches the teacher score on synthetic validation
(correlation 0.973 versus 0.941 for the tree), but transfers less safely to real
data. This repeats the earlier result from the independently trained five-member
MLP ensemble: synthetic fit is not the limiting factor; synthetic-to-real risk
calibration is.

## Size and latency

On the same server and identical feature batches, model-only inference for one
series and its 18 candidate windows was 286x faster for the depth-8 tree and
149x faster for the tiny MLP than for ExtraTrees. Batched over 256 series, the
tree used 1.16 microseconds per series versus 1,191 microseconds for ExtraTrees.

Feature extraction takes approximately 1.4--1.5 ms per series and therefore
dominates end-to-end latency for every compact student. Further latency work
should simplify or vectorize feature extraction rather than reduce the student
model again.

## Interpretability

The depth-8 tree has 251 leaves and uses 49 of 94 features. It is locally
interpretable because every decision follows at most eight threshold tests,
but it is better described as auditable than globally simple. Its dominant
features are:

1. horizon/window ratio;
2. candidate-window fraction;
3. spectral entropy;
4. log candidate-window fraction;
5. spectral peak period;
6. 2,048-scale mean absolute differences;
7. 128-scale mean absolute differences;
8. effective-window/history fraction.

The Ridge endpoint is globally transparent, but its real risk/accuracy frontier
is inferior to the depth-8 tree. Its largest standardized coefficients also
emphasize horizon/window geometry, spectral entropy, 512-scale differences and
dispersion, spectral peak share, and effective-window fraction.

## Direct training versus distillation

Distillation was consistently stronger under the synthetic-only selection
criterion. For the 64x32 MLP, balanced synthetic context saving was 14.12% when
distilled and 4.18% when trained directly. The three-member directly trained MLP
ensemble improved that to 8.24%, but remained behind the single distilled MLP
while using about three times the memory. Ensembling reduced initialization
variance but did not remove noisy-target or domain-shift error.

## Dense intermediate-policy sweep across models

The same model-specific comparison was repeated on ten TSFMs with stable real
forecast caches. For each TSFM, a separate depth-8 tree was distilled from its
frozen ExtraTrees score. Each model family then received 21 thresholds spanning
score quantiles 0--100%, calibrated only on that model's same 2,000 held-out
synthetic series. GIFT-Eval was used only once for evaluation, over the same 97
cells per policy. Thus the intermediate thresholds were not selected using real
outcomes.

The tables report the maximum theoretical FLOPs saving attainable while the
observed real more-than-5% harm rate remains below each stated budget. Values
are `F / H / M`: FLOPs saved %, observed harm %, and aggregate MASE change %
relative to native context. Positive MASE change is worse. ExtraTrees and the
depth-8 tree are shown in adjacent columns.

| TSFM | ET, harm <=5% | T8, harm <=5% | ET, harm <=10% | T8, harm <=10% |
|---|---:|---:|---:|---:|
| Chronos2-Base | 51.67 / 4.57 / +0.05 | 20.66 / 1.60 / +0.02 | 68.89 / 8.47 / +0.45 | 74.54 / 9.81 / +0.99 |
| Chronos2-Small | 28.95 / 3.66 / -0.07 | 27.46 / 4.11 / -0.15 | 65.38 / 9.20 / +0.56 | 67.00 / 9.17 / +0.88 |
| Chronos2-Synth | 38.33 / 3.92 / +0.08 | 0.00 / 0.00 / 0.00 | 61.23 / 8.95 / +0.01 | 55.39 / 9.55 / +1.23 |
| ChronosBolt-Base | 14.29 / 4.81 / -0.03 | 16.17 / 4.62 / 0.00 | 20.71 / 7.89 / +0.07 | 29.56 / 9.25 / +0.36 |
| FlowState-R1 | 13.07 / 4.76 / +0.30 | 0.00 / 0.00 / 0.00 | 41.30 / 9.56 / +0.67 | 40.62 / 8.56 / +0.66 |
| Moirai2-Small | 10.24 / 3.93 / +0.02 | 0.00 / 0.00 / 0.00 | 39.15 / 9.19 / +0.11 | 39.58 / 9.16 / +0.37 |
| PatchTST-FM-R1 | 30.49 / 4.74 / +0.29 | 0.00 / 0.00 / 0.00 | 57.96 / 9.19 / +0.99 | 64.12 / 9.35 / +1.71 |
| Sundial-Base-128M | 1.34 / 4.76 / +0.17 | 0.00 / 0.00 / 0.00 | 2.78 / 8.97 / +0.28 | 0.00 / 0.00 / 0.00 |
| TimesFM2.5-200M | 37.69 / 3.96 / +0.22 | 35.85 / 4.31 / +0.29 | 66.90 / 9.89 / +0.74 | 67.45 / 8.88 / +1.05 |
| Toto-2.0-313m | 0.00 / 0.00 / 0.00 | 20.50 / 4.78 / +0.15 | 39.07 / 8.62 / +0.75 | 39.16 / 8.62 / +1.14 |

| TSFM | ET, harm <=15% | T8, harm <=15% | ET, harm <=20% | T8, harm <=20% |
|---|---:|---:|---:|---:|
| Chronos2-Base | 79.53 / 13.98 / +1.64 | 76.90 / 10.94 / +1.54 | 83.23 / 18.70 / +2.23 | 83.91 / 17.44 / +3.26 |
| Chronos2-Small | 79.15 / 14.18 / +2.05 | 80.99 / 13.23 / +2.40 | 85.21 / 19.80 / +3.14 | 84.88 / 17.85 / +3.12 |
| Chronos2-Synth | 68.60 / 11.96 / +0.42 | 72.46 / 14.16 / +1.70 | 78.32 / 18.49 / +2.53 | 80.20 / 18.66 / +2.99 |
| ChronosBolt-Base | 40.44 / 14.93 / +1.76 | 29.56 / 9.25 / +0.36 | 49.65 / 18.44 / +3.17 | 38.29 / 16.24 / +1.30 |
| FlowState-R1 | 60.32 / 13.71 / +2.67 | 67.11 / 14.31 / +2.77 | 75.73 / 19.36 / +4.90 | 75.82 / 19.11 / +4.23 |
| Moirai2-Small | 60.42 / 13.28 / +0.73 | 62.14 / 14.64 / +1.37 | 83.09 / 19.54 / +2.74 | 85.06 / 19.51 / +2.72 |
| PatchTST-FM-R1 | 71.23 / 13.20 / +2.40 | 74.10 / 12.60 / +2.57 | 80.71 / 19.29 / +4.42 | 78.81 / 16.97 / +4.72 |
| Sundial-Base-128M | 4.30 / 14.20 / +0.30 | 4.84 / 14.73 / +0.32 | 5.48 / 17.69 / +0.55 | 5.67 / 17.36 / +0.48 |
| TimesFM2.5-200M | 73.74 / 12.94 / +1.27 | 77.29 / 13.22 / +1.92 | 83.01 / 19.85 / +2.62 | 80.95 / 15.30 / +3.16 |
| Toto-2.0-313m | 59.51 / 14.62 / +2.51 | 61.24 / 14.06 / +3.30 | 64.95 / 17.97 / +3.53 | 61.24 / 14.06 / +3.30 |

Intermediate policies do create crossings, but not equal frontiers. At the 5%
harm budget, T8 saves 10.54 percentage points fewer FLOPs on average and beats
ET on compute in only 2/10 models. At 10% and 15%, T8 saves 1.41 and 0.94 points
more on average and wins the compute comparison in 7/10 and 8/10 models.
However, its aggregate MASE change is respectively 0.38 and 0.25 points worse
on average, and none of those matched-budget T8 policies dominates ET jointly
in compute, exact harm, and MASE.

At the 20% budget the result is mixed: T8 saves 1.46 points fewer FLOPs on
average and wins compute in 5/10 models, but it genuinely dominates the matched
ET endpoint in all three metrics for FlowState-R1, Moirai2-Small, and
Sundial-Base-128M. Across all 21 thresholds, at least one T8 point dominates an
ET point in 7/10 models, while ET dominates at least one T8 point in 6/10. The
fronts therefore cross rather than coincide. ExtraTrees retains the clearest
advantage in the conservative region; the tree becomes competitive mainly in
the medium-to-aggressive region.

TiRex2 is not included yet because its dense window-cache producer was still
actively writing forecasts during this analysis. Evaluating it concurrently
would allow the two policies to observe different cache snapshots.

## Recommendation

Keep ExtraTrees when the strongest conservative risk/compute frontier matters.
Use the distilled depth-8 tree when memory, latency, and per-decision auditability
matter more than recovering every percentage point of compute saving. Do not
replace the teacher with the current MLP or Ridge students.

Remote artifacts:

- `logs/experiments/master_recompute/calibrated_context_risk_extreme/chronos2_base`
- `logs/experiments/master_recompute/compact_context_risk/chronos2_base`
- `logs/experiments/master_recompute/compact_context_risk_all_models`

Implementation staging files:

- `.codex_remote_staging/experiments/distill_calibrated_context_risk.py`
- `.codex_remote_staging/experiments/benchmark_compact_context_risk.py`
- `.codex_remote_staging/experiments/evaluate_context_risk_profile_override.py`
- `.codex_remote_staging/experiments/summarize_compact_dense_frontier.py`
