# Theoretical FLOPs audit for low-saving TSFMs

Date: 2026-08-10

## Corrections

- **Chronos2 / Chronos2-Synth:** corrected from a 6+6 layer, width-512
  encoder-decoder proxy to the shipped 12-block, width-768 encoder. The model
  processes context, a REG token, and future patches; each block contains time
  attention, group attention, and an FFN.
- **ChronosBolt-Base:** corrected patch size from 32 to 16 and dimensions from
  512/2048/6+6 to 768/3072/12+12. The decoder uses one query to emit a direct
  64-step block. Horizons beyond 64 recurse over nine quantile histories.
- **Sundial-Base-128M:** the Transformer processes context patches, not a
  concatenated context+horizon sequence. Added the dominant context-independent
  TimeFlow cost: 100 samples, 50 flow steps, three residual blocks, and a
  fixed 720-value output.
- **TiRex2:** replaced the quadratic-attention proxy with linear xLSTM token
  scaling over context plus the checkpoint's fixed 928-step future region.
- **Toto-2.0-313m:** config values were correct. Its 23 time-attention layers
  plus one singleton-variate layer make the existing univariate proxy close;
  its saving percentage is unchanged.

Five regression tests verify checkpoint shapes, monotonicity, TiRex linear
scaling, Sundial multi-patch behavior, and ChronosBolt recursive decoding.

## Corrected alpha=0 savings

| Model | Full/native MASE | Alpha=0 MASE | Old saving | Audited saving | Change |
|---|---:|---:|---:|---:|---:|
| Chronos2-Synth | 0.735228 | 0.738722 | 35.73% | 34.66% | -1.07 pp |
| ChronosBolt-Base | 0.809586 | 0.822113 | 34.01% | 27.64% | -6.38 pp |
| Sundial-Base-128M | 0.750750 | 0.760866 | 23.15% | 2.91% | -20.24 pp |
| TiRex2 | 0.696689 | 0.705405 | 23.95% | 19.00% | -4.95 pp |
| Toto-2.0-313m | 0.703999 | 0.707185 | 21.65% | 21.65% | 0.00 pp |

Sundial's apparent context savings were mostly an accounting artifact: its
100-sample diffusion/flow head dominates inference and is unaffected by context
shortening. TiRex also had overstated savings because recurrent cost is linear.

## Model-specific alpha diagnostics

| Model | Candidate alpha | MASE | Audited saving | Interpretation vs alpha=0 |
|---|---:|---:|---:|---|
| Chronos2-Synth | 0.40 | 0.738090 | 40.27% | better MASE and +5.60 pp saving, but LODO MASE worsens 0.41% |
| ChronosBolt-Base | 0.10 | 0.822640 | 28.46% | +0.82 pp saving for +0.064% MASE; LODO worsens 0.19% |
| Sundial-Base-128M | 0.40 | 0.760808 | 3.21% | better MASE and +0.29 pp; LODO MASE is essentially tied (+0.005%) |
| TiRex2 | 0.60 | 0.705840 | 21.58% | +2.58 pp for +0.062% MASE; unstable on BitbrainsFS-5T |
| Toto-2.0-313m | 0.10 | 0.707032 | 21.71% | small in-sample dominance; no LODO compute gain |

Only Sundial alpha=0.40 is stable enough in the leave-one-dataset-out diagnostic
to recommend as a follow-up without qualification, although its audited saving
remains only 3.21%. The other model-specific alphas are test-set diagnostic
operating points and require external calibration.

## Cost-aware near-optimal selection

The follow-up keeps alpha=0, forms a predicted near-optimal action set using a
fraction of each row's predicted score range, and chooses the action with the
lowest audited MAC count inside that set. Tolerance zero reproduces alpha=0
MASE exactly (apart from cheaper tie-breaking), which is a useful implementation
check.

| Model | Suggested policy | MASE | Audited saving | Change vs alpha=0 | LODO result |
|---|---:|---:|---:|---:|---:|
| Chronos2-Synth | score tolerance 0.005 | 0.741709 | 52.41% | +17.74 pp saving, +0.404% MASE | same policy on all 55 held-out datasets |
| ChronosBolt-Base | score tolerance 0.02 | 0.823212 | 33.96% | +6.33 pp saving, +0.134% MASE | +6.33 pp saving, +0.481% MASE |
| Sundial-Base-128M | alpha 0.40 | 0.760808 | 3.21% | +0.29 pp and slightly better MASE | +0.29 pp saving, +0.005% MASE |
| TiRex2 | score tolerance 0.0025 | 0.705823 | 21.41% | +2.41 pp saving, +0.059% MASE | +2.42 pp saving, +0.130% MASE |
| Toto-2.0-313m | score tolerance 0.01 | 0.707424 | 24.03% | +2.38 pp saving, +0.034% MASE | same policy on all 55 held-out datasets |

The Synth point is an intentionally more aggressive operating point; alpha=0.40
remains the conservative choice at 0.738090 MASE and 40.27% saving. Bolt's
in-sample tradeoff weakens under dataset holdout and should be externally
calibrated. Cost-aware selection does not solve Sundial's low-saving issue:
even its more aggressive tolerances save little because the fixed flow head
dominates total computation.

## Artifacts

- `experiments/compare_window_strategies_gifteval.py`
- `experiments/tests/test_theoretical_flops_audit.py`
- `experiments/analyze_cost_aware_lodo.py`
- `logs/experiments/master_recompute/serieswise_predictor/all_models_shrinkage/low_saving_flops_audit_comparison.csv`
- `logs/experiments/master_recompute/serieswise_predictor/all_models_shrinkage/low_saving_alpha_sweep_audited_flops.csv`
- `logs/experiments/master_recompute/serieswise_predictor/all_models_shrinkage/low_saving_alpha_lodo.csv`
- `logs/experiments/master_recompute/serieswise_predictor/all_models_shrinkage/low_saving_cost_aware_audited_flops.csv`
- `logs/experiments/master_recompute/serieswise_predictor/all_models_shrinkage/low_saving_cost_aware_audited_flops_cells.csv`
- `logs/experiments/master_recompute/serieswise_predictor/all_models_shrinkage/low_saving_cost_aware_lodo.csv`
