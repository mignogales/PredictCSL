# Context interpretability experiments

The unified runner exposes nine experiments (`exp0`–`exp8`). See
`configs/experiments.yaml` for the complete grid. The two most recent additions
are below.

## Multi-model long-lag follow-up

`configs/experiments_long_lag.yaml` is the isolated follow-up to the initial
Chronos2-Small suite. It inherits the standard configuration but writes to
`logs/experiments/context_interpretability_long_lag`, uses planted lags
128/512/1,024/2,048, and extends both the generator and context grid to 8,192
timesteps. The separate output root is deliberate: resumable cells from the
earlier 32--256 lag experiment must not be mistaken for fresh long-lag cells.
The default launcher uses Chronos2-Small as the reference and compares it with
Moirai2-Small, TimesFM2.5-200M, PatchTST-FM-R1, Sundial-Base-128M, and
Toto-2.0-313m. These models span distinct architectures while retaining Exp7
masking support.

For the paper's Exp4 comparison, use
`configs/experiments_long_lag_comparison.yaml`. It fixes the synthetic cohort
at 48 series per condition, matching the completed Toto recovery run. Once a
GPU is available, the following helper runs only the five missing comparators:

```bash
CUDA_VISIBLE_DEVICES=3 \
  bash scripts/run_interpretability_long_lag_exp4_missing.sh
```

Launch the full run set through the environment-routing master:

```bash
bash scripts/run_interpretability_multimodel.sh
```

Override the space-separated model list when a smaller pilot is appropriate:

```bash
INTERPRETABILITY_MODELS="Chronos2-Small PatchTST-FM-R1 Moirai2-Small" \
  bash scripts/run_interpretability_multimodel.sh
```

Exp1 and Exp4 apply to every forecaster. Exp7 is capability-gated: FlowState
and TiRex are retained in the default comparison for perturbation and planted
lags, but their masking decomposition is recorded as unsupported rather than
treated as a null result.

### Plot map for the causal tests

- `00_exp1_temporal_perturbation_test.png`: exact clean-versus-one-block
  intervention used by Exp1.
- `03d_perturbation_heatmaps_by_type_abs_dloss.png`: one raw triangular
  W-by-lookback matrix per intervention type; columns fix input length, rows fix
  the perturbed block, and color is mean absolute loss change. White cells are
  structurally unavailable or omitted by the configured geometric block
  thinning, not zero effects.
- `03e_perturbation_block_sweep_raw.png`: the earlier one-panel-per-input-length
  view on raw loss units. Each point is one perturbed block; no per-column
  normalization is applied.
- `03c_perturbation_profiles*.png`: the original normalized block profiles,
  retained because their shapes are easy to compare even when effect scales
  differ by input length.
- `00_exp4_long_dependency_test.png`: oracle, context-curve, and intervention
  logic for the planted long-lag test.
- `07_sufficient_context_vs_lag_all_models.png`: cross-model recovery of
  planted dependency reach.
- `03e_perturbation_recency_all_models.png`: cross-model block-mean sensitivity
  profile at each model's largest supported input.

## Exp6 — predictor contrast saliency

Exp6 explains the learned context-length predictor using the scalar target

\[
S(x)=\widehat E(L_{long}\mid x,h)-\widehat E(L_{short}\mid x,h).
\]

The predictor always receives one fixed canonical input (8,192 or 15,360
timesteps, as recorded in its checkpoint) and emits the entire context-error
curve. `short` and `long` select two coordinates of that one output; the method
does **not** compare gradients from differently-sized inputs.

Signed integrated gradients are aggregated into predictor patches:

- positive attribution pushes the predicted long-window error upward relative
  to the short window;
- negative attribution favours the long window;
- raw timestep maps, scalar contrasts and completeness errors are saved in each
  cell's `attr_<baseline>.npz`.

Supported predictors:

- `PatchTSTContextLength` (Transformer predictor);
- `MambaContextLength` (bidirectional Mamba predictor; its environment must
  contain `mamba-ssm` and `causal-conv1d`).

This support is independent of whether the labelled TSFM itself has a
differentiable forecast path. Consequently, any TSFM with a trained context
predictor checkpoint can use exp6. For v3/v4 checkpoints, set
`predictor_contrast_saliency.predictor_dir` to the appropriate model directory.

Example (an explicit experiment selection overrides its conservative
`enabled: false` default):

```bash
python -m context_interpretability.run_experiment \
  --models PatchTST-FM-R1 --experiments exp6 --source synthetic
```

## Exp7 — context restriction decomposition

For every visible suffix `L` inside a fixed full context `W`, exp7 compares:

1. `sliced`: physically feed only the last `L` values;
2. `attention_mask/full_history_stats`: feed the genuine `W` values and hide
   the old prefix as attention keys;
3. `attention_mask/tail_matched_stats`: keep width/positions/mask identical to
   (2), but replace the hidden prefix by a deterministic sequence with the
   visible tail's mean and standard deviation.

It reports MAE and MSE by default and plots:

- absolute slicing/masking error;
- masking error minus slicing error;
- the normalization proxy: full-history-stat mask minus tail-stat-matched mask.

The final residual, tail-stat mask minus slicing, contains position, physical
width, padding and any model-specific preprocessing differences. The
normalization component is exact for global affine mean/std normalization. It
is an operational proxy for robust/nonlinear scalers or architectures that can
read nominally hidden token states through paths other than attention.

Exp7 applies to families with working attention restriction: PatchTST-FM,
Chronos-2/Bolt, Sundial, TimeMoE, Moirai, TimesFM and Toto as declared in
`configs/models/capabilities.yaml`. Several hooks are marked “verify on server”
there. FlowState and TiRex have no attention mask and are explicitly skipped;
their analogous intervention is recurrent-state reset/truncation.

```bash
python -m context_interpretability.run_experiment \
  --models PatchTST-FM-R1 --experiments exp7 --source synthetic
```

By default exp7 uses only the largest supported `W` and every smaller visible
grid length. Set `all_full_contexts: true`, or provide `full_contexts` and
`visible_lengths`, for a larger or more focused sweep.

## Exp8 — direct TSFM loss-contrast saliency

Exp8 is the direct forecasting-model counterpart to exp6 and is the more
mechanistic test of TSFM context use:

\[
S(x,y)=\ell(f(x_{-L_{long}:}),y)-\ell(f(x_{-L_{short}:}),y).
\]

The long window is the common differentiable input. The short branch consumes
its suffix, so both gradients share one `L_long` coordinate system. Timesteps
older than `L_short` affect only the long branch; suffix attribution captures
different processing of the same recent observations under the two lengths.
MAE and MSE contrasts are emitted separately and use the same future target.

Currently this exact gradient experiment supports PatchTST-FM only, because it
is the only TSFM adapter with a differentiable forecasting wrapper. Other
families use sampling, autoregressive `generate()`, NumPy pipeline boundaries,
or internal `no_grad`. Exp1 remains the gradient-free causal sensitivity map
for those models; a paired block-contrast extension can provide an all-model
analogue if needed.

```bash
python -m context_interpretability.run_experiment \
  --models PatchTST-FM-R1 --experiments exp8 --source synthetic
```

Exp8 includes normalization, position and padding differences between the two
branches. Interpret it alongside exp7, which estimates those components.
