"""
Interpretability analysis of context saturation in TSFMs.

Tests whether observations beyond the sufficient context length have negligible
forecast-relevant causal influence (predictive redundancy), while the model
retains the ability to use distant observations when they carry additional
predictive information.

Six experiments share one config interface, one tabular results schema and one
figure pipeline:

  exp0  attention_masking      — the EXISTING attention-span restriction
                                 (experiments.context_attention_mask), re-scored
                                 into the common schema.
  exp1  perturbation           — temporal-block perturbation heatmaps (causal,
                                 input level; the principal evidence).
  exp2  activation_patching    — where in the network block information becomes
                                 causally (ir)relevant.
  exp3  forecast_lens          — layer at which the forecast stabilizes, per
                                 context length (frozen-head lens).
  exp4  synthetic_controls     — mandatory control: distant-dependency data
                                 where ignoring far context WOULD be a mistake.
  exp5  integrated_gradients   — corroborative gradient sensitivity (not causal
                                 evidence on its own).

Entry point (SERVER — nothing runs locally, see the repo CLAUDE.md):

    python -m context_interpretability.run_experiment --help

Everything model-specific lives behind ``adapters``; experiment code is
model-agnostic. Unsupported (model, experiment) pairs are skipped explicitly
and logged, never silently approximated.
"""

RESULTS_ROOT = "logs/experiments/context_interpretability"
