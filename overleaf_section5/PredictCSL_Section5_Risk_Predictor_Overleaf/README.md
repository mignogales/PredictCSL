# Overleaf package: Section 5 risk-calibrated predictor

Upload `section5_zero_shot_predictor.tex` and the three PDF figures in
`figures/`, preserving that subdirectory. Include the section with:

```tex
\input{section5_zero_shot_predictor}
```

The manuscript preamble needs:

```tex
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{graphicx}
```

The text assumes the labels `sec:real-context`, `sec:context-mechanisms`, and
`eq:normalized-mase` from Sections 3--4.

## Figures

- `fig1_risk_selector_pipeline.pdf`: synthetic calibration and zero-shot
  deployment with native-context abstention.
- `fig2_compute_harm_dial.pdf`: the five aggregate compute--harm operating
  points.
- `fig3_model_profile_heatmaps.pdf`: profile behavior across 11 forecasters.

PNG twins are included for quick inspection. Run `generate_figures.py` to
regenerate both formats from `data/frozen/`.

## Frozen evidence

- `risk_profile_definitions.csv`: synthetic harm budgets and mean-ratio caps.
- `risk_profile_summary.csv`: aggregate real operating points across 11 TSFMs.
- `risk_profiles_all_models.csv`: model-by-profile results used in Figure 3 and
  Table 2.
- `chronos2_base_dataset_profiles.csv`: recognizable-dataset results for the
  Efficiency and Max-efficiency profiles.

The internal experiment keys remain `extreme` and `very_extreme`; the paper
uses the presentation names **Efficiency** and **Max efficiency**.

## Evidential scope and accounting

- Every predictor is trained on synthetic labels produced by its own frozen
  TSFM. No GiftEval values enter training or checkpoint selection.
- Each model has its own frozen ExtraTrees risk predictor and synthetic
  calibration thresholds. The real benchmark is used only for final evaluation.
- Reported FLOPs and measured time are for the TSFM forecast stage. They exclude
  predictor inference; the separate overhead table prevents those values from
  being presented as end-to-end savings.
- TiRex internally pads to a fixed context, so its theoretical context FLOPs
  reduction is not a realized hardware-saving claim.
- Synthetic harm budgets are control targets, not real-domain guarantees. The
  section reports observed tail risk and discusses calibration failures.
