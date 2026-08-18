# Overleaf package: Section 4

Copy `section4_mechanisms.tex` into the Overleaf project and upload these four
figure files while preserving the `figures/visual_draft/` subdirectory:

- `figures/fig1_perturbation_recency_all_models.png`
- `figures/visual_draft/fig2_sufficient_context_vs_lag_with_toto_placeholder.pdf`
- `figures/visual_draft/fig3_slicing_vs_masking_all_models.pdf`
- `figures/visual_draft/fig4_masking_gap_heatmap_all_models.pdf`

Include the section with:

```tex
\input{section4_mechanisms}
```

The manuscript preamble needs:

```tex
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{graphicx}
```

The text assumes that Section 3 uses `\label{sec:real-context}`. The companion
edit in `overleaf_section3/section3_useful_context.tex` changes its final bridge
so that interpretability is Section 4 and the learned selector follows it.

## Frozen evidence

`data/frozen/` contains the compact, publication-facing evidence tables:

- exact planted-lag controls and the ten-model lag-tracking rollup;
- all eight completed Exp7 cells aggregated by model, metric, and condition;
- the complete Chronos2-Small Exp1 clean curve and its four region ratios;
- the completed nine-model strong long-lag follow-up plus Toto's weak series;
- `source_inventory.json` and `manifest.json` with SHA-256 hashes.

`data/frozen_input/` retains the small source JSON files and the eight raw Exp7
cells from which the frozen tables were derived; `data/frozen_input_long_lag/`
retains the Toto pilot sources. The package intentionally excludes the much
larger raw Exp1 cells; the exact Chronos2-Small values used in the text are
frozen separately, and the
cross-model Exp1 plot is preserved as the server-generated figure.

## Shared figure style

Figures 1--3 import typography, axis/legend treatment, and the canonical model
color--marker--line assignments from `../paper_plot_style.py`.  Figure 7 in
Section 3 is the visual reference.  Model aesthetics must be changed in that
shared module rather than inside an individual figure generator.

Run `generate_figures.py` in the project environment to regenerate the two
vector figures from `data/frozen/`. The downloaded Exp1 and worked-example
Exp7 PNGs are already publication-ready exports from the unified analysis.

## Evidential scope

- Exp1: 11 models; model-specific largest completed contexts.
- Exp4: 10 models with the complete 32--256 lag grid. The separate 128--2,048
  follow-up contains nine models at dependency strength 1 (48 series per
  condition), plus Toto at strength 0.5. Chronos-Bolt and TiRex lack completed
  long-lag controls in this cohort.
- Exp7: eight models. FlowState and TiRex lack the required attention-mask
  path in this suite; Toto's comparable cell was not part of the frozen main
  cohort.

Do not describe missing or capability-gated experiments as null results.
