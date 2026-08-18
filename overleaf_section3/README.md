# Overleaf package: Section 3

Upload `section3_useful_context.tex` and the required PDF figures in `figures/`
while preserving the `figures/` subdirectory. Then include the section with:

```tex
\input{section3_useful_context}
```

The manuscript preamble needs:

```tex
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{graphicx}
```

The draft assumes the GiftEval bibliography key is `auer2024gifteval`; rename
that one citation if the main `.bib` file uses a different key.

Both PDF (preferred for Overleaf) and PNG versions are provided. The figures
can be regenerated from the audited CSV/JSON exports with
`generate_figures.py` in the project environment.

## Shared figure style

Paper plots import typography, axis/legend treatment, and the canonical model
color--marker--line assignments from `../paper_plot_style.py`.  Add or revise a
model identity there rather than defining plot-local aesthetics; Figure 7 is
the visual reference for this shared style.

`fig6_multimodel_oracle_forest.tex` is a ready-to-paste figure block for the
11-model oracle comparison.

`fig7_multimodel_pareto_overlay.tex` is a ready-to-paste figure block for the
superposed 11-model supported Pareto frontiers.  The corresponding audited
frontier exports are stored under `data/oracle_pareto_frontier_multimodel/`.

`fig8_multimodel_pareto_overlay_absolute.tex` provides the corresponding
absolute-score version: GiftEval normalized MASE is not divided by each
model's native value.
