# PredictCSL paper package

This directory is a self-contained LaTeX project for Sections 3--5. Upload the
contents of `PredictCSL_Overleaf.zip` directly to a new Overleaf project;
`main.tex` is the root document.

## Layout

- `main.tex`: root document and shared preamble.
- `sections/`: editable manuscript text.
- `figures/`: every image referenced by the manuscript.
- `references.bib`: bibliography used by the current text.
- `reproducibility/`: frozen evidence and Python scripts used to validate or
  regenerate the section figures. These files are not required for LaTeX
  compilation and can be omitted from a smaller Overleaf upload.

## Compile

Overleaf should detect `main.tex` automatically. Locally, use:

```bash
latexmk -pdf main.tex
```

The document currently begins at Section 3 because Sections 1--2 were not
present in the repository. Replace the placeholder title and author in
`main.tex`, and add earlier sections above the existing `\input` lines when
they are ready.

## Reproducibility files

The reproducibility folders preserve each source bundle's frozen evidence.
The Python scripts came from the original section packages; some figure scripts
also depend on experiment exports outside this upload. The supplied publication
figures are already complete, so these scripts are not needed to edit or compile
the manuscript in Overleaf.
