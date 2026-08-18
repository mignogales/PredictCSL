#!/usr/bin/env python3
"""Validate the numerical claims made by Section 5's frozen evidence."""

from __future__ import annotations

import csv
from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "frozen"


def close(actual: float, expected: float, tol: float = 5e-7) -> None:
    if not np.isclose(actual, expected, atol=tol, rtol=0):
        raise AssertionError(f"Expected {expected}, found {actual}")


def validate_latex_structure() -> None:
    """Catch local packaging errors without requiring a TeX installation."""
    tex = (ROOT / "section5_zero_shot_predictor.tex").read_text()
    uncommented = re.sub(r"(?<!\\)%.*$", "", tex, flags=re.MULTILINE)

    depth = 0
    for match in re.finditer(r"(?<!\\)[{}]", uncommented):
        depth += 1 if match.group() == "{" else -1
        assert depth >= 0, f"Unmatched closing brace at byte {match.start()}"
    assert depth == 0, f"Unbalanced LaTeX braces: final depth {depth}"

    stack: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", uncommented):
        kind, environment = match.groups()
        if kind == "begin":
            stack.append(environment)
        else:
            assert stack, f"Closing unopened environment: {environment}"
            opened = stack.pop()
            assert opened == environment, (
                f"Environment mismatch: began {opened}, ended {environment}"
            )
    assert not stack, f"Unclosed LaTeX environments: {stack}"

    labels = re.findall(r"\\label\{([^}]+)\}", uncommented)
    assert len(labels) == len(set(labels)), "Duplicate LaTeX label"
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", uncommented))
    external_refs = {
        "eq:normalized-mase", "sec:context-mechanisms", "sec:real-context"
    }
    assert refs <= set(labels) | external_refs, f"Unknown references: {refs - set(labels)}"

    figure_paths = re.findall(
        r"\\includegraphics(?:\[[^]]*\])?\s*\{([^}]+)\}", uncommented
    )
    assert len(figure_paths) == 3
    assert all((ROOT / path).is_file() for path in figure_paths)


def main() -> None:
    validate_latex_structure()
    multi = pd.read_csv(DATA / "zero_shot_multimodel.csv")
    assert len(multi) == 22
    assert set(multi.variant) == {"PatchTST", "Mamba"}
    assert (multi.groupby("variant").model.nunique() == 11).all()
    assert (multi.n_cells == 97).all()
    assert (multi.n_instances == 371330).all()

    mamba = multi[multi.variant == "Mamba"].set_index("model")
    patch = multi[multi.variant == "PatchTST"].set_index("model")
    assert set(mamba[mamba.relative_nmase_gain_pct > 0].index) == {
        "Chronos2-Small", "Moirai2-Small"
    }
    assert set(patch[patch.relative_nmase_gain_pct > 0].index) == {
        "Moirai2-Small", "Sundial-Base-128M"
    }
    close(mamba.loc["Chronos2-Small", "pred_nmase"], 0.724343722)
    close(mamba.loc["Chronos2-Small", "cell_flops_saved_pct"], 56.492235)
    close(mamba.loc["Moirai2-Small", "relative_nmase_gain_pct"], 0.788962)
    close(mamba.relative_nmase_gain_pct.min(), -1.906213)

    obj = pd.read_csv(DATA / "chronos2_small_objectives.csv").set_index("policy")
    assert len(obj) == 5
    assert obj.policy_nmase.idxmin() == "Soft top-k classification"
    assert obj.cell_flops_saved_pct.idxmax() == "Risk-aware"
    close(obj.loc["Mamba curve", "tsfm_wall_time_saved_pct"], 52.616982)
    close(obj.loc["Soft top-k classification", "policy_nmase"], 0.722412185)
    assert obj.beats_full_eligible.max() == 58
    assert (obj.n_eligible == 95).all()

    overhead = pd.read_csv(DATA / "mamba_predictor_overhead.csv")
    assert len(overhead) == 11
    close(overhead.pct_of_full_h48.min(), 0.091791)
    close(overhead.pct_of_full_h48.max(), 2.355090)
    close(overhead.pct_of_full_h48.mean(), 0.566393, tol=2e-6)

    topk = pd.read_csv(DATA / "chronos2_small_topk.csv")
    assert len(topk) == 10
    mt = topk[topk.variant == "Mamba"]
    pt = topk[topk.variant == "PatchTST"]
    close(mt.real_nmase.mean(), 0.726532704)
    # Report sample standard deviations across the five retained checkpoints.
    close(mt.real_nmase.std(ddof=1), 0.003880551)
    close(pt.real_nmase.std(ddof=1), 0.001611632)
    assert int((mt.real_nmase < 0.726707598).sum()) == 3
    assert int((pt.real_nmase < 0.726707598).sum()) == 4

    with (DATA / "source_inventory.csv").open(newline="") as handle:
        sources = list(csv.DictReader(handle))
    assert len(sources) == 30
    assert all(len(row["sha256"]) == 64 for row in sources)
    assert len({row["source"] for row in sources}) == len(sources)

    print("All frozen Section 5 claims validated.")


if __name__ == "__main__":
    main()
