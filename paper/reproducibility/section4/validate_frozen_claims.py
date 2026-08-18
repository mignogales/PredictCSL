#!/usr/bin/env python3
"""Fail if a numerical claim in section4_mechanisms.tex drifts from the freeze."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA = Path(__file__).resolve().parent / "data" / "frozen"


def close(actual: float, expected: float, atol: float = 5e-4) -> None:
    if not np.isclose(actual, expected, atol=atol, rtol=0):
        raise AssertionError(f"{actual} != {expected} (atol={atol})")


def main() -> None:
    curve = pd.read_csv(DATA / "chronos2_small_exp1_curve.csv")
    close(curve.loc[curve.context_length.eq(32), "mean_mae"].iloc[0], 0.797, 5e-4)
    close(curve.mean_mae.min(), 0.312, 5e-4)
    assert int(curve.loc[curve.within_5pct_of_minimum, "context_length"].min()) == 1536
    assert int(curve.loc[curve.mean_mae.idxmin(), "context_length"]) == 3072

    ratios = pd.read_csv(DATA / "chronos2_small_exp1_region_ratios.csv")
    assert ratios.distant_recent_mean_abs_effect_ratio.round(3).tolist() == [
        0.146, 0.189, 0.222, 0.376]

    lag = pd.read_csv(DATA / "exp4_lag_tracking_summary.csv")
    assert len(lag) == 10
    close(lag.spearman_lag_vs_sufficient.min(), 0.730, 5e-4)
    close(lag.spearman_lag_vs_sufficient.max(), 0.908, 5e-4)
    assert int(lag.fraction_reaching_lag.eq(1.0).sum()) == 3
    assert int(lag.fraction_reaching_lag.eq(0.875).sum()) == 6
    assert int(lag.fraction_reaching_lag.eq(0.625).sum()) == 1
    assert int(lag.zero_min_context.min()) == 48
    assert int(lag.zero_max_context.max()) == 256
    assert int(lag.local_min_context.min()) == 32
    assert int(lag.local_max_context.max()) == 384
    assert int(lag.n_broken_controls.sum()) == 0

    long_lag = pd.read_csv(DATA / "exp4_long_lag_toto.csv")
    assert long_lag.model.unique().tolist() == ["Toto-2.0-313m"]
    assert sorted(long_lag.lag.unique().tolist()) == [128, 512, 1024, 2048]
    assert int(long_lag.n_instances.unique()[0]) == 48
    assert int(long_lag.config_broken.sum()) == 0
    strong = long_lag[long_lag.strength.eq(1.0)].sort_values("lag")
    assert strong.sufficient_context.tolist() == [384, 1024, 1536, 2560]
    assert bool((strong.sufficient_context >= strong.lag).all())
    moderate = long_lag[long_lag.strength.eq(0.5)].sort_values("lag")
    assert moderate.sufficient_context.tolist() == [192, 192, 128, 192]
    assert int((moderate.sufficient_context >= moderate.lag).sum()) == 1
    assert int(moderate.limitation_flag.sum()) == 3

    dec = pd.read_csv(DATA / "exp7_decomposition_summary.csv")
    mae = dec[dec.metric.eq("mae")]
    pivot = mae.pivot(index="model", columns="variant", values="mean_abs_loss_delta")
    full = pivot["attention_mask/full_history_stats"]
    tail = pivot["attention_mask/tail_matched_stats"]
    reduction = 100 * (1 - tail / full)
    residual = 100 * tail / full
    assert len(reduction) == 8
    close(reduction.min(), 48.2, 0.05)
    close(reduction.median(), 97.2, 0.05)
    close(residual["TimesFM2.5-200M"], 51.8, 0.05)
    close(residual["PatchTST-FM-R1"], 34.5, 0.05)
    close(residual["Sundial-Base-128M"], 17.3, 0.05)
    assert float(residual.drop([
        "TimesFM2.5-200M", "PatchTST-FM-R1", "Sundial-Base-128M"
    ]).max()) <= 3.4
    close(full["Chronos2-Small"], 0.03115, 5e-6)
    close(tail["Chronos2-Small"], 0.00105, 5e-6)

    print("All frozen Section 4 claims validated.")


if __name__ == "__main__":
    main()
