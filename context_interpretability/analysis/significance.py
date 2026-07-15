"""
Significance layer (spec §4.4 + §12): paired tests per (block, context) with
Benjamini–Hochberg correction across the tested blocks/layers, bootstrap CIs
over instances, and practical effect sizes alongside p-values.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from context_interpretability.analysis.aggregate import BLOCK_KEYS, block_effects
from context_interpretability.metrics.statistics import (
    benjamini_hochberg, summarize_paired_effects)


def significance_table(df: pd.DataFrame, value: str = "loss_delta",
                       n_boot: int = 2000, alpha: float = 0.05,
                       seed: int = 0) -> pd.DataFrame:
    """One row per (model, dataset, method, context, block[, layer]) with the
    full §4.4 summary; BH correction applied within each
    (model, dataset, method, context_length) family of comparisons."""
    eff = block_effects(df, value=value)
    if eff.empty:
        return eff
    rows: List[dict] = []
    for i, rec in eff.iterrows():
        stats = summarize_paired_effects(rec["values"], n_boot=n_boot,
                                         alpha=alpha, seed=seed + int(i))
        row = {k: rec[k] for k in eff.columns if k not in ("values",)}
        row.update(stats)
        rows.append(row)
    out = pd.DataFrame(rows)

    fam_keys = [k for k in ("model", "dataset", "method", "context_length")
                if k in out.columns]
    out["p_adj"] = np.nan
    out["significant"] = False
    for _key, idx in out.groupby(fam_keys).groups.items():
        rej, adj = benjamini_hochberg(out.loc[idx, "p_wilcoxon"].to_numpy(),
                                      alpha=alpha)
        out.loc[idx, "p_adj"] = adj
        out.loc[idx, "significant"] = rej
    return out


def save_significance(df: pd.DataFrame, out_csv: str, value: str = "loss_delta",
                      **kwargs) -> Optional[pd.DataFrame]:
    tab = significance_table(df, value=value, **kwargs)
    if tab is None or tab.empty:
        return None
    tab.to_csv(out_csv, index=False)
    return tab
