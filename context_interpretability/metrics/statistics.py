"""
Statistical machinery (spec §12): bootstrap CIs over forecasting instances,
paired tests for clean-vs-intervened effects, Benjamini–Hochberg correction and
practical effect sizes.

Aggregation contract: callers aggregate repeated perturbation seeds WITHIN a
sample first (they are not independent), then hand this module one paired
effect per instance.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as _scipy_stats
except Exception:  # noqa: BLE001 — scipy is expected; degrade explicitly
    _scipy_stats = None


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0, stat=np.nanmean) -> Tuple[float, float]:
    """Percentile bootstrap CI of ``stat`` over instances."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot = stat(v[idx], axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def wilcoxon_signed_rank(deltas: np.ndarray) -> float:
    """p-value that paired deltas are symmetric about zero (two-sided)."""
    d = np.asarray(deltas, dtype=np.float64)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if d.size < 5:
        return np.nan
    if _scipy_stats is None:
        raise ImportError("scipy is required for wilcoxon_signed_rank")
    try:
        return float(_scipy_stats.wilcoxon(d).pvalue)
    except ValueError:
        return np.nan


def paired_permutation_test(deltas: np.ndarray, n_perm: int = 5000,
                            seed: int = 0) -> float:
    """Sign-flip permutation test of mean(delta) == 0 (two-sided)."""
    d = np.asarray(deltas, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = abs(np.mean(d))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    perm = np.abs((signs * d).mean(axis=1))
    return float((np.sum(perm >= obs) + 1) / (n_perm + 1))


def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rejected mask, adjusted p-values); NaNs pass through unrejected."""
    p = np.asarray(pvals, dtype=np.float64)
    adj = np.full_like(p, np.nan)
    rej = np.zeros(p.shape, dtype=bool)
    ok = np.isfinite(p)
    if not ok.any():
        return rej, adj
    ps = p[ok]
    m = ps.size
    order = np.argsort(ps)
    ranked = ps[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]   # monotone adjustment
    adj_ok = np.empty(m)
    adj_ok[order] = np.minimum(ranked, 1.0)
    adj[ok] = adj_ok
    rej[ok] = adj_ok <= alpha
    return rej, adj


def cohens_d_paired(deltas: np.ndarray) -> float:
    """Paired Cohen's d: mean(delta) / std(delta)."""
    d = np.asarray(deltas, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 2 or np.std(d, ddof=1) == 0:
        return np.nan
    return float(np.mean(d) / np.std(d, ddof=1))


def rank_biserial(deltas: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (Wilcoxon effect size)."""
    d = np.asarray(deltas, dtype=np.float64)
    d = d[np.isfinite(d) & (d != 0)]
    if d.size == 0:
        return np.nan
    ranks = _rankdata(np.abs(d))
    total = ranks.sum()
    pos = ranks[d > 0].sum()
    return float(2.0 * pos / total - 1.0)


def _rankdata(a: np.ndarray) -> np.ndarray:
    if _scipy_stats is not None:
        return _scipy_stats.rankdata(a)
    order = np.argsort(a)
    ranks = np.empty(a.size)
    ranks[order] = np.arange(1, a.size + 1)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    if _scipy_stats is not None:
        return float(_scipy_stats.spearmanr(a[ok], b[ok]).statistic)
    ra, rb = _rankdata(a[ok]), _rankdata(b[ok])
    return float(np.corrcoef(ra, rb)[0, 1])


def summarize_paired_effects(deltas: np.ndarray, n_boot: int = 2000,
                             alpha: float = 0.05, seed: int = 0
                             ) -> Dict[str, float]:
    """Spec §4.4 per-(block, context) statistical summary of paired deltas."""
    d = np.asarray(deltas, dtype=np.float64)
    d = d[np.isfinite(d)]
    lo, hi = bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)
    return {
        "n": int(d.size),
        "mean": float(np.mean(d)) if d.size else np.nan,
        "median": float(np.median(d)) if d.size else np.nan,
        "std": float(np.std(d, ddof=1)) if d.size > 1 else np.nan,
        "ci_low": lo,
        "ci_high": hi,
        "prop_positive": float(np.mean(d > 0)) if d.size else np.nan,
        "p_wilcoxon": wilcoxon_signed_rank(d),
        "p_permutation": paired_permutation_test(d, seed=seed),
        "cohens_d": cohens_d_paired(d),
        "rank_biserial": rank_biserial(d),
    }
