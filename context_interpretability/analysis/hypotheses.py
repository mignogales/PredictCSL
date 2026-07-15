"""
Hypothesis evaluation (spec §15) — the code TESTS H1–H5, it never assumes them.

Each hypothesis gets a verdict in {"supporting", "contradicting", "mixed",
"insufficient_data"} plus the quantitative evidence behind it, written to
``hypotheses_report.json`` and a human-readable ``hypotheses_report.md``.
Decision thresholds are explicit module constants, echoed into the report.

Also embeds the integrated-gradients limitations statement (spec §8.7) so no
report ships without it.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from context_interpretability.analysis import aggregate as agg
from context_interpretability.analysis.significance import significance_table
from context_interpretability.experiments.integrated_gradients import LIMITATIONS
from context_interpretability.metrics.statistics import spearman
from context_interpretability.schema import load_results

# explicit, reported decision thresholds
DISTANT_TO_RECENT_RATIO = 0.2     # H1: distant effect < 20% of recent effect
CONCENTRATION_FRACTION = 0.8      # H2: >= 80% of effect mass inside suff ctx
H4_MIN_SPEARMAN = 0.6             # H4: rank corr(suff ctx, true lag d)
H5_MIN_MEDIAN_SPEARMAN = 0.5      # H5: cross-method agreement


def _sufficient_contexts(run_dir: str, tolerance: float = 0.05
                         ) -> Dict[str, int]:
    """dataset tag -> sufficient context, from every clean cache in the run."""
    out: Dict[str, int] = {}
    for dirpath, dirs, _files in os.walk(run_dir):
        for d in dirs:
            if not d.startswith("clean_cache"):
                continue
            cache = os.path.join(dirpath, d)
            Ws, means = [], []
            for f in sorted(os.listdir(cache)):
                if f.startswith("clean_w") and f.endswith(".npz"):
                    z = np.load(os.path.join(cache, f))
                    Ws.append(int(f[len("clean_w"):-len(".npz")]))
                    means.append(float(np.nanmean(z["loss"])))
            if len(Ws) >= 3:
                order = np.argsort(Ws)
                curve = np.array(means)[order]
                wins = [Ws[i] for i in order]
                tag = os.path.relpath(dirpath, run_dir)
                out[tag] = agg.sufficient_context_from_curve(
                    wins, curve, tolerance)
    return out


def _match_suff(suff: Dict[str, int], dataset: str) -> Optional[int]:
    for tag, v in suff.items():
        if dataset in tag:
            return v
    return None


def _split_effect(df: pd.DataFrame, value: str, suff: Dict[str, int]
                  ) -> Optional[Dict[str, float]]:
    """Mean |effect| for blocks inside vs beyond the sufficient context."""
    d = agg.collapse_seeds(df)
    if d.empty:
        return None
    recent, distant = [], []
    for ds, g in d.groupby("dataset"):
        s = _match_suff(suff, str(ds))
        if s is None:
            continue
        v = g[np.isfinite(g[value])]
        recent.append(v.loc[v["lookback_start"] < s, value].abs().mean())
        distant.append(v.loc[v["lookback_start"] >= s, value].abs().mean())
    r = float(np.nanmean(recent)) if recent else np.nan
    dd = float(np.nanmean(distant)) if distant else np.nan
    if not np.isfinite(r) or r == 0:
        return None
    return {"recent_effect": r, "distant_effect": dd,
            "ratio": dd / r if np.isfinite(dd) else np.nan}


def _verdict_from_ratio(ratio: float, threshold: float) -> str:
    if not np.isfinite(ratio):
        return "insufficient_data"
    if ratio <= threshold:
        return "supporting"
    if ratio <= 2 * threshold:
        return "mixed"
    return "contradicting"


def evaluate_h1(df: pd.DataFrame, suff: Dict[str, int]) -> dict:
    """H1: perturbing beyond the sufficient context degrades ~nothing."""
    pert = df[df["method"] == "perturbation"]
    ev = _split_effect(pert, "loss_delta", suff)
    if ev is None:
        return {"verdict": "insufficient_data"}
    # significance backstop: distant blocks individually significant?
    sig = significance_table(pert, value="loss_delta", n_boot=500)
    n_sig_distant = 0
    if not sig.empty:
        for _i, row in sig.iterrows():
            s = _match_suff(suff, str(row["dataset"]))
            if s is not None and row["lookback_start"] >= s \
                    and bool(row["significant"]) and row["mean"] > 0:
                n_sig_distant += 1
    ev.update({"n_significant_distant_blocks": n_sig_distant,
               "threshold": DISTANT_TO_RECENT_RATIO,
               "verdict": _verdict_from_ratio(ev["ratio"],
                                              DISTANT_TO_RECENT_RATIO)})
    return ev


def evaluate_h2(df: pd.DataFrame, suff: Dict[str, int]) -> dict:
    """H2: attribution + intervention effect mass concentrates inside the
    sufficient context."""
    fracs = {}
    for method, value in [("perturbation", "loss_delta"),
                          ("integrated_gradients", "attribution_score"),
                          ("attention_masking", "loss_delta")]:
        d = agg.collapse_seeds(df[df["method"] == method])
        if d.empty:
            continue
        per_ds = []
        for ds, g in d.groupby("dataset"):
            s = _match_suff(suff, str(ds))
            if s is None:
                continue
            mass = g[value].abs()
            total = mass.sum()
            if total > 0:
                per_ds.append(
                    mass[g["lookback_start"] < s].sum() / total)
        if per_ds:
            fracs[method] = float(np.nanmean(per_ds))
    if not fracs:
        return {"verdict": "insufficient_data"}
    mean_frac = float(np.nanmean(list(fracs.values())))
    verdict = ("supporting" if mean_frac >= CONCENTRATION_FRACTION else
               "mixed" if mean_frac >= 0.5 else "contradicting")
    return {"per_method_fraction_inside": fracs,
            "mean_fraction_inside": mean_frac,
            "threshold": CONCENTRATION_FRACTION, "verdict": verdict}


def evaluate_h3(df: pd.DataFrame, run_dir: str) -> dict:
    """H3: intermediate predictions stabilize with depth; the final forecast
    stops changing with extra context while early layers still differ."""
    lens = df[df["method"] == "forecast_lens"]
    if lens.empty:
        return {"verdict": "insufficient_data"}
    mats = agg.lens_matrices(lens)
    dist = mats.get("dist_to_same_norm")
    if dist is None or dist.empty or dist.shape[0] < 3:
        return {"verdict": "insufficient_data"}
    arr = dist.to_numpy(dtype=float)
    # depth-monotonicity: correlation of distance with layer index, per context
    depth_corrs = [spearman(np.arange(arr.shape[0]), -arr[:, j])
                   for j in range(arr.shape[1])]
    med_corr = float(np.nanmedian(depth_corrs))
    sat_files = []
    for dirpath, _dirs, files in os.walk(run_dir):
        if "saturation_layers.json" in files:
            with open(os.path.join(dirpath, "saturation_layers.json")) as f:
                sat_files.append(json.load(f))
    sat_layers = [v for s in sat_files
                  for v in s.get("saturation_layer_by_context", {}).values()
                  if v >= 0]
    verdict = ("supporting" if med_corr > 0.5 and sat_layers else
               "mixed" if med_corr > 0 else "contradicting")
    return {"median_depth_spearman": med_corr,
            "saturation_layers": sat_layers, "verdict": verdict}


def evaluate_h4(run_dir: str) -> dict:
    """H4: when distant info is genuinely predictive, effective context
    expands toward the dependency lag."""
    path = os.path.join(run_dir, "exp4_synthetic_controls",
                        "controls_summary.json")
    if not os.path.exists(path):
        return {"verdict": "insufficient_data"}
    with open(path) as f:
        summ = json.load(f)
    rows = [c for c in summ.get("controls", [])
            if c["spec"]["family"] == "B" and not c.get("config_broken")]
    strong = [c for c in rows if c["spec"]["strength"] >= 0.5]
    zero = [c for c in rows if c["spec"]["strength"] == 0.0]
    if not strong:
        return {"verdict": "insufficient_data"}
    lags = [c["spec"]["distant_lag"] for c in strong]
    suffs = [c["sufficient_context"] for c in strong]
    rho = spearman(np.array(lags, float), np.array(suffs, float))
    reach = float(np.mean([s >= d for s, d in zip(suffs, lags)]))
    limitations = [c["dataset"] for c in rows if c.get("limitation_flag")]
    zero_ok = all(c["sufficient_context"] <= min(lags, default=1e9)
                  for c in zero) if zero else None
    verdict = ("supporting" if (np.isfinite(rho) and rho >= H4_MIN_SPEARMAN
                                and reach >= 0.5) else
               "contradicting" if limitations and reach < 0.25 else "mixed")
    return {"spearman_lag_vs_sufficient": rho,
            "fraction_reaching_lag": reach,
            "zero_strength_stays_local": zero_ok,
            "architectural_limitation_flags": limitations,
            "threshold": H4_MIN_SPEARMAN, "verdict": verdict}


def evaluate_h5(df: pd.DataFrame) -> dict:
    """H5: the methods agree qualitatively on the effective context."""
    from context_interpretability.analysis.figures import cross_method_table
    rhos = []
    for W in sorted(df["context_length"].dropna().unique()):
        t = cross_method_table(df, int(W))
        if t is not None:
            rhos.extend(t["spearman"].dropna().tolist())
    if not rhos:
        return {"verdict": "insufficient_data"}
    med = float(np.nanmedian(rhos))
    verdict = ("supporting" if med >= H5_MIN_MEDIAN_SPEARMAN else
               "mixed" if med > 0 else "contradicting")
    return {"median_cross_method_spearman": med, "n_pairs": len(rhos),
            "threshold": H5_MIN_MEDIAN_SPEARMAN, "verdict": verdict}


HYPOTHESES = {
    "H1": "Perturbing observations beyond the sufficient context causes "
          "negligible forecast degradation.",
    "H2": "Forecast-relevant attribution and intervention effects concentrate "
          "inside the sufficient context.",
    "H3": "Intermediate predictions stabilize across context lengths as "
          "representations progress through the network.",
    "H4": "When distant information is made genuinely predictive, the model's "
          "effective context expands toward that dependency.",
    "H5": "Attention masking, perturbation, activation patching, forecast "
          "lens and integrated gradients give qualitatively consistent "
          "estimates of the effective context.",
}


def evaluate(run_dir: str, tolerance: float = 0.05) -> dict:
    df = load_results(run_dir)
    suff = _sufficient_contexts(run_dir, tolerance)
    report = {
        "run_dir": run_dir,
        "sufficient_contexts": suff,
        "ig_limitations": LIMITATIONS,
        "hypotheses": {
            "H1": {"statement": HYPOTHESES["H1"], **evaluate_h1(df, suff)},
            "H2": {"statement": HYPOTHESES["H2"], **evaluate_h2(df, suff)},
            "H3": {"statement": HYPOTHESES["H3"], **evaluate_h3(df, run_dir)},
            "H4": {"statement": HYPOTHESES["H4"], **evaluate_h4(run_dir)},
            "H5": {"statement": HYPOTHESES["H5"], **evaluate_h5(df)},
        },
    }
    with open(os.path.join(run_dir, "hypotheses_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    _write_markdown(report, os.path.join(run_dir, "hypotheses_report.md"))
    return report


def _write_markdown(report: dict, path: str) -> None:
    lines = ["# Context-saturation hypothesis report", ""]
    for name, h in report["hypotheses"].items():
        lines += [f"## {name} — **{h.get('verdict', '?')}**",
                  "", f"> {h['statement']}", ""]
        for k, v in h.items():
            if k in ("statement", "verdict"):
                continue
            lines.append(f"- `{k}`: {v}")
        lines.append("")
    lines += ["## Integrated-gradients limitations (always applicable)", "",
              report["ig_limitations"], ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[hypotheses] wrote {path}")
