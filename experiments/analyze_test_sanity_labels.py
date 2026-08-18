#!/usr/bin/env python3
"""Audit the labels/loss geometry behind the contaminated full-grid sanity run."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.evaluate_instance_windows import (
    _cell_metric_path, _ground_tree, _load_vector, discover_cells)

ROOT = "logs/experiments/window_ablation_gifteval"
WINDOWS = np.asarray((32,48,64,96,128,192,256,384,512,768,1024,1536,
                      2048,2560,3072,4096,6144,8192))


def q(x):
    x=np.asarray(x); x=x[np.isfinite(x)]
    return {str(p):float(np.quantile(x,p)) for p in (0,.001,.01,.05,.5,.95,.99,.999,1)}


def main():
    rng=np.random.RandomState(42); ground=_ground_tree(ROOT); curves=[]; rows=[]
    for cell in discover_cells(ROOT,["Chronos2-Small"]):
        with np.load(cell.anchor_npz) as a: n=int(a["predicted_curves"].shape[0])
        e=np.full((n,len(WINDOWS)),np.nan); c=np.zeros_like(e)
        for j,w in enumerate(WINDOWS):
            e[:,j],c[:,j],_=_load_vector(_cell_metric_path(ground,cell,int(w)),n,"mase_gluonts_real")
        ok=np.isfinite(e)&(c>0)&(e>0); has=ok.any(1)
        last=np.where(ok,np.arange(len(WINDOWS))[None,:],-1).max(1)
        idx=np.flatnonzero(has)
        if not len(idx): continue
        native=e[idx,last[idx]]; filled=np.where(ok[idx],e[idx],native[:,None])
        pick=rng.choice(len(idx),size=min(32,len(idx)),replace=False)
        curves.append(filled[pick])
        rows.append({"cell":f"{cell.dataset}/{cell.term}","available_rows":len(idx),
                     "sampled":len(pick),"median_available_actions":float(np.median(ok[idx].sum(1))),
                     "max_available_actions":int(ok[idx].sum(1).max())})
    y=np.concatenate(curves); native=y[:,-1]; oracle=y.min(1); action=y.argmin(1)
    regret=(y-oracle[:,None])/np.maximum(oracle[:,None],1e-12)
    harm=np.maximum((y-native[:,None])/np.maximum(native[:,None],1e-12),0)
    logrel=np.log(np.maximum(y,1e-12)/np.maximum(native[:,None],1e-12))
    report={
        "n_cells":len(rows),"n_rows":len(y),"cells":rows,
        "error_quantiles":q(y),"native_error_quantiles":q(native),
        "oracle_error_quantiles":q(oracle),"oracle_over_native_quantiles":q(oracle/native),
        "oracle_action_counts":dict(zip(map(str,WINDOWS),np.bincount(action,minlength=len(WINDOWS)).tolist())),
        "regret_all_actions_quantiles":q(regret),"regret_row_max_quantiles":q(regret.max(1)),
        "harm_all_actions_quantiles":q(harm),"log_error_over_native_quantiles":q(logrel),
        "initial_uniform_policy_cost_quantiles":q((regret+2*harm).mean(1)),
        "per_action":[{"window":int(w),"win_rate_vs_native":float(np.mean(y[:,j]<native)),
                       "harm5_rate":float(np.mean(y[:,j]>1.05*native)),
                       "median_error_ratio":float(np.median(y[:,j]/native)),
                       "p99_regret":float(np.quantile(regret[:,j],.99))}
                      for j,w in enumerate(WINDOWS)],
    }
    out=Path("logs/experiments/test_leak_sanity_v1/label_audit.json")
    out.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
