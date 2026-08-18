#!/usr/bin/env python3
"""Deliberately test-contaminated capacity sanity check for the selector."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gift_eval.data import Dataset as GiftEvalDataset
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import datasets_config
from experiments.evaluate_instance_windows import (
    _cell_metric_path, _ground_tree, _load_vector, discover_cells)
from experiments.predict_context_length import MambaContextLength
from experiments.pretrain_bounded_selector import bounded_risk_loss
from experiments.test_window_ablation_gifteval_v5 import (
    GiftEvalCache, _closest_horizon_idx, _prepare_predictor_inputs)

WINDOWS = (32, 48, 64, 96, 128, 192, 256, 384, 512,
           768, 1024, 1536, 2048, 2560, 3072, 4096, 6144, 8192)
HORIZONS = (24, 96, 192)


class Tasks(Dataset):
    def __init__(self, x, h, y):
        self.x = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(-1)
        self.h = torch.from_numpy(np.ascontiguousarray(h)).long()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).float()
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.h[i], self.y[i]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); chosen, curves = [], []
    for x, h, y in loader:
        pred = model(x.to(device), h.to(device))[0]
        chosen.append(pred.argmin(1).cpu().numpy()); curves.append(y.numpy())
    action = np.concatenate(chosen); error = np.concatenate(curves)
    row = np.arange(len(error)); selected = error[row, action]
    native = error[:, -1]; oracle = error.min(1)
    return {
        "n": len(error),
        "mean_error_vs_native": float(np.mean(selected / np.maximum(native, 1e-8))),
        "mean_regret": float(np.mean((selected-oracle) / np.maximum(oracle, 1e-8))),
        "improvement_rate": float(np.mean(selected < native)),
        "harm_gt_5pct_rate": float(np.mean((selected-native)/np.maximum(native,1e-8) > .05)),
        "native_rate": float(np.mean(action == len(WINDOWS)-1)),
        "oracle_improvement": float(1-np.mean(oracle)/np.mean(native)),
    }


def main():
    p=argparse.ArgumentParser(); p.add_argument("--ablation-root",default="logs/experiments/window_ablation_gifteval")
    p.add_argument("--output-dir",required=True); p.add_argument("--device",default="cuda:0")
    p.add_argument("--rows-per-cell",type=int,default=32); p.add_argument("--epochs",type=int,default=15)
    p.add_argument("--seed",type=int,default=42); a=p.parse_args()
    rng=np.random.RandomState(a.seed); ground=_ground_tree(a.ablation_root)
    specs={(d,str(t)):(n,u) for n,t,d,u in datasets_config.datasets_to_run()}
    xs={k:[] for k in ("train","val","test")}; hs={k:[] for k in xs}; ys={k:[] for k in xs}
    cells=discover_cells(a.ablation_root,["Chronos2-Small"])
    for number,cell in enumerate(cells,1):
        spec=specs.get((cell.dataset,str(cell.term)))
        if not spec: continue
        try: cache=GiftEvalCache(GiftEvalDataset(name=spec[0],term=cell.term,to_univariate=spec[1]),cell.dataset)
        except Exception: continue
        n=cache.n_total; error=np.full((n,len(WINDOWS)),np.nan); counts=np.zeros_like(error)
        for j,w in enumerate(WINDOWS):
            error[:,j],counts[:,j],_=_load_vector(_cell_metric_path(ground,cell,w),n,"mase_gluonts_real")
        eligible=np.isfinite(error)&(counts>0)&(error>0)
        has_any=eligible.any(1)
        last=np.where(eligible,np.arange(len(WINDOWS))[None,:],-1).max(1)
        rows=np.arange(n)[has_any]; native=error[rows,last[has_any]]
        # Match the deployed native-capped policy: an unsupported requested
        # width has the same outcome as the largest available genuine context.
        error[rows]=np.where(eligible[rows],error[rows],native[:,None])
        valid=np.flatnonzero(has_any)
        if not len(valid): continue
        pick=rng.choice(valid,size=min(a.rows_per_cell,len(valid)),replace=False); rng.shuffle(pick)
        prepared=_prepare_predictor_inputs([cache.contexts[i] for i in pick],8192)
        h=np.full(len(pick),_closest_horizon_idx(cache.horizon,list(HORIZONS)),np.int64)
        ntr=max(1,int(.6*len(pick))); nv=max(1,int(.2*len(pick)))
        boundaries={"train":(0,ntr),"val":(ntr,min(ntr+nv,len(pick))),"test":(min(ntr+nv,len(pick)),len(pick))}
        for split,(lo,hi) in boundaries.items():
            if hi<=lo: continue
            xs[split].append(prepared[lo:hi]); hs[split].append(h[lo:hi]); ys[split].append(error[pick[lo:hi]])
        print(f"[{number}/{len(cells)}] {cell.dataset}/{cell.term}: {len(pick)}",flush=True)
    data={k:Tasks(np.concatenate(xs[k]),np.concatenate(hs[k]),np.concatenate(ys[k])) for k in xs}
    loaders={k:DataLoader(v,batch_size=32,shuffle=k=="train") for k,v in data.items()}
    model=MambaContextLength(8192,128,128,2,16,4,2,.1,.15,len(WINDOWS),len(HORIZONS)).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    best=float("inf"); best_state=None
    for epoch in range(1,a.epochs+1):
        model.train(); total=0
        for x,h,y in loaders["train"]:
            pred=model(x.to(a.device),h.to(a.device))[0]
            loss=bounded_risk_loss(pred,y.to(a.device),native_harm_weight=2.0)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); total+=float(loss)
        val=evaluate(model,loaders["val"],a.device)
        print(f"epoch={epoch} loss={total/len(loaders['train']):.4f} val={val}",flush=True)
        if val["mean_regret"]<best: best=val["mean_regret"]; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    report={"warning":"DELIBERATELY TEST-CONTAMINATED; diagnostic only","windows":WINDOWS,
            "train":evaluate(model,loaders["train"],a.device),"val":evaluate(model,loaders["val"],a.device),
            "test_same_cells":evaluate(model,loaders["test"],a.device)}
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);torch.save(best_state,out/"model.pt")
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))

if __name__=="__main__": main()
