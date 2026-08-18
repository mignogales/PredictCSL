#!/usr/bin/env python3
"""Measure stability of full window-error curves across consecutive origins."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from experiments.series_aware_window_splits import build

TOLS=(.01,.03,.05,.10)
def rank(x):
 order=np.argsort(x);r=np.empty(len(x),float);r[order]=np.arange(len(x));return r
def corr(a,b):
 if len(a)<2 or np.std(a)==0 or np.std(b)==0:return np.nan
 return float(np.corrcoef(a,b)[0,1])
def summarize(values):
 a=np.asarray(values,float);a=a[np.isfinite(a)]
 return {'n':len(a),'mean':float(np.mean(a)),'median':float(np.median(a)),'p10':float(np.quantile(a,.1)),'p90':float(np.quantile(a,.9))} if len(a) else {'n':0}

def main():
 p=argparse.ArgumentParser();p.add_argument('--ablation-root',default='logs/experiments/window_ablation_gifteval');p.add_argument('--output',required=True);p.add_argument('--max-cells',type=int,default=8);p.add_argument('--max-series',type=int,default=12);p.add_argument('--max-origins',type=int,default=8);a=p.parse_args()
 d=build(a.ablation_root,a.max_cells,4,3,a.max_series,a.max_origins);rows=[]
 for cell in np.unique(d['cell']):
  ci=np.flatnonzero(d['cell']==cell)
  for item in np.unique(d['item'][ci]):
   ii=ci[d['item'][ci]==item];ii=ii[np.argsort(d['start'][ii])]
   for prev,cur in zip(ii[:-1],ii[1:]):
    mask=d['mask'][prev]&d['mask'][cur];ep=d['errors'][prev,mask].astype(float);ec=d['errors'][cur,mask].astype(float)
    if len(ep)<2:continue
    # Shape comparison is invariant to the very different error scales by origin.
    zp=(ep-ep.mean())/max(ep.std(),1e-12);zc=(ec-ec.mean())/max(ec.std(),1e-12)
    bp=int(np.argmin(ep));bc=int(np.argmin(ec));row={'cell':cell,'item':item,'pearson_shape':corr(zp,zc),'spearman':corr(rank(ep),rank(ec)),'argmin_same':bp==bc,'previous_ratio_current':float(ec[bp]/max(ec[bc],1e-12))}
    for tol in TOLS:
     sp=ep<=ep.min()*(1+tol);sc=ec<=ec.min()*(1+tol);union=(sp|sc).sum();row[f'jaccard_{int(tol*100)}']=float((sp&sc).sum()/union) if union else np.nan;row[f'prev_safe_{int(tol*100)}']=bool(ec[bp]<=ec.min()*(1+tol))
    rows.append(row)
 def aggregate(rr):
  out={'pairs':len(rr),'argmin_same_rate':float(np.mean([r['argmin_same'] for r in rr])),'pearson_shape':summarize([r['pearson_shape'] for r in rr]),'spearman':summarize([r['spearman'] for r in rr]),'previous_ratio_current':summarize([r['previous_ratio_current'] for r in rr])}
  for tol in TOLS:
   k=int(tol*100);out[f'near_opt_{k}pct_jaccard']=summarize([r[f'jaccard_{k}'] for r in rr]);out[f'previous_window_safe_{k}pct_rate']=float(np.mean([r[f'prev_safe_{k}'] for r in rr]))
  out['argmin_changed_but_prev_within_1pct_rate']=float(np.mean([(not r['argmin_same']) and r['prev_safe_1'] for r in rr]));return out
 report={'cells':np.unique(d['cell']).tolist(),'pooled':aggregate(rows),'per_cell':{c:aggregate([r for r in rows if r['cell']==c]) for c in np.unique(d['cell'])}}
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
