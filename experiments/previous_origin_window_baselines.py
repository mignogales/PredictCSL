#!/usr/bin/env python3
"""Temporal persistence baselines for per-instance oracle windows."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from experiments.series_aware_window_splits import build,splits,LOGW

def score(data,test,pred):
 actions=[]
 for q,mask in zip(pred,data['mask'][test]):
  candidates=np.flatnonzero(mask);actions.append(candidates[np.argmin(abs(LOGW[candidates]-q))])
 actions=np.asarray(actions);err=data['errors'][test];mask=data['mask'][test];rows=np.arange(len(test));chosen=err[rows,actions]
 native=np.asarray([e[np.flatnonzero(m)[-1]] for e,m in zip(err,mask)]);oracle=np.nanmin(np.where(mask,err,np.nan),axis=1);target=data['y'][test]
 return {'n':len(test),'grid_accuracy':float(np.mean(np.argmin(abs(pred[:,None]-LOGW),1)==np.argmin(abs(target[:,None]-LOGW),1))),
  'mae_log2':float(np.mean(abs(pred-target))),'median_ae_log2':float(np.median(abs(pred-target))),
  'error_vs_native':float(np.mean(chosen/np.maximum(native,1e-8))),'improvement_rate':float(np.mean(chosen<native)),
  'mean_regret_vs_oracle':float(np.mean((chosen-oracle)/np.maximum(oracle,1e-8)))}

def main():
 p=argparse.ArgumentParser();p.add_argument('--ablation-root',default='logs/experiments/window_ablation_gifteval');p.add_argument('--output',required=True);p.add_argument('--max-cells',type=int,default=8);p.add_argument('--max-series',type=int,default=12);p.add_argument('--max-origins',type=int,default=8);a=p.parse_args()
 d=build(a.ablation_root,a.max_cells,4,3,a.max_series,a.max_origins);train,test=splits(d,'intra',42);pred={'previous_origin':[],'series_median':[],'cell_median':[]}
 for i in test:
  same_series=train[(d['cell'][train]==d['cell'][i])&(d['item'][train]==d['item'][i])];same_series=same_series[np.argsort(d['start'][same_series])]
  same_cell=train[d['cell'][train]==d['cell'][i]]
  pred['previous_origin'].append(d['y'][same_series[-1]]);pred['series_median'].append(np.median(d['y'][same_series]));pred['cell_median'].append(np.median(d['y'][same_cell]))
 report={'cells':np.unique(d['cell']).tolist(),'n_series':len(test),'baselines':{}}
 for name,values in pred.items():
  values=np.asarray(values);result={'pooled':score(d,test,values),'per_cell':{}}
  for cell in np.unique(d['cell']):
   take=np.flatnonzero(d['cell'][test]==cell);result['per_cell'][cell]=score(d,test[take],values[take])
  report['baselines'][name]=result;print(name,json.dumps(result),flush=True)
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
