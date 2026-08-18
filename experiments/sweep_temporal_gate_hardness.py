#!/usr/bin/env python3
"""Recalibrate extra hardness profiles from a persisted temporal gate run."""
import argparse,json,sys
from pathlib import Path
import joblib,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from experiments.train_real_temporal_worth_gate import metrics
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();root=Path(a.run_dir);z=np.load(root/'temporal_gate_dataset.npz');x=z['x'];action=z['action'];error=z['error'];save=z['save'];cells=z['cells'];items=z['items'];starts=z['starts'];val=[];test=[]
 for cell in np.unique(cells):
  ci=np.flatnonzero(cells==cell)
  for item in np.unique(items[ci]):
   ii=ci[items[ci]==item];ii=ii[np.argsort(starts[ii])]
   if len(ii)>=3:val.append(ii[-2]);test.append(ii[-1])
 val=np.asarray(val);test=np.asarray(test);gate=joblib.load(root/'gate.joblib')['model'];vs=gate.predict(x[val]);ts=gate.predict(x[test]);cand=np.unique(np.r_[np.inf,np.quantile(vs,np.linspace(0,1,1001))]);spec={'conservative':(.005,1.0),'balanced':(.01,1.001),'aggressive':(.03,1.005),'compute_5pct':(.05,1.01),'compute_10pct':(.10,1.02)};report={'profiles':{},'raw_test':metrics(error[test],action[test],ts,-np.inf,save[test])}
 for name,(harm,lim) in spec.items():
  rows=[metrics(error[val],action[val],vs,t,save[val]) for t in cand];ok=[r for r in rows if r['harm5_rate']<=harm and r['mean_error_vs_native']<=lim];v=max(ok,key=lambda r:(r['flops_saved_pct'],-r['mean_error_vs_native']));report['profiles'][name]={'validation':v,'test':metrics(error[test],action[test],ts,v['threshold'],save[test])}
 (root/'hardness_sweep.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
