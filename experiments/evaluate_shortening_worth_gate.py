#!/usr/bin/env python3
"""Apply a validation-calibrated worth-shortening gate to real GIFT-Eval."""
import argparse,json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from gift_eval.data import Dataset as GiftEvalDataset
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from experiments import datasets_config
from experiments.evaluate_instance_windows import _cell_metric_path,_ground_tree,_load_vector,discover_cells
from experiments.test_window_ablation_gifteval_v5 import GiftEvalCache,_closest_horizon_idx,_prepare_predictor_inputs,predict_curves_for_dataset
from experiments.train_shortening_worth_gate import load_selector,features

def weighted(v,c):
 ok=np.isfinite(v)&np.isfinite(c)&(c>0);return float(np.average(v[ok],weights=c[ok])) if ok.any() else np.nan
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--gate',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--ablation-root',default='logs/experiments/window_ablation_gifteval');p.add_argument('--device',default='cuda:0');a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);m,ck=load_selector(a.checkpoint,a.device);bundle=joblib.load(a.gate);windows=np.asarray(ck['windows']);horizons=list(ck['horizons']);profiles={k:v['threshold'] for k,v in bundle['profiles'].items()};profiles['raw']=-np.inf;specs={(d,str(t)):(n,u) for n,t,d,u in datasets_config.datasets_to_run()};ground=_ground_tree(a.ablation_root);rows=[]
 for num,cell in enumerate(discover_cells(a.ablation_root,['Chronos2-Small']),1):
  spec=specs.get((cell.dataset,str(cell.term)))
  if not spec:continue
  try:cache=GiftEvalCache(GiftEvalDataset(name=spec[0],term=cell.term,to_univariate=spec[1]),cell.dataset)
  except Exception as exc:print('skip',cell.dataset,exc,flush=True);continue
  n=cache.n_total;hidx=_closest_horizon_idx(cache.horizon,horizons);pred=predict_curves_for_dataset(m,cache,ck['model_config']['context_length'],hidx,a.device,training_objective='risk',batch_size=128);ctx=_prepare_predictor_inputs(cache.contexts,ck['model_config']['context_length']);xf,action=features(ctx,pred,np.full(n,horizons[hidx]),windows);score=bundle['model'].predict(xf);error=np.full((n,len(windows)),np.nan);counts=np.zeros_like(error)
  for j,w in enumerate(windows):error[:,j],counts[:,j],_=_load_vector(_cell_metric_path(ground,cell,int(w)),n,'mase_gluonts_real')
  native,native_counts,_=_load_vector(_cell_metric_path(ground,cell,'full_native'),n,'mase_gluonts_real');eligible=np.isfinite(error)&np.isfinite(counts)&(counts>0);last=np.where(eligible,np.arange(len(windows))[None,:],-1).max(1);missing_native=~np.isfinite(native)|~np.isfinite(native_counts)|(native_counts<=0);fill=missing_native&(last>=0);native[fill]=error[np.arange(n)[fill],last[fill]];native_counts[fill]=counts[np.arange(n)[fill],last[fill]];native_mase=weighted(native,native_counts);row={'dataset':cell.dataset,'term':str(cell.term),'n':n,'native_mase':native_mase}
  for name,t in profiles.items():
   use=(score>=t)&(action<len(windows)-1);chosen=native.copy();cc=native_counts.copy();chosen[use]=error[np.arange(n)[use],action[use]];cc[use]=counts[np.arange(n)[use],action[use]];missing=~np.isfinite(chosen)|~np.isfinite(cc)|(cc<=0);chosen[missing]=native[missing];cc[missing]=native_counts[missing];effective=use&~missing;row[f'{name}_mase']=weighted(chosen,cc);row[f'{name}_native_rate']=float(np.mean(~effective));row[f'{name}_context_reduction']=float(np.mean(np.where(effective,1-windows[action]/windows[-1],0)))
  rows.append(row);print(f'[{num}] {cell.dataset}/{cell.term}',flush=True)
 frame=pd.DataFrame(rows);frame.to_csv(out/'cell_results.csv',index=False);summary={}
 for name in profiles:
  ratio=frame[f'{name}_mase']/frame.native_mase;weights=frame.n/frame.n.sum();summary[name]={'n_cells':len(frame),'macro_relative_change_pct':float(100*np.mean(ratio-1)),'geomean_relative_change_pct':float(100*np.expm1(np.mean(np.log(ratio)))),'cell_win_rate':float(np.mean(ratio<1)),'instance_weighted_native_rate':float(np.sum(weights*frame[f'{name}_native_rate'])),'instance_weighted_context_reduction':float(np.sum(weights*frame[f'{name}_context_reduction']))}
 report={'thresholds_fixed_from_pretrain_validation':profiles,'official_test_used_for_tuning':False,'summary':summary};(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
