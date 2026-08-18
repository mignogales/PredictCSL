#!/usr/bin/env python3
"""Deliberately test-contaminated upper bound for the worth-shortening gate."""
import argparse,json,sys
from pathlib import Path
import joblib,numpy as np,torch
from sklearn.ensemble import ExtraTreesRegressor
from gift_eval.data import Dataset as GiftEvalDataset
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from experiments import datasets_config
from experiments.evaluate_instance_windows import _cell_metric_path,_ground_tree,_load_vector,discover_cells
from experiments.test_window_ablation_gifteval_v5 import GiftEvalCache,_closest_horizon_idx,_prepare_predictor_inputs,predict_curves_for_dataset
from experiments.train_shortening_worth_gate import load_selector,features,calibrate,policy_metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--ablation-root',default='logs/experiments/window_ablation_gifteval');p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=42);p.add_argument('--max-train',type=int,default=100000);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);m,ck=load_selector(a.checkpoint,a.device);windows=np.asarray(ck['windows']);horizons=list(ck['horizons']);specs={(d,str(t)):(n,u) for n,t,d,u in datasets_config.datasets_to_run()};ground=_ground_tree(a.ablation_root);X=[];A=[];E=[];CELL=[]
 for num,cell in enumerate(discover_cells(a.ablation_root,['Chronos2-Small']),1):
  spec=specs.get((cell.dataset,str(cell.term)))
  if not spec:continue
  try:cache=GiftEvalCache(GiftEvalDataset(name=spec[0],term=cell.term,to_univariate=spec[1]),cell.dataset)
  except Exception as exc:print('skip',cell.dataset,exc,flush=True);continue
  n=cache.n_total;hi=_closest_horizon_idx(cache.horizon,horizons);pred=predict_curves_for_dataset(m,cache,ck['model_config']['context_length'],hi,a.device,training_objective='risk',batch_size=128);ctx=_prepare_predictor_inputs(cache.contexts,ck['model_config']['context_length']);xf,action=features(ctx,pred,np.full(n,horizons[hi]),windows);err=np.full((n,len(windows)),np.nan);cnt=np.zeros_like(err)
  for j,w in enumerate(windows):err[:,j],cnt[:,j],_=_load_vector(_cell_metric_path(ground,cell,int(w)),n,'mae')
  native,nc,_=_load_vector(_cell_metric_path(ground,cell,'full_native'),n,'mae');available=np.isfinite(err)&np.isfinite(cnt)&(cnt>0);last=np.where(available,np.arange(len(windows))[None,:],-1).max(1);missing=~np.isfinite(native)|~np.isfinite(nc)|(nc<=0);fill=missing&(last>=0);native[fill]=err[np.arange(n)[fill],last[fill]];nc[fill]=cnt[np.arange(n)[fill],last[fill]];proposed=np.where(action==len(windows)-1,native,err[np.arange(n),action]);proposed_count=np.where(action==len(windows)-1,nc,cnt[np.arange(n),action]);valid=np.isfinite(native)&(native>0)&np.isfinite(nc)&(nc>0)&np.isfinite(proposed)&(proposed_count>0);ev=err[valid].copy();ev[:,-1]=native[valid];X.append(xf[valid]);A.append(action[valid]);E.append(ev);CELL.extend([f'{cell.dataset}/{cell.term}']*int(valid.sum()));print(f'[{num}] {cell.dataset}/{cell.term}: {valid.sum()}',flush=True)
 x=np.concatenate(X);action=np.concatenate(A);error=np.concatenate(E);cells=np.asarray(CELL);rng=np.random.RandomState(a.seed);split=np.empty(len(x),dtype='U5')
 for cell in np.unique(cells):
  ix=np.flatnonzero(cells==cell);rng.shuffle(ix);n=len(ix);ntr=max(1,int(.6*n));nv=max(1,int(.2*n));split[ix[:ntr]]='train';split[ix[ntr:ntr+nv]]='val';split[ix[ntr+nv:]]='test'
 realized=(error[:,-1]-error[np.arange(len(error)),action])/np.maximum(error[:,-1],1e-8);tr=np.flatnonzero(split=='train');val=split=='val';test=split=='test'
 if len(tr)>a.max_train:tr=rng.choice(tr,a.max_train,replace=False)
 gate=ExtraTreesRegressor(n_estimators=700,min_samples_leaf=8,max_features=.7,n_jobs=-1,random_state=a.seed);gate.fit(x[tr],realized[tr]);vs=gate.predict(x[val]);profiles=calibrate(error[val],action[val],vs,windows);ts=gate.predict(x[test]);report={'warning':'DELIBERATELY TRAINED ON REAL GIFT-EVAL TEST','n':len(x),'n_cells':len(np.unique(cells)),'n_train':len(tr),'n_val':int(val.sum()),'n_test':int(test.sum()),'profiles':{},'raw_test':policy_metrics(error[test],action[test],ts,-np.inf,windows)}
 for name,v in profiles.items():report['profiles'][name]={'validation':v,'test':policy_metrics(error[test],action[test],ts,v['threshold'],windows)}
 joblib.dump({'model':gate,'profiles':profiles,'windows':windows},out/'gate.joblib');(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
