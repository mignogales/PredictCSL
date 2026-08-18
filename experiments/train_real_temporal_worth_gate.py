#!/usr/bin/env python3
"""Test-contaminated but chronological: earlier origins -> last origin per series."""
import argparse,json,sys
from pathlib import Path
import joblib,numpy as np,torch
from sklearn.ensemble import ExtraTreesRegressor
from gift_eval.data import Dataset as GiftEvalDataset
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from experiments import datasets_config
from experiments import models_config
from experiments.compare_window_strategies_gifteval import theoretical_flops,DEFAULT_PATCH_SIZES
from experiments.evaluate_instance_windows import _cell_metric_path,_ground_tree,_load_vector,discover_cells
from experiments.test_window_ablation_gifteval_v5 import GiftEvalCache,_closest_horizon_idx,_prepare_predictor_inputs,predict_curves_for_dataset
from experiments.test_window_ablation_gifteval_v5 import _full_native_context_cap
from experiments.train_shortening_worth_gate import load_selector,features
from experiments.predict_context_length import MambaContextLength

def metrics(error,action,score,threshold,raw_save):
 use=(score>=threshold)&(action<error.shape[1]-1);final=np.where(use,action,error.shape[1]-1);ratio=error[np.arange(len(error)),final]/np.maximum(error[:,-1],1e-8)
 return {'threshold':float(threshold),'coverage':float(use.mean()),'mean_error_vs_native':float(ratio.mean()),'harm5_rate':float(np.mean(ratio>1.05)),'harm_rate':float(np.mean(ratio>1)),'improvement_rate':float(np.mean(ratio<1)),'flops_saved_pct':float(100*np.sum(np.where(use,raw_save[:,0],0))/np.sum(raw_save[:,1]))}
def calibrate(error,action,score,raw_save):
 cand=np.unique(np.r_[np.inf,np.quantile(score,np.linspace(0,1,401))]);spec={'conservative':(.005,1.0),'balanced':(.01,1.001),'aggressive':(.03,1.005)};out={}
 for name,(harm,lim) in spec.items():
  rows=[metrics(error,action,score,t,raw_save) for t in cand];ok=[r for r in rows if r['harm5_rate']<=harm and r['mean_error_vs_native']<=lim];out[name]=max(ok,key=lambda r:(r['flops_saved_pct'],-r['mean_error_vs_native'])) if ok else metrics(error,action,score,np.inf,raw_save)
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--legacy-config',default='');p.add_argument('--training-objective',default='risk',choices=['risk','curve']);p.add_argument('--model-short',default='Chronos2-Small');p.add_argument('--output-dir',required=True);p.add_argument('--ablation-root',default='logs/experiments/master_recompute/window_ablation_gifteval');p.add_argument('--device',default='cuda:0');p.add_argument('--max-train-per-cell',type=int,default=5000);p.add_argument('--seed',type=int,default=42);p.add_argument('--dataset-only',action='store_true',help='Build the reusable feature/oracle cache without fitting the deliberately test-contaminated sanity gate.');a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 model_specs={display:(model_id,family) for model_id,family,display in models_config.models_to_run()}
 if a.model_short not in model_specs:raise ValueError(f'Unknown run model {a.model_short!r}; choose from {sorted(model_specs)}')
 model_id,_model_family=model_specs[a.model_short]
 if a.legacy_config:
  c=json.loads(Path(a.legacy_config).read_text());model=MambaContextLength(c['context_length'],c['patch_length'],c['d_model'],c['num_hidden_layers'],c['d_state'],c['d_conv'],c['expand'],c['dropout'],c['mask_ratio'],c['n_windows'],c['n_horizons']);model.load_state_dict(torch.load(a.checkpoint,map_location='cpu'));model=model.to(a.device).eval();ck={'windows':c['window_grid'],'horizons':c['horizon_grid'],'model_config':c}
 else:model,ck=load_selector(a.checkpoint,a.device)
 windows=np.asarray(ck['windows']);horizons=list(ck['horizons']);specs={(d,str(t)):(n,u) for n,t,d,u in datasets_config.datasets_to_run()};ground=_ground_tree(a.ablation_root);X=[];A=[];E=[];CELL=[];ITEM=[];START=[];SAVE=[]
 for num,cell in enumerate(discover_cells(a.ablation_root,[a.model_short]),1):
  spec=specs.get((cell.dataset,str(cell.term)))
  if not spec:continue
  gd=GiftEvalDataset(name=spec[0],term=cell.term,to_univariate=spec[1]);cache=GiftEvalCache(gd,cell.dataset);meta=[]
  for inp,lbl in gd.test_data:
   if len(lbl['target'])>=cache.horizon:meta.append((str(inp.get('item_id','unknown')),str(lbl['start'])))
  n=cache.n_total;hi=_closest_horizon_idx(cache.horizon,horizons);pred=predict_curves_for_dataset(model,cache,ck['model_config']['context_length'],hi,a.device,training_objective=a.training_objective,batch_size=128);ctx=_prepare_predictor_inputs(cache.contexts,ck['model_config']['context_length']);xf,action=features(ctx,pred,np.full(n,horizons[hi]),windows);err=np.full((n,len(windows)),np.nan);cnt=np.zeros_like(err)
  for j,w in enumerate(windows):err[:,j],cnt[:,j],_=_load_vector(_cell_metric_path(ground,cell,int(w)),n,'mase_gluonts_real')
  native,nc,_=_load_vector(_cell_metric_path(ground,cell,'full_native'),n,'mase_gluonts_real');proposed=np.where(action==len(windows)-1,native,err[np.arange(n),action]);pc=np.where(action==len(windows)-1,nc,cnt[np.arange(n),action]);effective_action=action.copy();proposal_valid=np.isfinite(proposed)&(pc>0);effective_action[~proposal_valid]=len(windows)-1;proposed[~proposal_valid]=native[~proposal_valid];pc[~proposal_valid]=nc[~proposal_valid];valid=np.isfinite(native)&(native>0)&(nc>0);ix=np.flatnonzero(valid);ev=err[ix].copy();ev[:,-1]=native[ix];native_cap=_full_native_context_cap(_model_family,cache.horizon,models_config.context_limit(_model_family));native_ctx=np.minimum(cache.context_lengths[ix],native_cap);selected_ctx=np.where(effective_action[ix]==len(windows)-1,native_ctx,np.minimum(windows[effective_action[ix]],native_ctx));full=np.asarray([theoretical_flops(model_id,int(c),cache.horizon,DEFAULT_PATCH_SIZES) for c in native_ctx]);sel=np.asarray([theoretical_flops(model_id,int(c),cache.horizon,DEFAULT_PATCH_SIZES) for c in selected_ctx]);X.append(xf[ix]);A.append(effective_action[ix]);E.append(ev);SAVE.append(np.column_stack([full-sel,full]));CELL.extend([f'{cell.dataset}/{cell.term}']*len(ix));ITEM.extend([meta[i][0] for i in ix]);START.extend([meta[i][1] for i in ix]);print(f'[{num}] {cell.dataset}/{cell.term}: {len(ix)} ({int(np.sum(~proposal_valid[ix]))} native fallbacks)',flush=True)
 x=np.concatenate(X);action=np.concatenate(A);error=np.concatenate(E);save=np.concatenate(SAVE);cells=np.asarray(CELL);items=np.asarray(ITEM);starts=np.asarray(START);np.savez_compressed(out/'temporal_gate_dataset.npz',x=x,action=action,error=error,save=save,cells=cells,items=items,starts=starts)
 if a.dataset_only:
  print(json.dumps({'dataset_only':True,'n_rows':len(x),'n_cells':len(np.unique(cells)),'path':str(out/'temporal_gate_dataset.npz')},indent=2),flush=True);return
 train=[];val=[];test=[]
 for cell in np.unique(cells):
  ci=np.flatnonzero(cells==cell)
  for item in np.unique(items[ci]):
   ii=ci[items[ci]==item];ii=ii[np.argsort(starts[ii])]
   if len(ii)>=3:train.extend(ii[:-2]);val.append(ii[-2]);test.append(ii[-1])
 rng=np.random.RandomState(a.seed);balanced=[]
 for cell in np.unique(cells):
  ii=np.asarray(train)[cells[np.asarray(train)]==cell];balanced.extend(rng.choice(ii,min(len(ii),a.max_train_per_cell),replace=False))
 train=np.asarray(balanced);val=np.asarray(val);test=np.asarray(test);realized=(error[:,-1]-error[np.arange(len(error)),action])/np.maximum(error[:,-1],1e-8);gate=ExtraTreesRegressor(n_estimators=700,min_samples_leaf=8,max_features=.7,n_jobs=-1,random_state=a.seed);gate.fit(x[train],realized[train]);vs=gate.predict(x[val]);profiles=calibrate(error[val],action[val],vs,save[val]);ts=gate.predict(x[test]);report={'warning':'REAL GIFT-EVAL USED FOR TRAINING; CHRONOLOGICAL SPLIT','n_cells':len(np.unique(cells)),'n_train':len(train),'n_val':len(val),'n_test':len(test),'profiles':{},'raw_test':metrics(error[test],action[test],ts,-np.inf,save[test])}
 for name,v in profiles.items():report['profiles'][name]={'validation':v,'test':metrics(error[test],action[test],ts,v['threshold'],save[test])}
 joblib.dump({'model':gate,'profiles':profiles},out/'gate.joblib');(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
