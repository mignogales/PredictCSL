#!/usr/bin/env python3
"""Learn whether to trust an existing context-window selector proposal."""
import argparse,json,sys
from pathlib import Path
import joblib,numpy as np,torch
from sklearn.ensemble import ExtraTreesRegressor
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from experiments.predict_context_length import MambaContextLength

def load_selector(path,device):
 ck=torch.load(path,map_location='cpu');c=ck['model_config'];m=MambaContextLength(c['context_length'],c['patch_length'],c['d_model'],c['num_hidden_layers'],c['d_state'],c['d_conv'],c['expand'],c['dropout'],c['mask_ratio'],len(ck['windows']),len(ck['horizons']));m.load_state_dict(ck['state_dict']);return m.to(device).eval(),ck
@torch.no_grad()
def predict(m,x,h,device):
 out=[]
 for s in range(0,len(x),128):
  z=torch.from_numpy(np.ascontiguousarray(x[s:s+128])).float().unsqueeze(-1).to(device);hi=torch.from_numpy(h[s:s+128]).long().to(device);out.append(m(z,hi)[0].cpu().numpy())
 return np.concatenate(out)
def cheap_series_features(context):
 x=np.asarray(context,dtype=np.float32);cols=[]
 for width in (32,128,512,2048,8192):
  y=x[:,-width:];d=np.diff(y,axis=1);cols.extend([y.mean(1),y.std(1),y[:,-1],np.mean(np.abs(d),1),d.std(1),y[:,-1]-y[:,0]])
 for lag in (1,24,96):
  a=x[:,-2048:-lag];b=x[:,-2048+lag:];a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True);cols.append(np.mean(a*b,1)/np.maximum(a.std(1)*b.std(1),1e-6))
 return np.nan_to_num(np.column_stack(cols),copy=False)
def features(context,pred,h,windows):
 base=cheap_series_features(context);action=pred.argmin(1);native=pred[:,-1];chosen=pred[np.arange(len(pred)),action];sorted_pred=np.sort(pred,axis=1)
 extra=np.column_stack([pred,native-chosen,sorted_pred[:,1]-sorted_pred[:,0],action/(len(windows)-1),np.log1p(np.asarray(windows)[action]),np.log1p(np.asarray(h,float))])
 return np.column_stack([base,extra]).astype(np.float32),action
def policy_metrics(error,action,score,threshold,windows):
 use=(score>=threshold)&(action<len(windows)-1);final=np.where(use,action,len(windows)-1);row=np.arange(len(error));chosen=error[row,final];native=error[:,-1];ratio=chosen/np.maximum(native,1e-8);saving=np.where(use,1-np.asarray(windows)[final]/windows[-1],0)
 return {'threshold':float(threshold),'coverage':float(use.mean()),'mean_error_vs_native':float(ratio.mean()),'harm5_rate':float(np.mean(ratio>1.05)),'harm_rate':float(np.mean(ratio>1)),'improvement_rate':float(np.mean(ratio<1)),'mean_context_reduction':float(saving.mean()),'mean_selected_window':float(np.mean(np.where(use,np.asarray(windows)[final],windows[-1])))}
def calibrate(error,action,score,windows):
 candidates=np.unique(np.r_[np.inf,np.quantile(score,np.linspace(0,1,401))]);profiles={'conservative':(.005,1.000),'balanced':(.01,1.001),'aggressive':(.03,1.005)};out={}
 for name,(harm,mean_limit) in profiles.items():
  feasible=[]
  for t in candidates:
   m=policy_metrics(error,action,score,t,windows)
   if m['harm5_rate']<=harm and m['mean_error_vs_native']<=mean_limit:feasible.append(m)
  out[name]=max(feasible,key=lambda m:(m['mean_context_reduction'],-m['mean_error_vs_native'])) if feasible else policy_metrics(error,action,score,np.inf,windows)
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=42);a=p.parse_args();root=Path(a.root);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);m,ck=load_selector(a.checkpoint,a.device);windows=np.asarray(ck['windows']);horizons=np.asarray(ck['horizons']);ctx=np.load(root/'prepared/contexts.npy');splits=np.load(root/'prepared/splits.npy');curves=np.load(root/'labels/chronos2_small/curves_mae.npy');n=len(ctx);task_ctx=np.repeat(ctx,len(horizons),axis=0);hidx=np.tile(np.arange(len(horizons)),n);hval=horizons[hidx];error=curves.transpose(0,2,1).reshape(-1,len(windows));task_split=np.repeat(splits,len(horizons));pred=predict(m,task_ctx,hidx,a.device);x,action=features(task_ctx,pred,hval,windows);realized=(error[:,-1]-error[np.arange(len(error)),action])/np.maximum(error[:,-1],1e-8)
 train=task_split=='train';val=task_split=='val';test=task_split=='test';gate=ExtraTreesRegressor(n_estimators=700,min_samples_leaf=8,max_features=.7,n_jobs=-1,random_state=a.seed);gate.fit(x[train],realized[train]);val_score=gate.predict(x[val]);profiles=calibrate(error[val],action[val],val_score,windows);test_score=gate.predict(x[test]);report={'method':'two-stage existing selector plus learned worth-shortening gate','checkpoint':a.checkpoint,'features':int(x.shape[1]),'train_tasks':int(train.sum()),'val_tasks':int(val.sum()),'test_tasks':int(test.sum()),'profiles':{}}
 for name,v in profiles.items():report['profiles'][name]={'validation':v,'internal_test':policy_metrics(error[test],action[test],test_score,v['threshold'],windows)}
 report['raw_validation']=policy_metrics(error[val],action[val],val_score,-np.inf,windows);report['raw_internal_test']=policy_metrics(error[test],action[test],test_score,-np.inf,windows);joblib.dump({'model':gate,'windows':windows,'horizons':horizons,'profiles':profiles},out/'gate.joblib');(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
