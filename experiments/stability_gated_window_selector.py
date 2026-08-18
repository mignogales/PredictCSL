#!/usr/bin/env python3
"""Train-only temporal-stability gate for the intra-series Mamba selector."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from experiments.series_aware_window_splits import build,splits,net,LOGW

def corr(a,b):
 m=np.isfinite(a)&np.isfinite(b)
 if m.sum()<2 or np.std(a[m])==0 or np.std(b[m])==0:return np.nan
 return float(np.corrcoef(a[m],b[m])[0,1])
def series_stability(d,indices):
 ii=indices[np.argsort(d['start'][indices])];vals=[]
 for a,b in zip(ii[:-1],ii[1:]):
  mask=d['mask'][a]&d['mask'][b];ea=d['errors'][a,mask];eb=d['errors'][b,mask]
  vals.append(corr(ea,eb))
 return float(np.nanmedian(vals)) if np.isfinite(vals).any() else -1.0
@torch.no_grad()
def predictions(m,d,ix,device):
 m.eval();out=[]
 for s in range(0,len(ix),64):
  j=ix[s:s+64];x=torch.from_numpy(np.ascontiguousarray(d['x'][j])).float().unsqueeze(-1).to(device);h=torch.from_numpy(d['h'][j]).long().to(device);out.append((5+8*m(x,h)[0].squeeze(1)).cpu().numpy())
 return np.concatenate(out)
def score(d,ix,pred,enabled):
 actions=[]
 for q,mask,on in zip(pred,d['mask'][ix],enabled):
  candidates=np.flatnonzero(mask);actions.append(candidates[np.argmin(abs(LOGW[candidates]-q))] if on else candidates[-1])
 actions=np.asarray(actions);err=d['errors'][ix];mask=d['mask'][ix];row=np.arange(len(ix));chosen=err[row,actions];native=np.asarray([e[np.flatnonzero(mm)[-1]] for e,mm in zip(err,mask)]);oracle=np.nanmin(np.where(mask,err,np.nan),1)
 return {'n':len(ix),'coverage':float(np.mean(enabled)),'error_vs_native':float(np.mean(chosen/np.maximum(native,1e-8))),'improvement_rate':float(np.mean(chosen<native)),'harm5_rate':float(np.mean(chosen>1.05*native)),'mean_regret':float(np.mean((chosen-oracle)/np.maximum(oracle,1e-8)))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--ablation-root',default='logs/experiments/window_ablation_gifteval');p.add_argument('--device',default='cuda:0');p.add_argument('--updates',type=int,default=1200);a=p.parse_args();d=build(a.ablation_root,8,4,3,12,8);tr,te=splits(d,'intra',42);stability={}
 for i in te:
  key=(d['cell'][i],d['item'][i]);hist=tr[(d['cell'][tr]==key[0])&(d['item'][tr]==key[1])];stability[key]=series_stability(d,hist)
 torch.manual_seed(42);m=net().to(a.device);ds=TensorDataset(torch.from_numpy(np.ascontiguousarray(d['x'][tr])).float().unsqueeze(-1),torch.from_numpy(d['h'][tr]).long(),torch.from_numpy(((d['y'][tr]-5)/8).astype(np.float32)).unsqueeze(1));loader=DataLoader(ds,batch_size=32,shuffle=True);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0);it=iter(loader);m.train()
 for _ in range(a.updates):
  try:x,h,y=next(it)
  except StopIteration:it=iter(loader);x,h,y=next(it)
  loss=torch.nn.functional.mse_loss(m(x.to(a.device),h.to(a.device))[0],y.to(a.device));opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10);opt.step()
 pred=predictions(m,d,te,a.device);s=np.asarray([stability[(d['cell'][i],d['item'][i])] for i in te]);thresholds=[-1,.0,.2,.4,.6,.8];report={'n_test':len(te),'stability':{'median':float(np.median(s)),'p25':float(np.quantile(s,.25)),'p75':float(np.quantile(s,.75))},'thresholds':{str(t):score(d,te,pred,s>=t) for t in thresholds}}
 # Representative same-series curves: highest/lowest train-only stability.
 representatives=[]
 for label,pos in [('stable',int(np.argmax(s))),('unstable',int(np.argmin(s)))]:
  i=te[pos];allix=np.flatnonzero((d['cell']==d['cell'][i])&(d['item']==d['item'][i]));allix=allix[np.argsort(d['start'][allix])];curves=[]
  for j in allix:
   mask=d['mask'][j];native=d['errors'][j,np.flatnonzero(mask)[-1]];curves.append({'start':str(d['start'][j]),'values':[float(v/native) if ok else None for v,ok in zip(d['errors'][j],mask)]})
  representatives.append({'label':label,'cell':str(d['cell'][i]),'item':str(d['item'][i]),'train_stability':float(s[pos]),'windows':np.exp2(LOGW).astype(int).tolist(),'curves':curves})
 report['representatives']=representatives;Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
