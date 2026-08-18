#!/usr/bin/env python3
"""Nested same-cell learning curve for the continuous window regressor."""

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.predict_context_length import MambaContextLength

WINDOWS = np.asarray((32,48,64,96,128,192,256,384,512,768,1024,1536,2048,2560,3072,4096,6144,8192))
LOGW = np.log2(WINDOWS)

def model():
 return MambaContextLength(8192,128,128,2,16,4,2,0.0,0.0,1,3)

@torch.no_grad()
def evaluate(net, data, ix, device):
 net.eval(); preds=[]
 for start in range(0,len(ix),64):
  j=ix[start:start+64];x=torch.from_numpy(np.ascontiguousarray(data['x'][j])).float().unsqueeze(-1).to(device);h=torch.from_numpy(data['h'][j]).long().to(device)
  preds.append((5+8*net(x,h)[0].squeeze(1)).cpu().numpy())
 pred=np.concatenate(preds);target=data['y'][ix];actions=[]
 for p,m in zip(pred,data['mask'][ix]):
  candidates=np.flatnonzero(m);actions.append(candidates[np.argmin(abs(LOGW[candidates]-p))])
 actions=np.asarray(actions);errors=data['errors'][ix];mask=data['mask'][ix];rows=np.arange(len(ix))
 chosen=errors[rows,actions];native=np.asarray([e[np.flatnonzero(m)[-1]] for e,m in zip(errors,mask)])
 oracle=np.nanmin(np.where(mask,errors,np.nan),axis=1)
 target_actions=np.argmin(abs(pred[:,None]-LOGW[None,:]),axis=1)
 true_actions=np.argmin(abs(target[:,None]-LOGW[None,:]),axis=1)
 return {'n':len(ix),'mae_log2':float(np.mean(abs(pred-target))),'median_ae_log2':float(np.median(abs(pred-target))),
  'grid_accuracy':float(np.mean(target_actions==true_actions)),'error_vs_native':float(np.mean(chosen/np.maximum(native,1e-8))),
  'regret_vs_oracle':float(np.mean((chosen-oracle)/np.maximum(oracle,1e-8))),'improvement_rate':float(np.mean(chosen<native))}

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--updates',type=int,default=600);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 root=Path(a.data_dir);data={'x':np.load(root/'contexts.npy'),'h':np.load(root/'horizon_idx.npy'),'y':np.load(root/'target_log2.npy'),
  'cell':np.load(root/'cell.npy'),'split':np.load(root/'split.npy'),'errors':np.load(root/'errors_mae.npy'),'mask':np.load(root/'available.npy')}
 cells=np.unique(data['cell']);ranked=sorted(cells,key=lambda c:(-len(np.unique(data['y'][data['cell']==c])),c))
 sizes=sorted(set([1,4,8,16,len(ranked)]));report={'ranked_cells':ranked,'updates':a.updates,'runs':[]}
 for ncell in sizes:
  torch.manual_seed(a.seed);np.random.seed(a.seed);selected=set(ranked[:ncell]);incell=np.asarray([c in selected for c in data['cell']])
  train_ix=np.flatnonzero(incell&(data['split']=='train'));test_ix=np.flatnonzero(incell&(data['split']=='test'))
  ds=TensorDataset(torch.from_numpy(np.ascontiguousarray(data['x'][train_ix])).float().unsqueeze(-1),torch.from_numpy(data['h'][train_ix]).long(),torch.from_numpy(((data['y'][train_ix]-5)/8).astype(np.float32)).unsqueeze(1))
  loader=DataLoader(ds,batch_size=min(32,len(ds)),shuffle=True,drop_last=False);net=model().to(a.device);opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=0)
  iterator=iter(loader);net.train()
  for step in range(1,a.updates+1):
   try:x,h,y=next(iterator)
   except StopIteration:iterator=iter(loader);x,h,y=next(iterator)
   pred=net(x.to(a.device),h.to(a.device))[0];loss=torch.nn.functional.mse_loss(pred,y.to(a.device));opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10);opt.step()
  row={'n_cells':ncell,'n_train':len(train_ix),'cells':ranked[:ncell],'train':evaluate(net,data,train_ix,a.device),'test_same_cells':evaluate(net,data,test_ix,a.device)}
  report['runs'].append(row);print(json.dumps(row),flush=True);del net;torch.cuda.empty_cache()
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
