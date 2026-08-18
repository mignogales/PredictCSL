#!/usr/bin/env python3
"""Compare temporal intra-series and held-out inter-series window prediction."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from gift_eval.data import Dataset as GiftEvalDataset
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from experiments import datasets_config
from experiments.evaluate_instance_windows import _cell_metric_path,_ground_tree,_load_vector,discover_cells
from experiments.predict_context_length import MambaContextLength
from experiments.test_window_ablation_gifteval_v5 import GiftEvalCache,_closest_horizon_idx,_prepare_predictor_inputs

WINDOWS=np.asarray((32,48,64,96,128,192,256,384,512,768,1024,1536,2048,2560,3072,4096,6144,8192));LOGW=np.log2(WINDOWS);HORIZONS=(24,96,192)

def build(root,max_cells,min_series,min_origins,max_series,max_origins):
 specs={(d,str(t)):(n,u) for n,t,d,u in datasets_config.datasets_to_run()};ground=_ground_tree(root);parts=[]
 for cell in discover_cells(root,['Chronos2-Small']):
  spec=specs.get((cell.dataset,str(cell.term)))
  if not spec:continue
  gd=GiftEvalDataset(name=spec[0],term=cell.term,to_univariate=spec[1]);cache=GiftEvalCache(gd,cell.dataset);meta=[]
  for inp,lbl in gd.test_data:
   if len(lbl['target'])>=cache.horizon:meta.append((str(inp.get('item_id','unknown')),str(lbl['start'])))
  if len(meta)!=cache.n_total:raise RuntimeError('metadata/cache alignment failed')
  counts={}
  for item,_ in meta:counts[item]=counts.get(item,0)+1
  keep_items=sorted(k for k,v in counts.items() if v>=min_origins)[:max_series]
  if len(keep_items)<min_series:continue
  e=np.full((cache.n_total,len(WINDOWS)),np.nan);cnt=np.zeros_like(e)
  for j,w in enumerate(WINDOWS):e[:,j],cnt[:,j],_=_load_vector(_cell_metric_path(ground,cell,int(w)),cache.n_total,'mae')
  avail=np.isfinite(e)&(cnt>0)&(e>=0);selected=[]
  for item in keep_items:
   candidates=[i for i,(it,_) in enumerate(meta) if it==item and avail[i].any()]
   selected.extend(sorted(candidates,key=lambda i:meta[i][1])[-max_origins:])
  idx=np.asarray(selected)
  if not len(idx):continue
  native=np.asarray([e[i,np.flatnonzero(avail[i])[-1]] for i in idx]);scale=max(float(np.median(native)),1e-8);target=[]
  for i,nat in zip(idx,native):
   best=float(np.min(e[i,avail[i]]));tol=max(.01*max(float(nat),best),.001*scale);target.append(LOGW[avail[i]&(e[i]<=best+tol)].max())
  parts.append({'cell':f'{cell.dataset}/{cell.term}','x':_prepare_predictor_inputs([cache.contexts[i] for i in idx],8192),'h':np.full(len(idx),_closest_horizon_idx(cache.horizon,list(HORIZONS))),
   'y':np.asarray(target,np.float32),'errors':e[idx].astype(np.float32),'mask':avail[idx],'item':np.asarray([meta[i][0] for i in idx]),'start':np.asarray([meta[i][1] for i in idx])})
  print('selected',parts[-1]['cell'],len(keep_items),'series',len(idx),'origins',flush=True)
  if len(parts)>=max_cells:break
 if not parts:raise RuntimeError('no suitable cells')
 out={k:np.concatenate([p[k] for p in parts]) for k in ('x','h','y','errors','mask','item','start')};out['cell']=np.concatenate([np.repeat(p['cell'],len(p['y'])) for p in parts]);return out

def splits(data,protocol,seed):
 train=[];test=[];rng=np.random.RandomState(seed)
 for cell in np.unique(data['cell']):
  ci=np.flatnonzero(data['cell']==cell);items=np.unique(data['item'][ci])
  if protocol=='intra':
   for item in items:
    ii=ci[data['item'][ci]==item];ii=ii[np.argsort(data['start'][ii])];train.extend(ii[:-1]);test.append(ii[-1])
  else:
   items=items.copy();rng.shuffle(items);n_test=max(1,int(round(.25*len(items))));held=set(items[:n_test])
   for i in ci:(test if data['item'][i] in held else train).append(i)
 return np.asarray(train),np.asarray(test)

def net():return MambaContextLength(8192,128,128,2,16,4,2,0.,0.,1,len(HORIZONS))
@torch.no_grad()
def evaluate(m,d,ix,device):
 m.eval();pred=[]
 for s in range(0,len(ix),64):
  j=ix[s:s+64];x=torch.from_numpy(np.ascontiguousarray(d['x'][j])).float().unsqueeze(-1).to(device);h=torch.from_numpy(d['h'][j]).long().to(device);pred.append((5+8*m(x,h)[0].squeeze(1)).cpu().numpy())
 p=np.concatenate(pred);y=d['y'][ix];actions=[]
 for q,mask in zip(p,d['mask'][ix]):c=np.flatnonzero(mask);actions.append(c[np.argmin(abs(LOGW[c]-q))])
 actions=np.asarray(actions);err=d['errors'][ix];mask=d['mask'][ix];row=np.arange(len(ix));chosen=err[row,actions];native=np.asarray([z[np.flatnonzero(mm)[-1]] for z,mm in zip(err,mask)])
 return {'n':len(ix),'mae_log2':float(np.mean(abs(p-y))),'median_ae_log2':float(np.median(abs(p-y))),'grid_accuracy':float(np.mean(np.argmin(abs(p[:,None]-LOGW),1)==np.argmin(abs(y[:,None]-LOGW),1))),'error_vs_native':float(np.mean(chosen/np.maximum(native,1e-8))),'improvement_rate':float(np.mean(chosen<native))}

def main():
 p=argparse.ArgumentParser();p.add_argument('--ablation-root',default='logs/experiments/window_ablation_gifteval');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--max-cells',type=int,default=8);p.add_argument('--min-series',type=int,default=4);p.add_argument('--min-origins',type=int,default=3);p.add_argument('--max-series',type=int,default=12);p.add_argument('--max-origins',type=int,default=8);p.add_argument('--updates',type=int,default=1200);p.add_argument('--seed',type=int,default=42);a=p.parse_args();d=build(a.ablation_root,a.max_cells,a.min_series,a.min_origins,a.max_series,a.max_origins);report={'cells':np.unique(d['cell']).tolist(),'protocols':{}}
 for protocol in ('intra','inter'):
  tr,te=splits(d,protocol,a.seed);torch.manual_seed(a.seed);m=net().to(a.device);ds=TensorDataset(torch.from_numpy(np.ascontiguousarray(d['x'][tr])).float().unsqueeze(-1),torch.from_numpy(d['h'][tr]).long(),torch.from_numpy(((d['y'][tr]-5)/8).astype(np.float32)).unsqueeze(1));loader=DataLoader(ds,batch_size=min(32,len(ds)),shuffle=True);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0);it=iter(loader);m.train()
  for _ in range(a.updates):
   try:x,h,y=next(it)
   except StopIteration:it=iter(loader);x,h,y=next(it)
   loss=torch.nn.functional.mse_loss(m(x.to(a.device),h.to(a.device))[0],y.to(a.device));opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10);opt.step()
  result={'n_train':len(tr),'n_test':len(te),'train':evaluate(m,d,tr,a.device),'test':evaluate(m,d,te,a.device),'per_cell':{}}
  for cell in np.unique(d['cell']):
   ci=te[d['cell'][te]==cell];result['per_cell'][cell]=evaluate(m,d,ci,a.device)
  report['protocols'][protocol]=result;print(protocol,json.dumps(result),flush=True);del m;torch.cuda.empty_cache()
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
