#!/usr/bin/env python3
"""Build a clean instance-oracle dataset and regress continuous log2(window).

This is deliberately test-contaminated and is only a capacity/mechanics check.
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from gift_eval.data import Dataset as GiftEvalDataset
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from experiments import datasets_config
from experiments.evaluate_instance_windows import (_cell_metric_path,_ground_tree,_load_vector,discover_cells)
from experiments.predict_context_length import MambaContextLength
from experiments.test_window_ablation_gifteval_v5 import GiftEvalCache,_closest_horizon_idx,_prepare_predictor_inputs

WINDOWS=np.asarray((32,48,64,96,128,192,256,384,512,768,1024,1536,2048,2560,3072,4096,6144,8192))
LOGW=np.log2(WINDOWS); HORIZONS=(24,96,192)

class OracleTasks(Dataset):
 def __init__(self,x,h,lo,hi,target,errors,mask):
  self.x=torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(-1);self.h=torch.from_numpy(h).long()
  self.lo=torch.from_numpy(lo).float();self.hi=torch.from_numpy(hi).float();self.target=torch.from_numpy(target).float()
  self.errors=errors;self.mask=mask
 def __len__(self):return len(self.x)
 def __getitem__(self,i):return self.x[i],self.h[i],self.lo[i],self.hi[i],self.target[i],i

def build(root,out,rows_per_cell,seed):
 rng=np.random.RandomState(seed);ground=_ground_tree(root)
 specs={(d,str(t)):(n,u) for n,t,d,u in datasets_config.datasets_to_run()}
 X=[];H=[];LO=[];HI=[];TARGET=[];ERROR=[];MASK=[];SPLIT=[];CELL=[];manifest=[]
 for num,cell in enumerate(discover_cells(root,["Chronos2-Small"]),1):
  spec=specs.get((cell.dataset,str(cell.term)))
  if not spec:continue
  try:cache=GiftEvalCache(GiftEvalDataset(name=spec[0],term=cell.term,to_univariate=spec[1]),cell.dataset)
  except Exception as exc:print("skip",cell.dataset,exc);continue
  n=cache.n_total;e=np.full((n,len(WINDOWS)),np.nan);counts=np.zeros_like(e)
  for j,w in enumerate(WINDOWS):e[:,j],counts[:,j],_=_load_vector(_cell_metric_path(ground,cell,int(w)),n,"mae")
  available=np.isfinite(e)&(counts>0)&(e>=0);valid=np.flatnonzero(available.any(1))
  if not len(valid):continue
  pick=rng.choice(valid,size=min(rows_per_cell,len(valid)),replace=False);rng.shuffle(pick)
  x=_prepare_predictor_inputs([cache.contexts[i] for i in pick],8192)
  cell_native=[]
  for i in pick:
   av=available[i];err=e[i];native=err[np.flatnonzero(av)[-1]];cell_native.append(native)
  scale=max(float(np.median(cell_native)),1e-8)
  los=[];his=[];targets=[]
  for i in pick:
   av=available[i];err=e[i];best=float(np.min(err[av]));native=float(err[np.flatnonzero(av)[-1]])
   tolerance=max(.01*max(native,best),.001*scale)
   near=av&(err<=best+tolerance);logs=LOGW[near]
   los.append(logs.min());his.append(logs.max());targets.append(logs.max())
  m=len(pick);ntr=max(1,int(.6*m));nv=max(1,int(.2*m));split=np.array(["train"]*ntr+["val"]*nv+["test"]*(m-ntr-nv))
  X.append(x);H.append(np.full(m,_closest_horizon_idx(cache.horizon,list(HORIZONS)),np.int64));LO.append(los);HI.append(his);TARGET.append(targets)
  ERROR.append(e[pick]);MASK.append(available[pick]);SPLIT.append(split);CELL.extend([f"{cell.dataset}/{cell.term}"]*m)
  manifest.append({"cell":f"{cell.dataset}/{cell.term}","rows":m,"scale":scale,"horizon":cache.horizon})
  print(f"[{num}] {cell.dataset}/{cell.term}: {m}",flush=True)
 out.mkdir(parents=True,exist_ok=True)
 arrays={"contexts":np.concatenate(X),"horizon_idx":np.concatenate(H),"lower_log2":np.concatenate(LO).astype(np.float32),
         "upper_log2":np.concatenate(HI).astype(np.float32),"target_log2":np.concatenate(TARGET).astype(np.float32),
         "errors_mae":np.concatenate(ERROR).astype(np.float32),"available":np.concatenate(MASK),"split":np.concatenate(SPLIT),
         "cell":np.asarray(CELL)}
 for k,v in arrays.items():np.save(out/f"{k}.npy",v)
 (out/"manifest.json").write_text(json.dumps({"warning":"TEST CONTAMINATED", "windows":WINDOWS.tolist(),"cells":manifest,"n":len(arrays["split"])},indent=2)+"\n")
 return arrays

def load(out):return {k:np.load(out/f"{k}.npy") for k in ("contexts","horizon_idx","lower_log2","upper_log2","target_log2","errors_mae","available","split","cell")}

def pred_logw(raw):return 5.0+8.0*torch.sigmoid(raw.squeeze(1))

@torch.no_grad()
def evaluate(model,ds,loader,device):
 model.eval();preds=[];indices=[]
 for x,h,lo,hi,target,idx in loader:preds.append(pred_logw(model(x.to(device),h.to(device))[0]).cpu().numpy());indices.append(idx.numpy())
 pred=np.concatenate(preds);idx=np.concatenate(indices);errors=ds.errors[idx];mask=ds.mask[idx]
 action=[]
 for p,m in zip(pred,mask):
  candidates=np.flatnonzero(m);action.append(candidates[np.argmin(np.abs(LOGW[candidates]-p))])
 action=np.asarray(action);row=np.arange(len(idx));chosen=errors[row,action];native=np.array([e[np.flatnonzero(m)[-1]] for e,m in zip(errors,mask)])
 oracle=np.nanmin(np.where(mask,errors,np.nan),1)
 inside=(pred>=ds.lo[idx].numpy())&(pred<=ds.hi[idx].numpy())
 return {"n":len(idx),"mean_error_vs_native":float(np.mean(chosen/np.maximum(native,1e-8))),"mean_regret":float(np.mean((chosen-oracle)/np.maximum(oracle,1e-8))),
         "interval_accuracy":float(inside.mean()),"median_abs_log2_error":float(np.median(np.abs(pred-ds.target[idx].numpy()))),
         "improvement_rate":float(np.mean(chosen<native)),"harm5_rate":float(np.mean(chosen>1.05*native)),"native_rate":float(np.mean(action==np.array([np.flatnonzero(m)[-1] for m in mask])))}

def main():
 p=argparse.ArgumentParser();p.add_argument("--ablation-root",default="logs/experiments/window_ablation_gifteval");p.add_argument("--output-dir",required=True)
 p.add_argument("--rows-per-cell",type=int,default=32);p.add_argument("--epochs",type=int,default=30);p.add_argument("--device",default="cuda:0");p.add_argument("--seed",type=int,default=42);p.add_argument("--rebuild",action="store_true");a=p.parse_args()
 out=Path(a.output_dir);data=build(a.ablation_root,out,a.rows_per_cell,a.seed) if a.rebuild or not (out/"contexts.npy").exists() else load(out)
 sets={};loaders={}
 for s in ("train","val","test"):
  ix=np.flatnonzero(data["split"]==s);ds=OracleTasks(data["contexts"][ix],data["horizon_idx"][ix],data["lower_log2"][ix],data["upper_log2"][ix],data["target_log2"][ix],data["errors_mae"][ix],data["available"][ix]);sets[s]=ds;loaders[s]=DataLoader(ds,batch_size=32,shuffle=s=="train")
 model=MambaContextLength(8192,128,128,2,16,4,2,.1,.15,1,len(HORIZONS)).to(a.device);torch.nn.init.constant_(model.curve_head[-1].bias,2.0)
 opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3);best=1e9;state=None;history=[]
 for epoch in range(1,a.epochs+1):
  model.train();total=0
  for x,h,lo,hi,target,idx in loaders["train"]:
   pred=pred_logw(model(x.to(a.device),h.to(a.device))[0]);lo=lo.to(a.device);hi=hi.to(a.device);target=target.to(a.device)
   outside=F.smooth_l1_loss(pred,torch.clamp(pred.detach(),lo,hi));anchor=.1*F.smooth_l1_loss(pred,target);loss=outside+anchor
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=float(loss)
  val=evaluate(model,sets["val"],loaders["val"],a.device);history.append({"epoch":epoch,"loss":total/len(loaders["train"]),"val":val});print(history[-1],flush=True)
  score=val["median_abs_log2_error"]
  if score<best:best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(state);report={"warning":"DELIBERATELY TEST CONTAMINATED","train":evaluate(model,sets["train"],loaders["train"],a.device),"val":evaluate(model,sets["val"],loaders["val"],a.device),"test_same_cells":evaluate(model,sets["test"],loaders["test"],a.device),"n_by_split":{k:len(v) for k,v in sets.items()}}
 torch.save(state,out/"model.pt");(out/"report.json").write_text(json.dumps(report,indent=2)+"\n");(out/"history.json").write_text(json.dumps(history,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
