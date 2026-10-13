import os,sys,json,time,random
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[2];A=Path(__file__).parent;OUT=R/'training/outputs/final-score-maximization'
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
state={'phase':'HEAD_TRAINING','completed':18657,'total':18657,'fast_features_completed':192};started=json.loads((A/'resume_state.json').read_text())['original_started_at'];state['started_at']=started;save(A/'heads_status.json',state)
try:
 import numpy as np,torch
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import f1_score
 torch.manual_seed(62);random.seed(62);np.random.seed(62);assert torch.cuda.is_available(),'GPU_UNAVAILABLE'
 X=np.load(OUT/'train_features.npy',mmap_mode='r');V=np.load(OUT/'fast_features.npy',mmap_mode='r');y=np.load(OUT/'train_labels.npy');assert len(X)==len(y)==18657 and len(V)==192;assert np.isfinite(X).all() and np.isfinite(V).all();dim=X.shape[1]//2;truncated=None
 labels=['joy','gratitude','relaxed','care','pride','neutral','surprise','mixed','sadness','loneliness','anxiety','anger','fear','disgust','shame','helplessness'];eta=json.loads((A/'feature_eta.json').read_text());fast_latencies=[eta['estimate_minutes']*60000/18657]
 features=np.asarray(X,dtype=np.float32);train_ix,valid_ix=train_test_split(np.arange(len(y)),test_size=.2,random_state=62,stratify=y);mean=features[train_ix].mean(axis=0);std=features[train_ix].std(axis=0).clip(.01);features=(features-mean)/std;np.savez(OUT/'feature_scaler.npz',mean=mean,std=std)
 device='cuda:0';tx=torch.tensor(features[train_ix],device=device);ty=torch.tensor(y[train_ix],device=device);vx=torch.tensor(features[valid_ix],device=device);vy=torch.tensor(y[valid_ix],device=device);fx=torch.tensor((np.asarray(V,dtype=np.float32)-mean)/std,device=device);results={};probs={}
 for name in ['linear','mlp']:
  torch.manual_seed(62);head=torch.nn.Linear(dim*2,16) if name=='linear' else torch.nn.Sequential(torch.nn.Linear(dim*2,256),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(256,16));head=head.to(device);optimizer=torch.optim.AdamW(head.parameters(),lr=.001 if name=='mlp' else .003,weight_decay=.01);best=-1;stale=0;history=[]
  for epoch in range(30):
   head.train();order=torch.randperm(len(tx),device=device);losses=[]
   for batch in order.split(256):
    optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(head(tx[batch]),ty[batch]);loss.backward();optimizer.step();losses.append(loss.item())
   head.eval()
   with torch.no_grad():
    tp=torch.cat([head(b).argmax(-1) for b in tx.split(512)]);vp=torch.cat([head(b).argmax(-1) for b in vx.split(512)]);train_acc=(tp==ty).float().mean().item();valid_acc=(vp==vy).float().mean().item()
   history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'train_accuracy':train_acc,'internal_valid_accuracy':valid_acc});print('HEAD',name,history[-1],flush=True)
   if valid_acc>best:best=valid_acc;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};best_epoch=epoch+1;stale=0
   else:stale+=1
   if stale>=5:break
  head.load_state_dict(best_state);head.eval()
  with torch.no_grad():
   tp=torch.cat([head(b).argmax(-1) for b in tx.split(512)]).cpu().numpy();vp=torch.cat([head(b).argmax(-1) for b in vx.split(512)]).cpu().numpy();p=head(fx).softmax(-1).cpu().numpy()
  latencies=[]
  for row in fx[:96]:
   torch.cuda.synchronize(0);stamp=time.perf_counter()
   with torch.no_grad():head(row.unsqueeze(0))
   torch.cuda.synchronize(0);latencies.append((time.perf_counter()-stamp)*1000)
  torch.save(best_state,OUT/(name+'_head.pt'));probs[name]=p.tolist();results[name]={'head_avg_latency_ms':float(np.mean(latencies)),'best_epoch':best_epoch,'train_accuracy':float(np.mean(tp==y[train_ix])),'internal_valid_accuracy':float(np.mean(vp==y[valid_ix])),'internal_macro_f1':float(f1_score(y[valid_ix],vp,labels=list(range(16)),average='macro',zero_division=0)),'history':history}
 save(A/'head_results.json',{'labels':labels,'heads':results,'feature_avg_latency_ms':float(np.mean(fast_latencies)),'feature_latency_note':'Measured first-100 train feature timing proxy; not an end-to-end classifier runtime benchmark','frozen_backbone':True,'feature_definition':'last content token + mean of preceding 128 context tokens, final normalized decoder layer','max_feature_length':4096,'truncated_inputs':truncated,'train_count':len(y),'internal_train_count':len(train_ix),'internal_valid_count':len(valid_ix),'elapsed_minutes':(time.time()-started)/60});save(A/'fast_head_probabilities.json',{'labels':labels,'probabilities':probs});state['phase']='COMPLETED';save(A/'heads_status.json',state)
except Exception as e:state.update(phase='BLOCKED',blocking_issue=str(e));save(A/'heads_status.json',state);raise
