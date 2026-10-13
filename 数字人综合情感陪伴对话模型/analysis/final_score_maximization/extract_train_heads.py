import os,sys,json,time,gc,random,copy,hashlib
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1');sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[2];A=Path(__file__).parent;OUT=R/'training/outputs/final-score-maximization';OUT.mkdir(exist_ok=True)
load=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
started=time.time();state={'phase':'FEATURE_MODEL_LOAD','started_at':started};save(A/'heads_status.json',state)
try:
 import numpy as np,torch
 from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
 from peft import PeftModel
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import f1_score
 torch.manual_seed(62);random.seed(62);np.random.seed(62)
 assert torch.cuda.is_available() and torch.cuda.device_count()==2,'GPU_UNAVAILABLE';torch.cuda.set_device(0)
 base='/root/participant/model/Qwen2___5-7B-Instruct';adapter=R/'training/outputs/v2-e1-emotion-rebalance-fixed18657-rerun2'
 tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True);quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
 model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,torch_dtype=torch.bfloat16,quantization_config=quant,device_map={'':'cuda:0'},attn_implementation='sdpa');model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False);model.eval()
 for parameter in model.parameters():parameter.requires_grad_(False)
 assert not any(p.requires_grad for p in model.parameters())
 decoder=model.get_base_model().model;dim=model.config.hidden_size;labels=['joy','gratitude','relaxed','care','pride','neutral','surprise','mixed','sadness','loneliness','anxiety','anger','fear','disgust','shame','helplessness'];data=load(R/'training/data/train_sft.jsonl');fast=load(A/'fast192.jsonl');assert len(data)==18657 and len(fast)==192
 X=np.lib.format.open_memmap(OUT/'train_features.npy',mode='w+',dtype='float16',shape=(len(data),dim*2));V=np.lib.format.open_memmap(OUT/'fast_features.npy',mode='w+',dtype='float16',shape=(192,dim*2));y=np.array([labels.index(json.loads(q['messages'][2]['content'])['emotion_label']) for q in data]);np.save(OUT/'train_labels.npy',y);truncated=0
 def feature(row):
  global truncated
  text=tokenizer.apply_chat_template(row['messages'][:2],tokenize=False,add_generation_prompt=False);ids=tokenizer.encode(text,add_special_tokens=False)
  if len(ids)>4096:truncated+=1;ids=ids[-4096:]
  end_token=tokenizer.convert_tokens_to_ids('<|im_end|>');end=max(i for i,v in enumerate(ids) if v==end_token);assert end>0
  t=torch.tensor([ids],device='cuda:0')
  with torch.no_grad():out=decoder(input_ids=t,use_cache=False,output_hidden_states=False,return_dict=True);hidden=out.last_hidden_state[0];f=torch.cat([hidden[end-1],hidden[max(0,end-128):end].mean(dim=0)]).float().cpu().numpy()
  assert np.isfinite(f).all();del t,out,hidden;return f
 feature_started=time.time()
 for i,q in enumerate(data):
  X[i]=feature(q)
  if (i+1)%100==0:
   X.flush();state.update(phase='TRAIN_FEATURES',completed=i+1,total=len(data),elapsed_minutes=(time.time()-started)/60);save(A/'heads_status.json',state);print('FEATURE',i+1,flush=True)
   if i==99:
    estimate=(time.time()-feature_started)*len(data)/100;save(A/'feature_eta.json',{'estimate_minutes':estimate/60,'head_and_eval_reserve_minutes':45})
    assert estimate+45*60<4*3600,'FEATURE_TIME_BUDGET_EXCEEDED'
  assert time.time()-started<3*3600,'FEATURE_TIME_BUDGET_EXCEEDED'
 fast_latencies=[]
 for i,q in enumerate(fast):
  stamp=time.perf_counter();V[i]=feature(q);torch.cuda.synchronize(0);fast_latencies.append((time.perf_counter()-stamp)*1000)
 X.flush();V.flush();del decoder,model;gc.collect();torch.cuda.empty_cache();state['phase']='HEAD_TRAINING';save(A/'heads_status.json',state)
 features=np.asarray(X,dtype=np.float32);train_ix,valid_ix=train_test_split(np.arange(len(y)),test_size=.2,random_state=62,stratify=y);mean=features[train_ix].mean(axis=0);std=features[train_ix].std(axis=0).clip(.01);features=(features-mean)/std;np.savez(OUT/'feature_scaler.npz',mean=mean,std=std)
 device='cuda:1';tx=torch.tensor(features[train_ix],device=device);ty=torch.tensor(y[train_ix],device=device);vx=torch.tensor(features[valid_ix],device=device);vy=torch.tensor(y[valid_ix],device=device);fx=torch.tensor((np.asarray(V,dtype=np.float32)-mean)/std,device=device);results={};probs={}
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
   torch.cuda.synchronize(1);stamp=time.perf_counter()
   with torch.no_grad():head(row.unsqueeze(0))
   torch.cuda.synchronize(1);latencies.append((time.perf_counter()-stamp)*1000)
  torch.save(best_state,OUT/(name+'_head.pt'));probs[name]=p.tolist();results[name]={'head_avg_latency_ms':float(np.mean(latencies)),'best_epoch':best_epoch,'train_accuracy':float(np.mean(tp==y[train_ix])),'internal_valid_accuracy':float(np.mean(vp==y[valid_ix])),'internal_macro_f1':float(f1_score(y[valid_ix],vp,labels=list(range(16)),average='macro',zero_division=0)),'history':history}
 save(A/'head_results.json',{'labels':labels,'heads':results,'feature_avg_latency_ms':float(np.mean(fast_latencies)),'frozen_backbone':True,'feature_definition':'last content token + mean of preceding 128 context tokens, final normalized decoder layer','max_feature_length':4096,'truncated_inputs':truncated,'train_count':len(y),'internal_train_count':len(train_ix),'internal_valid_count':len(valid_ix),'elapsed_minutes':(time.time()-started)/60});save(A/'fast_head_probabilities.json',{'labels':labels,'probabilities':probs});state['phase']='COMPLETED';save(A/'heads_status.json',state)
except Exception as e:state.update(phase='BLOCKED',blocking_issue=str(e));save(A/'heads_status.json',state);raise
