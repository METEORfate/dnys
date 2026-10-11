import os,json,random,time,argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from collections import Counter,defaultdict
ap=argparse.ArgumentParser();ap.add_argument('--mode');ap.add_argument('--gpu');a=ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
R=Path(__file__).resolve().parents[1];MP='/root/development/assets/bert-base-multilingual-cased';O=R/'analysis/v2_c23_dual_input';O.mkdir(exist_ok=True)
labels=['joy','gratitude','relaxed','care','pride','neutral','surprise','mixed','sadness','loneliness','anxiety','anger','fear','disgust','shame','helplessness'];li={x:i for i,x in enumerate(labels)}
random.seed(49);torch.manual_seed(49);torch.cuda.manual_seed_all(49);torch.set_num_threads(4)
tok=AutoTokenizer.from_pretrained(MP,local_files_only=True);dev='cuda:0'
def read(p):
 out=[]
 for line in open(p,encoding='utf8'):
  o=json.loads(line);ms=o['messages'];idx=max(i for i,m in enumerate(ms) if m['role']=='user');target=json.loads(next(m['content'] for m in reversed(ms) if m['role']=='assistant'));out.append((ms[:idx],ms[idx]['content'],li[target['emotion_label']]))
 return out
tr=read(R/'training/data/train_sft.jsonl');va=read(R/'training/data/val_dev_sft.jsonl');assert len(tr)==18657 and len(va)==1649
by=defaultdict(list)
for i,x in enumerate(tr):by[x[2]].append(i)
rng=random.Random(49);tiny=sum([rng.sample(by[i],8) for i in range(16)],[]);probe=sum([rng.sample(by[i],min(300,len(by[i]))) for i in range(16)],[])
def encode(row):
 prev,text,y=row;lt=tok.encode(text,add_special_tokens=False)
 if a.mode=='latest':
  if len(lt)>510:lt=lt[:160]+lt[-350:]
  return tok.build_inputs_with_special_tokens(lt),y
 if len(lt)>448:lt=lt[:128]+lt[-320:]
 marker=tok.encode('[USER_LATEST]',add_special_tokens=False);budget=512-tok.num_special_tokens_to_add(False)-len(marker)-len(lt);ctx=[]
 for m in reversed(prev):
  if m['role'] not in ['user','assistant']:continue
  z=tok.encode(('[USER_PREV]' if m['role']=='user' else '[ASSISTANT_PREV]')+'\n'+m['content'],add_special_tokens=False)
  if budget<=0:break
  take=z[-budget:];ctx=take+ctx;budget-=len(take)
  if len(take)<len(z):break
 result=tok.build_inputs_with_special_tokens(ctx+marker+lt);assert len(result)<=512
 return result,y
train=[encode(x) for x in tr];val=[encode(x) for x in va]
def batches(ix,data,shuffle=False):
 ix=list(ix)
 if shuffle:random.shuffle(ix)
 for k in range(0,len(ix),32):
  b=[data[i] for i in ix[k:k+32]];z=tok.pad({'input_ids':[x for x,y in b]},padding=True,return_tensors='pt').to(dev);y=torch.tensor([y for x,y in b],device=dev);yield z,y
def model():return AutoModelForSequenceClassification.from_pretrained(MP,num_labels=16,local_files_only=True,ignore_mismatched_sizes=True).float().to(dev)
def optimizer(m,e,h):return torch.optim.AdamW([{'params':m.base_model.parameters(),'lr':e},{'params':m.classifier.parameters(),'lr':h}],weight_decay=.01)
m=model();opt=optimizer(m,2e-5,2e-4);hist=[];first=None;step=0
for ep in range(30):
 m.train();losses=0;correct=0
 for z,y in batches(tiny,train,True):
  opt.zero_grad();logits=m(**z).logits;loss=torch.nn.functional.cross_entropy(logits,y);loss.backward();opt.step();step+=1;losses+=loss.item()*len(y);correct+=(logits.argmax(1)==y).sum().item()
 acc=correct/128;hist.append({'epoch':ep+1,'loss':losses/128,'accuracy':acc});print(a.mode,hist[-1],flush=True)
 if acc>=.9 and first is None:first=step
result={'mode':a.mode,'sanity_pass':hist[-1]['accuracy']>=.9,'final_train_accuracy':hist[-1]['accuracy'],'best_train_accuracy':max(x['accuracy'] for x in hist),'steps_to_90pct':first,'history':hist,'tiny_indices':tiny}
if result['sanity_pass']:
 del m,opt;torch.cuda.empty_cache();m=model();opt=optimizer(m,1e-5,3e-4);m.train()
 for z,y in batches(probe,train,True):
  opt.zero_grad();loss=torch.nn.functional.cross_entropy(m(**z).logits,y);loss.backward();opt.step()
 m.eval();ys=[];pr=[]
 with torch.no_grad():
  for z,y in batches(range(len(val)),val):pr.extend(m(**z).logits.argmax(1).cpu().tolist());ys.extend(y.cpu().tolist())
 fs=[]
 for k in range(16):
  tp=sum(x==k and q==k for x,q in zip(ys,pr));fp=sum(x!=k and q==k for x,q in zip(ys,pr));fn=sum(x==k and q!=k for x,q in zip(ys,pr));fs.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
 result.update(probe_val_accuracy=sum(x==q for x,q in zip(ys,pr))/len(ys),probe_macro_f1=sum(fs)/16,prediction_distribution=dict(Counter(labels[i] for i in pr)))
json.dump(result,open(O/(a.mode+'.json'),'w'),ensure_ascii=False,indent=2);print('DONE',result['sanity_pass'],flush=True)
