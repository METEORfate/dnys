import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import json,re,sys,time,pathlib,hashlib,numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
R=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from training.evaluate_lora import sentence_bleu4,rouge_l_f1
A=R/'analysis/v2_rq1'
def load(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def turns(row):
 out=[]
 for m in row['messages']:
  if m['role']!='user':continue
  text=m['content']
  if text.startswith('对话历史:'):
   ms=list(re.finditer(r'(?m)^(user|assistant): ',text));assert ms
   out.extend({'role':q.group(1),'content':text[q.end():ms[i+1].start() if i+1<len(ms) else len(text)].strip()} for i,q in enumerate(ms))
  else:out.append(m)
 return out
def query(t,mode):
 u=[m['content'] for m in t if m['role']=='user'];assert u
 return u[-1] if mode=='LATEST_USER' else '\n\n'.join(('[USER_LATEST]' if i==len(u)-1 else '[USER]')+'\n'+v for i,v in enumerate(u))
def target(row):return json.loads(next(m['content'] for m in reversed(row['messages']) if m['role']=='assistant'))['response_text']
def h(t):return hashlib.sha256(json.dumps(t,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
tr=load(R/'training/data/train_sft.jsonl');va=load(R/'training/data/val_dev_sft.jsonl');bp=load(R/'training/eval_results/v2_e1_rerun2_valdev/prediction.jsonl');assert len(tr)==18657 and len(va)==len(bp)==1649
tt=[turns(x) for x in tr];vt=[turns(x) for x in va];ts={h(x) for x in tt};vs={h(x) for x in vt};tpre={h(x[:i]) for x in tt for i in range(1,len(x)+1)};vpre={h(x[:i]) for x in vt for i in range(1,len(x)+1)};overlap=len((ts & vpre)|(vs & tpre));assert overlap==0,'conversation overlap STOP'
refs=[target(x) for x in va];trainrefs=[target(x) for x in tr];baseline=[x['response_text'] for x in bp];retrieval={}
for mode in ['LATEST_USER','USER_HISTORY']:
 print('TFIDF',mode,flush=True);vec=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=150000,sublinear_tf=True,dtype=np.float32);X=vec.fit_transform([query(x,mode) for x in tt]);V=vec.transform([query(x,mode) for x in vt]);indices=[];sims=[]
 for start in range(0,len(va),32):
  scores=(V[start:start+32] @ X.T).toarray();ix=scores.argmax(axis=1);indices.extend(ix.tolist());sims.extend(scores[np.arange(len(ix)),ix].tolist())
 retrieval[mode]={'indices':indices,'similarities':np.array(sims),'responses':[trainrefs[i] for i in indices]};print('RETRIEVED',mode,'max',max(sims),flush=True)

def lexical(cands):return {'bleu':np.array([sentence_bleu4(r,c) for r,c in zip(refs,cands)]),'rouge':np.array([rouge_l_f1(r,c) for r,c in zip(refs,cands)])}
metrics={'BASELINE':lexical(baseline)}
for mode,d in retrieval.items():metrics[mode]=lexical(d['responses'])
print('LEXICAL_DONE',flush=True)
import torch
from bert_score import score
device='cpu'
if torch.cuda.is_available():
 try:torch.ones(1,device='cuda:0');torch.cuda.synchronize();device='cuda:0'
 except RuntimeError:device='cpu'
print('BERTSCORE_DEVICE',device,flush=True)
allc=baseline+retrieval['LATEST_USER']['responses']+retrieval['USER_HISTORY']['responses']
P,Q,F=score(allc,refs*3,model_type='/root/development/assets/bert-base-multilingual-cased',num_layers=9,lang='zh',rescale_with_baseline=False,device=device,batch_size=32,verbose=False)
f=F.cpu().numpy()
for i,mode in enumerate(['BASELINE','LATEST_USER','USER_HISTORY']):metrics[mode]['bert']=f[i*1649:(i+1)*1649].astype(float)
summary=lambda m:{k:float(v.mean()) for k,v in m.items()}
bas=summary(metrics['BASELINE']);best=max(retrieval,key=lambda mode:sum(summary(metrics[mode]).values())/3);d=retrieval[best];mm=metrics[best];sim=d['similarities']
# Conservative recognizable contact and explicit named-entity patterns. No NER.
contact=re.compile(r'(?:https?://|www\.)\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\d)1[3-9]\d{9}(?!\d)|(?<!\d)\d{14,18}[0-9Xx]?(?!\d)')
entities=re.compile(r'(?:我叫|名叫|叫我)[：: ]?([\u4e00-\u9fff]{2,4})(?=[，。！？、\s]|$)|([赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦许何吕施张孔曹严华金魏陶姜谢邹苏潘董袁邓罗叶刘黄][\u4e00-\u9fff]{0,2}(?:先生|女士|老师|医生|教授))|([\u4e00-\u9fffA-Za-z]{2,16}(?:大学|学院|医院|集团|银行|研究所|有限公司))')
def unsafe(resp,q):
 if contact.search(resp):return True
 for m in entities.finditer(resp):
  entity=next(x for x in m.groups() if x is not None)
  if entity not in q:return True
 return False
safety=np.array([not unsafe(resp,query(turn,'USER_HISTORY')) for resp,turn in zip(d['responses'],vt)])
buckets=[]
for lo,hi,name in [(.95,1.01,'>=0.95'),(.9,.95,'0.90-0.95'),(.85,.9,'0.85-0.90'),(.8,.85,'0.80-0.85'),(.75,.8,'0.75-0.80'),(-1,.75,'<0.75')]:
 mask=(sim>=lo)&(sim<hi);buckets.append({'bucket':name,'count':int(mask.sum()),'v2':{k:float(v[mask].mean()) if mask.any() else None for k,v in metrics['BASELINE'].items()},'retrieval':{k:float(v[mask].mean()) if mask.any() else None for k,v in mm.items()}})
gates=[]
for threshold in [.8,.85,.9,.92,.95,.97]:
 eligible=sim>=threshold;mask=eligible & safety;hybrid={k:np.where(mask,mm[k],metrics['BASELINE'][k]) for k in bas};vals=summary(hybrid);delta={k:vals[k]-bas[k] for k in bas};paired={}
 for metric in ['rouge','bert']:
  change=(mm[metric]-metrics['BASELINE'][metric])[mask];better=int((change>1e-6).sum());worse=int((change< -1e-6).sum());paired[metric]={'retrieval_better':better,'v2_better':worse,'equal':int((np.abs(change)<=1e-6).sum()),'net_better':better-worse}
 promote=delta['bleu']>=.003 and delta['rouge']>=.003 and delta['bert']>=0 and mask.sum()<=1649*.15
 gates.append({'threshold':threshold,'override_count':int(mask.sum()),'override_percent':float(mask.mean()*100),'metrics':vals,'delta':delta,'paired_overridden_samples':paired,'unsafe_filtered_count':int((eligible & ~safety).sum()),'promote':bool(promote)})
selected=max(gates,key=lambda g:(g['promote'],sum(g['delta'].values()),-g['override_count']))
out={'samples':1649,'train_samples':18657,'train_val_conversation_overlap':overlap,'overlap_method':'canonical complete histories and cross-set history-prefix hashes','baseline':bas,'retrieval':{mode:summary(metrics[mode]) for mode in retrieval},'best_query':best,'query_selection':'highest mean of BLEU/ROUGE/BERTScore; internal proxy only','similarity_buckets':buckets,'gates':gates,'best_threshold':selected,'unsafe_filtered_count':int((~safety).sum()),'result':'PROMOTE' if selected['promote'] else 'REJECT','device':device,'safety_filter_limit':'heuristic contact/explicit-name/institution regex; not comprehensive NER','holdout_used':False,'runtime_modified':False,'blocking_issue':'NONE'}
(A/'result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
with (A/'retrieval_rows.jsonl').open('w') as handle:
 for i in range(1649):handle.write(json.dumps({'row_index':i,**{mode:{'top1_index':int(rd['indices'][i]),'top1_similarity':float(rd['similarities'][i]),'retrieval_response':rd['responses'][i]} for mode,rd in retrieval.items()}},ensure_ascii=False)+'\n')
print('DONE',json.dumps(out,ensure_ascii=False),flush=True)
