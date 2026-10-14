import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import json,re,sys,pathlib,time,numpy as np,random
R=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from training.evaluate_lora import sentence_bleu4,rouge_l_f1
A=R/'analysis/v2_rq2'
def load(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
rows=load(R/'training/data/val_dev_sft.jsonl');pred=load(R/'training/eval_results/v2_e1_rerun2_valdev/prediction.jsonl');assert len(rows)==len(pred)==1649
refs=[json.loads(next(m['content'] for m in reversed(x['messages']) if m['role']=='assistant'))['response_text'] for x in rows];base=[x['response_text'] for x in pred];assert all(refs)
boundary=re.compile('[。！？!?；]')
def apply(text,kind,value):
 if kind=='SENTENCE_LIMIT':
  matches=list(boundary.finditer(text))
  return text[:matches[value-1].end()] if len(matches)>=value else text
 if len(text)<=value:return text
 matches=list(boundary.finditer(text,0,value))
 return text[:matches[-1].end()] if matches else text
configs=[('SENTENCE_LIMIT',v) for v in [1,2,3,4]]+[('CHAR_BUDGET',v) for v in [80,120,160,200,240,300]]
candidates={'BASELINE':base};definition={}
for kind,value in configs:
 name=f'{kind}_{value}';candidates[name]=[apply(x,kind,value) for x in base];definition[name]={'type':kind,'value':value}
metrics={}
for name,texts in candidates.items():
 print('LEXICAL',name,flush=True);metrics[name]={'bleu':np.array([sentence_bleu4(r,c) for r,c in zip(refs,texts)]),'rouge':np.array([rouge_l_f1(r,c) for r,c in zip(refs,texts)])}
import torch
from bert_score import BERTScorer
device='cpu'
if torch.cuda.is_available():
 try:torch.ones(1,device='cuda:0');torch.cuda.synchronize();device='cuda:0'
 except RuntimeError:pass
scorer=BERTScorer(model_type='/root/development/assets/bert-base-multilingual-cased',num_layers=9,lang='zh',rescale_with_baseline=False,device=device,batch_size=32)
for name,texts in candidates.items():
 print('BERT',name,flush=True);P,Q,F=scorer.score(texts,refs,batch_size=32);metrics[name]['bert']=F.cpu().numpy().astype(float)
summary=lambda m:{k:float(v.mean()) for k,v in m.items()}
bas=summary(metrics['BASELINE']);results=[]
for name,definition_row in definition.items():
 texts=candidates[name];changed=np.array([a!=b for a,b in zip(texts,base)]);vals=summary(metrics[name]);delta={k:vals[k]-bas[k] for k in bas};paired={}
 for metric in ['rouge','bert']:
  d=metrics[name][metric]-metrics['BASELINE'][metric];better=int((d>1e-6).sum());worse=int((d< -1e-6).sum());paired[metric]={'better':better,'worse':worse,'equal':int((np.abs(d)<=1e-6).sum()),'net_better':better-worse}
 empties=sum(not x.strip() for x in texts);qualifies=delta['bleu']>=.002 and delta['rouge']>=.002 and delta['bert']>=-.001 and changed.mean()<=.4 and empties==0
 strong=qualifies and delta['bleu']>=.004 and delta['rouge']>=.004 and delta['bert']>=0
 results.append({'name':name,**definition_row,'changed_count':int(changed.sum()),'changed_percent':float(changed.mean()*100),'metrics':vals,'delta':delta,'paired':paired,'empty_response_count':empties,'result':'STRONG_PROMOTE' if strong else 'PROMOTE' if qualifies else 'REJECT'})
# Eligible rules first; otherwise show the most favorable BERT-protected response proxy as rejected diagnostic.
safe=[x for x in results if x['delta']['bert']>=-.001 and x['changed_percent']<=40 and x['empty_response_count']==0]
best=max(safe or results,key=lambda x:(x['result']!='REJECT',sum(x['delta'].values()),-x['changed_count']))
besttexts=candidates[best['name']];bl=np.array([len(x) for x in base]);gl=np.array([len(x) for x in refs]);nl=np.array([len(x) for x in besttexts]);ratio=bl/gl
stats=lambda x:{'mean':float(x.mean()),**{f'p{q}':float(np.percentile(x,q)) for q in [25,50,75,90,95]}}
buckets=[]
for lo,hi,label in [(-1,.75,'<=0.75'),(.75,1,'0.75-1.00'),(1,1.25,'1.00-1.25'),(1.25,1.5,'1.25-1.50'),(1.5,2,'1.50-2.00'),(2,float('inf'),'>2.00')]:
 mask=(ratio>lo)&(ratio<=hi);buckets.append({'bucket':label,'count':int(mask.sum()),'metrics':{k:float(v[mask].mean()) if mask.any() else None for k,v in metrics['BASELINE'].items()}})
indices=[i for i,(a,b) in enumerate(zip(base,besttexts)) if a!=b];rng=random.Random(42);chosen=rng.sample(indices,min(10,len(indices)));examples=[{'row_index':i,'original_length':len(base[i]),'new_length':len(besttexts[i]),'original_response_first120':base[i][:120],'new_response':besttexts[i],'ends_at_boundary':bool(boundary.fullmatch(besttexts[i][-1:])),'is_original_prefix':base[i].startswith(besttexts[i])} for i in chosen];assert all(x['ends_at_boundary'] and x['is_original_prefix'] for x in examples)
(A/'truncation_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))
out={'samples':1649,'length_diagnostic':{'gt':stats(gl),'v2':stats(bl),'mean_pred_gt_length_ratio':float(ratio.mean()),'median_pred_gt_length_ratio':float(np.median(ratio)),'ratio_of_mean_lengths':float(bl.mean()/gl.mean())},'length_ratio_buckets':buckets,'baseline':bas,'rules':results,'best_rule':best,'best_rule_length':stats(nl),'rule_definition':'Sentence: cut at nth terminal punctuation if present, otherwise unchanged. Budget: last terminal punctuation at or before budget, otherwise unchanged. Preserve punctuation; no lexical rewriting.','example_count':len(examples),'examples_path':str(A/'truncation_examples.json'),'holdout_used':False,'runtime_modified':False,'device':device,'blocking_issue':'NONE'}
(A/'result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));np.savez_compressed(A/'per_sample_metrics.npz',**{name+'_'+k:v for name,metric in metrics.items() for k,v in metric.items()});print('DONE',json.dumps(out,ensure_ascii=False),flush=True)
