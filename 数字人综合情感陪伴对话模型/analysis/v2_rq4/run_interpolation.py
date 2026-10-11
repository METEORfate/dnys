import pathlib,json,sys,os,copy,hashlib,subprocess,time,concurrent.futures
R=pathlib.Path(__file__).resolve().parents[2];A=R/'analysis/v2_rq4'
PY='/opt/conda/envs/conda_env/bin/python';BP='/root/development/runtime_candidates/v2_p1_style/python_env/bin/python'
BASE=R/'training/outputs/v2-e1-emotion-rebalance-fixed18657-rerun2';TARGET=R/'training/outputs/v2-rq3-response-coverage-recovery/checkpoint-80'
ALPHAS=[.20,.35,.50,.65,.80]
load=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def prepare():
 import torch
 from safetensors.torch import load_file,save_file
 assert torch.cuda.is_available() and torch.cuda.device_count()==2
 for i in range(2):
  x=torch.ones(2,2,device=f'cuda:{i}');x@x;torch.cuda.synchronize(i)
 c=json.loads((BASE/'adapter_config.json').read_text());d=json.loads((TARGET/'adapter_config.json').read_text())
 for k in set(c)|set(d):
  assert (set(c[k])==set(d[k]) if k=='target_modules' else c.get(k)==d.get(k)),k
 x=load_file(str(BASE/'adapter_model.safetensors'));y=load_file(str(TARGET/'adapter_model.safetensors'))
 assert x.keys()==y.keys() and all(x[k].shape==y[k].shape for k in x)
 assert all('lora_A' in k or 'lora_B' in k for k in x)
 assert all(torch.isfinite(v).all() for v in list(x.values())+list(y.values()))
 assert all(torch.equal(torch.lerp(x[k].float(),y[k].float(),0).to(x[k].dtype),x[k]) and torch.equal(torch.lerp(x[k].float(),y[k].float(),1).to(y[k].dtype),y[k]) for k in x)
 for alpha in ALPHAS:
  out=R/f'training/outputs/v2-rq4-interp-a{round(alpha*100):03d}';assert not out.exists(),str(out);out.mkdir()
  z={k:torch.lerp(x[k].float(),y[k].float(),alpha).contiguous() for k in x}
  assert all(torch.isfinite(v).all() for v in z.values());save_file(z,str(out/'adapter_model.safetensors'),metadata={'format':'pt'})
  (out/'adapter_config.json').write_text((BASE/'adapter_config.json').read_text())
 dump(A/'compatibility.json',{'compatible':True,'parameter_keys':len(x),'endpoint_parameters_exact':True,'rank':c['r'],'lora_alpha':c['lora_alpha'],'base':str(BASE),'target':str(TARGET)})
 sys.path.insert(0,str(R));from training.preprocess import build_samples
 mini=load(R/'analysis/fast_iter_mini_dev_96.jsonl');assert len(mini)==96
 key=lambda q:json.dumps(q['messages'],ensure_ascii=False,sort_keys=True,separators=(',',':'))
 wanted={key(q) for q in mini};found={}
 for raw in load(R/'训练-验证-数据集/val/val_public.jsonl'):
  for q,point in zip(build_samples(raw,False),raw['prediction_points']):
   k=key(q)
   if k in wanted:
    hist=[{'role':t['role'],'content':t['content']} for t in sorted(raw['turns'],key=lambda t:t['turn_id']) if t['turn_id']<=point['target_user_turn_id'] and t['turn_id']!=point['assistant_response_turn_id']]
    if k in found:assert hist==found[k]
    found[k]=hist
 assert len(found)==len(wanted)
 (A/'histories.jsonl').write_text(''.join(json.dumps({'row_index':i,'history':found[key(q)]},ensure_ascii=False)+'\n' for i,q in enumerate(mini)))
 import random
 indexes=sorted(random.Random(42).sample(range(96),3));(A/'sanity3.jsonl').write_text(''.join(json.dumps(mini[i],ensure_ascii=False)+'\n' for i in indexes));dump(A/'sanity_indices.json',indexes)
def post(dest):
 sys.path.insert(0,str(R));from training.evaluate_lora import exact_set_accuracy
 sys.path.insert(0,'/root/participant');from style_postprocessor import StylePostprocessor
 pp=StylePostprocessor();assert pp.loaded and pp.personality_loaded
 rows=load(R/'analysis/fast_iter_mini_dev_96.jsonl');pred=load(dest/'prediction.jsonl');hist=load(A/'histories.jsonl');assert len(rows)==len(pred)==len(hist)==96
 final=[]
 for q,h in zip(pred,hist):
  f=pp.apply(copy.deepcopy(q),h['history'])
  for k in ['id','emotion_label','response_text','memory_refs']:assert q[k]==f[k]
  assert q['user_profile']['interests']==f['user_profile']['interests'];final.append(f)
 assert pp.fallback_count==pp.personality_fallback_count==0
 targets=[json.loads(q['messages'][2]['content']) for q in rows];s=json.loads((dest/'summary.json').read_text());assert s['samples']==96
 metrics={'emotion':s['emotion_accuracy'],'bleu':s['bleu_4'],'rouge':s['rouge_l'],'json_valid':s['json_parse_success_rate'],'nonempty':s['response_text_nonempty_rate']}
 for dim,name in [('personality_traits','personality'),('interests','interests'),('style','style')]:metrics[name]=sum(exact_set_accuracy(f,t,dim) for f,t in zip(final,targets))/96
 metrics['profile_mean']=sum(metrics[k] for k in ['personality','interests','style'])/3
 from bert_score import score
 import torch
 P,Q,F=score([q['response_text'] for q in final],[q['response_text'] for q in targets],model_type='/root/development/assets/bert-base-multilingual-cased',num_layers=9,lang='zh',rescale_with_baseline=False,device='cuda:0',batch_size=32)
 metrics['bert_f1']=float(F.mean());dump(dest/'final_metrics.json',metrics)
 (dest/'final_prediction.jsonl').write_text(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in final))
def pipeline():
 status={'phase':'PREPARING','blocking_issue':'NONE','started_at':time.time()};dump(A/'status.json',status)
 env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
 def run(cmd,log,gpu):
  e=env.copy();e['CUDA_VISIBLE_DEVICES']=str(gpu)
  with (A/log).open('w') as f:
   p=subprocess.Popen(cmd,cwd=R,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   code=p.wait()
  if code:raise RuntimeError(f'{log} exit {code}')
 def evaluate(name,adapter,data,limit,gpu):
  dest=R/f'training/eval_results/v2_rq4_{name}';assert not dest.exists()
  run([PY,str(R/'training/evaluate_lora.py'),'--model-path',str(R/'participant/model/Qwen2___5-7B-Instruct'),'--adapter-path',str(adapter),'--data-file',str(data),'--output-dir',str(dest),'--limit',str(limit)],name+'.log',gpu)
  s=json.loads((dest/'summary.json').read_text());assert s['samples']==limit and s['json_parse_success_rate']==1 and s['response_text_nonempty_rate']==1
  return dest
 try:
  prepare();status['phase']='ENDPOINT_SANITY';dump(A/'status.json',status)
  with concurrent.futures.ThreadPoolExecutor(2) as pool:
   jobs=[pool.submit(evaluate,n,p,A/'sanity3.jsonl',3,i) for i,(n,p) in enumerate([('sanity_a000',BASE),('sanity_a100',TARGET)])]
   for job in jobs:job.result()
  status['phase']='FAST96';dump(A/'status.json',status)
  tasks=[('baseline',BASE)]+[(f'a{round(a*100):03d}',R/f'training/outputs/v2-rq4-interp-a{round(a*100):03d}') for a in ALPHAS]
  def worker(gpu):
   for name,p in tasks[gpu::2]:evaluate(name,p,R/'analysis/fast_iter_mini_dev_96.jsonl',96,gpu)
  with concurrent.futures.ThreadPoolExecutor(2) as pool:
   jobs=[pool.submit(worker,i) for i in range(2)]
   for job in jobs:job.result()
  status['phase']='PROFILE_AND_BERTSCORE';dump(A/'status.json',status)
  for name,p in tasks:run([BP,str(pathlib.Path(__file__)),'post',str(R/f'training/eval_results/v2_rq4_{name}')],'post_'+name+'.log',0)
  baseline=json.loads((R/'training/eval_results/v2_rq4_baseline/final_metrics.json').read_text());checks=[]
  for a in ALPHAS:
   m=json.loads((R/f'training/eval_results/v2_rq4_a{round(a*100):03d}/final_metrics.json').read_text());delta={k:m[k]-baseline[k] for k in ['emotion','profile_mean','bleu','rouge','bert_f1']}
   guard=round(m['emotion']*96)>=round(baseline['emotion']*96)-1 and m['profile_mean']>=baseline['profile_mean']-.007 and m['json_valid']==m['nonempty']==1
   result='REJECT'
   if guard:
    if delta['bleu']>=.006 and delta['rouge']>=.005 and delta['bert_f1']>=0:result='STRONG_PROMOTE'
    elif delta['bleu']>=.003 and delta['rouge']>=.003 and delta['bert_f1']>=-.0005:result='PROMOTE'
    elif delta['bleu']>=.002 and delta['rouge']>=.002 and delta['bert_f1']>=-.0005:result='PROMOTE_WEAK'
   checks.append({'alpha':a,'metrics':m,'delta':delta,'guardrail_pass':guard,'response_gain_score':sum(delta[k] for k in ['bleu','rouge','bert_f1']),'result':result})
  eligible=[q for q in checks if q['guardrail_pass']];best=max(eligible,key=lambda q:({'REJECT':0,'PROMOTE_WEAK':1,'PROMOTE':2,'STRONG_PROMOTE':3}[q['result']],q['response_gain_score'])) if eligible else None
  dump(A/'result.json',{'adapter_compatible':True,'baseline':baseline,'candidates':checks,'best':best,'result':best['result'] if best else 'REJECT','blocking_issue':'NONE','weak_bert_tolerance':-.0005})
  status['phase']='COMPLETED';status['result']=best['result'] if best else 'REJECT';dump(A/'status.json',status)
 except Exception as e:
  status['phase']='BLOCKED';status['blocking_issue']=str(e);dump(A/'status.json',status);raise
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='post':post(pathlib.Path(sys.argv[2]))
 else:pipeline()
