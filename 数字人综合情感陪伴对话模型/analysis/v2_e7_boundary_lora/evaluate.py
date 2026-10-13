import os,sys,json,time,subprocess,hashlib,urllib.request,copy
from pathlib import Path
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[2];A=Path(__file__).parent;O=Path('/root/participant')
load=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
sys.path.insert(0,str(O));import run_inference as rt
from style_postprocessor import StylePostprocessor
from core.message_builder import build_messages
import torch
state={'phase':'PREFLIGHT','started_at':time.time()};save(A/'evaluation_status.json',state);server=None
try:
 assert torch.cuda.is_available() and torch.cuda.device_count()==2,'GPU_UNAVAILABLE'
 for i in range(2):
  x=torch.ones(2,2,device='cuda:'+str(i));x@x;torch.cuda.synchronize(i)
 paths={'p0':O/'configs/system_prompt_v1.txt'}
 expected={'p0':json.loads((A/'setup.json').read_text())['runtime_hashes']['configs/system_prompt_v1.txt']}
 for k,p in paths.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[k]
 prompts={k:p.read_text().strip() for k,p in paths.items()};config=json.loads((O/'configs/inference_config.json').read_text());assert config['backend']=='openai_compatible'
 inputs=load(R/'analysis/final_score_maximization/inputs192.jsonl');rows=load(R/'analysis/final_score_maximization/fast192.jsonl');assert len(inputs)==len(rows)==192
 # Verify previously fixed history inputs against the mini-dev, without reading full validation.
 for q,r in zip(inputs,rows):assert build_messages({**q,'id':None})[1]['content']==r['messages'][1]['content']
 pp=StylePostprocessor();assert pp.loaded and pp.personality_loaded
 env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1');env['PATH']='/opt/conda/envs/conda_env/bin:'+env['PATH']
 cmd=['/opt/conda/envs/conda_env/bin/vllm','serve',str(O/'model/Qwen2___5-7B-Instruct'),'--host','127.0.0.1','--port','8000','--served-model-name','emotion-companion-lora','--enable-lora','--lora-modules','emotion-companion-lora='+str(O/'model/lora_adapter'),'e7-40='+str(R/'training/outputs/v2-e7-emotion-boundary/checkpoint-40'),'e7-80='+str(R/'training/outputs/v2-e7-emotion-boundary/checkpoint-80'),'--max-loras','3','--tensor-parallel-size','2','--max-num-seqs','8']
 log=(A/'server.log').open('w');server=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);state['server_pid']=server.pid;state['phase']='SERVER_START';save(A/'evaluation_status.json',state)
 for _ in range(600):
  assert server.poll() is None,'SERVER_EXITED'
  try:
   urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).read();break
  except Exception:time.sleep(1)
 else:raise RuntimeError('SERVER_START_TIMEOUT')
 models=json.loads(urllib.request.urlopen('http://127.0.0.1:8000/v1/models').read());assert any(x['id']=='emotion-companion-lora' and x.get('parent') for x in models['data']),models
 assert {'emotion-companion-lora','e7-40','e7-80'} <= {x['id'] for x in models['data'] if x.get('parent')},'LORA_REGISTRATION_FAILED'
 save(A/'server_setup.json',{'prompt_sha256':expected,'command':cmd,'models':models,'input_sha256':hashlib.sha256((R/'analysis/final_score_maximization/inputs192.jsonl').read_bytes()).hexdigest(),'server_restart_count':0})



 names=['p0','e7-40','e7-80'];pp={n:StylePostprocessor() for n in names};assert all(p.loaded and p.personality_loaded for p in pp.values());dest=A/'predictions.jsonl';assert not dest.exists()
 def generate(start,end,variants):
  for i in range(start,end):
   order=variants[i%len(variants):]+variants[:i%len(variants)]
   for name in order:
    assert server.poll() is None,'SERVER_EXITED';state.update(phase='FAST96_GENERATION' if end==96 else 'FAST192_GENERATION',row_index=i,variant=name);save(A/'evaluation_status.json',state)
    c=copy.deepcopy(config);c['api_model']='emotion-companion-lora' if name=='p0' else name;q=inputs[i];stamp=time.perf_counter();raw=rt.request_chat_completion(c,build_messages(q),c['generation']);pred,failed=rt.parse_prediction(q['id'],raw);pre=copy.deepcopy(pred)
    if not failed:pred=pp[name].apply(pred,q['history'])
    for key in ['id','emotion_label','response_text','memory_refs']:assert pre[key]==pred[key]
    assert pre['user_profile']['interests']==pred['user_profile']['interests']
    with dest.open('a') as f:f.write(json.dumps({'row_index':i,'variant':name,'prediction':pred,'parse_failed':failed,'latency_ms':(time.perf_counter()-stamp)*1000},ensure_ascii=False)+'\n')
   print('GENERATE',i+1,end,flush=True)
 import numpy as np
 from bert_score import score
 sys.path.insert(0,str(R));from training.evaluate_lora import sentence_bleu4,rouge_l_f1,classification_diagnostics
 torch.set_num_threads(8);targets=[json.loads(q['messages'][-1]['content']) for q in rows]
 setup=json.loads((A/'setup.json').read_text());guard=setup['guardrails']
 def evaluate(N,variants):
  state.update(phase='FAST'+str(N)+'_METRICS');save(A/'evaluation_status.json',state);records=load(dest);metrics={};preds={};refs=[q['response_text'] for q in targets[:N]];gt=[q['emotion_label'] for q in targets[:N]]
  for name in variants:
   rec=sorted([q for q in records if q['variant']==name and q['row_index']<N],key=lambda q:q['row_index']);assert [q['row_index'] for q in rec]==list(range(N));pred=[q['prediction'] for q in rec];preds[name]=pred;text=[q['response_text'] for q in pred]
   _,_,F=score(text,refs,model_type='/root/development/assets/bert-base-multilingual-cased',num_layers=9,lang='zh',rescale_with_baseline=False,device='cpu',batch_size=32,verbose=False)
   lat=[q['latency_ms'] for q in rec];m={'emotion':sum(q['emotion_label']==t for q,t in zip(pred,gt))/N,'macro_f1':classification_diagnostics([q['emotion_label'] for q in pred],gt)['f1'],'bleu':sum(sentence_bleu4(r,p) for r,p in zip(refs,text))/N,'rouge':sum(rouge_l_f1(r,p) for r,p in zip(refs,text))/N,'bert':float(F.mean()),'json_valid':1-sum(q['parse_failed'] for q in rec)/N,'nonempty':sum(bool(q.strip()) for q in text)/N,'avg_latency_ms':float(np.mean(lat)),'p95_latency_ms':float(np.percentile(lat,95)),'p1_fallback':pp[name].fallback_count,'p2_fallback':pp[name].personality_fallback_count}
   for d,k in [('personality_traits','personality'),('interests','interests'),('style','style')]:m[k]=sum(set(q['user_profile'][d])==set(t['user_profile'][d]) for q,t in zip(pred,targets[:N]))/N
   m['profile_mean']=sum(m[k] for k in ['personality','interests','style'])/3;m['recall']={l:{'support':gt.count(l),'recall':sum(q['emotion_label']==t==l for q,t in zip(pred,gt))/gt.count(l) if gt.count(l) else None} for l in ['anxiety','fear','helplessness','mixed','relaxed','gratitude']};metrics[name]=m;save(A/f'partial_fast{N}_metrics.json',metrics);print('METRICS',N,name,m['emotion'],m['macro_f1'],flush=True)
  result={'samples':N,'baseline':metrics['p0'],'checkpoints':{}}
  for name in variants[1:]:
   m=metrics[name];b=metrics['p0'];delta={k:m[k]-b[k] for k in ['emotion','macro_f1','personality','interests','style','profile_mean','bleu','rouge','bert']};wr=rw=0;changes=[]
   for i,(p,c,t) in enumerate(zip(preds['p0'],preds[name],gt)):
    bc=p['emotion_label']==t;cc=c['emotion_label']==t;wr+=not bc and cc;rw+=bc and not cc
    if p['emotion_label']!=c['emotion_label']:changes.append({'id':inputs[i]['id'],'gt':t,'before':p['emotion_label'],'after':c['emotion_label']})
   failures=[k for k in guard if delta[k.removesuffix('_delta_min')]<guard[k]-1e-9]
   if m['json_valid']<1 or m['nonempty']<1:failures.append('json_or_nonempty')
   passed=not failures and m['p1_fallback']==m['p2_fallback']==0
   result['checkpoints'][name]={'adapter_path':str(R/'training/outputs/v2-e7-emotion-boundary'/('checkpoint-'+name.split('-')[-1])),'metrics':m,'delta':delta,'wrong_to_right':wr,'right_to_wrong':rw,'net':wr-rw,'changes':changes,'guardrail_failures':failures,'passed':passed}
  save(A/f'fast{N}_result.json',result);return result
 generate(0,96,names);first=evaluate(96,names);survivors=[n for n in names[1:] if first['checkpoints'][n]['passed']];second=None
 if survivors:
  generate(96,192,['p0']+survivors);second=evaluate(192,['p0']+survivors)
 server.terminate();server.wait(timeout=60);log.close();server=None
 assert all(hashlib.sha256((O/p).read_bytes()).hexdigest()==h for p,h in setup['runtime_hashes'].items())
 eligible=[n for n in survivors if second['checkpoints'][n]['passed'] and second['checkpoints'][n]['delta']['emotion']>0] if second else []
 best=max(eligible,key=lambda n:(second['checkpoints'][n]['metrics']['emotion'],second['checkpoints'][n]['metrics']['macro_f1'])) if eligible else 'CURRENT'
 train=json.loads((A/'training_completion.json').read_text());result={'dataset':setup,'training_steps':80,'training_minutes':train['training_minutes'],'adapter_path':setup['output_adapter'],'fast96':first,'fast192':second,'best_candidate':best,'experiment_result':'CANDIDATE_FOR_REVIEW' if best!='CURRENT' else 'REJECT','automatic_promotion':False,'formal_runtime_modified':False,'blocking_issue':'NONE','evaluation_minutes':(time.time()-state['started_at'])/60};save(A/'result.json',result)
 lines=['# V2-E7 Emotion Boundary LoRA','',f"Result: {result['experiment_result']}",f'Best: {best}','FORMAL_RUNTIME_MODIFIED=NO','',f"Dataset: 2800 = 1960 boundary + 840 anchor; seed=66; unique original=2800",'Selection: train GT + fixed latest-user keyword boundary cues; not audited LLM errors. No relabeling or synthetic examples.',f"Distribution: {json.dumps(setup['distribution'],ensure_ascii=False)}",f"Adapter: {setup['output_adapter']}",f"Training: 80 steps, lr=1e-5, minutes={train['training_minutes']:.2f}; exact V2-E1 weights continued; full assistant-only CE.",'', '## Metrics']
 for stage,res in [('FAST96',first),('FAST192',second)]:
  lines+=['',f'### {stage}']
  if not res:lines+=['Not run: no checkpoint passed FAST96.'];continue
  lines+=['|adapter|emotion|macro_f1|personality|interests|style|profile_mean|BLEU|ROUGE|BERT|avg_ms|p95_ms|','|---|---|---|---|---|---|---|---|---|---|---|---|']
  for n,m in [('CURRENT',res['baseline'])]+[(n,c['metrics']) for n,c in res['checkpoints'].items()]:lines+=['|'+n+'|'+'|'.join(f'{m[k]:.6f}' for k in ['emotion','macro_f1','personality','interests','style','profile_mean','bleu','rouge','bert','avg_latency_ms','p95_latency_ms'])+'|']
  for n,c in res['checkpoints'].items():lines += [f"{n}: wrong->right {c['wrong_to_right']}, right->wrong {c['right_to_wrong']}, failed guards {c['guardrail_failures']}; recall {json.dumps(c['metrics']['recall'])}"]
 (A/'V2_E7_RESULT.md').write_text('\n'.join(lines)+'\n');state.update(phase='COMPLETED',best_candidate=best);save(A/'evaluation_status.json',state)
except Exception as e:
 state.update(phase='BLOCKED',blocking_issue=str(e));save(A/'evaluation_status.json',state);raise
finally:
 if server is not None and server.poll() is None:server.terminate();server.wait(timeout=60)
