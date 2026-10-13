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
state={'phase':'PREFLIGHT','started_at':time.time()};save(A/'status.json',state);server=None
try:
 assert torch.cuda.is_available() and torch.cuda.device_count()==2,'GPU_UNAVAILABLE'
 for i in range(2):
  x=torch.ones(2,2,device='cuda:'+str(i));x@x;torch.cuda.synchronize(i)
 paths={n:A/(n+'.txt') for n in ['p0','minimal','format_only']}
 expected=json.loads((A/'setup.json').read_text())['prompt_sha256']
 for k,p in paths.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[k]
 prompts={k:p.read_text().strip() for k,p in paths.items()};config=json.loads((O/'configs/inference_config.json').read_text());assert config['backend']=='openai_compatible'
 inputs=load(R/'analysis/final_score_maximization/inputs192.jsonl');rows=load(R/'analysis/final_score_maximization/fast192.jsonl');assert len(inputs)==len(rows)==192
 # Verify previously fixed history inputs against the mini-dev, without reading full validation.
 for q,r in zip(inputs,rows):assert build_messages({**q,'id':None})[1]['content']==r['messages'][1]['content']
 pp=StylePostprocessor();assert pp.loaded and pp.personality_loaded
 env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1');env['PATH']='/opt/conda/envs/conda_env/bin:'+env['PATH']
 cmd=['/opt/conda/envs/conda_env/bin/vllm','serve',str(O/'model/Qwen2___5-7B-Instruct'),'--host','127.0.0.1','--port','8000','--served-model-name','emotion-companion-lora','--enable-lora','--lora-modules','emotion-companion-lora='+str(O/'model/lora_adapter'),'--tensor-parallel-size','2','--max-num-seqs','8']
 log=(A/'server.log').open('w');server=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);state['server_pid']=server.pid;state['phase']='SERVER_START';save(A/'status.json',state)
 for _ in range(600):
  assert server.poll() is None,'SERVER_EXITED'
  try:
   urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).read();break
  except Exception:time.sleep(1)
 else:raise RuntimeError('SERVER_START_TIMEOUT')
 models=json.loads(urllib.request.urlopen('http://127.0.0.1:8000/v1/models').read());assert any(x['id']=='emotion-companion-lora' and x.get('parent') for x in models['data']),models
 save(A/'server_setup.json',{'prompt_sha256':expected,'command':cmd,'models':models,'input_sha256':hashlib.sha256((R/'analysis/final_score_maximization/inputs192.jsonl').read_bytes()).hexdigest(),'server_restart_count':0})


 names=['p0','minimal','format_only'];postprocessors={n:StylePostprocessor() for n in names};assert all(p.loaded and p.personality_loaded for p in postprocessors.values());N=192;dest=A/'predictions.jsonl';assert not dest.exists()
 for i,q in enumerate(inputs):
  order=names[i%3:]+names[:i%3]
  for name in order:
   assert server.poll() is None,'SERVER_EXITED'
   state.update(phase='INFERENCE',row_index=i,variant=name);save(A/'status.json',state)
   messages=build_messages(q);messages[0]['content']=prompts[name];start=time.perf_counter();raw=rt.request_chat_completion(config,messages,config['generation']);pred,failed=rt.parse_prediction(q['id'],raw);pre=copy.deepcopy(pred)
   if not failed:pred=postprocessors[name].apply(pred,q['history'])
   for key in ['id','emotion_label','response_text','memory_refs']:assert pre[key]==pred[key]
   assert pre['user_profile']['interests']==pred['user_profile']['interests']
   with dest.open('a') as f:f.write(json.dumps({'row_index':i,'variant':name,'prediction':pred,'parse_failed':failed,'latency_ms':(time.perf_counter()-start)*1000},ensure_ascii=False)+'\n')
  print(i+1,'/192 triples',flush=True)
 server.terminate();server.wait(timeout=60);log.close();server=None
 state['phase']='METRICS';save(A/'status.json',state)
 sys.path.insert(0,str(R));from training.evaluate_lora import sentence_bleu4,rouge_l_f1,classification_diagnostics
 from bert_score import score
 import numpy as np
 torch.set_num_threads(8);targets=[json.loads(q['messages'][2]['content']) for q in rows];gt=[q['emotion_label'] for q in targets];records=load(dest);assert len(records)==3*N;metrics={};preds={};per={}
 for name in names:
  rec=sorted([q for q in records if q['variant']==name],key=lambda q:q['row_index']);assert [q['row_index'] for q in rec]==list(range(N));pred=[q['prediction'] for q in rec];preds[name]=pred;text=[q['response_text'] for q in pred];ref=[q['response_text'] for q in targets]
  _,_,F=score(text,ref,model_type='/root/development/assets/bert-base-multilingual-cased',num_layers=9,lang='zh',rescale_with_baseline=False,device='cpu',batch_size=32,verbose=False)
  lat=[q['latency_ms'] for q in rec];m={'emotion':sum(q['emotion_label']==t for q,t in zip(pred,gt))/N,'macro_f1':classification_diagnostics([q['emotion_label'] for q in pred],gt)['f1'],'bleu':sum(sentence_bleu4(r,p) for r,p in zip(ref,text))/N,'rouge':sum(rouge_l_f1(r,p) for r,p in zip(ref,text))/N,'bert':float(F.mean()),'json_valid':1-sum(q['parse_failed'] for q in rec)/N,'nonempty':sum(bool(q.strip()) for q in text)/N,'avg_latency_ms':float(np.mean(lat)),'p95_latency_ms':float(np.percentile(lat,95)),'p1_fallback':postprocessors[name].fallback_count,'p2_fallback':postprocessors[name].personality_fallback_count}
  for d,k in [('personality_traits','personality'),('interests','interests'),('style','style')]:m[k]=sum(set(q['user_profile'][d])==set(t['user_profile'][d]) for q,t in zip(pred,targets))/N
  m['profile_mean']=sum(m[k] for k in ['personality','interests','style'])/3;m['per_class']={}
  for label in sorted(set(gt)):
   tp=sum(q['emotion_label']==label and t==label for q,t in zip(pred,gt));sp=gt.count(label);pc=sum(q['emotion_label']==label for q in pred);pr=tp/pc if pc else 0;re=tp/sp;m['per_class'][label]={'support':sp,'precision':pr,'recall':re,'f1':2*pr*re/(pr+re) if pr+re else 0}
  metrics[name]=m;save(A/'partial_metrics.json',metrics);print('METRICS',name,m['emotion'],flush=True)
 transitions={};deltas={}
 for name in names[1:]:
  wr=rw=0;changes=[]
  for i,(b,c,t) in enumerate(zip(preds['p0'],preds[name],gt)):
   bc=b['emotion_label']==t;cc=c['emotion_label']==t;wr+=not bc and cc;rw+=bc and not cc
   if b['emotion_label']!=c['emotion_label']:changes.append({'id':inputs[i]['id'],'gt':t,'before':b['emotion_label'],'after':c['emotion_label']})
  transitions[name]={'wrong_to_right':wr,'right_to_wrong':rw,'net':wr-rw,'changes':changes};deltas[name]={k:metrics[name][k]-metrics['p0'][k] for k in ['emotion','macro_f1','personality','interests','style','profile_mean','bleu','rouge','bert','avg_latency_ms','p95_latency_ms']}
 setup=json.loads((A/'setup.json').read_text());assert all(hashlib.sha256((O/p).read_bytes()).hexdigest()==h for p,h in setup['runtime_hashes'].items())
 result={'samples':N,'metrics':metrics,'delta_vs_p0':deltas,'emotion_transitions':transitions,'prompt_sha256':expected,'formal_runtime_modified':False,'blocking_issue':'NONE','elapsed_minutes':(time.time()-state['started_at'])/60};save(A/'result.json',result);state.update(phase='COMPLETED');save(A/'status.json',state)
except Exception as e:
 state.update(phase='BLOCKED',blocking_issue=str(e));save(A/'status.json',state);save(A/'result.json',{'completed':False,'formal_runtime_modified':False,'blocking_issue':str(e)});raise
finally:
 if server is not None and server.poll() is None:server.terminate();server.wait(timeout=60)
