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
 paths={'p0':O/'configs/system_prompt_v1.txt'}
 expected={'p0':'1ade81131d0983a1224ddcb9b60bd2aaa6979f7a75ffa26cee711602b738d873'}
 for k,p in paths.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[k]
 prompts={k:p.read_text().strip() for k,p in paths.items()};config=json.loads((O/'configs/inference_config.json').read_text());assert config['backend']=='openai_compatible'
 inputs=load(R/'analysis/v2_pr1/inputs96.jsonl');rows=load(R/'analysis/fast_iter_mini_dev_96.jsonl');assert len(inputs)==len(rows)==96
 # Verify previously fixed history inputs against the mini-dev, without reading full validation.
 for q,r in zip(inputs,rows):assert build_messages({**q,'id':None})[1]['content']==r['messages'][1]['content']
 pp=StylePostprocessor();assert pp.loaded and pp.personality_loaded
 env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1');env['PATH']='/opt/conda/envs/conda_env/bin:'+env['PATH']
 cmd=['/opt/conda/envs/conda_env/bin/vllm','serve',str(O/'model/Qwen2___5-7B-Instruct'),'--host','127.0.0.1','--port','8000','--served-model-name','emotion-companion-lora','--enable-lora','--lora-modules','emotion-companion-lora='+str(O/'model/lora_adapter'),'rq3-response='+str(R/'training/outputs/v2-rq3-response-coverage-recovery/checkpoint-80'),'--max-loras','2','--tensor-parallel-size','2','--max-num-seqs','8']
 log=(A/'server.log').open('w');server=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);state['server_pid']=server.pid;state['phase']='SERVER_START';save(A/'status.json',state)
 for _ in range(600):
  assert server.poll() is None,'SERVER_EXITED'
  try:
   urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).read();break
  except Exception:time.sleep(1)
 else:raise RuntimeError('SERVER_START_TIMEOUT')
 models=json.loads(urllib.request.urlopen('http://127.0.0.1:8000/v1/models').read());assert all(any(x['id']==n and x.get('parent') for x in models['data']) for n in ['emotion-companion-lora','rq3-response']),models
 save(A/'server_setup.json',{'prompt_sha256':expected,'command':cmd,'models':models,'input_sha256':hashlib.sha256((R/'analysis/v2_pr1/inputs96.jsonl').read_bytes()).hexdigest(),'server_restart_count':0})

 import concurrent.futures,statistics
 from core.output_schema import EMOTIONS,PROFILE_LABELS
 pools=concurrent.futures.ThreadPoolExecutor(2);pp=StylePostprocessor();assert pp.loaded and pp.personality_loaded
 def request(name,messages):
  c=dict(config);c['api_model']='emotion-companion-lora' if name=='a' else 'rq3-response';start=time.perf_counter();raw=rt.request_chat_completion(c,messages,config['generation']);return raw,(time.perf_counter()-start)*1000
 for mode in ['sequential','concurrent']:
  dest=A/(mode+'.jsonl');assert not dest.exists()
  for i,q in enumerate(inputs):
   assert server.poll() is None,'SERVER_EXITED';state.update(phase='FAST96',mode=mode,row_index=i);save(A/'status.json',state);messages=build_messages(q);start=time.perf_counter()
   if mode=='sequential':ar,al=request('a',messages);br,bl=request('b',messages)
   else:
    f=pools.submit(request,'a',messages);g=pools.submit(request,'b',messages);ar,al=f.result();br,bl=g.result()
   baseline,af=rt.parse_prediction(q['id'],ar);bp,bf=rt.parse_prediction(q['id'],br);baseline=pp.apply(baseline,q['history']) if not af else baseline
   combined=copy.deepcopy(baseline);combined['response_text']=bp['response_text'];combined['memory_refs']=[]
   assert combined['emotion_label']==baseline['emotion_label'] and combined['user_profile']==baseline['user_profile']
   assert combined['memory_refs']==[] and combined['id']==baseline['id']
   with dest.open('a') as f:f.write(json.dumps({'row_index':i,'baseline':baseline,'combined':combined,'a_parse_failed':af,'b_parse_failed':bf,'a_request_ms':al,'b_request_ms':bl,'wall_ms':(time.perf_counter()-start)*1000},ensure_ascii=False)+'\n')
   print(mode,i+1,'/96',flush=True)
 pools.shutdown();assert pp.fallback_count==pp.personality_fallback_count==0
 server.terminate();server.wait(timeout=60);server=None;log.close();state['phase']='METRICS';save(A/'status.json',state)
 sys.path.insert(0,str(R));from training.evaluate_lora import sentence_bleu4,rouge_l_f1,classification_diagnostics
 from bert_score import score
 import numpy as np
 torch.set_num_threads(8);targets=[json.loads(q['messages'][2]['content']) for q in rows];gt=[q['emotion_label'] for q in targets];result={};baseline_metrics={}
 for mode in ['sequential','concurrent']:
  records=load(A/(mode+'.jsonl'));assert len(records)==96;metrics={};arrays={}
  for label,key in [('baseline','baseline'),('candidate','combined')]:
   pred=[q[key] for q in records];texts=[q['response_text'] for q in pred];refs=[q['response_text'] for q in targets];_,_,F=score(texts,refs,model_type='/root/development/assets/bert-base-multilingual-cased',num_layers=9,lang='zh',rescale_with_baseline=False,device='cpu',batch_size=32,verbose=False)
   b=np.array([sentence_bleu4(r,p) for r,p in zip(refs,texts)]);ro=np.array([rouge_l_f1(r,p) for r,p in zip(refs,texts)]);be=F.numpy();lat=[q['wall_ms'] if label=='candidate' else q['a_request_ms'] for q in records]
   m={'emotion':sum(q['emotion_label']==t for q,t in zip(pred,gt))/96,'macro_f1':classification_diagnostics([q['emotion_label'] for q in pred],gt)['f1'],'bleu':float(b.mean()),'rouge':float(ro.mean()),'bert':float(be.mean()),'avg_latency':statistics.mean(lat),'p50_latency':statistics.median(lat),'p95_latency':float(np.percentile(lat,95)),'mean_response_chars':statistics.mean(map(len,texts)),'nonempty':sum(bool(t.strip()) for t in texts)/96,'json_valid':sum(not q['a_parse_failed'] and (label=='baseline' or not q['b_parse_failed']) for q in records)/96,'memory_refs_valid':sum(q['memory_refs']==[] for q in pred)/96}
   for d,k in [('personality_traits','personality'),('interests','interests'),('style','style')]:m[k]=sum(set(q['user_profile'][d])==set(t['user_profile'][d]) for q,t in zip(pred,targets))/96
   m['profile_mean']=sum(m[k] for k in ['personality','interests','style'])/3;metrics[label]=m;arrays[label]={'rouge':ro,'bert':be}
  delta={k:metrics['candidate'][k]-metrics['baseline'][k] for k in ['emotion','profile_mean','bleu','rouge','bert']};paired={}
  for k in ['rouge','bert']:
   d=arrays['candidate'][k]-arrays['baseline'][k];better=int((d>1e-6).sum());worse=int((d< -1e-6).sum());paired[k]={'better':better,'worse':worse,'equal':96-better-worse,'net':better-worse}
  decision='REJECT';m=metrics['candidate']
  if delta['emotion']<-1/96-1e-9 or delta['profile_mean']<-.007 or m['json_valid']!=1 or m['nonempty']!=1:decision='IMPLEMENTATION_FAILURE'
  elif delta['bleu']>=.006 and delta['rouge']>=.005 and delta['bert']>=0:decision='STRONG_PROMOTE'
  elif delta['bleu']>=.003 and delta['rouge']>=.003 and delta['bert']>=-.0005:decision='PROMOTE'
  elif delta['bleu']>0 and delta['rouge']>0:decision='PROMOTE_WEAK'
  result[mode]={'metrics':metrics,'delta':delta,'paired':paired,'decision':decision,'a_request_avg_ms':statistics.mean(q['a_request_ms'] for q in records),'b_request_avg_ms':statistics.mean(q['b_request_ms'] for q in records),'latency_increase_percent':(m['avg_latency']/metrics['baseline']['avg_latency']-1)*100,'causal_classification_preserved':True};save(A/'partial_result.json',result)
 setup=json.loads((A/'setup.json').read_text());assert all(hashlib.sha256((O/p).read_bytes()).hexdigest()==h for p,h in setup['official_runtime_hashes'].items())
 rank={'IMPLEMENTATION_FAILURE':-1,'REJECT':0,'PROMOTE_WEAK':1,'PROMOTE':2,'STRONG_PROMOTE':3};best=max(result,key=lambda m:(rank[result[m]['decision']],sum(result[m]['delta'][k] for k in ['bleu','rouge','bert'])));save(A/'result.json',{'setup':setup,'modes':result,'best_mode':best,'decision':result[best]['decision'],'formal_runtime_modified':False,'blocking_issue':'NONE','elapsed_minutes':(time.time()-state['started_at'])/60});state['phase']='COMPLETED';state['result']=result[best]['decision'];save(A/'status.json',state)
except Exception as e:
 state['phase']='BLOCKED';state['blocking_issue']=str(e);save(A/'status.json',state);raise
finally:
 if server is not None and server.poll() is None:server.terminate();server.wait(timeout=60)
