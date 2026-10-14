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
 paths={'p0':O/'configs/system_prompt_v1.txt','a':Path('/root/development/runtime_candidates/pr2_a/configs/system_prompt_v1.txt')}
 expected={'p0':'1ade81131d0983a1224ddcb9b60bd2aaa6979f7a75ffa26cee711602b738d873','a':'0e4d79a39c0c3a0ff70f0111bd21c76941ae89540dfed517ff32601b0a72da8a'}
 for k,p in paths.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[k]
 prompts={k:p.read_text().strip() for k,p in paths.items()};config=json.loads((O/'configs/inference_config.json').read_text());assert config['backend']=='openai_compatible'
 inputs=load(A/'inputs1649.jsonl');rows=load(R/'training/data/val_dev_sft.jsonl');assert len(inputs)==len(rows)==1649
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
 save(A/'server_setup.json',{'prompt_sha256':expected,'command':cmd,'models':models,'input_sha256':hashlib.sha256((A/'inputs1649.jsonl').read_bytes()).hexdigest(),'server_restart_count':0})

 postprocessors={n:StylePostprocessor() for n in ['p0','a']};assert all(p.loaded and p.personality_loaded for p in postprocessors.values());N=1649;dest=A/'paired_predictions.jsonl';assert not dest.exists()
 for i,q in enumerate(inputs):
  order=['p0','a'] if i%2==0 else ['a','p0'] # odd ordinal sample P0 first
  for name in order:
   assert server.poll() is None,'SERVER_EXITED'
   state.update(phase='PAIRED_INFERENCE',row_index=i,variant=name);save(A/'status.json',state)
   messages=build_messages(q);messages[0]['content']=prompts[name];start=time.perf_counter();raw=rt.request_chat_completion(config,messages,config['generation']);pred,failed=rt.parse_prediction(q['id'],raw);pre=copy.deepcopy(pred)
   if not failed:pred=postprocessors[name].apply(pred,q['history'])
   for key in ['id','emotion_label','response_text','memory_refs']:assert pre[key]==pred[key]
   assert pre['user_profile']['interests']==pred['user_profile']['interests']
   with dest.open('a') as f:f.write(json.dumps({'row_index':i,'variant':name,'prediction':pred,'parse_failed':failed,'latency_ms':(time.perf_counter()-start)*1000},ensure_ascii=False)+'\n')
  print(i+1,'/1649 pairs',flush=True)
 server.terminate();server.wait(timeout=60);log.close();server=None
 state['phase']='METRICS';save(A/'status.json',state)
 sys.path.insert(0,str(R));from training.evaluate_lora import sentence_bleu4,rouge_l_f1,classification_diagnostics
 from bert_score import score
 import numpy as np,statistics
 torch.set_num_threads(8);targets=[json.loads(q['messages'][2]['content']) for q in rows];gt=[q['emotion_label'] for q in targets];records=load(dest);assert len(records)==2*N;metrics={};per={};preds={}
 for name in ['p0','a']:
  rec=sorted([q for q in records if q['variant']==name],key=lambda q:q['row_index']);assert [q['row_index'] for q in rec]==list(range(N));pred=[q['prediction'] for q in rec];preds[name]=pred;text=[q['response_text'] for q in pred];ref=[q['response_text'] for q in targets]
  _,_,F=score(text,ref,model_type='/root/development/assets/bert-base-multilingual-cased',num_layers=9,lang='zh',rescale_with_baseline=False,device='cpu',batch_size=32,verbose=False)
  emo=np.array([q['emotion_label']==t for q,t in zip(pred,gt)],dtype=float);b=np.array([sentence_bleu4(r,p) for r,p in zip(ref,text)]);ro=np.array([rouge_l_f1(r,p) for r,p in zip(ref,text)]);bf=F.numpy().astype(float);lat=[q['latency_ms'] for q in rec]
  m={'emotion':float(emo.mean()),'macro_f1':classification_diagnostics([q['emotion_label'] for q in pred],gt)['f1'],'bleu':float(b.mean()),'rouge':float(ro.mean()),'bert':float(bf.mean()),'json_valid':1-sum(q['parse_failed'] for q in rec)/N,'nonempty':sum(bool(q.strip()) for q in text)/N,'avg_latency':statistics.mean(lat),'p50_latency':statistics.median(lat),'p95_latency':float(np.percentile(lat,95)),'p1_fallback':postprocessors[name].fallback_count,'p2_fallback':postprocessors[name].personality_fallback_count}
  pp=[]
  for d,k in [('personality_traits','personality'),('interests','interests'),('style','style')]:
   v=np.array([set(q['user_profile'][d])==set(t['user_profile'][d]) for q,t in zip(pred,targets)],dtype=float);pp.append(v);m[k]=float(v.mean())
  profile=np.mean(pp,axis=0);m['profile_mean']=float(profile.mean());classes={}
  for l in sorted(rt.PROFILE_LABELS) if False else sorted(set(gt)|{q['emotion_label'] for q in pred}):
   tp=sum(q['emotion_label']==l and t==l for q,t in zip(pred,gt));fp=sum(q['emotion_label']==l and t!=l for q,t in zip(pred,gt));fn=gt.count(l)-tp;p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0;classes[l]={'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0,'support':gt.count(l)}
  m['per_class']=classes;metrics[name]=m;per[name]={'emotion':emo,'profile_mean':profile,'bleu':b,'rouge':ro,'bert':bf};save(A/'partial_metrics.json',metrics)
 keys=['emotion','macro_f1','personality','interests','style','profile_mean','bleu','rouge','bert'];delta={k:metrics['a'][k]-metrics['p0'][k] for k in keys};delta['latency_percent']=(metrics['a']['avg_latency']/metrics['p0']['avg_latency']-1)*100
 rng=np.random.default_rng(61);ci={};boot={k:[] for k in per['p0']}
 for _ in range(100):
  ix=rng.integers(0,N,size=(100,N))
  for k in boot:boot[k].extend((per['a'][k]-per['p0'][k])[ix].mean(axis=1).tolist())
 for k,v in boot.items():ci[k]=np.percentile(v,[2.5,97.5]).tolist()
 wr=[];rw=[];changes=[];patterns={k:{'wrong_to_right':0,'right_to_wrong':0} for k in ['fear_to_anxiety','gratitude_pride_to_relaxed']}
 for i,(p,c,t) in enumerate(zip(preds['p0'],preds['a'],gt)):
  pc=p['emotion_label']==t;cc=c['emotion_label']==t;x={'id':inputs[i]['id'],'gt':t,'p0':p['emotion_label'],'a':c['emotion_label']}
  if not pc and cc:wr.append(x)
  if pc and not cc:rw.append(x)
  if p['emotion_label']!=c['emotion_label']:changes.append(x)
  if p['emotion_label']=='fear' and c['emotion_label']=='anxiety':patterns['fear_to_anxiety']['wrong_to_right']+=not pc and cc;patterns['fear_to_anxiety']['right_to_wrong']+=pc and not cc
  if p['emotion_label'] in ['gratitude','pride'] and c['emotion_label']=='relaxed':patterns['gratitude_pride_to_relaxed']['wrong_to_right']+=not pc and cc;patterns['gratitude_pride_to_relaxed']['right_to_wrong']+=pc and not cc
 paired={}
 for k in ['rouge','bert']:
  d=per['a'][k]-per['p0'][k];better=int((d>1e-6).sum());worse=int((d< -1e-6).sum());paired[k]={'better':better,'worse':worse,'equal':N-better-worse,'net':better-worse,'mean':float(d.mean()),'median':float(np.median(d))}
 decision='INCONCLUSIVE';response_ok=delta['bleu']>=0 and delta['rouge']>=0 and delta['bert']>=-.0005;positive=delta['emotion']>=.005 or delta['profile_mean']>=.005 or delta['bleu']>=.002 or delta['rouge']>=.002 or delta['bert']>=.002
 if delta['emotion']<-.005 or delta['profile_mean']<-.007 or (delta['bleu']<-.002 and delta['rouge']<-.002):decision='REJECT'
 elif delta['emotion']>=0 and delta['profile_mean']>=-.005 and response_ok and positive:
  decision='STRONG_PROMOTE' if delta['emotion']>0 and ci['emotion'][0]>=-.005 and delta['bert']>=0 else 'PROMOTE'
 setup=json.loads((A/'setup.json').read_text());assert all(hashlib.sha256((O/p).read_bytes()).hexdigest()==h for p,h in setup['runtime_hashes'].items())
 result={'samples':N,'metrics':metrics,'delta':delta,'bootstrap_ci95':ci,'bootstrap_seed':61,'bootstrap_iterations':10000,'transitions':{'wrong_to_right':len(wr),'right_to_wrong':len(rw),'net':len(wr)-len(rw),'improvements':wr,'regressions':rw,'changes':changes,'patterns':patterns},'response_paired':paired,'decision':decision,'holdout_recommended':decision in ['PROMOTE','STRONG_PROMOTE'],'formal_runtime_modified':False,'blocking_issue':'NONE','elapsed_minutes':(time.time()-state['started_at'])/60,'decision_interpretation':{'response_obvious_drop':-.002,'clear_emotion_profile_gain':.005,'clear_response_gain':.002,'emotion_ci_obvious_negative':-.005}};save(A/'result.json',result)
 (A/'emotion_changes.jsonl').write_text(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in changes));state['phase']='COMPLETED';state['result']=decision;save(A/'status.json',state)
except Exception as e:
 state['phase']='BLOCKED';state['blocking_issue']=str(e);save(A/'status.json',state);raise
finally:
 if server is not None and server.poll() is None:server.terminate();server.wait(timeout=60)
