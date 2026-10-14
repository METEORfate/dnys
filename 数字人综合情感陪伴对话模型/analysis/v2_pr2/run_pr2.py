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
 paths={'p0':O/'configs/system_prompt_v1.txt','v2':Path('/root/prompt_research_package/prompt_research/prompts/v2_system.txt')}
 expected={'p0':'1ade81131d0983a1224ddcb9b60bd2aaa6979f7a75ffa26cee711602b738d873','v2':'ac3056972e84776da4910ae1fa96e42a81477cc82815fefd06d8dcf599d1335b'}
 for k,p in paths.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[k]
 prompts={k:p.read_text().strip() for k,p in paths.items()};config=json.loads((O/'configs/inference_config.json').read_text());assert config['backend']=='openai_compatible'
 inputs=load(R/'analysis/v2_pr1/inputs96.jsonl');rows=load(R/'analysis/fast_iter_mini_dev_96.jsonl');assert len(inputs)==len(rows)==96
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
 save(A/'server_setup.json',{'prompt_sha256':expected,'command':cmd,'models':models,'input_sha256':hashlib.sha256((R/'analysis/v2_pr1/inputs96.jsonl').read_bytes()).hexdigest(),'server_restart_count':0})

 sys.path.insert(0,str(R));from training.evaluate_lora import sentence_bleu4,rouge_l_f1,classification_diagnostics
 from bert_score import score
 import statistics
 torch.set_num_threads(8);targets=[json.loads(q['messages'][2]['content']) for q in rows];gt=[q['emotion_label'] for q in targets]
 setup=json.loads((A/'setup.json').read_text());results={}
 def evaluate_records(records):
  metrics={};preds={}
  for name in ['p0','candidate']:
   rec=sorted([q for q in records if q['variant']==name],key=lambda q:q['row_index']);assert len(rec)==96;pred=[q['prediction'] for q in rec];preds[name]=pred
   text=[q['response_text'] for q in pred];ref=[q['response_text'] for q in targets];_,_,F=score(text,ref,model_type='/root/development/assets/bert-base-multilingual-cased',num_layers=9,lang='zh',rescale_with_baseline=False,device='cpu',batch_size=32,verbose=False)
   latency=sorted(q['latency_ms'] for q in rec)
   m={'emotion':sum(q['emotion_label']==t for q,t in zip(pred,gt))/96,'macro_f1':classification_diagnostics([q['emotion_label'] for q in pred],gt)['f1'],'bleu':sum(sentence_bleu4(r,p) for r,p in zip(ref,text))/96,'rouge':sum(rouge_l_f1(r,p) for r,p in zip(ref,text))/96,'bert':float(F.mean()),'json_valid':1-sum(q['parse_failed'] for q in rec)/96,'nonempty':sum(bool(q.strip()) for q in text)/96,'avg_latency':statistics.mean(latency),'p50_latency':statistics.median(latency),'p95_latency':latency[91]}
   for d,k in [('personality_traits','personality'),('interests','interests'),('style','style')]:m[k]=sum(set(q['user_profile'][d])==set(t['user_profile'][d]) for q,t in zip(pred,targets))/96
   m['profile_mean']=sum(m[k] for k in ['personality','interests','style'])/3
   m['recall']={l:sum(q['emotion_label']==l and t==l for q,t in zip(pred,gt))/gt.count(l) for l in ['anxiety','fear','helplessness','mixed','relaxed','sadness']};metrics[name]=m
  wr=[];rw=[];changes=[]
  for i,(p,c,t) in enumerate(zip(preds['p0'],preds['candidate'],gt)):
   pc=p['emotion_label']==t;cc=c['emotion_label']==t;item={'id':inputs[i]['id'],'gt':t,'p0':p['emotion_label'],'candidate':c['emotion_label']}
   if not pc and cc:wr.append(item)
   if pc and not cc:rw.append(item)
   if p['emotion_label']!=c['emotion_label']:changes.append(item)
  delta={k:metrics['candidate'][k]-metrics['p0'][k] for k in ['emotion','profile_mean','bleu','rouge','bert']}
  return {'metrics':metrics,'delta':delta,'wrong_to_right':wr,'right_to_wrong':rw,'changes':changes}
 def decide(d,m):
  guard=d['emotion']>=-1/96-1e-9 and d['profile_mean']>=-.007 and m['json_valid']==m['nonempty']==1 and not (d['bleu']<-.002 and d['rouge']<-.002)
  stable=d['bleu']>=-.002 and d['rouge']>=-.002 and d['bert']>=-.001
  if not guard:return 'REJECT'
  if (d['emotion']>=2/96-1e-9 and stable) or (d['emotion']>=-1e-9 and d['bleu']>=.003 and d['rouge']>=.003 and d['bert']>=-.0005):return 'STRONG_PROMOTE'
  if (d['emotion']>=1/96-1e-9 and stable) or (d['emotion']>=-1e-9 and d['bleu']>=.002 and d['rouge']>=.002 and d['bert']>=-.0005):return 'PROMOTE'
  if d['emotion']>=1/96-1e-9 or (d['bleu']>0 and d['rouge']>0 and max(d['bleu'],d['rouge'])>=.002 and d['bert']>=-.001):return 'PROMOTE_WEAK'
  return 'REJECT'
 def paired(k,roundnum):
  candidate=Path(setup['candidates'][k]['prompt_path']).read_text().strip();dest=A/(k+'_pass'+str(roundnum)+'.jsonl');assert not dest.exists()
  for i,q in enumerate(inputs):
   order=['p0','candidate'] if (i%2==0)==(roundnum==1) else ['candidate','p0']
   for name in order:
    assert server.poll() is None,'SERVER_EXITED'
    state.update(phase='PAIRED_INFERENCE',candidate=k,pass_number=roundnum,row_index=i,variant=name);save(A/'status.json',state)
    messages=build_messages(q);messages[0]['content']=prompts['p0'] if name=='p0' else candidate;start=time.perf_counter();raw=rt.request_chat_completion(config,messages,config['generation']);pred,failed=rt.parse_prediction(q['id'],raw);pre=copy.deepcopy(pred)
    if not failed:pred=pp.apply(pred,q['history'])
    for key in ['id','emotion_label','response_text','memory_refs']:assert pre[key]==pred[key]
    assert pre['user_profile']['interests']==pred['user_profile']['interests']
    with dest.open('a') as f:f.write(json.dumps({'row_index':i,'variant':name,'prediction':pred,'parse_failed':failed,'latency_ms':(time.perf_counter()-start)*1000},ensure_ascii=False)+'\n')
   print(k,roundnum,i+1,'/96',flush=True)
  state['phase']='METRICS';save(A/'status.json',state);res=evaluate_records(load(dest));res['decision']=decide(res['delta'],res['metrics']['candidate']);save(A/(k+'_pass'+str(roundnum)+'_metrics.json'),res);return res
 for k in ['a','b','c']:
  first=paired(k,1);results[k]={'first':first,'retest':None,'decision':first['decision']}
  if first['decision'] in ['PROMOTE','STRONG_PROMOTE']:
   second=paired(k,2);mean={d:(first['delta'][d]+second['delta'][d])/2 for d in first['delta']};mc={d:(first['metrics']['candidate'][d]+second['metrics']['candidate'][d])/2 for d in first['metrics']['candidate'] if isinstance(first['metrics']['candidate'][d],(int,float))};decision=decide(mean,mc)
   si=sorted({x['id'] for x in first['wrong_to_right']}&{x['id'] for x in second['wrong_to_right']});sr=sorted({x['id'] for x in first['right_to_wrong']}&{x['id'] for x in second['right_to_wrong']})
   results[k].update(retest=second,mean_delta=mean,decision=decision,stable_improvement_ids=si,stable_regression_ids=sr)
  save(A/'partial_result.json',results)
 assert pp.fallback_count==pp.personality_fallback_count==0
 for p,h in setup['runtime_hashes'].items():assert hashlib.sha256((O/p).read_bytes()).hexdigest()==h
 eligible=[k for k in results if results[k]['decision']!='REJECT'];winner=max(eligible,key=lambda k:((results[k].get('mean_delta') or results[k]['first']['delta'])['emotion'],sum((results[k].get('mean_delta') or results[k]['first']['delta'])[d] for d in ['bleu','rouge','bert']))) if eligible else None
 final=results[winner]['decision'] if winner else 'REJECT_ALL';save(A/'result.json',{'candidates':setup['candidates'],'results':results,'winner':winner,'final_decision':final,'prompt_optimization_exhausted':not eligible,'formal_runtime_modified':False,'blocking_issue':'NONE','elapsed_minutes':(time.time()-state['started_at'])/60,'tolerances':{'response_stable_bleu':-.002,'response_stable_rouge':-.002,'response_stable_bert':-.001}});state['phase']='COMPLETED';state['result']=final;save(A/'status.json',state)
except Exception as e:
 state['phase']='BLOCKED';state['blocking_issue']=str(e);save(A/'status.json',state);raise
finally:
 if server is not None and server.poll() is None:server.terminate();server.wait(timeout=60)
