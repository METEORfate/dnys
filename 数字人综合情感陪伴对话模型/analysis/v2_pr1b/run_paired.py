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
 cmd=['/opt/conda/envs/conda_env/bin/vllm','serve',str(O/'model/Qwen2___5-7B-Instruct'),'--host','127.0.0.1','--port','8000','--served-model-name','emotion-companion-lora','--enable-lora','--lora-modules','emotion-companion-lora='+str(O/'model/lora_adapter'),'--tensor-parallel-size','2']
 log=(A/'server.log').open('w');server=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);state['server_pid']=server.pid;state['phase']='SERVER_START';save(A/'status.json',state)
 for _ in range(600):
  assert server.poll() is None,'SERVER_EXITED'
  try:
   urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).read();break
  except Exception:time.sleep(1)
 else:raise RuntimeError('SERVER_START_TIMEOUT')
 models=json.loads(urllib.request.urlopen('http://127.0.0.1:8000/v1/models').read());assert any(x['id']=='emotion-companion-lora' and x.get('parent') for x in models['data']),models
 save(A/'setup.json',{'prompt_sha256':expected,'command':cmd,'models':models,'input_sha256':hashlib.sha256((R/'analysis/v2_pr1/inputs96.jsonl').read_bytes()).hexdigest(),'server_restart_count':0})
 for passname in ['A','B']:
  dest=A/('pass_'+passname+'.jsonl');assert not dest.exists()
  for i,q in enumerate(inputs):
   order=['p0','v2'] if (i%2==0)==(passname=='A') else ['v2','p0']
   for name in order:
    assert server.poll() is None,'SERVER_EXITED'
    state.update(phase='PAIRED_INFERENCE',pass_name=passname,row_index=i,variant=name);save(A/'status.json',state)
    messages=build_messages(q);messages[0]['content']=prompts[name];start=time.perf_counter();raw=rt.request_chat_completion(config,messages,config['generation']);pred,failed=rt.parse_prediction(q['id'],raw);pre=copy.deepcopy(pred)
    if not failed:pred=pp.apply(pred,q['history'])
    for k in ['id','emotion_label','response_text','memory_refs']:assert pre[k]==pred[k]
    assert pre['user_profile']['interests']==pred['user_profile']['interests']
    with dest.open('a') as f:f.write(json.dumps({'row_index':i,'variant':name,'prediction':pred,'parse_failed':failed,'latency_ms':(time.perf_counter()-start)*1000},ensure_ascii=False)+'\n')
   print(passname,i+1,'/96',flush=True)
 assert pp.fallback_count==pp.personality_fallback_count==0
 state['phase']='METRICS';save(A/'status.json',state)
 server.terminate();server.wait(timeout=60);log.close();server=None
 sys.path.insert(0,str(R));from training.evaluate_lora import sentence_bleu4,rouge_l_f1,classification_diagnostics
 from bert_score import score
 torch.set_num_threads(8);targets=[json.loads(q['messages'][2]['content']) for q in rows];gt=[q['emotion_label'] for q in targets];metrics={};preds={}
 for passname in ['A','B']:
  records=load(A/('pass_'+passname+'.jsonl'));assert len(records)==192
  for name in ['p0','v2']:
   rec=sorted([q for q in records if q['variant']==name],key=lambda q:q['row_index']);assert [q['row_index'] for q in rec]==list(range(96));pred=[q['prediction'] for q in rec];key=name+'_'+passname;preds[key]=pred
   text=[q['response_text'] for q in pred];ref=[q['response_text'] for q in targets];_,_,F=score(text,ref,model_type='/root/development/assets/bert-base-multilingual-cased',num_layers=9,lang='zh',rescale_with_baseline=False,device='cpu',batch_size=32,verbose=False)
   m={'emotion':sum(q['emotion_label']==t for q,t in zip(pred,gt))/96,'macro_f1':classification_diagnostics([q['emotion_label'] for q in pred],gt)['f1'],'bleu':sum(sentence_bleu4(r,p) for r,p in zip(ref,text))/96,'rouge':sum(rouge_l_f1(r,p) for r,p in zip(ref,text))/96,'bert':float(F.mean()),'json_valid':1-sum(q['parse_failed'] for q in rec)/96,'nonempty':sum(bool(q.strip()) for q in text)/96}
   for d,k in [('personality_traits','personality'),('interests','interests'),('style','style')]:m[k]=sum(set(q['user_profile'][d])==set(t['user_profile'][d]) for q,t in zip(pred,targets))/96
   m['profile_mean']=sum(m[k] for k in ['personality','interests','style'])/3;metrics[key]=m;save(A/'partial_metrics.json',metrics)
 repeat={k:sum(a[k]==b[k] for a,b in zip(preds['p0_A'],preds['p0_B']))/96 for k in ['emotion_label','response_text','user_profile']};repeat['emotion_delta']=metrics['p0_B']['emotion']-metrics['p0_A']['emotion']
 transitions={};sets={}
 for pn in ['A','B']:
  wr=[];rw=[]
  for i,(p,v,t) in enumerate(zip(preds['p0_'+pn],preds['v2_'+pn],gt)):
   pc=p['emotion_label']==t;vc=v['emotion_label']==t
   if not pc and vc:wr.append(inputs[i]['id'])
   if pc and not vc:rw.append(inputs[i]['id'])
  transitions[pn]={'wrong_to_right':len(wr),'right_to_wrong':len(rw),'improvement_ids':wr,'regression_ids':rw};sets[pn]=(set(wr),set(rw))
 stablei=sorted(sets['A'][0]&sets['B'][0]);stabler=sorted(sets['A'][1]&sets['B'][1]);delta={k:sum(metrics['v2_'+pn][k]-metrics['p0_'+pn][k] for pn in ['A','B'])/2 for k in ['emotion','profile_mean','bleu','rouge','bert']}
 result='REJECT'
 if abs(repeat['emotion_delta'])>=3/96-1e-9:result='EXPERIMENT_UNSTABLE'
 elif delta['emotion']<=-2/96+1e-9 and len(stabler)>len(stablei):result='CONFIRMED_REJECT'
 elif abs(delta['emotion'])<=1/96+1e-9 and delta['bleu']>=.002 and delta['rouge']>=.002 and delta['bert']>=-.0005:result='PROMPT_CANDIDATE'
 save(A/'result.json',{'metrics':metrics,'repeatability':repeat,'transitions':transitions,'mean_delta':delta,'stable_improvement_ids':stablei,'stable_regression_ids':stabler,'historical_p0_67_reproduced':all(round(metrics['p0_'+pn]['emotion']*96)==67 for pn in ['A','B']),'result':result,'blocking_issue':'NONE','elapsed_minutes':(time.time()-state['started_at'])/60})
 state['phase']='COMPLETED';state['result']=result;save(A/'status.json',state)
except Exception as e:
 state['phase']='BLOCKED';state['blocking_issue']=str(e);save(A/'status.json',state);raise
finally:
 if server is not None and server.poll() is None:server.terminate();server.wait(timeout=60)
