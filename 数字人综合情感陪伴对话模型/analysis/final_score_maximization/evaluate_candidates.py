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
 inputs=load(A/'inputs192.jsonl');rows=load(A/'fast192.jsonl');assert len(inputs)==len(rows)==192
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

 import re,statistics,numpy as np
 from bert_score import score
 sys.path.insert(0,str(R));from training.evaluate_lora import sentence_bleu4,rouge_l_f1,classification_diagnostics
 setup=json.loads((A/'setup.json').read_text());heads=json.loads((A/'head_results.json').read_text());hp=json.loads((A/'fast_head_probabilities.json').read_text());labels=hp['labels'];prob=hp['probabilities'];records=[];results={};allpreds={};targets=[json.loads(q['messages'][2]['content']) for q in rows]
 def rule(latest,b):
  if b in ['gratitude','pride'] and re.search(r'松了一口气|释然|轻松(?:多了|一些|了些)|没那么堵|心里.*石头.*轻|不用再.*猜',latest):return 'relaxed'
  if b=='fear' and re.search(r'万一|会不会|最担心',latest) and not re.search(r'医生|体检|确诊|癌|刀|打我|死亡|危险',latest):return 'anxiety'
  if b=='anxiety' and re.search(r'无能为力|无路可走|怎么选都不对|根本没办法|不管.*都没用',latest):return 'helplessness'
  return b
 def select(i,name,b):
  if name in ['linear','mlp']:return labels[int(np.argmax(prob[name][i]))]
  latest=[m['content'] for m in inputs[i]['history'] if m['role']=='user'][-1];rp=rule(latest,b)
  if name=='rule':return rp
  votes=[]
  for head in ['linear','mlp']:
   ps=np.array(prob[head][i]);order=ps.argsort();votes.append((labels[int(order[-1])],ps[order[-1]]>=.9 and ps[order[-1]]-ps[order[-2]]>=.5))
  if votes[0][0]==votes[1][0] and all(v[1] for v in votes) and (rp==b or rp==votes[0][0]):return votes[0][0]
  if rp!=b and any(v[0]==rp and v[1] for v in votes):return rp
  return b
 def generate(start,end):
  for i in range(start,end):
   assert server.poll() is None,'SERVER_EXITED';state.update(phase='FAST96_GENERATION' if end==96 else 'FAST192_GENERATION',row_index=i);save(A/'evaluation_status.json',state);q=inputs[i];st=time.perf_counter();raw=rt.request_chat_completion(config,build_messages(q),config['generation']);baseline,failed=rt.parse_prediction(q['id'],raw);baseline=pp.apply(baseline,q['history']) if not failed else baseline;bl=(time.perf_counter()-st)*1000
   messages=build_messages(q);messages[0]['content']=prompts['p0']+'\n\n'+setup['response_wrapper'];st=time.perf_counter();raw=rt.request_chat_completion(config,messages,config['generation']);d,df=rt.parse_prediction(q['id'],raw);dl=(time.perf_counter()-st)*1000
   rec={'row_index':i,'baseline':baseline,'baseline_parse_failed':failed,'baseline_latency':bl,'wrapper_response':d['response_text'],'wrapper_parse_failed':df,'wrapper_latency':dl};records.append(rec)
   with (A/'generation.jsonl').open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
   print('GENERATE',i+1,flush=True)
 def evaluate(N,names):
  torch.set_num_threads(8);ref=[q['response_text'] for q in targets[:N]];basepred=[q['baseline'] for q in records[:N]];baseline_text=[q['response_text'] for q in basepred];_,_,bf=score(baseline_text,ref,model_type='/root/development/assets/bert-base-multilingual-cased',num_layers=9,lang='zh',rescale_with_baseline=False,device='cpu',batch_size=32,verbose=False);bbert=bf.numpy()
  def measure(pred,bert,latency,parsefailed):
   gt=[q['emotion_label'] for q in targets[:N]];p=[q['emotion_label'] for q in pred];response=[q['response_text'] for q in pred];m={'emotion':sum(a==b for a,b in zip(p,gt))/N,'macro_f1':classification_diagnostics(p,gt)['f1'],'bleu':sum(sentence_bleu4(r,p) for r,p in zip(ref,response))/N,'rouge':sum(rouge_l_f1(r,p) for r,p in zip(ref,response))/N,'bert':float(np.mean(bert)),'avg_latency':float(np.mean(latency)),'p95_latency':float(np.percentile(latency,95)),'json_valid':1-sum(parsefailed)/N,'nonempty':sum(bool(s.strip()) for s in response)/N}
   for dim,key in [('personality_traits','personality'),('interests','interests'),('style','style')]:m[key]=sum(set(p['user_profile'][dim])==set(t['user_profile'][dim]) for p,t in zip(pred,targets[:N]))/N
   m['profile_mean']=sum(m[k] for k in ['personality','interests','style'])/3;return m
  b=measure(basepred,bbert,[q['baseline_latency'] for q in records[:N]],[q['baseline_parse_failed'] for q in records[:N]]);res={'baseline':b,'candidates':{}}
  for name in names:
   pred=copy.deepcopy(basepred);lat=[];failed=[]
   if name=='response_wrapper':
    for i,p in enumerate(pred):p['response_text']=records[i]['wrapper_response']
    _,_,F=score([q['response_text'] for q in pred],ref,model_type='/root/development/assets/bert-base-multilingual-cased',num_layers=9,lang='zh',rescale_with_baseline=False,device='cpu',batch_size=32,verbose=False);bert=F.numpy();lat=[q['baseline_latency']+q['wrapper_latency'] for q in records[:N]];failed=[q['baseline_parse_failed'] or q['wrapper_parse_failed'] for q in records[:N]]
   else:
    rule_latencies=[]
    for i,p in enumerate(pred):
     stamp=time.perf_counter();p['emotion_label']=select(i,name,p['emotion_label']);rule_latencies.append((time.perf_counter()-stamp)*1000)
    extra=float(np.mean(rule_latencies)) if name=='rule' else heads['feature_avg_latency_ms']+sum(heads['heads'][h]['head_avg_latency_ms'] for h in (['linear','mlp'] if name=='hybrid' else [name]));lat=[q['baseline_latency']+extra for q in records[:N]];bert=bbert;failed=[q['baseline_parse_failed'] for q in records[:N]]
   for p,q in zip(pred,basepred):
    for k in ['id','user_profile','memory_refs']:assert p[k]==q[k]
    assert (p['emotion_label']==q['emotion_label'] if name=='response_wrapper' else p['response_text']==q['response_text'])
   m=measure(pred,bert,lat,failed);delta={k:m[k]-b[k] for k in ['emotion','macro_f1','profile_mean','bleu','rouge','bert']};wr=rw=0;trans=[]
   for i,(p,q,t) in enumerate(zip(basepred,pred,targets[:N])):
    pc=p['emotion_label']==t['emotion_label'];qc=q['emotion_label']==t['emotion_label'];wr+=not pc and qc;rw+=pc and not qc
    if p['emotion_label']!=q['emotion_label']:trans.append({'id':inputs[i]['id'],'gt':t['emotion_label'],'baseline':p['emotion_label'],'candidate':q['emotion_label']})
   gain=delta['emotion']>=1/96-1e-9 or (delta['emotion']>=-1e-9 and delta['bleu']>=.002 and delta['rouge']>=.002 and delta['bert']>=-.0005);guard=delta['profile_mean']>=-.007 and m['json_valid']==1 and m['nonempty']==1
   res['candidates'][name]={'metrics':m,'delta':delta,'wrong_to_right':wr,'right_to_wrong':rw,'transitions':trans,'passes':bool(gain and guard)}
   if N==192:
    extra_gt=[q['emotion_label'] for q in targets[96:192]];bc=sum(p['emotion_label']==t for p,t in zip(basepred[96:],extra_gt));cc=sum(p['emotion_label']==t for p,t in zip(pred[96:],extra_gt));res['candidates'][name]['fresh96_emotion_delta']=(cc-bc)/96
    fresh={
     'bleu':sum(sentence_bleu4(t['response_text'],c['response_text'])-sentence_bleu4(t['response_text'],b['response_text']) for t,c,b in zip(targets[96:192],pred[96:],basepred[96:]))/96,
     'rouge':sum(rouge_l_f1(t['response_text'],c['response_text'])-rouge_l_f1(t['response_text'],b['response_text']) for t,c,b in zip(targets[96:192],pred[96:],basepred[96:]))/96,
     'bert':float(np.mean(bert[96:]-bbert[96:]))}
    res['candidates'][name]['fresh96_response_delta']=fresh
    res['candidates'][name]['fresh96_passes']=(cc-bc)>=1 or ((cc-bc)>=0 and fresh['bleu']>=.002 and fresh['rouge']>=.002 and fresh['bert']>=-.0005)
  save(A/('fast'+str(N)+'_result.json'),res);return res
 generate(0,96);state['phase']='FAST96_METRICS';save(A/'evaluation_status.json',state);first=evaluate(96,['linear','mlp','rule','hybrid','response_wrapper']);passed=[n for n,x in first['candidates'].items() if x['passes']];second=None
 if passed:
  generate(96,192);state['phase']='FAST192_METRICS';save(A/'evaluation_status.json',state);second=evaluate(192,passed)
 candidates=[n for n in passed if second['candidates'][n]['passes'] and second['candidates'][n]['fresh96_passes']] if second else []
 best=max(candidates,key=lambda n:(second['candidates'][n]['metrics']['emotion'],sum(second['candidates'][n]['delta'][k] for k in ['bleu','rouge','bert']))) if candidates else 'CURRENT'
 assert pp.fallback_count==pp.personality_fallback_count==0
 for p,h in setup['runtime_hashes'].items():assert hashlib.sha256((O/p).read_bytes()).hexdigest()==h
 save(A/'result.json',{'heads':heads,'fast96':first,'fast192':second,'best_candidate':best,'worth_full_valdev':best!='CURRENT','formal_runtime_modified':False,'blocking_issue':'NONE','elapsed_minutes':(time.time()-state['started_at'])/60});state.update(phase='COMPLETED',best_candidate=best);save(A/'evaluation_status.json',state)
except Exception as e:state.update(phase='BLOCKED',blocking_issue=str(e));save(A/'evaluation_status.json',state);raise
finally:
 if server is not None and server.poll() is None:server.terminate();server.wait(timeout=60)
