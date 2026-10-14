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
 cmd=['/opt/conda/envs/conda_env/bin/vllm','serve',str(O/'model/Qwen2___5-7B-Instruct'),'--host','127.0.0.1','--port','8000','--served-model-name','emotion-companion-lora','--enable-lora','--lora-modules','emotion-companion-lora='+str(O/'model/lora_adapter'),'--tensor-parallel-size','2','--max-num-seqs','8']
 log=(A/'server.log').open('w');server=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);state['server_pid']=server.pid;state['phase']='SERVER_START';save(A/'status.json',state)
 for _ in range(600):
  assert server.poll() is None,'SERVER_EXITED'
  try:
   urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).read();break
  except Exception:time.sleep(1)
 else:raise RuntimeError('SERVER_START_TIMEOUT')
 models=json.loads(urllib.request.urlopen('http://127.0.0.1:8000/v1/models').read());assert any(x['id']=='emotion-companion-lora' and x.get('parent') for x in models['data']),models
 save(A/'setup.json',{'prompt_sha256':expected,'command':cmd,'models':models,'input_sha256':hashlib.sha256((R/'analysis/v2_pr1/inputs96.jsonl').read_bytes()).hexdigest(),'server_restart_count':0})

 from transformers import AutoTokenizer
 from core.output_schema import EMOTIONS
 import math,statistics
 labels=['joy','gratitude','relaxed','care','pride','neutral','surprise','mixed','sadness','loneliness','anxiety','anger','fear','disgust','shame','helplessness'];assert set(labels)==set(EMOTIONS)
 classifier_prompt='判断最新用户消息当前最主要的情绪。历史只用于理解上下文和消歧。必须从以下16个合法英文标签中选择一个：'+', '.join(labels)+'。只判断当前主导情绪，不生成回复、画像或解释。'
 (A/'classifier_prompt.txt').write_text(classifier_prompt);tokenizer=AutoTokenizer.from_pretrained(str(O/'model/Qwen2___5-7B-Instruct'),local_files_only=True);label_ids={l:tokenizer.encode(l,add_special_tokens=False) for l in labels};save(A/'label_tokens.json',label_ids)
 def score_labels(q):
  m=build_messages(q);m[0]['content']=classifier_prompt;prefix_text=tokenizer.apply_chat_template(m,tokenize=False,add_generation_prompt=True);prefix_ids=tokenizer.encode(prefix_text,add_special_tokens=False);sequences=[prefix_ids+label_ids[l] for l in labels]
  assert all(tokenizer.encode(prefix_text+l,add_special_tokens=False)==ids for l,ids in zip(labels,sequences)),'PREFIX_LABEL_TOKEN_BOUNDARY_MISMATCH'
  payload={'model':config['api_model'],'prompt':sequences,'max_tokens':0,'echo':True,'prompt_logprobs':0,'logprobs':0,'return_token_ids':True,'add_special_tokens':False,'temperature':0}
  req=urllib.request.Request('http://127.0.0.1:8000/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  start=time.perf_counter()
  with urllib.request.urlopen(req,timeout=600) as resp:data=json.loads(resp.read())
  latency=(time.perf_counter()-start)*1000;choices=sorted(data['choices'],key=lambda x:x['index']);assert len(choices)==16;out={}
  for l,ids,c in zip(labels,sequences,choices):
   assert c['prompt_token_ids']==ids;lp=c['prompt_logprobs'];assert len(lp)==len(ids)
   values=[]
   for j in range(len(prefix_ids),len(ids)):
    d=lp[j];entry=d.get(str(ids[j]),d.get(ids[j]));assert entry is not None,(l,j,d);v=entry['logprob'];assert math.isfinite(v);values.append(v)
   assert len(values)==len(label_ids[l])>0;out[l]={'token_ids':label_ids[l],'token_logprobs':values,'sum':sum(values),'mean':sum(values)/len(values)}
  return out,latency
 pp=StylePostprocessor();assert pp.loaded and pp.personality_loaded
 dest=A/'scores96.jsonl';assert not dest.exists()
 for i,q in enumerate(inputs):
  assert server.poll() is None,'SERVER_EXITED';state.update(phase='FAST96',row_index=i);save(A/'status.json',state)
  start=time.perf_counter();raw=rt.request_chat_completion(config,build_messages(q),config['generation']);baseline,failed=rt.parse_prediction(q['id'],raw);baseline=pp.apply(baseline,q['history']) if not failed else baseline;baseline_ms=(time.perf_counter()-start)*1000
  scores,extra=score_labels(q);preds={}
  for name in ['sum','mean']:
   ranking=sorted(labels,key=lambda l:scores[l][name],reverse=True);preds[name]={'label':ranking[0],'margin':scores[ranking[0]][name]-scores[ranking[1]][name],'top2':ranking[1]}
  with dest.open('a') as f:f.write(json.dumps({'row_index':i,'id':q['id'],'baseline':baseline,'parse_failed':failed,'scores':scores,'predictions':preds,'baseline_ms':baseline_ms,'scorer_ms':extra},ensure_ascii=False)+'\n')
  print(i+1,'/96',flush=True)
 assert pp.fallback_count==pp.personality_fallback_count==0
 # A single additional sample tests concurrent latency only; generation output is discarded.
 import concurrent.futures
 with concurrent.futures.ThreadPoolExecutor(2) as pool:
  start=time.perf_counter();f=pool.submit(rt.request_chat_completion,config,build_messages(inputs[0]),config['generation']);g=pool.submit(score_labels,inputs[0]);f.result();g.result();concurrent_ms=(time.perf_counter()-start)*1000
 server.terminate();server.wait(timeout=60);server=None;log.close();state['phase']='ANALYSIS';save(A/'status.json',state)
 sys.path.insert(0,str(R));from training.evaluate_lora import classification_diagnostics
 records=load(dest);assert len(records)==96;targets=[json.loads(q['messages'][2]['content']) for q in rows];gt=[q['emotion_label'] for q in targets];bp=[q['baseline']['emotion_label'] for q in records]
 def metrics(pred):
  wr=[];rw=[]
  for i,(b,c,t) in enumerate(zip(bp,pred,gt)):
   item={'id':records[i]['id'],'gt':t,'baseline':b,'candidate':c}
   if b!=t and c==t:wr.append(item)
   if b==t and c!=t:rw.append(item)
  return {'emotion':sum(p==t for p,t in zip(pred,gt))/96,'macro_f1':classification_diagnostics(pred,gt)['f1'],'wrong_to_right':len(wr),'right_to_wrong':len(rw),'improvements':wr,'regressions':rw,'agreement':sum(p==b for p,b in zip(pred,bp))/96}
 base=metrics(bp);stand={};gates=[]
 for name in ['sum','mean']:
  pred=[q['predictions'][name]['label'] for q in records];stand[name]=metrics(pred)
  for threshold in [0,.25,.5,.75,1,1.5]:
   hybrid=[q['predictions'][name]['label'] if q['predictions'][name]['label']!=b and q['predictions'][name]['margin']>=threshold else b for q,b in zip(records,bp)];m=metrics(hybrid);m.update(scoring=name,threshold=threshold,override_count=sum(p!=b for p,b in zip(hybrid,bp)),delta=m['emotion']-base['emotion'])
   # Only emotion may change. All response/profile/memory metrics are identical by construction.
   for rec,l in zip(records,hybrid):
    final=copy.deepcopy(rec['baseline']);final['emotion_label']=l;assert final['response_text']==rec['baseline']['response_text'] and final['user_profile']==rec['baseline']['user_profile'];assert rec['baseline']['memory_refs']==[];assert final['memory_refs']==[]
   gates.append(m)
 best=max(gates,key=lambda m:(m['emotion'],m['wrong_to_right']-m['right_to_wrong'],-m['override_count']));name=best['scoring'];threshold=best['threshold'];bestpred=[q['predictions'][name]['label'] if q['predictions'][name]['label']!=b and q['predictions'][name]['margin']>=threshold else b for q,b in zip(records,bp)]
 recall={l:{'support':gt.count(l),'baseline':sum(p==l and t==l for p,t in zip(bp,gt))/gt.count(l),'candidate':sum(p==l and t==l for p,t in zip(bestpred,gt))/gt.count(l)} for l in ['anxiety','fear','helplessness','mixed','relaxed','sadness','gratitude','pride']}
 decision='STRONG_PROMOTE' if round(best['delta']*96)>=2 else 'PROMOTE' if round(best['delta']*96)>=1 and best['right_to_wrong']<=best['wrong_to_right'] else 'REJECT'
 bms=statistics.mean(q['baseline_ms'] for q in records);sms=statistics.mean(q['scorer_ms'] for q in records)
 hashes=json.loads((R/'analysis/v2_pr2/setup.json').read_text())['runtime_hashes'];assert all(hashlib.sha256((O/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
 save(A/'result.json',{'gpu_health':'PASS','label_likelihood_supported':True,'classifier_prompt':classifier_prompt,'label_tokens':label_ids,'baseline':base,'standalone':stand,'gates':gates,'best':best,'recall':recall,'latency':{'baseline_ms':bms,'scorer_extra_ms':sms,'combined_ms':bms+sms,'increase_percent':sms/bms*100,'concurrent_single_sample_ms':concurrent_ms},'non_emotion_fields_identical':True,'decision':decision,'formal_runtime_modified':False,'blocking_issue':'NONE','elapsed_minutes':(time.time()-state['started_at'])/60});state['phase']='COMPLETED';state['result']=decision;save(A/'status.json',state)
except Exception as e:
 state['phase']='BLOCKED';state['blocking_issue']=str(e);save(A/'status.json',state);raise
finally:
 if server is not None and server.poll() is None:server.terminate();server.wait(timeout=60)
