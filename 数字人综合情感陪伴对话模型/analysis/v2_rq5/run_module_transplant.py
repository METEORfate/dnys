import pathlib,json,sys,os,re,time,subprocess,concurrent.futures,copy
R=pathlib.Path(__file__).resolve().parents[2];A=R/'analysis/v2_rq5';PY='/opt/conda/envs/conda_env/bin/python';BP='/root/development/runtime_candidates/v2_p1_style/python_env/bin/python'
BASE=R/'training/outputs/v2-e1-emotion-rebalance-fixed18657-rerun2';TARGET=R/'training/outputs/v2-rq3-response-coverage-recovery/checkpoint-80'
NAMES=['ATTN_ALL','MLP_ALL','LATE_HALF_ALL','LATE_HALF_ATTN','LATE_HALF_MLP','LATE_QUARTER_ALL']
load=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def prepare():
 import torch
 from safetensors.torch import load_file,save_file
 c=json.loads((BASE/'adapter_config.json').read_text());d=json.loads((TARGET/'adapter_config.json').read_text())
 for k in set(c)|set(d):assert (set(c[k])==set(d[k]) if k=='target_modules' else c.get(k)==d.get(k)),k
 layers=json.loads((R/'participant/model/Qwen2___5-7B-Instruct/config.json').read_text())['num_hidden_layers']
 x=load_file(str(BASE/'adapter_model.safetensors'));y=load_file(str(TARGET/'adapter_model.safetensors'))
 assert x.keys()==y.keys() and all(x[k].shape==y[k].shape for k in x)
 parsed={k:re.search(r'\.layers\.(\d+)\.(?:self_attn|mlp)\.([a-z_]+)\.lora_([AB])\.weight$',k) for k in x};assert all(parsed.values())
 assert {int(m[1]) for m in parsed.values()}==set(range(layers));assert all(torch.isfinite(v).all() for v in list(x.values())+list(y.values()))
 attn={'q_proj','k_proj','v_proj','o_proj'};mlp={'gate_proj','up_proj','down_proj'};total=sum(v.numel() for v in x.values());audit={}
 for name in NAMES:
  selected=[]
  for k,m in parsed.items():
   layer=int(m[1]);module=m[2]
   select=(name=='ATTN_ALL' and module in attn) or (name=='MLP_ALL' and module in mlp) or (name=='LATE_HALF_ALL' and layer>=layers//2) or (name=='LATE_HALF_ATTN' and layer>=layers//2 and module in attn) or (name=='LATE_HALF_MLP' and layer>=layers//2 and module in mlp) or (name=='LATE_QUARTER_ALL' and layer>=layers-layers//4)
   if select:selected.append(k)
  z={k:(y[k] if k in selected else x[k]).contiguous() for k in x};out=R/f'training/outputs/v2-rq5-{name.lower()}'
  if out.exists():
   saved=load_file(str(out/'adapter_model.safetensors'));assert all(torch.equal(saved[k],z[k]) and saved[k].dtype==z[k].dtype for k in z)
  else:
   out.mkdir();save_file(z,str(out/'adapter_model.safetensors'),metadata={'format':'pt'});(out/'adapter_config.json').write_text((BASE/'adapter_config.json').read_text());saved=load_file(str(out/'adapter_model.safetensors'))
  assert all(torch.equal(saved[k],x[k]) and saved[k].dtype==x[k].dtype for k in x if k not in selected)
  assert all(torch.equal(saved[k],y[k]) and saved[k].dtype==y[k].dtype for k in selected)
  assert all(torch.isfinite(v).all() for v in saved.values())
  audit[name]={'selected_key_count':len(selected),'total_lora_key_count':len(x),'selected_parameter_percent':100*sum(x[k].numel() for k in selected)/total,'unselected_tensor_and_dtype_exact':True,'nan_count':0,'inf_count':0,'path':str(out)}
 dump(A/'parameter_audit.json',{'compatible':True,'layer_count':layers,'half_start':layers//2,'quarter_start':layers-layers//4,'rank':c['r'],'alpha':c['lora_alpha'],'candidates':audit})
 return audit
def pipeline():
 state={'phase':'PARAMETER_AUDIT','blocking_issue':'NONE','started_at':time.time()};dump(A/'status.json',state)
 env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
 def run(cmd,log,gpu):
  e=env.copy();e['CUDA_VISIBLE_DEVICES']=str(gpu)
  with (A/log).open('w') as f:
   p=subprocess.Popen(cmd,cwd=R,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);code=p.wait()
  if code:raise RuntimeError(f'{log} exited {code}')
 try:
  audit=prepare();state['phase']='GPU_PREFLIGHT';dump(A/'status.json',state)
  import torch
  assert torch.cuda.is_available() and torch.cuda.device_count()==2,'GPU_UNAVAILABLE'
  for i in range(2):
   x=torch.ones(2,2,device=f'cuda:{i}');x@x;torch.cuda.synchronize(i)
  state['phase']='FAST96';dump(A/'status.json',state)
  def worker(gpu):
   for name in NAMES[gpu::2]:
    dest=R/f'training/eval_results/v2_rq5_{name.lower()}'
    if not (dest/'summary.json').exists():
     assert not dest.exists(),'incomplete output requires explicit recovery: '+str(dest)
     run([PY,str(R/'training/evaluate_lora.py'),'--model-path',str(R/'participant/model/Qwen2___5-7B-Instruct'),'--adapter-path',audit[name]['path'],'--data-file',str(R/'analysis/fast_iter_mini_dev_96.jsonl'),'--output-dir',str(dest),'--limit','96'],name+'.log',gpu)
    summary=json.loads((dest/'summary.json').read_text());assert summary['samples']==96 and len(load(dest/'prediction.jsonl'))==96
  with concurrent.futures.ThreadPoolExecutor(2) as pool:
   jobs=[pool.submit(worker,i) for i in range(2)]
   for job in jobs:job.result()
  state['phase']='PROFILE_AND_BERTSCORE';dump(A/'status.json',state)
  for name in NAMES:
   dest=R/f'training/eval_results/v2_rq5_{name.lower()}'
   if not (dest/'final_metrics.json').exists():run([BP,str(pathlib.Path(__file__)),'post',str(dest)],'post_'+name+'.log',0)
  base=json.loads((R/'analysis/v2_rq4/result.json').read_text())['baseline'];bp=load(R/'training/eval_results/v2_rq4_baseline/prediction.jsonl');rows=load(R/'analysis/fast_iter_mini_dev_96.jsonl');assert len(bp)==len(rows)==96
  checks=[]
  for name in NAMES:
   dest=R/f'training/eval_results/v2_rq5_{name.lower()}';m=json.loads((dest/'final_metrics.json').read_text());pred=load(dest/'prediction.jsonl');delta={k:m[k]-base[k] for k in ['emotion','profile_mean','bleu','rouge','bert_f1']}
   guard=round(m['emotion']*96)>=66 and m['profile_mean']>=base['profile_mean']-.007 and m['json_valid']==m['nonempty']==1;result='REJECT'
   if guard:
    if delta['bleu']>=.006 and delta['rouge']>=.005 and delta['bert_f1']>=0:result='STRONG_PROMOTE'
    elif delta['bleu']>=.003 and delta['rouge']>=.003 and delta['bert_f1']>=-.0005:result='PROMOTE'
    elif delta['bleu']>=.002 and delta['rouge']>=.002 and delta['bert_f1']>=-.001:result='PROMOTE_WEAK'
   diagnostic=None
   if round(m['emotion']*96)>=66:
    wr=rw=0;regressions=[]
    for i,(b,p,q) in enumerate(zip(bp,pred,rows)):
     gt=json.loads(q['messages'][2]['content'])['emotion_label'];bc=b['emotion_label']==gt;pc=p['emotion_label']==gt;wr+=not bc and pc;rw+=bc and not pc
     if bc and not pc:regressions.append({'row_index':i,'sample_id':q.get('id') or f'fast96_row_{i:03d}','gt':gt,'baseline':b['emotion_label'],'candidate':p['emotion_label']})
    diagnostic={'wrong_to_right':wr,'right_to_wrong':rw,'regressions':regressions,'sample_id_note':'row identifier used when source id absent'}
   checks.append({'name':name,'parameter_audit':audit[name],'metrics':m,'delta':delta,'guardrail_pass':guard,'response_gain_score':sum(delta[k] for k in ['bleu','rouge','bert_f1']),'decision':result,'emotion_diagnostic':diagnostic})
  eligible=[q for q in checks if q['guardrail_pass']];best=max(eligible,key=lambda q:({'REJECT':0,'PROMOTE_WEAK':1,'PROMOTE':2,'STRONG_PROMOTE':3}[q['decision']],q['response_gain_score'])) if eligible else None
  dump(A/'result.json',{'adapter_compatible':True,'layer_count':json.loads((A/'parameter_audit.json').read_text())['layer_count'],'baseline':base,'candidates':checks,'best':best,'result':best['decision'] if best else 'REJECT','blocking_issue':'NONE'})
  state['phase']='COMPLETED';state['result']=best['decision'] if best else 'REJECT';dump(A/'status.json',state)
 except Exception as exc:
  state['phase']='BLOCKED';state['blocking_issue']=str(exc) or type(exc).__name__;dump(A/'status.json',state);raise
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='post':
  sys.path.insert(0,str(R/'analysis/v2_rq4'));from run_interpolation import post;post(pathlib.Path(sys.argv[2]))
 else:pipeline()
