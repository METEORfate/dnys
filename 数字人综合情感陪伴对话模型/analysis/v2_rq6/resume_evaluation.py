import pathlib,sys,os,json,time,subprocess,concurrent.futures
R=pathlib.Path(__file__).resolve().parents[2];A=R/'analysis/v2_rq6';PY='/opt/conda/envs/conda_env/bin/python';BP='/root/development/runtime_candidates/v2_p1_style/python_env/bin/python'
load=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def pipeline():
 state={'phase':'MASK_AUDIT','blocking_issue':'NONE','started_at':time.time()};dump(A/'status.json',state)
 env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false');env['PATH']='/opt/conda/envs/conda_env/bin:'+env['PATH']
 def run(cmd,name,gpu=None):
  e=env.copy()
  if gpu is not None:e['CUDA_VISIBLE_DEVICES']=str(gpu)
  with (A/name).open('w') as log:
   p=subprocess.Popen(cmd,cwd=R,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['child_pid']=p.pid;dump(A/'status.json',state);rc=p.wait()
  if rc:raise RuntimeError(name+' exit '+str(rc))
 try:
  if not (A/'mask_audit.json').exists():run([PY,str(A/'prepare_audit.py')],'prepare.log')
  audit=json.loads((A/'mask_audit.json').read_text());assert audit['status']=='PASS' and audit['zero_supervised_samples']==0
  run([PY,'-m','unittest','training.test_dual_objective'],'tests.log')
  state['phase']='GPU_PREFLIGHT';dump(A/'status.json',state)
  preflight=A/'dual_rank_preflight.py';preflight.write_text("import os,torch\nr=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(r);x=torch.ones(4,4,device='cuda:'+str(r));y=x@x;torch.cuda.synchronize(r);print('RANK_PASS',r,flush=True)\n")
  run(['/opt/conda/envs/conda_env/bin/torchrun','--standalone','--nproc_per_node=2',str(preflight)],'gpu_preflight.log')
  output=R/'training/outputs/v2-rq6-response-emotion-anchor'
  assert all((output/f'checkpoint-{n}'/'adapter_model.safetensors').exists() for n in [40,80,120,160])
  previous=json.loads((A/'prior_status.json').read_text());state['training_minutes']=previous['training_minutes'];state['phase']='FAST96';dump(A/'status.json',state)
  def worker(gpu):
   for step in [40,80,120,160][gpu::2]:
    dest=R/f'training/eval_results/v2_rq6_checkpoint_{step}_fast96'
    if not (dest/'summary.json').exists():
     run([PY,str(R/'training/evaluate_lora.py'),'--model-path',str(R/'participant/model/Qwen2___5-7B-Instruct'),'--adapter-path',str(output/f'checkpoint-{step}'),'--data-file',str(R/'analysis/fast_iter_mini_dev_96.jsonl'),'--output-dir',str(dest),'--limit','96'],f'eval_{step}.log',gpu)
    assert json.loads((dest/'summary.json').read_text())['samples']==96 and len(load(dest/'prediction.jsonl'))==96
  with concurrent.futures.ThreadPoolExecutor(2) as pool:
   jobs=[pool.submit(worker,i) for i in range(2)]
   for job in jobs:job.result()
  state['phase']='PROFILE_AND_BERTSCORE';dump(A/'status.json',state)
  for step in [40,80,120,160]:run([BP,str(R/'analysis/v2_rq5/run_module_transplant.py'),'post',str(R/f'training/eval_results/v2_rq6_checkpoint_{step}_fast96')],f'post_{step}.log',0)
  baseline=json.loads((R/'analysis/v2_rq4/result.json').read_text())['baseline'];bp=load(R/'training/eval_results/v2_rq4_baseline/prediction.jsonl');rows=load(R/'analysis/fast_iter_mini_dev_96.jsonl');assert len(bp)==len(rows)==96
  checks=[]
  for step in [40,80,120,160]:
   dest=R/f'training/eval_results/v2_rq6_checkpoint_{step}_fast96';m=json.loads((dest/'final_metrics.json').read_text());pred=load(dest/'prediction.jsonl');delta={k:m[k]-baseline[k] for k in ['emotion','profile_mean','bleu','rouge','bert_f1']};guard=round(m['emotion']*96)>=66 and m['profile_mean']>=baseline['profile_mean']-.007 and m['json_valid']==m['nonempty']==1;decision='REJECT'
   if guard:
    if delta['bleu']>=.006 and delta['rouge']>=.005 and delta['bert_f1']>=0:decision='STRONG_PROMOTE'
    elif delta['bleu']>=.003 and delta['rouge']>=.003 and delta['bert_f1']>=-.0005:decision='PROMOTE'
    elif delta['bleu']>=.002 and delta['rouge']>=.002 and delta['bert_f1']>=-.001:decision='PROMOTE_WEAK'
   wr=rw=0;reg=[];recall={}
   for i,(b,p,q) in enumerate(zip(bp,pred,rows)):
    gt=json.loads(q['messages'][2]['content'])['emotion_label'];bc=b['emotion_label']==gt;pc=p['emotion_label']==gt;wr+=not bc and pc;rw+=bc and not pc
    if bc and not pc:reg.append({'row_index':i,'sample_id':q.get('id') or f'fast96_row_{i:03d}','gt':gt,'baseline':b['emotion_label'],'candidate':p['emotion_label']})
   for label in ['anxiety','mixed','relaxed','fear','helplessness','sadness']:
    ids=[i for i,q in enumerate(rows) if json.loads(q['messages'][2]['content'])['emotion_label']==label];recall[label]={'support':len(ids),'baseline':sum(bp[i]['emotion_label']==label for i in ids)/len(ids) if ids else None,'candidate':sum(pred[i]['emotion_label']==label for i in ids)/len(ids) if ids else None}
   checks.append({'checkpoint':step,'metrics':m,'delta':delta,'guardrail_pass':guard,'decision':decision,'response_gain_score':sum(delta[k] for k in ['bleu','rouge','bert_f1']),'emotion_diagnostic':{'wrong_to_right':wr,'right_to_wrong':rw,'regressions':reg} if guard else None,'recall':recall})
  eligible=[q for q in checks if q['guardrail_pass']];best=max(eligible,key=lambda q:({'REJECT':0,'PROMOTE_WEAK':1,'PROMOTE':2,'STRONG_PROMOTE':3}[q['decision']],q['response_gain_score'])) if eligible else None
  result={'dataset':{'response_data_count':audit['response_data_count'],'emotion_anchor_count':audit['emotion_anchor_count'],'anchor_distribution':audit['anchor_distribution']},'mask_audit':audit,'baseline':baseline,'checkpoints':checks,'best':best,'result':best['decision'] if best else 'REJECT','training_steps':160,'training_minutes':state['training_minutes'],'blocking_issue':'NONE'};dump(A/'result.json',result);state['phase']='COMPLETED';state['result']=result['result'];dump(A/'status.json',state)
 except Exception as e:state['phase']='BLOCKED';state['blocking_issue']=str(e) or type(e).__name__;dump(A/'status.json',state);raise
if __name__=='__main__':pipeline()
