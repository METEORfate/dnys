import os,json,sys,time,pathlib,subprocess,signal,hashlib
R=pathlib.Path(__file__).resolve().parents[2];A=R/'analysis/v2_rq3';PY='/opt/conda/envs/conda_env/bin/python';BERTPY='/root/development/runtime_candidates/v2_p1_style/python_env/bin/python'
env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false');env['PATH']='/opt/conda/envs/conda_env/bin:/opt/conda/bin:'+env['PATH']
status={'phase':'MASK_AUDIT','started_at':time.time(),'blocking_issue':'NONE'}
def save(): (A/'pipeline_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
def run(cmd,logname):
 with (A/logname).open('w') as log:
  child=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);status['child_pid']=child.pid;save();code=child.wait()
 if code:raise RuntimeError(logname+' exited '+str(code)+'; inspect saved log')
try:
 save()
 while not (A/'mask_audit.json').exists():
  try:
   pid=4716;state=pathlib.Path(f'/proc/{pid}/stat').read_text().split()[2]
   if state=='Z':raise RuntimeError('Mask audit exited before PASS; see mask.log')
  except FileNotFoundError:raise RuntimeError('Mask audit exited before PASS; see mask.log')
  time.sleep(15)
 audit=json.loads((A/'mask_audit.json').read_text());assert audit['status']=='PASS' and audit['zero_response_samples']==0
 output=R/'training/outputs/v2-rq3-response-coverage-recovery';assert not output.exists(),'output already exists; do not overwrite'
 status['phase']='GPU_PREFLIGHT';save()
 run([PY,'-c','import torch;assert torch.cuda.is_available() and torch.cuda.device_count()==2;[torch.ones(1,device="cuda:"+str(i)) for i in range(2)];torch.cuda.synchronize();print("GPU_PASS")'],'preflight.log')
 status['phase']='TRAINING';status['train_started_at']=time.time();save()
 run(['/opt/conda/envs/conda_env/bin/torchrun','--standalone','--nproc_per_node=2',str(R/'training/train_response_coverage.py')],'training.log')
 status['training_minutes']=(time.time()-status['train_started_at'])/60;status['phase']='FAST96';save()
 baseline=json.loads((R/'analysis/fast_decoding_sweep_96.json').read_text())['A'];baseline={'emotion':baseline['emotion'],'profile_mean':baseline['profile'],'bleu':baseline['BLEU'],'rouge':baseline['ROUGE'],'bert_f1':baseline['BERTScore']}
 checks={}
 for step in [80,160,240]:
  cp=output/f'checkpoint-{step}';assert (cp/'adapter_model.safetensors').exists()
  dest=R/f'training/eval_results/v2_rq3_checkpoint_{step}_fast96';assert not dest.exists()
  status['evaluating_checkpoint']=step;save()
  run([PY,str(R/'training/evaluate_lora.py'),'--model-path',str(R/'participant/model/Qwen2___5-7B-Instruct'),'--adapter-path',str(cp),'--data-file',str(R/'analysis/fast_iter_mini_dev_96.jsonl'),'--output-dir',str(dest),'--limit','96'],f'eval_{step}.log')
  summary=json.loads((dest/'summary.json').read_text());assert summary['samples_evaluated']==96
  metrics={'emotion':summary['emotion_accuracy'],'profile_mean':summary['profile_mean_accuracy'],'bleu':summary['bleu_4'],'rouge':summary['rouge_l'],'json_valid':summary['json_parse_success_rate'],'nonempty':summary['response_text_nonempty_rate']}
  run([BERTPY,str(A/'score_checkpoint.py'),str(dest)],f'bert_{step}.log');metrics['bert_f1']=json.loads((dest/'local_bertscore.json').read_text())['f1']
  delta={k:metrics[k]-baseline[k] for k in baseline};guard=metrics['emotion']>=baseline['emotion']-.0105 and metrics['profile_mean']>=baseline['profile_mean']-.007 and metrics['json_valid']==1 and metrics['nonempty']==1
  strong=guard and delta['bleu']>=.004 and delta['rouge']>=.004 and delta['bert_f1']>=0;promote=guard and delta['bleu']>=.002 and delta['rouge']>=.002 and delta['bert_f1']>=-.001
  checks[str(step)]={'checkpoint':str(cp),'metrics':metrics,'delta':delta,'guardrail_pass':guard,'result':'STRONG_PROMOTE' if strong else 'PROMOTE' if promote else 'REJECT'}
  status['completed_checkpoints']=checks;save()
 eligible=[x for x in checks.values() if x['result']!='REJECT'];winner=max(eligible,key=lambda x:(x['result']=='STRONG_PROMOTE',sum(x['metrics'][k] for k in ['bleu','rouge','bert_f1']))) if eligible else None
 result={'dataset':json.loads((A/'dataset_audit.json').read_text()),'mask_audit':audit,'baseline':baseline,'checkpoints':checks,'winner':winner,'result':winner['result'] if winner else 'REJECT','training_minutes':status['training_minutes'],'training_steps':240,'blocking_issue':'NONE'}
 (A/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));status['phase']='COMPLETED';status['result']=result['result'];save()
except Exception as exc:
 status['phase']='BLOCKED';status['blocking_issue']=str(exc);save();raise
