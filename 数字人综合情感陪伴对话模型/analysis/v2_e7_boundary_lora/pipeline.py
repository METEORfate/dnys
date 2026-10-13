
import os,sys,json,time,subprocess,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];A=Path(__file__).parent;PY='/opt/conda/envs/conda_env/bin/python';O=Path('/root/participant');started=time.time();state={'phase':'DATA_PREPARATION','started_at':started,'blocking_issue':'NONE'}
def save(): (A/'status.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1');env['PATH']='/opt/conda/envs/conda_env/bin:'+env['PATH']
def run(cmd,name):
 with (A/name).open('w') as log:
  p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['child_pid']=p.pid;save();rc=p.wait()
 if rc:raise RuntimeError(name+' exit '+str(rc))
try:
 save()
 while not (A/'setup.json').exists():
  proc=Path('/proc/8411/stat');assert proc.exists() and proc.read_text().split()[2]!='Z','DATA_PREPARATION_FAILED';time.sleep(2)
 setup=json.loads((A/'setup.json').read_text());state.update(phase='GPU_PREFLIGHT',dataset_count=setup['dataset_count']);save()
 assert not (R/'training/outputs/v2-e7-emotion-boundary').exists(),'OUTPUT_ALREADY_EXISTS'
 (A/'dual_rank_preflight.py').write_text("import os,torch\nr=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(r);x=torch.ones(4,4,device='cuda:'+str(r));x@x;torch.cuda.synchronize(r);print('RANK_PASS',r,flush=True)\n")
 run(['/opt/conda/envs/conda_env/bin/torchrun','--standalone','--nproc_per_node=2',str(A/'dual_rank_preflight.py')],'gpu_preflight.log')
 run([PY,'-m','unittest','training.test_loss_mask','training.test_truncation'],'mask_tests.log')
 state.update(phase='TRAINING',training_started_at=time.time());save();run(['/opt/conda/envs/conda_env/bin/torchrun','--standalone','--nproc_per_node=2',str(A/'train_boundary.py')],'training.log');minutes=(time.time()-state['training_started_at'])/60
 for step in [40,80]:
  p=R/'training/outputs/v2-e7-emotion-boundary'/f'checkpoint-{step}';assert (p/'adapter_model.safetensors').is_file() and (p/'adapter_config.json').is_file()
 (A/'training_completion.json').write_text(json.dumps({'steps':80,'training_minutes':minutes}));state.update(phase='EVALUATION',training_steps=80,training_minutes=minutes);save();run(['/root/participant/python_env/bin/python',str(A/'evaluate.py')],'evaluation.log')
 result=json.loads((A/'result.json').read_text());assert result['blocking_issue']=='NONE';state.update(phase='COMPLETED',result=result['experiment_result'],best_candidate=result['best_candidate']);save()
except Exception as e:
 state.update(phase='BLOCKED',blocking_issue=str(e));save()
 if not (A/'result.json').exists():
  partial={'completed':False,'phase':state['phase'],'blocking_issue':str(e),'formal_runtime_modified':False,'training_minutes':state.get('training_minutes'),'training_steps':state.get('training_steps'),'fast96':None,'fast192':None}
  for n in ['setup.json','fast96_result.json','fast192_result.json']:
   p=A/n
   if p.exists():partial[n]=json.loads(p.read_text())
  (A/'result.json').write_text(json.dumps(partial,ensure_ascii=False,indent=2));(A/'V2_E7_RESULT.md').write_text('# V2-E7 incomplete\n\nblocking_issue: '+str(e)+'\nFORMAL_RUNTIME_MODIFIED=NO\n')
 raise
