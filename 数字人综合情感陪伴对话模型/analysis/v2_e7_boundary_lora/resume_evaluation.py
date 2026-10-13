import os,json,time,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[2];A=Path(__file__).parent
state=json.loads((A/'status.json').read_text());state.update(phase='EVALUATION_RESUMING',blocking_issue='NONE',evaluation_resumed_at=time.time())
def save(): (A/'status.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
base=json.loads((A/'result.json').read_text());base.update(experiment_result='EVALUATION_IN_PROGRESS',blocking_issue='NONE',evaluation_completed=False);(A/'result.json').write_text(json.dumps(base,ensure_ascii=False,indent=2))
with (A/'evaluation.log').open('w') as log:
 p=subprocess.Popen(['/root/participant/python_env/bin/python',str(A/'evaluate.py')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['child_pid']=p.pid;save();rc=p.wait()
if rc:
 failure=json.loads((A/'evaluation_status.json').read_text());state.update(phase='EVALUATION_BLOCKED',blocking_issue=failure.get('blocking_issue','evaluation exit '+str(rc)));base.update(experiment_result='INCOMPLETE',blocking_issue=state['blocking_issue'],evaluation_completed=False)
 for n in ['fast96_result.json','fast192_result.json']:
  f=A/n
  if f.exists():base[n]=json.loads(f.read_text())
 (A/'result.json').write_text(json.dumps(base,ensure_ascii=False,indent=2))
 (A/'V2_E7_RESULT.md').write_text('# V2-E7 evaluation incomplete\n\nTraining completed: 80 steps.\nblocking_issue: '+state['blocking_issue']+'\nFORMAL_RUNTIME_MODIFIED=NO\n')
else:
 r=json.loads((A/'result.json').read_text());state.update(phase='COMPLETED',result=r['experiment_result'],best_candidate=r['best_candidate'],blocking_issue='NONE')
save()
raise SystemExit(rc)
