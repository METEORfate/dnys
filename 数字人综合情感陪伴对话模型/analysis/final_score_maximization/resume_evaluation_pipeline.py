import pathlib,subprocess,time,json,os,signal
A=pathlib.Path(__file__).parent;PY='/root/participant/python_env/bin/python';budget=json.loads((A/'evaluation_resume_budget.json').read_text());started=budget['started_at']
def state(phase,**kwargs):(A/'status.json').write_text(json.dumps({'phase':phase,'started_at':started,'new_budget_hours':4,**kwargs},indent=2))
try:
 state('CANDIDATE_EVALUATION')
 with (A/'evaluate_candidates.py.log').open('w') as log:
  child=subprocess.Popen([PY,'-u',str(A/'evaluate_candidates.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state('CANDIDATE_EVALUATION',child_pid=child.pid)
  while child.poll() is None:
   if time.time()-started>=4*3600:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
    raise RuntimeError('AUTHORIZED_FOUR_HOUR_EVALUATION_BUDGET_REACHED')
   time.sleep(5)
  if child.returncode:raise RuntimeError('evaluate_candidates.py exit '+str(child.returncode))
 state('COMPLETED')
except Exception as e:state('BLOCKED',blocking_issue=str(e));raise
