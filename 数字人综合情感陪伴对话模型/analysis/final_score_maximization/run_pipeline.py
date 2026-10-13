import pathlib,subprocess,time,json,os
A=pathlib.Path(__file__).parent;PY='/opt/conda/envs/conda_env/bin/python';EP='/root/participant/python_env/bin/python';started=time.time()
def state(phase,**kwargs):(A/'status.json').write_text(json.dumps({'phase':phase,'started_at':started,**kwargs},indent=2))
try:
 for script,python in [('extract_train_heads.py',PY),('evaluate_candidates.py',EP)]:
  state('HEAD_FEATURES_AND_TRAINING' if script.startswith('extract') else 'CANDIDATE_EVALUATION')
  with (A/(script+'.log')).open('w') as log:
   child=subprocess.Popen([python,'-u',str(A/script)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state('HEAD_FEATURES_AND_TRAINING' if script.startswith('extract') else 'CANDIDATE_EVALUATION',child_pid=child.pid)
   while child.poll() is None:
    if time.time()-started>=4*3600:child.terminate();raise RuntimeError('FOUR_HOUR_BUDGET_REACHED')
    time.sleep(5)
   if child.returncode:raise RuntimeError(script+' exit '+str(child.returncode))
 state('COMPLETED')
except Exception as e:state('BLOCKED',blocking_issue=str(e));raise
