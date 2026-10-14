import os,sys,json,time,subprocess,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[2];A=Path(__file__).parent
load=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
state={'phase':'STARTING','started_at':time.time()};save(A/'status.json',state)
try:
 env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false');env['PATH']='/opt/conda/bin:/opt/conda/envs/conda_env/bin:'+env['PATH']
 for name in ['p0','v1','v2']:
  state['phase']='INFERENCE';state['variant']=name;save(A/'status.json',state)
  dest=A/('results_'+name)
  if (dest/'performance_report.json').exists():
   report=json.loads((dest/'performance_report.json').read_text());assert report['samples']==96 and len(load(dest/'submission.jsonl'))==96
   continue
  assert not dest.exists()
  with (A/(name+'.log')).open('w') as f:
   q=subprocess.Popen(['bash','/root/development/runtime_candidates/prompt_pr1_'+name+'/start.sh',str(A/'inputs96.jsonl'),str(dest)],env=env,stdout=f,stderr=subprocess.STDOUT)
   state['child_pid']=q.pid;save(A/'status.json',state);rc=q.wait()
  assert rc==0,f'{name} runtime exit {rc}'
  report=json.loads((dest/'performance_report.json').read_text());assert report['samples']==96
  assert report['style_classifier_loaded'] and report['personality_classifier_loaded']
  assert report['style_classifier_fallback_count']==report['personality_fallback_count']==0
 state['phase']='METRICS';save(A/'status.json',state)
 sys.path.insert(0,str(R));from training.evaluate_lora import sentence_bleu4,rouge_l_f1,classification_diagnostics
 from bert_score import score
 import torch;torch.set_num_threads(8)
 rows=load(R/'analysis/fast_iter_mini_dev_96.jsonl');targets=[json.loads(q['messages'][2]['content']) for q in rows];inputs=load(A/'inputs96.jsonl');gt=[q['emotion_label'] for q in targets];allm={};preds={};per={}
 for name in ['p0','v1','v2']:
  state['variant']=name;save(A/'status.json',state)
  dest=A/('results_'+name);pred=load(dest/'submission.jsonl');assert [q['id'] for q in pred]==[q['id'] for q in inputs]
  text=[q['response_text'] for q in pred];ref=[q['response_text'] for q in targets]
  _,_,F=score(text,ref,model_type='/root/development/assets/bert-base-multilingual-cased',num_layers=9,lang='zh',rescale_with_baseline=False,device='cpu',batch_size=32,verbose=False)
  b=[sentence_bleu4(r,p) for r,p in zip(ref,text)];ro=[rouge_l_f1(r,p) for r,p in zip(ref,text)];bf=F.tolist();report=json.loads((dest/'performance_report.json').read_text());emo=[q['emotion_label'] for q in pred]
  m={'emotion':sum(p==t for p,t in zip(emo,gt))/96,'macro_f1':classification_diagnostics(emo,gt)['f1'],'bleu':sum(b)/96,'rouge':sum(ro)/96,'bert':sum(bf)/96,'json_valid':1-report['parse_failure_count']/96,'nonempty':sum(bool(t.strip()) for t in text)/96,'mean_response_length':sum(map(len,text))/96,'avg_latency':report['average_latency_ms'],'p50_latency':report['median_latency_ms'],'p95_latency':report['p95_latency_ms']}
  for d,k in [('personality_traits','personality'),('interests','interests'),('style','style')]:m[k]=sum(set(p['user_profile'][d])==set(t['user_profile'][d]) for p,t in zip(pred,targets))/96
  m['profile_mean']=sum(m[k] for k in ['personality','interests','style'])/3
  m['recall']={l:sum(p==l and t==l for p,t in zip(emo,gt))/gt.count(l) if gt.count(l) else None for l in ['anxiety','mixed','relaxed','fear','helplessness','sadness']}
  allm[name]=m;preds[name]=pred;per[name]={'bleu':b,'rouge':ro,'bert':bf};save(dest/'metrics.json',m)
 base=allm['p0'];detail={}
 for name in ['v1','v2']:
  m=allm[name];d={k:m[k]-base[k] for k in ['emotion','profile_mean','bleu','rouge','bert']};d['latency_delta_percent']=(m['avg_latency']/base['avg_latency']-1)*100
  wr=rw=0;changes=[]
  for i,(p,c,t) in enumerate(zip(preds['p0'],preds[name],targets)):
   pc=p['emotion_label']==t['emotion_label'];cc=c['emotion_label']==t['emotion_label'];wr+=not pc and cc;rw+=pc and not cc
   if p['emotion_label']!=c['emotion_label']:changes.append({'id':p['id'],'gt':t['emotion_label'],'p0':p['emotion_label'],'candidate':c['emotion_label']})
  paired={}
  for k in ['rouge','bert']:
   delta=[x-y for x,y in zip(per[name][k],per['p0'][k])];better=sum(x>1e-6 for x in delta);worse=sum(x< -1e-6 for x in delta);paired[k]={'better':better,'worse':worse,'equal':96-better-worse,'net_better':better-worse}
  guard=wr-rw>=-1 and d['profile_mean']>=-.007 and m['json_valid']==m['nonempty']==1
  # Predeclared interpretation of response 'no obvious decline': BLEU/ROUGE >= -0.002, BERT >= -0.001.
  stable=d['bleu']>=-.002 and d['rouge']>=-.002 and d['bert']>=-.001
  decision='REJECT'
  if guard:
   if (wr-rw>=2 and stable) or (wr-rw>=0 and d['bleu']>=.004 and d['rouge']>=.004 and d['bert']>=0):decision='STRONG_PROMOTE'
   elif (wr-rw>=1 and stable) or (wr-rw>=0 and d['bleu']>=.002 and d['rouge']>=.002 and d['bert']>=-.0005):decision='PROMOTE'
   elif wr-rw>=1 or (d['bleu']>0 and d['rouge']>0 and max(d['bleu'],d['rouge'])>=.002):decision='PROMOTE_WEAK'
  m['decision']=decision;detail[name]={'delta':d,'wrong_to_right':wr,'right_to_wrong':rw,'emotion_changes':changes,'paired':paired,'guardrail':guard,'latency_warning':d['latency_delta_percent']>5}
 eligible=[n for n in ['v1','v2'] if allm[n]['decision']!='REJECT'];winner=max(eligible,key=lambda n:(allm[n]['emotion'],sum(detail[n]['delta'][k] for k in ['bleu','rouge','bert']),allm[n]['profile_mean'],-allm[n]['avg_latency'])) if eligible else 'p0'
 setup=json.loads((A/'setup.json').read_text());assert all(hashlib.sha256((Path('/root/participant')/p).read_bytes()).hexdigest()==h for p,h in setup['runtime_hashes'].items())
 save(A/'result.json',{'setup':setup,'metrics':allm,'detail':detail,'winner':winner,'result':allm[winner]['decision'] if winner!='p0' else 'REJECT','elapsed_minutes':(time.time()-state['started_at'])/60,'blocking_issue':'NONE','response_stability_tolerances':{'bleu':-.002,'rouge':-.002,'bert':-.001}})
 state['phase']='COMPLETED';state['winner']=winner;save(A/'status.json',state)
except Exception as e:
 state['phase']='BLOCKED';state['blocking_issue']=str(e);save(A/'status.json',state);raise
