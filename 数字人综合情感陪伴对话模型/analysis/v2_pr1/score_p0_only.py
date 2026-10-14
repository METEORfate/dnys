import os,sys,json,time,subprocess,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[2];A=Path(__file__).parent
load=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
state={'phase':'STARTING','started_at':time.time()};save(A/'status.json',state)
try:
 state['phase']='METRICS';save(A/'status.json',state)
 sys.path.insert(0,str(R));from training.evaluate_lora import sentence_bleu4,rouge_l_f1,classification_diagnostics
 from bert_score import score
 import torch;torch.set_num_threads(8)
 rows=load(R/'analysis/fast_iter_mini_dev_96.jsonl');targets=[json.loads(q['messages'][2]['content']) for q in rows];inputs=load(A/'inputs96.jsonl');gt=[q['emotion_label'] for q in targets];allm={};preds={};per={}
 for name in ['p0']:
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
 save(A/'partial_result.json',{'metrics':allm,'blocking_issue':'GPU_PLATFORM_UNSTABLE: NVML initialization failed; V1/V2 not evaluated'})
except Exception:
 raise
