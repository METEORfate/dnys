import sys,json,time,copy,hashlib,statistics
from pathlib import Path
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[2];A=Path(__file__).parent;F=R/'analysis/final_score_maximization';S=R/'analysis/v2_pr2a_valdev'
load=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
sys.path.insert(0,str(F));from conservative_emotion_rule import rule
sys.path.insert(0,str(R));from training.evaluate_lora import sentence_bleu4,rouge_l_f1,classification_diagnostics
setup=json.loads((A/'setup.json').read_text());assert sha(Path(setup['rule_file']))==setup['rule_set_hash'];assert sha(Path(setup['source_file']))==setup['source_sha256']
rows=load(R/'training/data/val_dev_sft.jsonl');inputs=load(S/'inputs1649.jsonl');records=sorted([q for q in load(S/'paired_predictions.jsonl') if q['variant']=='p0'],key=lambda q:q['row_index']);assert len(rows)==len(inputs)==len(records)==1649;N=len(rows);source_setup=json.loads((S/'setup.json').read_text());assert sha(R/'training/data/val_dev_sft.jsonl')==source_setup['valdev_sha256']
hashes=json.loads((F/'setup.json').read_text())['runtime_hashes'];assert source_setup['runtime_hashes']==hashes;assert all(sha(Path('/root/participant')/p)==h for p,h in hashes.items());source=json.loads((S/'result.json').read_text())['metrics']['p0']
# Verify the frozen module exactly reproduces the prior FAST192 experiment.
fast_records=load(F/'generation.jsonl');fast_inputs=load(F/'inputs192.jsonl');fast_rows=load(F/'fast192.jsonl');pred=[]
for q,h in zip(fast_records,fast_inputs):pred.append(rule([m['content'] for m in h['history'] if m['role']=='user'][-1],q['baseline']['emotion_label']))
assert sum(p==json.loads(q['messages'][2]['content'])['emotion_label'] for p,q in zip(pred,fast_rows))==139
baseline=[];candidate=[];changes=[];times=[];wr=rw=0;preserved={k:0 for k in ['response_text','user_profile','memory_refs','id']};gt=[]
for i,(row,h,rec) in enumerate(zip(rows,inputs,records)):
 assert rec['row_index']==i and rec['prediction']['id']==h['id'];target=json.loads(row['messages'][2]['content']);gt.append(target['emotion_label']);b=rec['prediction'];c=copy.deepcopy(b);latest=[m['content'] for m in h['history'] if m['role']=='user'][-1];stamp=time.perf_counter();c['emotion_label']=rule(latest,b['emotion_label']);times.append((time.perf_counter()-stamp)*1000)
 bc=b['emotion_label']==target['emotion_label'];cc=c['emotion_label']==target['emotion_label'];wr+=not bc and cc;rw+=bc and not cc
 for k in preserved:preserved[k]+=b[k]==c[k]
 assert all(b[k]==c[k] for k in preserved)
 if b['emotion_label']!=c['emotion_label']:changes.append({'id':h['id'],'row_index':i,'gt':target['emotion_label'],'before':b['emotion_label'],'after':c['emotion_label'],'wrong_to_right':bool(not bc and cc),'right_to_wrong':bool(bc and not cc)})
 baseline.append(b);candidate.append(c)
assert all(v==N for v in preserved.values())
# Recompute unchanged metrics against the recorded baseline; BERTScore is reused only after byte-identical responses are proven.
metrics={}
for name,ps in [('current',baseline),('candidate',candidate)]:
 em=[q['emotion_label'] for q in ps];m={'emotion':sum(p==t for p,t in zip(em,gt))/N,'macro_f1':classification_diagnostics(em,gt)['f1'],'bert':source['bert']}
 for dim,key in [('personality_traits','personality'),('interests','interests'),('style','style')]:m[key]=sum(set(p['user_profile'][dim])==set(json.loads(q['messages'][2]['content'])['user_profile'][dim]) for p,q in zip(ps,rows))/N
 m['profile_mean']=sum(m[k] for k in ['personality','interests','style'])/3
 if name=='current':
  m['bleu']=sum(sentence_bleu4(json.loads(q['messages'][2]['content'])['response_text'],p['response_text']) for p,q in zip(ps,rows))/N;m['rouge']=sum(rouge_l_f1(json.loads(q['messages'][2]['content'])['response_text'],p['response_text']) for p,q in zip(ps,rows))/N
 else:m.update(bleu=metrics['current']['bleu'],rouge=metrics['current']['rouge'])
 lat=[q['latency_ms']+(times[i] if name=='candidate' else 0) for i,q in enumerate(records)];m['avg_latency_ms']=statistics.mean(lat);ordered=sorted(lat);m['p95_latency_ms']=ordered[int((N-1)*.95)];m['latency_scope']='stored current HTTP latency plus measured CPU rule call; candidate end-to-end estimate, not a new runtime benchmark'
 classes={}
 for label in sorted(set(gt)):
  tp=sum(p==label and t==label for p,t in zip(em,gt));fp=sum(p==label and t!=label for p,t in zip(em,gt));fn=gt.count(label)-tp;precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0;classes[label]={'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0,'support':gt.count(label)}
 m['per_class']=classes;metrics[name]=m
for k in ['emotion','macro_f1','personality','interests','style','profile_mean','bleu','rouge','bert']:assert abs(metrics['current'][k]-source[k])<1e-10,(k,metrics['current'][k],source[k])
delta={k:metrics['candidate'][k]-metrics['current'][k] for k in ['emotion','macro_f1','personality','interests','style','profile_mean','bleu','rouge','bert']};decision='REJECT'
if delta['emotion']>=0 and wr>rw:
 decision='STRONG_PROMOTE' if delta['emotion']>=.005 and wr>=10 and rw<=5 else 'PROMOTE'
assert sha(Path(setup['rule_file']))==setup['rule_set_hash'];assert all(sha(Path('/root/participant')/p)==h for p,h in hashes.items())
result={'samples':N,'setup':setup,'metrics':metrics,'delta':delta,'paired':{'wrong_to_right':wr,'right_to_wrong':rw,'net':wr-rw,'changed_count':len(changes)},'preservation':{k:v/N for k,v in preserved.items()},'rule_latency_ms':{'avg':statistics.mean(times),'p95':sorted(times)[int((N-1)*.95)]},'decision':decision,'final_runtime_modified':False,'blocking_issue':'NONE','changes':changes,'validation_scope':'existing official-stack full val_dev P0 predictions, deterministic CPU replay; no new generation, no holdout'};save(A/'result.json',result)
(A/'emotion_changes.jsonl').write_text(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in changes));(A/'candidate_predictions.jsonl').write_text(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in candidate))
print('DECISION',decision);print('DELTA',delta);print('PAIRED',result['paired']);print('METRICS',{n:{k:v for k,v in m.items() if k not in ['per_class','latency_scope']} for n,m in metrics.items()});print('RULE_HASH',setup['rule_set_hash'])
