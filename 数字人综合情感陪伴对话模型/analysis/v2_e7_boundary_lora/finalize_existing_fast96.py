
import sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[2];A=Path(__file__).parent;O=Path('/root/participant');sys.path.insert(0,str(R))
from training.evaluate_lora import classification_diagnostics,sentence_bleu4,rouge_l_f1
load=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
s=json.loads((A/'setup.json').read_text());m=json.loads((A/'partial_fast96_metrics.json').read_text());t=[json.loads(q['messages'][-1]['content']) for q in load(R/'analysis/final_score_maximization/fast192.jsonl')[:96]];rec=load(A/'predictions.jsonl');assert len(rec)==288;pred={}
for n in ['p0','e7-40','e7-80']:
 rows=sorted([q for q in rec if q['variant']==n],key=lambda q:q['row_index']);assert [q['row_index'] for q in rows]==list(range(96));p=[q['prediction'] for q in rows];pred[n]=p
 assert abs(sum(a['emotion_label']==b['emotion_label'] for a,b in zip(p,t))/96-m[n]['emotion'])<1e-10
 assert abs(classification_diagnostics([a['emotion_label'] for a in p],[a['emotion_label'] for a in t])['f1']-m[n]['macro_f1'])<1e-10
 for d,k in [('personality_traits','personality'),('interests','interests'),('style','style')]:assert abs(sum(set(a['user_profile'][d])==set(b['user_profile'][d]) for a,b in zip(p,t))/96-m[n][k])<1e-10
 for fn,k in [(sentence_bleu4,'bleu'),(rouge_l_f1,'rouge')]:assert abs(sum(fn(b['response_text'],a['response_text']) for a,b in zip(p,t))/96-m[n][k])<1e-10
first={'samples':96,'baseline':m['p0'],'checkpoints':{}}
for n in ['e7-40','e7-80']:
 delta={k:m[n][k]-m['p0'][k] for k in ['emotion','macro_f1','personality','interests','style','profile_mean','bleu','rouge','bert']};fail=[k for k,v in s['guardrails'].items() if delta['profile_mean' if k=='profile_delta_min' else k.removesuffix('_delta_min')]<v-1e-9];wr=rw=0;changes=[]
 for b,c,gt in zip(pred['p0'],pred[n],t):
  bc=b['emotion_label']==gt['emotion_label'];cc=c['emotion_label']==gt['emotion_label'];wr+=not bc and cc;rw+=bc and not cc
  if b['emotion_label']!=c['emotion_label']:changes.append({'id':c['id'],'gt':gt['emotion_label'],'before':b['emotion_label'],'after':c['emotion_label']})
 first['checkpoints'][n]={'adapter_path':str(Path(s['output_adapter'])/('checkpoint-'+n.split('-')[-1])),'metrics':m[n],'delta':delta,'wrong_to_right':wr,'right_to_wrong':rw,'net':wr-rw,'changes':changes,'guardrail_failures':fail,'passed':not fail,'decision':'REJECT' if fail else 'PASS'}
assert all(not c['passed'] for c in first['checkpoints'].values());assert all(hashlib.sha256((O/p).read_bytes()).hexdigest()==h for p,h in s['runtime_hashes'].items())
train=json.loads((A/'training_completion.json').read_text());old=json.loads((A/'result.json').read_text());result={'dataset':s,'training_steps':80,'training_minutes':train['training_minutes'],'train_loss':0.9087026476860046,'continuation_tensor_exact_verified':True,'adapter_path':s['output_adapter'],'checkpoint_parameter_audit':old.get('checkpoints'),'fast96':first,'fast192':None,'fast192_reason':'Both checkpoints failed predefined FAST96 guardrails','best_candidate':'CURRENT','experiment_result':'REJECT','automatic_promotion':False,'formal_runtime_modified':False,'blocking_issue':'NONE','evaluation_completed':True,'evaluation_summary_fix':'profile_delta_min maps to profile_mean; no threshold or metric change; existing 288 predictions and BERT scores reused'}
for p,r in [(A/'result.json',result),(A/'fast96_result.json',first)]:p.write_text(json.dumps(r,ensure_ascii=False,indent=2))
lines=['# V2-E7 Emotion Boundary LoRA','','Result: REJECT; best: CURRENT','FORMAL_RUNTIME_MODIFIED=NO','',f"Source adapter: {s['source_adapter']}",f"Output adapter: {s['output_adapter']}",f"Training: 80 steps, {train['training_minutes']:.2f} minutes including loading; loss=0.908703; exact V2-E1 continuation verified on both ranks.",'r=16, alpha=32, original 7 modules; lr=1e-5 with existing linear scheduler, batch=1, grad_acc=8, max_length=4096, checkpointing=true, packing=false, assistant-only loss.','', '## Dataset','2800 unique original train samples = 1960 boundary + 840 V2-E1-distribution anchor; seed=66; no relabeling or synthetic data. Hard selection uses fixed latest-user keyword cues, not confirmed LLM errors. No validation/holdout used for selection.','Mask: 2800 retained, 226 context-trimmed, zero supervised=0; 7 existing CPU tests PASS.',json.dumps(s['distribution'],ensure_ascii=False),'', '## FAST96','One TP2 vLLM server, max_num_seqs=8, P0 and original generation, serial rotating requests; all outputs use P1/P2. Fresh CURRENT=61/96; historical 67/96 not substituted.','', '|adapter|emotion|macro_f1|personality|interests|style|profile_mean|BLEU|ROUGE|BERT|avg_ms|p95_ms|','|---|---|---|---|---|---|---|---|---|---|---|---|']
for n,v in m.items():lines+=['|'+n+'|'+'|'.join(f'{v[k]:.6f}' for k in ['emotion','macro_f1','personality','interests','style','profile_mean','bleu','rouge','bert','avg_latency_ms','p95_latency_ms'])+'|']
lines+=['','|class recall|CURRENT|checkpoint-40|checkpoint-80|','|---|---|---|---|']
for l in ['anxiety','fear','helplessness','mixed','relaxed','gratitude']:lines+=['|'+l+'|'+'|'.join(f"{m[n]['recall'][l]['recall']:.6f}" for n in ['p0','e7-40','e7-80'])+'|']
for n,c in first['checkpoints'].items():lines += ['',f"{n}: wrong->right={c['wrong_to_right']}, right->wrong={c['right_to_wrong']}, net={c['net']}; failures={c['guardrail_failures']}",json.dumps(c['delta'])]
lines+=['','All groups JSON/nonempty=100%, P1/P2 fallback=0. FAST192 skipped because both failed frozen guards.','checkpoint-80 improves emotion by 1/96 and all response metrics, but profile mean falls by 0.010417 beyond the fixed 0.007 guard. checkpoint-40 loses emotion and macro-F1.','No promotion, holdout or official file modification. Hashes unchanged.','Summary profile key mapping bug fixed; original generated predictions and thresholds unchanged. Non-BERT metrics rechecked against saved predictions, existing BERT scores retained.','blocking_issue: NONE']
(A/'V2_E7_RESULT.md').write_text('\n'.join(lines)+'\n')
for fn in ['status.json','evaluation_status.json']:
 state=json.loads((A/fn).read_text());state.update(phase='COMPLETED',best_candidate='CURRENT',result='REJECT',blocking_issue='NONE');(A/fn).write_text(json.dumps(state,ensure_ascii=False,indent=2))
for n,c in first['checkpoints'].items():print(n,'wrong_to_right',c['wrong_to_right'],'right_to_wrong',c['right_to_wrong'],'failures',c['guardrail_failures'])
print('COMPLETED_REJECT_FORMAL_UNCHANGED')
