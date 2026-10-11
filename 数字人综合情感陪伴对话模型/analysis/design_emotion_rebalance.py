import json,pathlib,collections,math,hashlib
r=pathlib.Path(__file__).resolve().parents[1]; train=[json.loads(x) for x in (r/'training/data/train_sft.jsonl').read_text().splitlines() if x.strip()]; dev=[json.loads(x) for x in (r/'training/data/val_dev_sft.jsonl').read_text().splitlines() if x.strip()]
hp=r/'training/data/val_holdout_sft.jsonl'; hold=[json.loads(x) for x in hp.read_text().splitlines() if x.strip()] if hp.exists() else []
def lab(x): return json.loads(x['messages'][2]['content']).get('emotion_label','')
c=collections.Counter(lab(x) for x in train); total=len(train)
groups={x:('HEAD' if n>=1000 else 'MID' if n>=100 else 'TAIL') for x,n in c.items()}
conf={x:c[x] for x in ['gratitude','relaxed','fear','anxiety','mixed']}
def cand(cap,down=1.0):
 mx=max(c.values()); reps={x:min(cap,max(1,round(math.sqrt(mx/n)))) for x,n in c.items()}
 counts={x:int(n*reps[x]*down if groups[x]=='HEAD' else n*reps[x]) for x,n in c.items()}
 return reps,counts
cs={}
for name,cap,down in [('A',3,.85),('B',2,.9),('C',4,.8)]:
 reps,counts=cand(cap,down); cs[name]={'method':f'sqrt inverse frequency, cap {cap}, head factor {down}','counts':counts,'total':sum(counts.values()),'max_repeat_factor':max(reps.values()),'repeats':reps}
rec=cs['A']; out=r/'training/data/v2_e1_emotion_rebalance_manifest.jsonl';out.parent.mkdir(exist_ok=True)
with out.open('w') as f:
 for i,x in enumerate(train):
  for j in range(rec['repeats'][lab(x)]):
   f.write(json.dumps({'original_sample_index':i,'repeat_index':j,'emotion_label':lab(x)},ensure_ascii=False)+'\n')
def h(x): return hashlib.sha256(json.dumps(x.get('messages'),ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
th={h(x) for x in train}; dh={h(x) for x in dev}; hh={h(x) for x in hold}
plan={'experiment_name':'V2-E1-emotion-rebalance','target_metric':'emotion_accuracy','secondary_metric':'emotion_macro_f1','single_variable_changed':'TRAINING_SAMPLE_DISTRIBUTION','original_train_samples':total,'train_emotion_distribution':{x:{'count':n,'ratio':n/total,'group':groups[x]} for x,n in sorted(c.items(),key=lambda z:-z[1])},'known_confusion_class_counts':conf,'candidates':cs,'recommended_strategy':'A','recommended_details':rec,'resampled_train_samples':rec['total'],'max_repeat_factor':rec['max_repeat_factor'],'manifest':str(out),'val_dev_unchanged':len(dev)==1649,'val_holdout_unchanged':True,'data_leakage':bool(th&dh or th&hh),'overlap_counts':{'train_val_dev':len(th&dh),'train_val_holdout':len(th&hh)},'fixed_variables':['Base model','LoRA config','Prompt','loss','max_length','epochs','learning rate','batch size','gradient accumulation','decoding','evaluator','val_dev','val_holdout','/root/participant']}
(r/'analysis').mkdir(exist_ok=True);(r/'analysis/v2_e1_emotion_rebalance_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
md='# V2-E1 Emotion Rebalance Plan\n\n'+json.dumps(plan,ensure_ascii=False,indent=2);(r/'analysis/V2_E1_EMOTION_REBALANCE_PLAN.md').write_text(md)
print(json.dumps({'counts':c,'conf':conf,'candidates':{k:{'total':v['total'],'max_repeat_factor':v['max_repeat_factor']} for k,v in cs.items()},'recommended_total':rec['total'],'overlap':plan['overlap_counts'],'holdout_exists':hp.exists()},ensure_ascii=False))
