import json,pathlib,collections,math,random
r=pathlib.Path(__file__).resolve().parents[1]; lines=[json.loads(x) for x in (r/'training/data/train_sft.jsonl').read_text().splitlines() if x.strip()]; N=len(lines); assert N==18657
cls=collections.defaultdict(list)
for i,x in enumerate(lines): cls[json.loads(x['messages'][2]['content']).get('emotion_label','')].append(i)
mx=max(map(len,cls.values())); base={e:math.sqrt(mx/len(v)) for e,v in cls.items()}
rng=random.Random(42); remaining={i:3 for i in range(N)}; avail=set(range(N)); chosen=[]
# weighted sampling with cap; class weights sqrt inverse frequency, dynamically exclude exhausted samples
for pos in range(N):
 weights=[]; inds=[]
 for e,vs in cls.items():
  live=[i for i in vs if remaining[i]>0]
  if not live: continue
  w=base[e]
  inds.extend(live); weights.extend([w]*len(live))
 x=rng.choices(inds,weights=weights,k=1)[0]; chosen.append(x); remaining[x]-=1
 if remaining[x]==0: avail.discard(x)
out=r/'training/data/v2_e1_emotion_rebalance_fixed18657_manifest.jsonl'
with out.open('w') as f:
 for pos,i in enumerate(chosen): f.write(json.dumps({'resampled_position':pos,'original_sample_index':i,'emotion_label':json.loads(lines[i]['messages'][2]['content']).get('emotion_label','')},ensure_ascii=False)+'\n')
before=collections.Counter(json.loads(x['messages'][2]['content']).get('emotion_label','') for x in lines); after=collections.Counter(json.loads(lines[i]['messages'][2]['content']).get('emotion_label','') for i in chosen); freq=collections.Counter(chosen)
def h(x): return json.dumps(x['messages'],ensure_ascii=False,sort_keys=True,separators=(',',':'))
th={h(x) for x in lines}; dev=[json.loads(x) for x in (r/'training/data/val_dev_sft.jsonl').read_text().splitlines() if x.strip()]; hold=[json.loads(x) for x in (r/'training/data/val_holdout_sft.jsonl').read_text().splitlines() if x.strip()]; dh={h(x) for x in dev}; hh={h(x) for x in hold}
dist={e:{'original_count':before[e],'original_ratio':before[e]/N,'resampled_count':after[e],'resampled_ratio':after[e]/N,'delta_count':after[e]-before[e],'delta_ratio':(after[e]-before[e])/N} for e in sorted(before)}
plan={'original_samples':N,'resampled_samples':len(chosen),'emotion_distribution_before':before,'emotion_distribution_after':after,'distribution':dist,'unique_original_samples':len(freq),'duplicate_sample_count':sum(v>1 for v in freq.values()),'max_repeat_factor':max(freq.values()),'seed':42,'sampling':'sqrt inverse-frequency weighted sampling with replacement, capped per original sample at 3','v1_expected_steps':'same as V1 because dataset length is 18657','v2_e1_expected_steps':'same as V1 because dataset length is 18657','steps_equal':True,'val_dev_unchanged':True,'val_holdout_unchanged':True,'data_leakage':bool(th&dh or th&hh),'overlap_counts':{'train_val_dev':len(th&dh),'train_val_holdout':len(th&hh)},'manifest':str(out),'single_variable_changed':'TRAINING_SAMPLE_DISTRIBUTION'}
(r/'analysis/V2_E1_EMOTION_REBALANCE_FIXED_PLAN.md').write_text('# V2-E1 Fixed Emotion Rebalance Plan\n\n'+json.dumps(plan,ensure_ascii=False,indent=2));(r/'analysis/v2_e1_emotion_rebalance_fixed_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
print(json.dumps({'before':before,'after':after,'unique':len(freq),'duplicates':sum(v>1 for v in freq.values()),'max':max(freq.values()),'overlap':plan['overlap_counts']},ensure_ascii=False))