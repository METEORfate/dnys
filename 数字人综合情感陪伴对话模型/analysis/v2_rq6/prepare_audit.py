import pathlib,sys,json,random,collections,os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
R=pathlib.Path(__file__).resolve().parents[2];A=R/'analysis/v2_rq6';sys.path.insert(0,str(R/'training'))
from emotion_anchor_loss import emotion_labels
from response_only_loss import response_labels
from dual_objective_stream import DualObjectiveStream
from datasets import Dataset,load_from_disk
from transformers import AutoTokenizer
load=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
train=load(R/'training/data/train_sft.jsonl');assert len(train)==18657
rng=random.Random(61);classes=['anxiety','mixed','relaxed','fear','helplessness','sadness'];indices=[];distribution={}
for label in classes:
 pool=[i for i,q in enumerate(train) if json.loads(q['messages'][2]['content'])['emotion_label']==label];chosen=rng.sample(pool,min(200,len(pool)));indices.extend(chosen);distribution[label]=len(chosen)
assert len(indices)==len(set(indices)) and 1000<=len(indices)<=1200
anchor=R/'training/data/v2_rq6_emotion_anchor.jsonl';text=''.join(json.dumps(train[i],ensure_ascii=False,separators=(',',':'))+'\n' for i in indices)
if anchor.exists():assert anchor.read_text()==text
else:anchor.write_text(text)
(A/'anchor_source_manifest.jsonl').write_text(''.join(json.dumps({'anchor_index':j,'original_sample_index':i,'emotion':json.loads(train[i]['messages'][2]['content'])['emotion_label']})+'\n' for j,i in enumerate(indices)))
tok=AutoTokenizer.from_pretrained(R/'participant/model/Qwen2___5-7B-Instruct',local_files_only=True,use_fast=True)
response=load_from_disk(str(R/'training/data/v2_rq3_response_features'));raw=load(R/'training/data/v2_rq3_missing_original_sft.jsonl');assert len(response)==len(raw)==7334
old=json.loads((R/'analysis/v2_rq3/mask_audit.json').read_text());assert old['status']=='PASS' and old['zero_response_samples']==0
features=[];audits=[]
for j,i in enumerate(indices):
 f,a=emotion_labels(tok,train[i]['messages']);assert a['emotion_tokens']>0 and a['non_emotion_supervised']==0;features.append(f);audits.append(a)
 if j%200==0:print('EMOTION_MASK_PROGRESS',j,len(indices),flush=True)
path=R/'training/data/v2_rq6_emotion_features'
if not path.exists():Dataset.from_list(features).save_to_disk(str(path))
r20=[]
for i in random.Random(61).sample(range(len(raw)),20):
 f,a=response_labels(tok,raw[i]['messages']);assert f==response[i] and a['response_tokens']>0 and a['non_response_supervised']==0;r20.append({'response_row_index':i,**a})
zero=sum(not any(v!=-100 for v in f['labels']) for f in response)+sum(not any(v!=-100 for v in f['labels']) for f in features);assert zero==0
streams=[]
for rank in range(2):
 stream=DualObjectiveStream(response,features,rank,2);schedule=list(stream.schedule());assert len(schedule)==1280 and [o for o,i in schedule]==['response','response','response','emotion']*320
 streams.append({'rank':rank,'microbatches':len(schedule),'response_batches':960,'emotion_batches':320})
report={'response_data_count':len(response),'emotion_anchor_count':len(indices),'anchor_distribution':distribution,'seed':61,'zero_supervised_samples':zero,'response_mask':'PASS','emotion_mask':'PASS','response_random20':r20,'emotion_random20':[{'anchor_index':i,**audits[i]} for i in random.Random(61).sample(range(len(indices)),20)],'mixing':streams,'status':'PASS','train_only':True,'anchor_gratitude_note':'six fixed focus classes only; relaxed anchors include boundary exposure, no extra seventh class added'}
(A/'mask_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('DUAL_MASK_AUDIT_PASS',flush=True)
