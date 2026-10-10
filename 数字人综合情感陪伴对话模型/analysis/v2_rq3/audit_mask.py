import os,sys,json,random,pathlib
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
R=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'training'))
from transformers import AutoTokenizer
from datasets import Dataset
from response_only_loss import response_labels
A=R/'analysis/v2_rq3';rows=[json.loads(x) for x in (R/'training/data/v2_rq3_missing_original_sft.jsonl').read_text().splitlines() if x.strip()]
tok=AutoTokenizer.from_pretrained(R/'participant/model/Qwen2___5-7B-Instruct',local_files_only=True,use_fast=True)
features=[];audits=[]
for i,row in enumerate(rows):
 f,a=response_labels(tok,row['messages']);features.append(f);audits.append(a)
 if i%500==0:print('MASK_PROGRESS',i,len(rows),flush=True)
zero=sum(x['response_tokens']==0 for x in audits);assert zero==0
sample=random.Random(42).sample(range(len(rows)),20)
out={'samples':len(rows),'zero_response_samples':zero,'emotion_tokens_supervised':sum(x['emotion_tokens_supervised'] for x in audits),'profile_tokens_supervised':sum(x['profile_tokens_supervised'] for x in audits),'memory_tokens_supervised':sum(x['memory_tokens_supervised'] for x in audits),'status':'PASS','sampled20':[{'missing_row_index':i,**audits[i]} for i in sample],'mask_method':'exact full-chat offsets; only tokens wholly inside response JSON string content, excluding quotes/boundary-crossing tokens','trimmed_samples':sum(x['trimmed'] for x in audits)}
assert all(x['response_tokens']>0 for x in audits)
Dataset.from_list(features).save_to_disk(str(R/'training/data/v2_rq3_response_features'));(A/'mask_audit.json').write_text(json.dumps(out,indent=2));print('MASK_AUDIT_PASS',flush=True)
