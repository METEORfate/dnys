
import os,sys,json,random,re,hashlib,collections,math
from pathlib import Path
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[2];A=Path(__file__).parent;O=Path('/root/participant');sys.path.insert(0,str(R/'training'))
from train_lora import prepare_dataset
from datasets import Dataset
from transformers import AutoTokenizer
load=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
rows=load(R/'training/data/train_sft.jsonl');manifest=load(R/'training/data/v2_e1_emotion_rebalance_fixed18657_manifest.jsonl');assert len(rows)==len(manifest)==18657
p0=(O/'configs/system_prompt_v1.txt').read_text().strip();assert all(r['messages'][0]['content']==p0 for r in rows)
labels=[json.loads(r['messages'][-1]['content'])['emotion_label'] for r in rows]
# Existing SFT history format: recover the final serialized user turn, never target text.
def latest(r):
 parts=re.split(r'(?:^|\n)(user|assistant): ',r['messages'][1]['content'])
 users=[parts[i+1] for i in range(1,len(parts)-1,2) if parts[i]=='user'];assert users;return users[-1]
cues={
'worry':r'担心|焦虑|万一|会不会|反复|不确定|紧张|怕.*结果',
'danger':r'威胁|危险|害怕|恐惧|医生|确诊|死亡|被.*排斥|失去|具体',
'powerless':r'无力|无能为力|没办法|不知道.*怎么办|无路|没有选择|改变不了|怎么.*都',
'gratitude':r'谢谢|感谢|感激|幸好.*你|多亏',
'relief':r'松.*气|放松|释然|轻松|压力.*小|终于.*不用|没那么',
'mixed':r'一方面|另一方面|又.*又|既.*又|同时|但|却|开心.*担心|高兴.*难过'}
quotas={'anxiety':480,'fear':360,'helplessness':300,'gratitude':280,'relaxed':280,'mixed':260};rng=random.Random(66)
chosen=[];priority_stats={}
for label,count in quotas.items():
 pool=[i for i,l in enumerate(labels) if l==label];rng.shuffle(pool)
 groups={'anxiety':['worry','danger','powerless'],'fear':['danger','worry'],'helplessness':['powerless','worry'],'gratitude':['gratitude','relief'],'relaxed':['relief','gratitude'],'mixed':['mixed','worry','relief','gratitude']}[label]
 def difficulty(i):
  text=latest(rows[i]);hits=[bool(re.search(cues[g],text)) for g in groups];return (sum(hits)>=2,sum(hits))
 pool.sort(key=difficulty,reverse=True);sel=pool[:count];assert len(sel)==count;chosen+=sel;priority_stats[label]={'selected':count,'two_or_more_cue_groups':sum(difficulty(i)[0] for i in sel)}
selected=set(chosen);manifest_counts=collections.Counter(m['emotion_label'] for m in manifest)
raw={l:840*c/18657 for l,c in manifest_counts.items()};anchor_quota={l:math.floor(c) for l,c in raw.items()}
for l in sorted(raw,key=lambda l:raw[l]-anchor_quota[l],reverse=True)[:840-sum(anchor_quota.values())]:anchor_quota[l]+=1
anchor=[]
for l,n in anchor_quota.items():
 # Manifest multiplicities supply the V2-E1 distribution; unique source IDs avoid extra repetitions.
 pool=list(dict.fromkeys(m['original_sample_index'] for m in manifest if m['emotion_label']==l and m['original_sample_index'] not in selected));rng.shuffle(pool);assert len(pool)>=n;anchor+=pool[:n]
allids=chosen+anchor;assert len(allids)==len(set(allids))==2800;rng.shuffle(allids)
data=R/'training/data/v2_e7_boundary_lora.jsonl';assert not data.exists();data.write_text(''.join(json.dumps(rows[i],ensure_ascii=False)+'\n' for i in allids))
(A/'dataset_manifest.jsonl').write_text(''.join(json.dumps({'original_sample_index':i,'emotion_label':labels[i],'group':'boundary' if i in selected else 'anchor'},ensure_ascii=False)+'\n' for i in allids))
tok=AutoTokenizer.from_pretrained(O/'model/Qwen2___5-7B-Instruct',local_files_only=True,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token
features=prepare_dataset(Dataset.from_list([rows[i] for i in allids]),tok,4096,'train');assert all(any(x!=-100 for x in f['labels']) for f in features);features.save_to_disk(str(R/'training/data/v2_e7_boundary_features'))
runtime_rel=['start.sh','run_inference.py','style_postprocessor.py','configs/system_prompt_v1.txt','configs/inference_config.json','model/lora_adapter/adapter_config.json','model/lora_adapter/adapter_model.safetensors']
setup={'seed':66,'dataset_count':2800,'boundary_count':1960,'anchor_count':840,'unique_original_count':2800,'boundary_quotas':quotas,'hard_selection':priority_stats,'distribution':dict(collections.Counter(labels[i] for i in allids)),'anchor_distribution':dict(collections.Counter(labels[i] for i in anchor)),'train_only':True,'selection_uses_validation':False,'system_prompt_effective_sha256':hashlib.sha256(p0.encode()).hexdigest(),'runtime_hashes':{p:hashlib.sha256((O/p).read_bytes()).hexdigest() for p in runtime_rel},'source_adapter':str(R/'training/outputs/v2-e1-emotion-rebalance-fixed18657-rerun2'),'output_adapter':str(R/'training/outputs/v2-e7-emotion-boundary'),'training':{'max_steps':80,'save_steps':40,'learning_rate':1e-5,'batch':1,'gradient_accumulation':8,'max_length':4096,'assistant_only_loss':True,'packing':False,'gradient_checkpointing':True},'guardrails':{'emotion_delta_min':0,'macro_f1_delta_min':0,'profile_delta_min':-.007,'bleu_delta_min':-.003,'rouge_delta_min':-.003,'bert_delta_min':-.002}}
(A/'setup.json').write_text(json.dumps(setup,ensure_ascii=False,indent=2));print('DATASET_PREPARED',setup,flush=True)
