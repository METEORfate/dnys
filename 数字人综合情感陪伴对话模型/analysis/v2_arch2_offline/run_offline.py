import os,sys,time,json,copy,gc,hashlib,collections
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1');sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[2];A=Path(__file__).parent;O=Path('/root/participant');load=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
state={'phase':'GPU_PREFLIGHT','started_at':time.time()};save(A/'status.json',state)
try:
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
 from peft import PeftModel
 assert torch.cuda.is_available() and torch.cuda.device_count()==2,'GPU_UNAVAILABLE'
 torch.cuda.set_device(0);x=torch.ones(2,2,device='cuda:0');x@x;torch.cuda.synchronize();del x
 def memory():
  torch.cuda.synchronize(0);free,total=torch.cuda.mem_get_info(0);return {'allocated_mib':torch.cuda.memory_allocated(0)/2**20,'reserved_mib':torch.cuda.memory_reserved(0)/2**20,'free_mib':free/2**20,'peak_allocated_mib':torch.cuda.max_memory_allocated(0)/2**20}
 state['phase']='MODEL_LOAD';save(A/'status.json',state)
 base=O/'model/Qwen2___5-7B-Instruct';adapter=R/'training/outputs/v2-e1-emotion-rebalance-fixed18657-rerun2'
 tokenizer=AutoTokenizer.from_pretrained(str(base),local_files_only=True);quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
 model=AutoModelForCausalLM.from_pretrained(str(base),local_files_only=True,torch_dtype=torch.bfloat16,quantization_config=quant,device_map={'':'cuda:0'},attn_implementation='sdpa');model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False);model.eval();assert not model.training
 for p in model.parameters():p.requires_grad_(False)
 sys.path.insert(0,str(O));from core.message_builder import build_messages
 prompt=(R/'analysis/v2_arch2/classifier_prompt.txt').read_text();label_ids=json.loads((R/'analysis/v2_arch2/label_tokens.json').read_text());labels=list(label_ids)
 assert all(tokenizer.encode(l,add_special_tokens=False)==ids for l,ids in label_ids.items())
 inputs=load(R/'analysis/v2_pr1/inputs96.jsonl');rows=load(R/'analysis/fast_iter_mini_dev_96.jsonl');assert len(inputs)==len(rows)==96
 before=memory();torch.cuda.reset_peak_memory_stats(0);records=[]
 def scorer(q,rowindex):
  messages=build_messages(q);messages[0]['content']=prompt;text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);ids=tokenizer.encode(text,add_special_tokens=False);assert all(tokenizer.encode(text+l,add_special_tokens=False)==ids+tokens for l,tokens in label_ids.items());start=time.perf_counter()
  scores={}
  with torch.no_grad():
   for label,tokens in label_ids.items():
    full_ids=ids+tokens;input_ids=torch.tensor([full_ids],device='cuda:0')
    # The retained logits cover the positions immediately before every forced label token.
    out=model(input_ids=input_ids,use_cache=False,logits_to_keep=len(tokens)+1)
    assert out.past_key_values is None and out.logits.shape[1]==len(tokens)+1
    label_logits=out.logits[0,:-1].float();logprobs=label_logits.log_softmax(dim=-1)
    values=[float(logprobs[j,tok].item()) for j,tok in enumerate(tokens)]
    assert len(values)==len(tokens)>0
    scores[label]={'token_logprobs':values,'sum':sum(values),'mean':sum(values)/len(values)}
    del input_ids,out,label_logits,logprobs
  gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();latency=(time.perf_counter()-start)*1000;pred={}
  for method in ['sum','mean']:
   ranking=sorted(labels,key=lambda l:scores[l][method],reverse=True);pred[method]={'label':ranking[0],'margin':scores[ranking[0]][method]-scores[ranking[1]][method]}
  return {'row_index':rowindex,'id':q['id'],'scores':scores,'predictions':pred,'latency_ms':latency,'memory_after':memory()}
 for i,q in enumerate(inputs):
  state.update(phase='FAST20' if i<20 else 'FAST96',row_index=i);save(A/'status.json',state);record=scorer(q,i);records.append(record)
  with (A/'scores96.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
  print(i+1,'/96',record['memory_after'],flush=True)
  if i==19:
   summary={'completed':20,'gpu_memory_before':before,'gpu_memory_after':memory(),'oom':'NO','prediction_distribution':{m:dict(collections.Counter(q['predictions'][m]['label'] for q in records)) for m in ['sum','mean']}};save(A/'fast20.json',summary)
   assert summary['gpu_memory_after']['allocated_mib']<=before['allocated_mib']+64,'CROSS_SAMPLE_CACHE_LEAK'
 gt=[json.loads(q['messages'][2]['content'])['emotion_label'] for q in rows];metrics={m:{'accuracy':sum(q['predictions'][m]['label']==t for q,t in zip(records,gt))/96,'prediction_distribution':dict(collections.Counter(q['predictions'][m]['label'] for q in records))} for m in ['sum','mean']}
 hashes=json.loads((R/'analysis/v2_pr2/setup.json').read_text())['runtime_hashes'];assert all(hashlib.sha256((O/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
 save(A/'result.json',{'offline_scorer':'PASS','fast20':json.loads((A/'fast20.json').read_text()),'fast96_completed':96,'gpu_memory_final':memory(),'oom':'NO','metrics':metrics,'model_loading':'same existing offline evaluator NF4 double quant, bf16 compute; cuda:0; eval/no_grad; batch=1; direct teacher-forcing, use_cache=False; logits_to_keep=label_count+1','prompt_unchanged':True,'formal_runtime_modified':False,'blocking_issue':'NONE','elapsed_minutes':(time.time()-state['started_at'])/60});state['phase']='COMPLETED';save(A/'status.json',state)
except Exception as e:
 state.update(phase='BLOCKED',blocking_issue=str(e) or type(e).__name__);save(A/'status.json',state);raise
