import os,argparse,json,time,re,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--prompt');ap.add_argument('--gpu');a=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;os.environ['HF_HUB_OFFLINE']='1'
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
from peft import PeftModel
R=Path(__file__).resolve().parents[1];O=R/'analysis/v2_c3_second_pass';O.mkdir(exist_ok=True)
rows=[json.loads(x) for x in open(R/'analysis/fast_iter_mini_dev_96.jsonl') if x.strip()];base=[json.loads(x) for x in open(R/'training/eval_results/fast_A_96/prediction.jsonl') if x.strip()];assert len(rows)==len(base)==96
labels=['anxiety','relaxed','helplessness','fear','sadness','mixed','loneliness','anger','shame','gratitude','neutral','pride','joy','disgust','care','surprise'];gt=[json.loads(x['messages'][-1]['content'])['emotion_label'] for x in rows];assert sum(y==x['emotion_label'] for y,x in zip(gt,base))==67
prompt='根据完整对话，判断最新一条用户消息的主要情绪。只能选择一个标签：'+', '.join(labels)+'。只输出标签本身，禁止解释。'
if a.prompt=='B':prompt+='\nanxiety：持续担忧、不确定性、预期性紧张；fear：针对较明确威胁或事件的害怕；relaxed：担忧解除、释然、放松；gratitude：明确感谢、感激对象；mixed：至少两种重要情绪同时占主导，不因语气复杂就使用；helplessness：明显无力、无控制感、不知道还能怎么办；sadness：失落、难过、低落为主；neutral：没有明显强情绪。其余标签按通常含义判断。'
mp=R/'participant/model/Qwen2___5-7B-Instruct';tok=AutoTokenizer.from_pretrained(mp,local_files_only=True);quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
m=AutoModelForCausalLM.from_pretrained(mp,local_files_only=True,torch_dtype=torch.bfloat16,quantization_config=quant,device_map={'':0});m=PeftModel.from_pretrained(m,R/'training/outputs/v2-e1-emotion-rebalance-fixed18657-rerun2',is_trainable=False);m.eval();out=[]
with open(O/(a.prompt+'_predictions.jsonl'),'w') as f:
 for i,row in enumerate(rows):
  # The SFT user content contains only serialized real conversation; exclude target and original system.
  conv=next(x['content'] for x in row['messages'] if x['role']=='user');z=tok.apply_chat_template([{'role':'system','content':prompt},{'role':'user','content':conv}],tokenize=True,add_generation_prompt=True,return_tensors='pt').to('cuda:0')
  assert z.shape[1]+8<=m.config.max_position_embeddings
  torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode():q=m.generate(z,attention_mask=torch.ones_like(z),max_new_tokens=8,do_sample=False,pad_token_id=tok.eos_token_id)
  torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000;raw=tok.decode(q[0,z.shape[1]:],skip_special_tokens=True).strip();found=re.fullmatch(r'[\s"\'`]*('+ '|'.join(labels)+r')[\s"\'`。.！!]*',raw,re.I);pred=found.group(1).lower() if found else base[i]['emotion_label'];x={'row_index':i,'gt':gt[i],'baseline':base[i]['emotion_label'],'raw':raw,'prediction':pred,'fallback':found is None,'latency_ms':ms};out.append(x);f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();print(a.prompt,i+1,96,flush=True)
json.dump(out,open(O/(a.prompt+'.json'),'w'),ensure_ascii=False,indent=2)
