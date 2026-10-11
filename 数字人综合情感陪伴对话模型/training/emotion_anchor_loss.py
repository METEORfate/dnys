"""Supervise only the top-level emotion_label JSON string value."""
import json
from loss_mask_utils import render_prompt_and_full,truncate_for_training

def emotion_labels(tokenizer,messages,max_length=4096):
 messages,details=truncate_for_training(tokenizer,messages,max_length)
 if messages is None or details['status']!='ok':raise ValueError('target/context cannot fit')
 prompt_ids,full_ids=render_prompt_and_full(tokenizer,messages)
 prompt=tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True)
 full=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
 answer=messages[-1]['content'];assert full.startswith(prompt) and full[len(prompt):].startswith(answer)
 decoder=json.JSONDecoder();pos=answer.index('{')+1;span=None
 while pos<len(answer):
  while pos<len(answer) and (answer[pos].isspace() or answer[pos]==','):pos+=1
  if answer[pos]=='}':break
  key,end=decoder.raw_decode(answer,pos);pos=end
  while answer[pos].isspace():pos+=1
  assert answer[pos]==':';pos+=1
  while answer[pos].isspace():pos+=1
  value,end=decoder.raw_decode(answer,pos)
  if key=='emotion_label':
   assert isinstance(value,str) and answer[pos]=='"' and answer[end-1]=='"';span=(len(prompt)+pos+1,len(prompt)+end-1)
  pos=end
 assert span is not None
 encoded=tokenizer(full,add_special_tokens=False,return_offsets_mapping=True);assert encoded['input_ids']==full_ids
 labels=[-100]*len(full_ids);count=0
 for i,(a,b) in enumerate(encoded['offset_mapping']):
  if b>a and a>=span[0] and b<=span[1]:
   assert i>=len(prompt_ids);labels[i]=full_ids[i];count+=1
 if not count:raise ValueError('zero emotion tokens')
 return {'input_ids':full_ids,'attention_mask':[1]*len(full_ids),'labels':labels},{'emotion_tokens':count,'response_tokens_supervised':0,'profile_tokens_supervised':0,'memory_tokens_supervised':0,'non_emotion_supervised':0,'emotion_span':span}
