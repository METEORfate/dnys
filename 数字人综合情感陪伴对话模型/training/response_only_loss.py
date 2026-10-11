"""Response-content-only labels, using offsets of the exact full chat rendering."""
import json,re
from loss_mask_utils import render_prompt_and_full,truncate_for_training

def response_labels(tokenizer,messages,max_length=4096):
    messages,details=truncate_for_training(tokenizer,messages,max_length)
    if messages is None or details['status']!='ok':raise ValueError('sample cannot fit without target truncation')
    prompt_ids,full_ids=render_prompt_and_full(tokenizer,messages)
    prompt=tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True)
    full=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    assert full.startswith(prompt)
    answer=messages[-1]['content'];assert full[len(prompt):].startswith(answer)
    match=re.search(r'"response_text"\s*:\s*',answer);assert match
    value_start=match.end();value,end=json.JSONDecoder().raw_decode(answer,value_start)
    assert isinstance(value,str) and answer[value_start]=='"' and answer[end-1]=='"'
    start=len(prompt)+value_start+1;stop=len(prompt)+end-1
    encoded=tokenizer(full,add_special_tokens=False,return_offsets_mapping=True)
    assert encoded['input_ids']==full_ids,'full-render offsets/token IDs mismatch'
    labels=[-100]*len(full_ids);supervised=[]
    for i,(a,b) in enumerate(encoded['offset_mapping']):
        if b>a and a>=start and b<=stop:
            assert i>=len(prompt_ids)
            labels[i]=full_ids[i];supervised.append((i,a,b))
    if not supervised:raise ValueError('zero response tokens')
    assert all(start<=a<b<=stop for _,a,b in supervised)
    return {'input_ids':full_ids,'attention_mask':[1]*len(full_ids),'labels':labels}, {'response_tokens':len(supervised),'non_response_supervised':0,'emotion_tokens_supervised':0,'profile_tokens_supervised':0,'memory_tokens_supervised':0,'response_span':[start,stop],'trimmed':bool(details.get('deleted_tokens',0))}
