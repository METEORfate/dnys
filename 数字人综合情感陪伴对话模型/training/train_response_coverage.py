"""Continued response-only QLoRA micro-SFT; isolated from the official runtime."""
import os,json,sys,inspect
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'training'))
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,set_seed
from peft import PeftModel,prepare_model_for_kbit_training
from loss_mask_utils import ExplicitLabelDataCollator
from train_lora import build_training_args,build_model_kwargs,import_training_stack
from types import SimpleNamespace
A=R/'analysis/v2_rq3';audit=json.loads((A/'mask_audit.json').read_text());assert audit['status']=='PASS' and audit['zero_response_samples']==0
rank=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(rank);assert torch.cuda.is_available();set_seed(42)
output=R/'training/outputs/v2-rq3-response-coverage-recovery'
options=SimpleNamespace(output_dir=output,num_train_epochs=1.,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,learning_rate=5e-6,gradient_checkpointing=True,seed=42,max_seq_length=4096,load_in_4bit=True)
stack=import_training_stack();args=build_training_args(stack,options);args.max_steps=240;args.save_steps=80;args.save_total_limit=3;args.eval_strategy='no';args.do_eval=False;args.logging_steps=10
modelpath=R/'participant/model/Qwen2___5-7B-Instruct';adapter=R/'training/outputs/v2-e1-emotion-rebalance-fixed18657-rerun2'
tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='right'
base=AutoModelForCausalLM.from_pretrained(modelpath,**build_model_kwargs(torch,stack,options,rank));base.config.use_cache=False;base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
model=PeftModel.from_pretrained(base,adapter,is_trainable=True)
config=json.loads((adapter/'adapter_config.json').read_text());pc=model.peft_config['default'];assert pc.r==config['r'] and pc.lora_alpha==config['lora_alpha'] and pc.lora_dropout==config['lora_dropout'] and set(pc.target_modules)==set(config['target_modules']);assert any(p.requires_grad for n,p in model.named_parameters() if 'lora_' in n)
print('CONTINUED_ADAPTER_LOADED',str(adapter),'RANK',rank,'WORLD_SIZE',os.environ.get('WORLD_SIZE'),'LR',args.learning_rate,'SCHEDULER',args.lr_scheduler_type,flush=True)
train=load_from_disk(str(R/'training/data/v2_rq3_response_features'));assert len(train)==audit['samples']
trainer=Trainer(model=model,args=args,train_dataset=train,data_collator=ExplicitLabelDataCollator(tok),processing_class=tok)
trainer.train();trainer.save_model(str(output));tok.save_pretrained(str(output));print('RQ3_TRAINING_COMPLETE',flush=True)
