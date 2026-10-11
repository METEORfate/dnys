"""Fixed 3-response/1-emotion rank-local continued QLoRA training."""
import os,json,sys
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'training'))
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,set_seed
from peft import PeftModel,prepare_model_for_kbit_training
from loss_mask_utils import ExplicitLabelDataCollator
from dual_objective_stream import DualObjectiveStream
from train_lora import build_training_args,build_model_kwargs,import_training_stack
from types import SimpleNamespace
class FixedStreamTrainer(Trainer):
 def get_train_dataloader(self):
  # Each rank owns its stream; do not let Accelerate shard it a second time.
  return DataLoader(self.train_dataset,batch_size=1,num_workers=0,collate_fn=self.data_collator)
def main():
 audit=json.loads((R/'analysis/v2_rq6/mask_audit.json').read_text());assert audit['status']=='PASS' and audit['zero_supervised_samples']==0
 rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2
 torch.cuda.set_device(rank);assert torch.cuda.is_available();set_seed(61)
 output=R/'training/outputs/v2-rq6-response-emotion-anchor';options=SimpleNamespace(output_dir=output,num_train_epochs=1.,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,learning_rate=3e-6,gradient_checkpointing=True,seed=61,max_seq_length=4096,load_in_4bit=True)
 stack=import_training_stack();args=build_training_args(stack,options);args.max_steps=160;args.save_steps=40;args.save_total_limit=4;args.eval_strategy='no';args.do_eval=False;args.logging_steps=10
 path=R/'participant/model/Qwen2___5-7B-Instruct';adapter=R/'training/outputs/v2-e1-emotion-rebalance-fixed18657-rerun2'
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='right'
 base=AutoModelForCausalLM.from_pretrained(path,**build_model_kwargs(torch,stack,options,rank));base.config.use_cache=False;base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
 model=PeftModel.from_pretrained(base,adapter,is_trainable=True);config=json.loads((adapter/'adapter_config.json').read_text());pc=model.peft_config['default'];assert pc.r==config['r'] and pc.lora_alpha==config['lora_alpha'] and pc.lora_dropout==config['lora_dropout'] and set(pc.target_modules)==set(config['target_modules'])
 response=load_from_disk(str(R/'training/data/v2_rq3_response_features'));emotion=load_from_disk(str(R/'training/data/v2_rq6_emotion_features'));stream=DualObjectiveStream(response,emotion,rank,world)
 print('CONTINUED_V2E1_LOADED','rank',rank,'world_size',world,'mix','RRRE','response_batches',960,'emotion_batches',320,'lr',args.learning_rate,'scheduler',args.lr_scheduler_type,flush=True)
 trainer=FixedStreamTrainer(model=model,args=args,train_dataset=stream,data_collator=ExplicitLabelDataCollator(tok),processing_class=tok)
 # Preserve per-microbatch mean CE, so long responses do not dilute anchors.
 trainer.model_accepts_loss_kwargs=False
 trainer.train();trainer.save_model(str(output))
 if rank==0:tok.save_pretrained(str(output))
 print('RQ6_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
