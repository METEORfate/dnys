
import sys,json,hashlib,torch
from pathlib import Path
from safetensors.torch import load_file,save_file
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[2];A=Path(__file__).parent;O=Path('/root/participant');BASE=R/'training/outputs/v2-e1-emotion-rebalance-fixed18657-rerun2';TARGET=R/'training/outputs/v2-e7-emotion-boundary/checkpoint-80'
c=json.loads((BASE/'adapter_config.json').read_text());d=json.loads((TARGET/'adapter_config.json').read_text())
fields=['r','lora_alpha','target_modules','rank_pattern','alpha_pattern','bias','lora_dropout','fan_in_fan_out','use_rslora','use_dora','modules_to_save','task_type','peft_type','layers_to_transform','layers_pattern']
for k in fields:assert (set(c[k])==set(d[k]) if k=='target_modules' else c.get(k)==d.get(k)),k
x=load_file(str(BASE/'adapter_model.safetensors'));y=load_file(str(TARGET/'adapter_model.safetensors'));assert x.keys()==y.keys();assert all(x[k].shape==y[k].shape for k in x);assert all('lora_A' in k or 'lora_B' in k for k in x);assert all(torch.isfinite(v).all() for v in list(x.values())+list(y.values()))
assert all(torch.equal(torch.lerp(x[k].float(),y[k].float(),0).to(x[k].dtype),x[k]) and torch.equal(torch.lerp(x[k].float(),y[k].float(),1).to(y[k].dtype),y[k]) for k in x)
assert hashlib.sha256((BASE/'adapter_model.safetensors').read_bytes()).hexdigest()==hashlib.sha256((O/'model/lora_adapter/adapter_model.safetensors').read_bytes()).hexdigest(),'OFFICIAL_MODEL_NOT_V2E1'
candidates={}
for alpha in [.10,.15,.20]:
 name=f'a{round(alpha*100):03d}';out=R/f'training/outputs/v2-e7-2-interp-{name}';assert not out.exists(),str(out);out.mkdir()
 z={k:torch.lerp(x[k].float(),y[k].float(),alpha).contiguous() for k in x};assert all(torch.isfinite(v).all() for v in z.values());save_file(z,str(out/'adapter_model.safetensors'),metadata={'format':'pt'});(out/'adapter_config.json').write_text((BASE/'adapter_config.json').read_text());back=load_file(str(out/'adapter_model.safetensors'));assert all(torch.equal(z[k],back[k]) for k in z)
 candidates[name]={'alpha':alpha,'adapter_path':str(out),'finite':True,'saved_tensor_exact_verified':True,'sha256':hashlib.sha256((out/'adapter_model.safetensors').read_bytes()).hexdigest()}
prior=json.loads((R/'analysis/v2_e7_boundary_lora/setup.json').read_text());assert all(hashlib.sha256((O/p).read_bytes()).hexdigest()==h for p,h in prior['runtime_hashes'].items())
setup={'base_adapter':str(BASE),'target_adapter':str(TARGET),'adapter_compatible':True,'keys':len(x),'rank':c['r'],'lora_alpha':c['lora_alpha'],'target_modules':c['target_modules'],'layer_count':json.loads((O/'model/Qwen2___5-7B-Instruct/config.json').read_text())['num_hidden_layers'],'endpoint_parameters_exact':True,'interpolation':'W = W_V2E1 + alpha * (W_E7_80 - W_V2E1), parameter space, FP32','candidates':candidates,'runtime_hashes':prior['runtime_hashes'],'guardrails':{'emotion_delta_min':0,'profile_delta_min':0,'bleu_delta_min':0,'rouge_delta_min':0,'bert_delta_min':0},'hard_profile_reject_delta':-.007,'fast192_rule':'Only emotion>=CURRENT AND profile_mean>=CURRENT AND response protection pass; JSON/nonempty/fallback also protected','training_performed':False,'formal_runtime_modified':False}
previous=json.loads((R/'analysis/v2_e7_1_interpolation/setup.json').read_text())['candidates']['a025'];assert hashlib.sha256((Path(previous['adapter_path'])/'adapter_model.safetensors').read_bytes()).hexdigest()==previous['sha256'];setup['candidates']['a025']=previous;setup['new_alphas']=[.10,.15,.20];setup['reference_alpha']=.25
(A/'setup.json').write_text(json.dumps(setup,ensure_ascii=False,indent=2));print(json.dumps(setup,ensure_ascii=False,indent=2))
