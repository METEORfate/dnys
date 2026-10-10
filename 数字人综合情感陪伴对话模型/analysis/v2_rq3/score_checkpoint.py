import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import pathlib,json,sys,torch
from bert_score import score
R=pathlib.Path(__file__).resolve().parents[2];dest=pathlib.Path(sys.argv[1]);load=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
rows=load(R/'analysis/fast_iter_mini_dev_96.jsonl');pred=load(dest/'prediction.jsonl');assert len(rows)==len(pred)==96
refs=[json.loads(x['messages'][2]['content'])['response_text'] for x in rows];cands=[x['response_text'] for x in pred];device='cuda:0' if torch.cuda.is_available() else 'cpu'
P,Q,F=score(cands,refs,model_type='/root/development/assets/bert-base-multilingual-cased',num_layers=9,lang='zh',rescale_with_baseline=False,device=device,batch_size=32)
(dest/'local_bertscore.json').write_text(json.dumps({'f1':float(F.mean()),'precision':float(P.mean()),'recall':float(Q.mean())}))
