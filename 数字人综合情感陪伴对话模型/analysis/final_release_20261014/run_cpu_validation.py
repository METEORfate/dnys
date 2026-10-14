"""Offline CPU validation; tokenizer assets are supplied externally, never downloaded."""
import argparse,sys,unittest,os
from pathlib import Path
from unittest.mock import patch
os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",PYTHONDONTWRITEBYTECODE="1")
R=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument("--tokenizer-path",required=True);p.add_argument("--train-data",required=True);args=p.parse_args()
from transformers import AutoTokenizer
original=AutoTokenizer.from_pretrained
def local_tokenizer(path,*a,**kw):
 if str(path)==str(R/"participant/model/Qwen2___5-7B-Instruct"):
  path=args.tokenizer_path
 kw["local_files_only"]=True
 return original(path,*a,**kw)
original_open=Path.open
def external_train_sample(path,*a,**kw):
 if str(path)==str(R/"training/data/train_sft.jsonl"):
  path=Path(args.train_data)
 return original_open(path,*a,**kw)
sys.path.insert(0,str(R));sys.path.insert(0,str(R/"participant"))
with patch.object(AutoTokenizer,"from_pretrained",side_effect=local_tokenizer), patch.object(Path,"open",external_train_sample):
 loader=unittest.TestLoader();suite=loader.discover(str(R/"training"),pattern="test_*.py");suite.addTests(loader.loadTestsFromNames(["test_lora_runtime","test_profile_postprocessor","test_style_postprocessor"]))
 result=unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
