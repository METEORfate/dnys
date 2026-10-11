"""CPU-only regression checks for prompt versioning, split leakage and paired metrics."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from core.prompt_loader import load_system_prompt, prompt_metadata
from prompt_work.run_ab import dataset, selected_rows, metrics, validity
from core.output_schema import normalize_prediction

class WorkflowTest(unittest.TestCase):
    def test_versions_and_default(self):
        self.assertEqual(load_system_prompt(), load_system_prompt("configs/system_prompt_v1.txt"))
        self.assertEqual(load_system_prompt("v0"), load_system_prompt())
        self.assertNotEqual(prompt_metadata("v0")["prompt_sha256"],prompt_metadata("v1")["prompt_sha256"])
        with patch.dict(os.environ, {"DNYS_PROMPT":"v1"}):
            self.assertEqual(load_system_prompt(),load_system_prompt("v1"))

    def test_no_future_or_gold_leak_and_stable_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp=Path(tmp)/"val.jsonl"
            records=[]
            for i in range(12):
                cid=f"val_conv_{i:06d}"
                target={"response_text":"SECRET_GOLD","emotion_label":"sadness",
                        "user_profile":{"personality_traits":[],"interests":[],"style":[]},"memory_refs":[]}
                records.append({"conversation_id":cid,
                    "turns":[{"turn_id":1,"role":"user","content":"不开心"},
                             {"turn_id":2,"role":"assistant","content":"SECRET_GOLD"},
                             {"turn_id":3,"role":"user","content":"FUTURE_SECRET"}],
                    "prediction_points":[{"target_user_turn_id":1,"assistant_response_turn_id":2,"target":target}]})
            fp.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in records),encoding="utf-8")
            dev=dataset(fp,"dev")
            holdout=dataset(fp,"holdout")
            self.assertEqual(len(dev)+len(holdout),12)
            self.assertFalse(set(r["conversation_id"] for r in dev)&set(r["conversation_id"] for r in holdout))
            self.assertTrue(all(r["history"]==[{"role":"user","content":"不开心"}] for r in dev+holdout))
            self.assertEqual(selected_rows(dev,"smoke",42),selected_rows(dev,"smoke",42))

    def test_raw_invalid_not_hidden(self):
        raw={"response_text":"hi","emotion_label":"WRONG","user_profile":{"personality_traits":[],"interests":[],"style":[]},"memory_refs":[]}
        flags=validity(raw)
        self.assertFalse(flags["emotion_valid"])
        self.assertFalse(flags["exact_fields"] is False)
        self.assertEqual(normalize_prediction("id",raw)["emotion_label"],"")

    def test_metric_shapes(self):
        gold={"response_text":"你好","emotion_label":"joy","user_profile":{"personality_traits":[],"interests":[],"style":[]},"memory_refs":[]}
        pred=normalize_prediction("a",gold)
        scores=metrics([{"prediction":pred,"target":gold,"latency_ms":3.0,"error":None,"validity":validity(gold)}])
        self.assertEqual(scores["emotion_accuracy"],1)
        self.assertEqual(scores["profile_mean_accuracy"],1)
        self.assertEqual(scores["inference_failures"],0)

if __name__ == "__main__":
    unittest.main()
