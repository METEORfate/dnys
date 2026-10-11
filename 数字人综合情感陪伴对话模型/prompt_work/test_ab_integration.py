"""CPU-only end-to-end A/B CLI test with a fake model endpoint.

No GPU, network service, model load, or real benchmark numbers are involved.
"""
from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prompt_work import run_ab

class NoopStyleProcessor:
    loaded = False
    def apply(self, normalized, history):
        return normalized

class ABIntegrationTest(unittest.TestCase):
    def test_full_pipeline_on_fake_model(self):
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            source = tmp / "val_public.jsonl"
            records = []
            for number in range(10):
                cid = f"val_conv_{number:06d}"
                gold = {
                    "response_text": "测试回答", "emotion_label": "sadness",
                    "user_profile": {"personality_traits": [], "interests": [], "style": []},
                    "memory_refs": []
                }
                records.append({
                    "conversation_id": cid,
                    "turns": [
                        {"turn_id": 1, "role": "user", "content": "我有点难过"},
                        {"turn_id": 2, "role": "assistant", "content": "测试回答"},
                        {"turn_id": 3, "role": "user", "content": "FUTURE_SECRET"}
                    ],
                    "prediction_points": [{
                        "target_user_turn_id": 1, "assistant_response_turn_id": 2, "target": gold
                    }]
                })
            source.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in records), encoding="utf-8")
            calls = []
            def fake_request(cfg, messages, generation):
                self.assertNotIn("FUTURE_SECRET", str(messages))
                self.assertNotIn("测试回答", str(messages))
                self.assertEqual(generation, {"max_new_tokens": 120, "do_sample": False})
                calls.append(messages[0]["content"])
                if "尤其当用户明确拒绝建议" in messages[0]["content"]:
                    return json.dumps({
                        "response_text": "我听见你很难过", "emotion_label": "sadness",
                        "user_profile": {"personality_traits": [], "interests": [], "style": []},
                        "memory_refs": []
                    }, ensure_ascii=False)
                return "invalid response"
            args = [
                "run_ab.py", "--prompts", "v0", "v1", "--tier", "smoke",
                "--data-file", str(source), "--output-dir", str(tmp / "reports")
            ]
            fake_config = {
                "backend": "openai_compatible", "api_model": "fake-lora",
                "base_url": "http://127.0.0.1:9001/v1",
                "api_key": "", "api_options": {},
                "generation": {"max_new_tokens": 120, "do_sample": False}
            }
            with patch.object(sys, "argv", args), \
                 patch.object(run_ab, "load_config", return_value=fake_config), \
                 patch.object(run_ab, "request_chat_completion", side_effect=fake_request), \
                 patch.object(run_ab, "StylePostprocessor", NoopStyleProcessor):
                run_ab.main()
            output_dirs = list((tmp / "reports").iterdir())
            self.assertEqual(len(output_dirs), 1)
            run_dir = output_dirs[0]
            report = json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(report["tier"], "smoke")
            self.assertEqual(len(report["versions"]), 2)
            self.assertEqual(report["paired_emotion_improvements"], len(calls)//2)
            self.assertEqual(report["paired_emotion_regressions"], 0)
            baseline = json.loads((run_dir / "prompt_v0" / "metrics.json").read_text(encoding="utf-8"))
            updated = json.loads((run_dir / "prompt_v1" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(baseline["model_raw_normalized"]["raw_validity_rates"]["json_object"], 0)
            self.assertEqual(updated["model_raw_normalized"]["raw_validity_rates"]["json_object"], 1)
            self.assertEqual(baseline["model_raw_normalized"]["json_parse_failures"], len(calls)//2)
            self.assertEqual(updated["model_raw_normalized"]["json_parse_failures"], 0)
            self.assertEqual(len(list((run_dir / "prompt_v0" / "predictions.jsonl").open(encoding="utf-8"))), len(calls)//2)
            self.assertEqual(len(calls) % 2, 0)

if __name__ == "__main__":
    unittest.main()
