import json
import tempfile
import unittest
from pathlib import Path

try:
    from .validate_sft_alignment import validate
    from .preprocess import build_samples
except ImportError:
    from validate_sft_alignment import validate
    from preprocess import build_samples


class SFTAlignmentTest(unittest.TestCase):
    def test_target_user_and_no_leaks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw.jsonl"
            sft = root / "sft.jsonl"
            raw_row = {"conversation_id": "train_conv_000001", "turns": [{"turn_id": 1, "role": "user", "content": "old"}, {"turn_id": 2, "role": "assistant", "content": "old answer"}, {"turn_id": 3, "role": "user", "content": "target user"}, {"turn_id": 4, "role": "assistant", "content": "target answer"}, {"turn_id": 5, "role": "user", "content": "future"}], "prediction_points": [{"target_user_turn_id": 3, "assistant_response_turn_id": 4, "target": {"response_text": "target answer", "emotion_label": "neutral", "user_profile": {"personality_traits": [], "interests": [], "style": []}, "memory_refs": ["mem_000001"]}}]}
            sft_row = {"messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "对话历史:\nuser: old\nassistant: old answer\nuser: target user"}, {"role": "assistant", "content": json.dumps(raw_row["prediction_points"][0]["target"], ensure_ascii=False, separators=(",", ":"))}]}
            raw.write_text(json.dumps(raw_row, ensure_ascii=False) + "\n", encoding="utf-8")
            sft.write_text(json.dumps(sft_row, ensure_ascii=False) + "\n", encoding="utf-8")
            result = validate(raw, sft, 0, 42)
            self.assertEqual(result["aligned_count"], 1)
            for key in ("missing_target_user_count", "target_assistant_leak_count", "future_turn_leak_count", "target_mismatch_count", "duplicate_count", "missing_count"):
                self.assertEqual(result[key], 0)

    def test_multiple_prediction_points_and_first_point(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw.jsonl"
            sft = root / "sft.jsonl"
            targets = [
                {"response_text": "answer one", "emotion_label": "neutral", "user_profile": {"personality_traits": [], "interests": [], "style": []}, "memory_refs": []},
                {"response_text": "answer two", "emotion_label": "joy", "user_profile": {"personality_traits": ["warm"], "interests": [], "style": ["direct"]}, "memory_refs": []},
            ]
            raw_row = {
                "conversation_id": "multi",
                "turns": [
                    {"turn_id": 1, "role": "user", "content": "first user"},
                    {"turn_id": 2, "role": "assistant", "content": "answer one"},
                    {"turn_id": 3, "role": "user", "content": "second user"},
                    {"turn_id": 4, "role": "assistant", "content": "answer two"},
                    {"turn_id": 5, "role": "user", "content": "future user"},
                ],
                "prediction_points": [
                    {"target_user_turn_id": 1, "assistant_response_turn_id": 2, "target": targets[0]},
                    {"target_user_turn_id": 3, "assistant_response_turn_id": 4, "target": targets[1]},
                ],
            }
            generated = list(build_samples(raw_row, False))
            raw.write_text(json.dumps(raw_row, ensure_ascii=False) + "\n", encoding="utf-8")
            sft.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in generated) + "\n", encoding="utf-8")
            result = validate(raw, sft, 0, 42)
            self.assertEqual(result["total_prediction_points"], 2)
            self.assertEqual(result["aligned_count"], 2)
            self.assertEqual(result["missing_target_user_count"], 0)
            self.assertEqual(result["target_assistant_leak_count"], 0)
            self.assertEqual(result["future_turn_leak_count"], 0)
            self.assertEqual(result["target_mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
