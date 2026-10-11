import json
import tempfile
import unittest
from pathlib import Path

try:
    from .create_validation_split import create_split
    from .preprocess import build_samples
except ImportError:
    from create_validation_split import create_split
    from preprocess import build_samples


def make_row(conversation_id: str, point_count: int) -> dict:
    turns = []
    points = []
    for index in range(point_count):
        user_id = index * 2 + 1
        assistant_id = user_id + 1
        response = f"answer-{conversation_id}-{index}"
        turns.extend([
            {"turn_id": user_id, "role": "user", "content": f"user-{conversation_id}-{index}"},
            {"turn_id": assistant_id, "role": "assistant", "content": response},
        ])
        points.append({
            "target_user_turn_id": user_id,
            "assistant_response_turn_id": assistant_id,
            "target": {
                "response_text": response,
                "emotion_label": "neutral",
                "user_profile": {"personality_traits": [], "interests": [], "style": []},
                "memory_refs": [],
            },
        })
    return {"conversation_id": conversation_id, "turns": turns, "prediction_points": points}


class ValidationSplitTest(unittest.TestCase):
    def test_conversation_disjoint_reproducible_and_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "val_public.jsonl"
            sft = root / "val_sft.jsonl"
            rows = [make_row("a", 2), make_row("b", 1), make_row("c", 2), make_row("d", 1)]
            raw.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
            sft_rows = []
            for row in rows:
                sft_rows.extend(build_samples(row, False))
            sft.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in sft_rows) + "\n", encoding="utf-8")
            first = create_split(raw, sft, root / "dev1.jsonl", root / "hold1.jsonl", root / "manifest1.json", seed=42)
            second = create_split(raw, sft, root / "dev2.jsonl", root / "hold2.jsonl", root / "manifest2.json", seed=42)
            self.assertEqual(first["source_sample_count"], 6)
            self.assertEqual(first["dev_sample_count"] + first["holdout_sample_count"], 6)
            self.assertEqual(first["conversation_overlap_count"], 0)
            self.assertEqual(first["duplicate_prediction_point_count"], 0)
            self.assertEqual(first["missing_prediction_point_count"], 0)
            self.assertEqual(first["dev_file_hash"], second["dev_file_hash"])
            self.assertEqual(first["holdout_file_hash"], second["holdout_file_hash"])

            dev = [json.loads(line) for line in (root / "dev1.jsonl").read_text(encoding="utf-8").splitlines()]
            holdout = [json.loads(line) for line in (root / "hold1.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(dev) + len(holdout), 6)


if __name__ == "__main__":
    unittest.main()
