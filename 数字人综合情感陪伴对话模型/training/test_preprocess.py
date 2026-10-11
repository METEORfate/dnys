import json
import tempfile
import unittest
from pathlib import Path

try:
    from .preprocess import build_samples, convert, preview
except ImportError:
    from preprocess import build_samples, convert, preview


ROW = {
    "conversation_id": "train_conv_000001",
    # Deliberately out of array order: truncation must use turn_id.
    "turns": [
        {"turn_id": 4, "role": "assistant", "content": "FUTURE_RESPONSE"},
        {"turn_id": 2, "role": "assistant", "content": "earlier reply"},
        {"turn_id": 3, "role": "user", "content": "current question"},
        {"turn_id": 1, "role": "user", "content": "earlier question"},
        {"turn_id": 5, "role": "user", "content": "FUTURE_USER"},
    ],
    "prediction_points": [{
        "target_user_turn_id": 3,
        "assistant_response_turn_id": 4,
        "target": {
            "response_text": "FUTURE_RESPONSE",
            "emotion_label": "neutral",
            "user_profile": {"personality_traits": [], "interests": [], "style": []},
            "memory_refs": ["mem_000001"],
        },
    }],
}


class PreprocessTest(unittest.TestCase):
    def test_turn_id_cutoff_and_target_order(self):
        sample = next(build_samples(ROW, False))
        prompt = sample["messages"][1]["content"]
        self.assertEqual(sample["messages"][0]["role"], "system")
        self.assertEqual(sample["messages"][2]["role"], "assistant")
        self.assertIn("earlier reply", prompt)
        self.assertLess(prompt.index("earlier question"), prompt.index("earlier reply"))
        self.assertIn("current question", prompt)
        self.assertNotIn("FUTURE_RESPONSE", prompt)
        self.assertNotIn("FUTURE_USER", prompt)
        answer = json.loads(sample["messages"][2]["content"])
        self.assertEqual(list(answer), ["response_text", "emotion_label", "user_profile", "memory_refs"])
        self.assertEqual(answer["memory_refs"], [])
        self.assertEqual(json.loads(next(build_samples(ROW, True))["messages"][2]["content"])["memory_refs"], ["mem_000001"])

    def test_preview_does_not_write_output(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.jsonl"
            output = Path(temp) / "output.jsonl"
            source.write_text(json.dumps(ROW) + "\n" + json.dumps(ROW) + "\n", encoding="utf-8")
            self.assertEqual(len(preview(source, 1, False)), 1)
            self.assertFalse(output.exists())
            stats = convert(source, output, False)
            self.assertEqual(stats["samples"], 2)
            self.assertEqual(stats["emotion_distribution"], {"neutral": 2})
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
