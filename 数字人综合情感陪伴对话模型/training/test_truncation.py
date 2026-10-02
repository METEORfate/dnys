import unittest

try:
    from .loss_mask_utils import build_explicit_labels, render_prompt_and_full, truncate_for_training
except ImportError:
    from loss_mask_utils import build_explicit_labels, render_prompt_and_full, truncate_for_training


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "|".join(f"{m['role']}:{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "|assistant:"
        return list(range(len(text)))


class TruncationTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_short_sample_unchanged(self):
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "对话历史:\nold\nnew"}, {"role": "assistant", "content": "answer"}]
        result, details = truncate_for_training(self.tokenizer, messages, 100)
        self.assertIsNotNone(result)
        self.assertEqual(details["deleted_tokens"], 0)

    def test_empty_history_is_reported_without_crashing(self):
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": ""}, {"role": "assistant", "content": "answer"}]
        result, details = truncate_for_training(self.tokenizer, messages, 100)
        self.assertIsNotNone(result)
        self.assertEqual(details["status"], "ok")

    def test_old_history_removed_first(self):
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "对话历史:\nold1\nold2\nnew"}, {"role": "assistant", "content": "answer"}]
        result, details = truncate_for_training(self.tokenizer, messages, 45)
        self.assertIsNotNone(result)
        self.assertNotIn("old1", result[1]["content"])
        self.assertIn("new", result[1]["content"])
        self.assertGreater(details["deleted_tokens"], 0)
        features = build_explicit_labels(self.tokenizer, result)
        prompt, full = render_prompt_and_full(self.tokenizer, result)
        self.assertEqual(full[:len(prompt)], prompt)
        self.assertEqual(features["labels"][:len(prompt)], [-100] * len(prompt))
        self.assertGreater(sum(label != -100 for label in features["labels"]), 0)

    def test_assistant_target_over_budget_is_reported(self):
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "对话历史:\nnew"}, {"role": "assistant", "content": "x" * 100}]
        result, details = truncate_for_training(self.tokenizer, messages, 20)
        self.assertIsNone(result)
        self.assertEqual(details["status"], "assistant_target_over_budget")


if __name__ == "__main__":
    unittest.main()
