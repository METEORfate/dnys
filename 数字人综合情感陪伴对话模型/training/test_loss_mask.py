import unittest

try:
    from .loss_mask_utils import (
        ExplicitLabelDataCollator,
        PromptPrefixMismatch,
        build_explicit_labels,
        render_prompt_and_full,
        template_has_generation_markers,
    )
except ImportError:
    from loss_mask_utils import (
        ExplicitLabelDataCollator,
        PromptPrefixMismatch,
        build_explicit_labels,
        render_prompt_and_full,
        template_has_generation_markers,
    )


class FakeTokenizer:
    chat_template = "{% generation %}assistant{% endgeneration %}"
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "|".join(f"{m['role']}:{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "|assistant:"
        return list(range(len(text)))


class PrefixMismatchTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        result = super().apply_chat_template(messages, tokenize, add_generation_prompt)
        if not add_generation_prompt and messages[-1]["role"] == "assistant":
            result.insert(0, -1)
        return result


class LossMaskTest(unittest.TestCase):
    def test_explicit_labels_mask_system_and_user(self):
        tokenizer = FakeTokenizer()
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}]
        features = build_explicit_labels(tokenizer, messages)
        prompt, full = render_prompt_and_full(tokenizer, messages)
        self.assertEqual(full[:len(prompt)], prompt)
        self.assertEqual(features["labels"][:len(prompt)], [-100] * len(prompt))
        self.assertEqual(features["labels"][len(prompt):], full[len(prompt):])
        self.assertGreater(sum(label != -100 for label in features["labels"]), 0)
        self.assertTrue(template_has_generation_markers(tokenizer))

    def test_prefix_mismatch_is_rejected(self):
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}]
        with self.assertRaises(PromptPrefixMismatch):
            build_explicit_labels(PrefixMismatchTokenizer(), messages)

    def test_collator_pads_labels_with_ignore_index(self):
        collator = ExplicitLabelDataCollator(FakeTokenizer())
        batch = collator([
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3], "attention_mask": [1], "labels": [3]},
        ])
        self.assertEqual(batch["input_ids"][1][-1].item() if hasattr(batch["input_ids"][1][-1], "item") else batch["input_ids"][1][-1], 0)
        self.assertEqual(batch["labels"][1][-1].item() if hasattr(batch["labels"][1][-1], "item") else batch["labels"][1][-1], -100)


if __name__ == "__main__":
    unittest.main()
