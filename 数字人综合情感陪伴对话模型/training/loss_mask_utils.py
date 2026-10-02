"""Tokenizer-backed explicit assistant-only labels and truncation utilities."""

from __future__ import annotations

from typing import Any


class PromptPrefixMismatch(ValueError):
    """Raised when the rendered prompt is not an exact prefix of the full chat."""


def _as_token_ids(value: Any) -> list[int]:
    """Normalize apply_chat_template output to a flat Python list of token ids."""
    if isinstance(value, dict):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value is None:
        raise ValueError("tokenizer.apply_chat_template 未返回 token ids")
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        value = list(value)
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("tokenizer.apply_chat_template 返回了批量 token ids")
        value = value[0]
    return [int(token_id) for token_id in value]


def render_prompt_and_full(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[list[int], list[int]]:
    """Render the exact training prompt and full target sequence.

    The prefix assertion is deliberately strict. If the tokenizer's template
    produces different prefixes for generation and complete rendering, callers
    must stop instead of guessing an assistant boundary.
    """
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("SFT messages 必须以 assistant target 结尾")
    prompt_ids = _as_token_ids(
        tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )
    )
    full_ids = _as_token_ids(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise PromptPrefixMismatch(
            "chat template 的 prompt token 不是完整序列的严格前缀；拒绝猜测 assistant 边界"
        )
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("assistant target 没有产生可监督 token")
    return prompt_ids, full_ids


def build_explicit_labels(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, list[int]]:
    """Create explicit input_ids, attention_mask and assistant-only labels."""
    prompt_ids, full_ids = render_prompt_and_full(tokenizer, messages)
    prompt_length = len(prompt_ids)
    labels = [-100] * prompt_length + full_ids[prompt_length:]
    if not any(label != -100 for label in labels):
        raise ValueError("assistant target labels 为空")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def token_ids(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> list[int]:
    return _as_token_ids(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt
        )
    )


def assistant_span(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[int, int, int]:
    prompt, full = render_prompt_and_full(tokenizer, messages)
    return len(full), len(prompt), len(full) - len(prompt)


def template_has_generation_markers(tokenizer: Any) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        template = "\n".join(str(value) for value in template.values())
    return isinstance(template, str) and "{% generation %}" in template and "{% endgeneration %}" in template


def truncate_for_training(
    tokenizer: Any, messages: list[dict[str, str]], max_length: int
) -> tuple[list[dict[str, str]] | None, dict[str, Any]]:
    """Trim oldest history while preserving the complete assistant target.

    Rendering is repeated after each deletion. The assistant target is measured
    from the explicit prompt/full prefix and is never truncated. ``None`` means
    the target or the irreducible system+target-user context cannot fit.
    """
    if max_length <= 0:
        raise ValueError("max_length 必须为正整数")
    working = [dict(message) for message in messages]
    total, _, target_tokens = assistant_span(tokenizer, working)
    if target_tokens > max_length:
        return None, {
            "status": "assistant_target_over_budget",
            "original_tokens": total,
            "truncated_tokens": total,
            "assistant_target_tokens": target_tokens,
            "deleted_tokens": 0,
            "target_user_preserved": False,
        }
    original = total
    history_lines = working[1]["content"].splitlines()
    newest_line = history_lines[-1] if history_lines else ""
    while total > max_length and len(history_lines) > 1:
        # Line 0 is the history header; the newest line is the target user.
        del history_lines[1]
        working[1]["content"] = "\n".join(history_lines)
        total, _, _ = assistant_span(tokenizer, working)
    if total > max_length:
        return None, {
            "status": "context_over_budget",
            "original_tokens": original,
            "truncated_tokens": total,
            "assistant_target_tokens": target_tokens,
            "deleted_tokens": original - total,
            "target_user_preserved": False,
        }
    return working, {
        "status": "ok",
        "original_tokens": original,
        "truncated_tokens": total,
        "assistant_target_tokens": target_tokens,
        "deleted_tokens": original - total,
        "target_user_preserved": newest_line in working[1]["content"],
    }


class ExplicitLabelDataCollator:
    """Pad already-tokenized explicit-label features for Trainer/SFTTrainer."""

    def __init__(self, tokenizer: Any, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("不能对空 batch 进行 collate")
        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("tokenizer 缺少 pad_token_id/eos_token_id")
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            ids = list(feature["input_ids"])
            mask = list(feature.get("attention_mask", [1] * len(ids)))
            target = list(feature["labels"])
            if len(ids) != len(mask) or len(ids) != len(target):
                raise ValueError("input_ids、attention_mask、labels 长度不一致")
            pad = max_length - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attention_mask.append(mask + [0] * pad)
            labels.append(target + [self.label_pad_token_id] * pad)
        try:
            import torch
        except ImportError:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
