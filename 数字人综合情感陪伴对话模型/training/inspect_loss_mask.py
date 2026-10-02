#!/usr/bin/env python3
"""Inspect the exact explicit labels used by the SFT training path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loss_mask_utils import (
    build_explicit_labels,
    render_prompt_and_full,
    template_has_generation_markers,
    token_ids,
    truncate_for_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples 必须为正整数")
    if args.max_length <= 0:
        raise ValueError("--max-length 必须为正整数")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("需要在 GPU/训练环境安装 transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, use_fast=True
    )
    if not tokenizer.chat_template:
        raise RuntimeError("tokenizer 没有 chat_template，无法渲染训练路径")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(json.dumps({
        "explicit_labels": True,
        "generation_markers_present_diagnostic_only": template_has_generation_markers(tokenizer),
        "max_length": args.max_length,
    }, ensure_ascii=False))

    checked = 0
    rejected = 0
    hard_failures = 0
    with args.data_file.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= args.num_samples:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["messages"]
            trimmed, details = truncate_for_training(tokenizer, messages, args.max_length)
            if trimmed is None:
                rejected += 1
                print(json.dumps({
                    "sample_index": index,
                    "status": details["status"],
                    "original_tokens": details.get("original_tokens"),
                    "assistant_target_tokens": details.get("assistant_target_tokens"),
                    "explicit_labels_checked": False,
                }, ensure_ascii=False))
                continue
            prompt_ids, full_ids = render_prompt_and_full(tokenizer, trimmed)
            features = build_explicit_labels(tokenizer, trimmed)
            labels = features["labels"]
            assistant_loss_tokens = sum(label != -100 for label in labels[len(prompt_ids):])
            system_tokens = len(token_ids(tokenizer, [trimmed[0]], add_generation_prompt=False))
            user_tokens = max(0, len(prompt_ids) - system_tokens)
            system_loss_tokens = sum(label != -100 for label in labels[:min(system_tokens, len(prompt_ids))])
            user_loss_tokens = sum(label != -100 for label in labels[system_tokens:len(prompt_ids)])
            if system_loss_tokens != 0 or user_loss_tokens != 0 or assistant_loss_tokens <= 0:
                hard_failures += 1
            print(json.dumps({
                "sample_index": index,
                "messages_roles": [message["role"] for message in trimmed],
                "token_total": len(full_ids),
                "system_token_count": system_tokens,
                "user_token_count": user_tokens,
                "assistant_token_count": assistant_loss_tokens,
                "prompt_token_count": len(prompt_ids),
                "system_loss_tokens": system_loss_tokens,
                "user_loss_tokens": user_loss_tokens,
                "assistant_loss_tokens": assistant_loss_tokens,
                "prompt_is_full_prefix": full_ids[:len(prompt_ids)] == prompt_ids,
                "condition": "system_loss_tokens == 0 and user_loss_tokens == 0 and assistant_loss_tokens > 0",
            }, ensure_ascii=False))
            checked += 1
    print(json.dumps({
        "checked": checked,
        "rejected": rejected,
        "hard_failures": hard_failures,
    }, ensure_ascii=False, indent=2))
    if hard_failures:
        raise SystemExit("explicit labels validation failed")


if __name__ == "__main__":
    main()
