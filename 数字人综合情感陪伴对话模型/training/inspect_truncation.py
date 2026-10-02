#!/usr/bin/env python3
"""Audit 4096-token truncation while preserving assistant targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loss_mask_utils import assistant_span, truncate_for_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("需要在 GPU/训练环境安装 transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    summary = {"total_samples": 0, "total_over_limit": 0, "assistant_target_over_budget_count": 0, "assistant_target_preserved_count": 0, "assistant_target_lost_count": 0, "target_user_preserved_count": 0, "deleted_tokens": 0}
    with args.data_file.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            messages = row["messages"]
            total, _, target_tokens = assistant_span(tokenizer, messages)
            summary["total_samples"] += 1
            if total <= args.max_length:
                continue
            summary["total_over_limit"] += 1
            truncated, details = truncate_for_training(tokenizer, messages, args.max_length)
            summary["deleted_tokens"] += details.get("deleted_tokens", 0)
            if details["status"] == "assistant_target_over_budget":
                summary["assistant_target_over_budget_count"] += 1
                summary["assistant_target_lost_count"] += 1
            elif truncated is not None:
                summary["assistant_target_preserved_count"] += 1
                # The newest user-history line is retained by the helper.
                summary["target_user_preserved_count"] += bool(details.get("target_user_preserved"))
            if args.details:
                print(json.dumps({"sample_index": index, "original_tokens": total, "assistant_target_tokens": target_tokens, **details}, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["assistant_target_lost_count"]:
        raise SystemExit("assistant_target_lost_count > 0: full training must remain blocked")


if __name__ == "__main__":
    main()
