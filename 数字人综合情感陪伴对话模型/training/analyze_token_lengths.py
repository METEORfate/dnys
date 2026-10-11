#!/usr/bin/env python3
"""Analyze Qwen SFT token lengths without modifying the dataset."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER = ROOT / "participant/model/Qwen2___5-7B-Instruct"
LIMITS = (1024, 2048, 4096)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--train", type=Path, default=ROOT / "training/data/train_sft.jsonl")
    p.add_argument("--val", type=Path, default=ROOT / "training/data/val_sft.jsonl")
    p.add_argument("--report", type=Path, help="可选：同时写入 JSON 报告")
    return p.parse_args()


def quantile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    # Nearest-rank percentile, matching a truncation safety check.
    rank = max(1, int((q * len(ordered)) + 0.999999))
    return ordered[rank - 1]


def stats(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "average": round(statistics.fmean(values), 2),
        "p50": quantile(values, 0.50),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values),
    }


def load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "未安装 transformers。请在训练环境安装 participant/requirements.txt，"
            "然后使用 Qwen2.5-7B-Instruct tokenizer 运行本脚本。"
        ) from exc
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer 目录不存在: {path}。请使用 --tokenizer-path 指向本地 Qwen2.5-7B-Instruct 目录。"
        )
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def count_file(path: Path, tokenizer: Any) -> dict[str, Any]:
    input_lengths: list[int] = []
    target_lengths: list[int] = []
    total_lengths: list[int] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                messages = row["messages"]
                system, user, assistant = messages
                input_tokens = tokenizer.apply_chat_template(
                    [system, user], tokenize=True, add_generation_prompt=True
                )
                target_tokens = tokenizer(
                    assistant["content"], add_special_tokens=False, return_attention_mask=False
                )["input_ids"]
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: 样本格式或 tokenizer 处理失败") from exc
            input_n = len(input_tokens)
            target_n = len(target_tokens)
            input_lengths.append(input_n)
            target_lengths.append(target_n)
            total_lengths.append(input_n + target_n)
    if not total_lengths:
        raise ValueError(f"{path}: 文件没有有效样本")
    total_stats = stats(total_lengths)
    return {
        "input_tokens": stats(input_lengths),
        "assistant_target_tokens": stats(target_lengths),
        "total_tokens": total_stats,
        "truncation": {
            str(limit): {
                "count": sum(length > limit for length in total_lengths),
                "ratio": round(sum(length > limit for length in total_lengths) / len(total_lengths), 6),
            }
            for limit in LIMITS
        },
    }


def recommendation(reports: dict[str, Any]) -> str:
    train = reports["train"]["total_tokens"]
    val = reports["val"]["total_tokens"]
    if train["p99"] <= 2048 and val["p99"] <= 2048:
        return "建议 max_seq_length=2048：train/val 的 P99 均不超过 2048。"
    if train["p99"] <= 4096 and val["p99"] <= 4096:
        return "建议 max_seq_length=4096：至少覆盖 train/val 的 P99，减少长样本截断。"
    return "建议 max_seq_length=4096，并单独处理超过 4096 的长样本；数据中仍存在 P99 超限样本。"


def main() -> None:
    options = args()
    tokenizer = load_tokenizer(options.tokenizer_path)
    reports = {
        "tokenizer_path": str(options.tokenizer_path),
        "length_definition": "input=system+user chat template with generation prompt; target=assistant content; total=input+target",
        "train": count_file(options.train, tokenizer),
        "val": count_file(options.val, tokenizer),
    }
    reports["recommendation"] = recommendation(reports)
    rendered = json.dumps(reports, ensure_ascii=False, indent=2)
    print(rendered)
    if options.report:
        options.report.parent.mkdir(parents=True, exist_ok=True)
        options.report.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
