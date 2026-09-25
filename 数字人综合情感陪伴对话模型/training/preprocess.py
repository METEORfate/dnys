#!/usr/bin/env python3
"""Convert conversation prediction points to Qwen chat SFT JSONL."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT = (
    "你是中文情感陪伴助手。根据截止到当前用户消息的对话历史，预测下一轮回复。"
    "只输出一个 JSON object，不要输出 Markdown 或解释。"
    "JSON 必须包含 response_text、emotion_label、user_profile、memory_refs。"
    "user_profile 必须包含 personality_traits、interests、style 三个字符串数组。"
)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须为 1/0、true/false、yes/no 或 on/off")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-input",
        type=Path,
        default=PROJECT_ROOT / "训练-验证-数据集/train/train_public.jsonl",
    )
    parser.add_argument(
        "--val-input",
        type=Path,
        default=PROJECT_ROOT / "训练-验证-数据集/val/val_public.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "training/data")
    parser.add_argument(
        "--dry-run", type=int, metavar="N",
        help="仅打印 train 和 val 各自前 N 条转换样本，不写输出文件",
    )
    return parser.parse_args()


def read_conversations(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: JSON 解析失败") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: 必须为 JSON object")
            yield line_number, row


def build_samples(row: dict[str, Any], enable_memory_refs: bool) -> Iterator[dict[str, Any]]:
    conversation_id = row.get("conversation_id", "<unknown>")
    turns = row.get("turns")
    points = row.get("prediction_points")
    if not isinstance(turns, list) or not isinstance(points, list):
        raise ValueError(f"{conversation_id}: turns 和 prediction_points 必须为数组")

    by_id: dict[int, dict[str, Any]] = {}
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError(f"{conversation_id}: turn 必须为 object")
        turn_id = turn.get("turn_id")
        if not isinstance(turn_id, int) or isinstance(turn_id, bool) or turn_id in by_id:
            raise ValueError(f"{conversation_id}: turn_id 无效或重复: {turn_id}")
        if turn.get("role") not in {"user", "assistant"} or not isinstance(turn.get("content"), str):
            raise ValueError(f"{conversation_id}: turn {turn_id} 的 role/content 无效")
        by_id[turn_id] = turn

    ordered_turns = [by_id[key] for key in sorted(by_id)]
    for point in points:
        if not isinstance(point, dict) or not isinstance(point.get("target"), dict):
            raise ValueError(f"{conversation_id}: prediction_point 无效")
        target_id = point.get("target_user_turn_id")
        response_id = point.get("assistant_response_turn_id")
        if (target_id not in by_id or by_id[target_id]["role"] != "user"
                or response_id not in by_id or by_id[response_id]["role"] != "assistant"):
            raise ValueError(f"{conversation_id}: 预测点轮次无效: {target_id} -> {response_id}")

        target = point["target"]
        required = ("response_text", "emotion_label", "user_profile", "memory_refs")
        if any(name not in target for name in required):
            raise ValueError(f"{conversation_id}: 预测点目标缺少字段")
        if target["response_text"] != by_id[response_id]["content"]:
            raise ValueError(f"{conversation_id}: 目标回复与对应 assistant 轮次不一致")
        if not isinstance(target["response_text"], str) or not isinstance(target["emotion_label"], str):
            raise ValueError(f"{conversation_id}: 目标文本或情绪类型无效")
        if not isinstance(target["user_profile"], dict) or not isinstance(target["memory_refs"], list):
            raise ValueError(f"{conversation_id}: 画像或记忆引用类型无效")

        # Use turn_id values, never array positions. Strictly exclude the target user
        # turn and every turn after the labelled assistant response.
        history = [
            turn for turn in ordered_turns
            if turn["turn_id"] < target_id and turn["turn_id"] <= response_id
        ]
        user_content = "对话历史:\n" + "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in history
        )
        answer = {
            "response_text": target["response_text"],
            "emotion_label": target["emotion_label"],
            "user_profile": target["user_profile"],
            "memory_refs": target["memory_refs"] if enable_memory_refs else [],
        }
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))},
            ]
        }


def convert(input_path: Path, output_path: Path, enable_memory_refs: bool) -> dict[str, Any]:
    count = 0
    total_length = 0
    max_length = 0
    emotions: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for line_number, row in read_conversations(input_path):
            try:
                samples = build_samples(row, enable_memory_refs)
                for sample in samples:
                    messages = sample["messages"]
                    input_length = len(messages[0]["content"]) + len(messages[1]["content"])
                    target = json.loads(messages[2]["content"])
                    total_length += input_length
                    max_length = max(max_length, input_length)
                    emotions[target["emotion_label"]] += 1
                    count += 1
                    output.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            except ValueError as exc:
                raise ValueError(f"{input_path}:{line_number}: {exc}") from exc
    if count == 0:
        raise ValueError(f"{input_path}: 没有可转换的 prediction_point")
    return {
        "samples": count,
        "average_input_characters": round(total_length / count, 2),
        "max_input_characters": max_length,
        "emotion_distribution": dict(sorted(emotions.items())),
    }


def preview(input_path: Path, limit: int, enable_memory_refs: bool) -> list[dict[str, Any]]:
    def samples() -> Iterator[dict[str, Any]]:
        for line_number, row in read_conversations(input_path):
            try:
                yield from build_samples(row, enable_memory_refs)
            except ValueError as exc:
                raise ValueError(f"{input_path}:{line_number}: {exc}") from exc

    return list(islice(samples(), limit))


def main() -> None:
    args = parse_args()
    enable_memory_refs = env_bool("ENABLE_MEMORY_REFS", False)
    if args.dry_run is not None and args.dry_run <= 0:
        raise ValueError("--dry-run N 要求 N 为正整数")
    for split, input_path in (("train", args.train_input), ("val", args.val_input)):
        if args.dry_run is not None:
            samples = preview(input_path, args.dry_run, enable_memory_refs)
            print(json.dumps({"split": split, "preview_samples": len(samples)}, ensure_ascii=False))
            for sample in samples:
                print(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
            continue
        output_path = args.output_dir / f"{split}_sft.jsonl"
        stats = convert(input_path, output_path, enable_memory_refs)
        print(json.dumps({"split": split, "output": str(output_path), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
