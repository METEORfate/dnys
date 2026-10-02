#!/usr/bin/env python3
"""Validate prediction-point to SFT sample alignment."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-file", type=Path, required=True)
    p.add_argument("--sft-file", type=Path, required=True)
    p.add_argument("--show-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def points(raw_path: Path):
    for line_number, conversation in rows(raw_path):
        turns = {turn["turn_id"]: turn for turn in conversation["turns"]}
        for point_index, point in enumerate(conversation["prediction_points"]):
            yield {
                "raw_line": line_number,
                "conversation_id": conversation["conversation_id"],
                "point_index": point_index,
                "turns": turns,
                "point": point,
            }


def expected_input(turns: dict[int, dict[str, Any]], user_id: int, response_id: int) -> str:
    visible = [turn for turn_id, turn in sorted(turns.items()) if turn_id <= user_id and turn_id != response_id]
    return "对话历史:\n" + "\n".join(f"{turn['role']}: {turn['content']}" for turn in visible)


def show_sample(item: dict[str, Any], sft: dict[str, Any]) -> None:
    point = item["point"]
    turns = item["turns"]
    user = turns[point["target_user_turn_id"]]
    assistant = turns[point["assistant_response_turn_id"]]
    messages = sft.get("messages", [])
    print(json.dumps({
        "conversation_id": item["conversation_id"],
        "target_user_turn_id": point["target_user_turn_id"],
        "target_user_text": user["content"],
        "assistant_response_turn_id": point["assistant_response_turn_id"],
        "target_response": point["target"]["response_text"],
        "converted_system": messages[0] if len(messages) > 0 else None,
        "converted_model_input": messages[1] if len(messages) > 1 else None,
        "converted_assistant_target": messages[2] if len(messages) > 2 else None,
    }, ensure_ascii=False, indent=2))


def validate(raw_path: Path, sft_path: Path, show_count: int, seed: int) -> dict[str, int]:
    source = list(points(raw_path))
    converted = [row for _, row in rows(sft_path)]
    counts = {
        "total_prediction_points": len(source),
        "aligned_count": 0,
        "missing_target_user_count": 0,
        "target_assistant_leak_count": 0,
        "future_turn_leak_count": 0,
        "target_mismatch_count": 0,
        "duplicate_count": max(0, len(converted) - len(source)),
        "missing_count": max(0, len(source) - len(converted)),
    }
    samples = set()
    if show_count and source:
        rng = random.Random(seed)
        indexes = {0, len(source) // 2, len(source) - 1}
        indexes.update(rng.sample(range(len(source)), min(show_count, len(source))))
        samples = indexes

    for index, item in enumerate(source):
        if index >= len(converted):
            continue
        sft = converted[index]
        item_errors = 0
        messages = sft.get("messages", []) if isinstance(sft, dict) else []
        if len(messages) != 3 or [message.get("role") for message in messages] != ["system", "user", "assistant"]:
            counts["target_mismatch_count"] += 1
            item_errors += 1
            if index in samples:
                show_sample(item, sft if isinstance(sft, dict) else {})
            continue

        point, turns = item["point"], item["turns"]
        user_id, response_id = point["target_user_turn_id"], point["assistant_response_turn_id"]
        actual_input = messages[1].get("content", "")
        if not isinstance(actual_input, str):
            actual_input = ""
            counts["target_mismatch_count"] += 1
            item_errors += 1
        user_content = turns[user_id]["content"]
        assistant_content = turns[response_id]["content"]

        expected = expected_input(turns, user_id, response_id)
        expected_lines = set(expected.splitlines())
        actual_lines = set(actual_input.splitlines())
        target_user_block = f"user: {user_content}"
        target_assistant_block = f"assistant: {assistant_content}"
        if target_user_block not in actual_input:
            counts["missing_target_user_count"] += 1
            item_errors += 1
        # Count a leak only when the labelled assistant turn line appears in the
        # actual input but is absent from the expected cutoff. Duplicate text in
        # an unrelated historical turn is not evidence of leakage.
        if target_assistant_block in actual_input and target_assistant_block not in expected:
            counts["target_assistant_leak_count"] += 1
            item_errors += 1

        # A future turn is a leak only when its serialized role/content line is
        # present in the actual input but absent from the expected cutoff input.
        future_leak = any(
            turn_id > user_id
            and f"{turn['role']}: {turn['content']}" in actual_input
            and f"{turn['role']}: {turn['content']}" not in expected
            for turn_id, turn in turns.items()
        )
        if future_leak:
            counts["future_turn_leak_count"] += 1
            item_errors += 1
        if actual_input != expected:
            counts["target_mismatch_count"] += 1
            item_errors += 1

        try:
            predicted_target = json.loads(messages[2]["content"])
        except (KeyError, TypeError, json.JSONDecodeError):
            predicted_target = None
        target = point["target"]
        if not isinstance(predicted_target, dict) or any(
            predicted_target.get(name) != target.get(name)
            for name in ("response_text", "emotion_label", "user_profile")
        ):
            counts["target_mismatch_count"] += 1
            item_errors += 1

        if item_errors == 0:
            counts["aligned_count"] += 1
        if index in samples:
            show_sample(item, sft)

    return counts

def main() -> None:
    options = args()
    result = validate(options.raw_file, options.sft_file, options.show_samples, options.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    hard_failures = ("missing_target_user_count", "target_assistant_leak_count", "future_turn_leak_count", "target_mismatch_count", "duplicate_count", "missing_count")
    if any(result[name] for name in hard_failures):
        raise SystemExit("SFT alignment Gate G1-A failed")


if __name__ == "__main__":
    main()
