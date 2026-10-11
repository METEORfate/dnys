#!/usr/bin/env python3
"""Create a conversation-disjoint deterministic dev/holdout split for public val."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .validate_sft_alignment import validate as validate_alignment
except ImportError:
    from validate_sft_alignment import validate as validate_alignment


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = ROOT / "训练-验证-数据集/val/val_public.jsonl"
DEFAULT_SFT = ROOT / "training/data/val_sft.jsonl"
DEFAULT_DEV = ROOT / "training/data/val_dev_sft.jsonl"
DEFAULT_HOLDOUT = ROOT / "training/data/val_holdout_sft.jsonl"
DEFAULT_MANIFEST = ROOT / "training/validation_split_manifest.json"
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--sft-file", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--dev-output", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--holdout-output", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dev-ratio", type=float, default=0.70)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def raw_prediction_points(raw_rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
    identities: list[tuple[str, int]] = []
    for row in raw_rows:
        conversation_id = row.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("原始 val 数据缺少有效 conversation_id")
        points = row.get("prediction_points")
        if not isinstance(points, list):
            raise ValueError(f"{conversation_id}: prediction_points 必须为数组")
        identities.extend((conversation_id, index) for index in range(len(points)))
    return identities


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_dev_conversations(
    counts: dict[str, int], seed: int, dev_ratio: float
) -> set[str]:
    """Choose a seeded conversation set nearest to the requested point ratio."""
    target = round(sum(counts.values()) * dev_ratio)
    ordered = sorted(counts)
    random.Random(seed).shuffle(ordered)
    selected: set[str] = set()
    selected_points = 0
    for conversation_id in ordered:
        size = counts[conversation_id]
        current_distance = abs(selected_points - target)
        added_distance = abs(selected_points + size - target)
        if added_distance < current_distance or (
            added_distance == current_distance and selected_points < target
        ):
            selected.add(conversation_id)
            selected_points += size
    return selected


def create_split(
    raw_file: Path,
    sft_file: Path,
    dev_output: Path,
    holdout_output: Path,
    manifest_file: Path,
    seed: int = DEFAULT_SEED,
    dev_ratio: float = 0.70,
) -> dict[str, Any]:
    if not 0 < dev_ratio < 1:
        raise ValueError("--dev-ratio 必须在 0 和 1 之间")
    raw_rows = read_jsonl(raw_file)
    sft_rows = read_jsonl(sft_file)
    identities = raw_prediction_points(raw_rows)
    if len(identities) != len(sft_rows):
        raise ValueError(
            f"原始 prediction point 数量 {len(identities)} 与 SFT 行数 {len(sft_rows)} 不一致"
        )
    if len(set(identities)) != len(identities):
        raise ValueError("原始 prediction point identity 重复")
    alignment = validate_alignment(raw_file, sft_file, 0, seed)
    hard_alignment_keys = (
        "missing_target_user_count",
        "target_assistant_leak_count",
        "future_turn_leak_count",
        "target_mismatch_count",
        "duplicate_count",
        "missing_count",
    )
    if any(alignment[key] for key in hard_alignment_keys):
        raise ValueError(f"val_sft 与原始 prediction point 未通过 alignment: {alignment}")

    by_conversation: dict[str, list[int]] = defaultdict(list)
    for index, (conversation_id, _) in enumerate(identities):
        by_conversation[conversation_id].append(index)
    dev_conversations = choose_dev_conversations(
        {conversation_id: len(indexes) for conversation_id, indexes in by_conversation.items()},
        seed,
        dev_ratio,
    )
    dev_indexes = {
        index
        for conversation_id, indexes in by_conversation.items()
        if conversation_id in dev_conversations
        for index in indexes
    }
    holdout_indexes = set(range(len(sft_rows))) - dev_indexes

    def write_rows(path: Path, indexes: set[int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for index in sorted(indexes):
                handle.write(json.dumps(sft_rows[index], ensure_ascii=False, separators=(",", ":")) + "\n")

    write_rows(dev_output, dev_indexes)
    write_rows(holdout_output, holdout_indexes)

    dev_ids = {identities[index] for index in dev_indexes}
    holdout_ids = {identities[index] for index in holdout_indexes}
    source_ids = set(identities)
    duplicate_count = len(identities) - len(source_ids)
    missing_count = len(source_ids - (dev_ids | holdout_ids))
    overlap_count = len(
        {conversation_id for conversation_id, _ in dev_ids}
        & {conversation_id for conversation_id, _ in holdout_ids}
    )
    manifest = {
        "split_version": "g2b-v1",
        "seed": seed,
        "split_method": "seeded conversation_id shuffle with nearest target point count",
        "dev_ratio_target": dev_ratio,
        "source_sample_count": len(sft_rows),
        "source_conversation_count": len(by_conversation),
        "dev_sample_count": len(dev_indexes),
        "dev_conversation_count": len(dev_conversations),
        "holdout_sample_count": len(holdout_indexes),
        "holdout_conversation_count": len(by_conversation) - len(dev_conversations),
        "conversation_overlap_count": overlap_count,
        "duplicate_count": duplicate_count,
        "missing_count": missing_count,
        "duplicate_prediction_point_count": duplicate_count,
        "missing_prediction_point_count": missing_count,
        "raw_file_hash": sha256_file(raw_file),
        "source_file_hash": sha256_file(sft_file),
        "dev_file_hash": sha256_file(dev_output),
        "holdout_file_hash": sha256_file(holdout_output),
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if (
        manifest["source_sample_count"] != len(dev_indexes) + len(holdout_indexes)
        or overlap_count
        or duplicate_count
        or missing_count
    ):
        raise RuntimeError("validation split 完整性检查失败")
    return manifest


def main() -> None:
    options = parse_args()
    manifest = create_split(
        options.raw_file,
        options.sft_file,
        options.dev_output,
        options.holdout_output,
        options.manifest,
        options.seed,
        options.dev_ratio,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
