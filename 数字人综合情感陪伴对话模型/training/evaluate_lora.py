#!/usr/bin/env python3
"""Evaluate a Qwen LoRA adapter on the prepared validation SFT set."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "participant/model/Qwen2___5-7B-Instruct"
DEFAULT_ADAPTER = ROOT / "training/outputs/qwen2.5-7b-lora"
DEFAULT_DATA = ROOT / "training/data/val_sft.jsonl"
DEFAULT_OUTPUT = ROOT / "training/eval_results"
EMOTIONS = {"joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise", "mixed", "sadness", "loneliness", "anxiety", "anger", "fear", "disgust", "shame", "helplessness"}
PROFILE_LABELS = {
    "personality_traits": {"extroverted", "introverted", "open", "conservative", "high_conscientiousness", "casual", "agreeable", "assertive", "emotionally_stable", "sensitive"},
    "interests": {"study_exam", "programming_technology", "reading_writing", "film_animation", "music", "games", "sports_fitness", "travel_outdoor", "pets", "social", "career_development", "art_design"},
    "style": {"brief", "detailed", "colloquial", "formal", "direct", "indirect", "humorous", "rational", "high_emotional_expression", "low_emotional_expression", "emoji_user"},
}
MEMORY_RE = re.compile(r"^mem_[0-9]{6}$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--max-new-tokens", type=int, default=1200)
    p.add_argument("--limit", type=int, default=None, help="只评估前 N 条，用于本地冒烟测试")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def load_stack():
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("评估需要 torch、transformers、peft 和 bitsandbytes。") from exc
    return torch, PeftModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def normalize(raw: Any) -> tuple[dict[str, Any], bool, bool]:
    if not isinstance(raw, dict):
        raw = {}
    response = raw.get("response_text")
    emotion = raw.get("emotion_label")
    profile = raw.get("user_profile")
    memories = raw.get("memory_refs")
    normalized_profile: dict[str, list[str]] = {}
    profile_complete = isinstance(profile, dict)
    for name, allowed in PROFILE_LABELS.items():
        values = profile.get(name) if isinstance(profile, dict) else None
        valid = isinstance(values, list) and len(values) == len(set(values)) and all(isinstance(v, str) and v in allowed for v in values)
        profile_complete = profile_complete and valid
        normalized_profile[name] = values if valid else []
    memories_valid = isinstance(memories, list) and len(memories) == len(set(memories)) and all(isinstance(v, str) and MEMORY_RE.fullmatch(v) for v in memories)
    result = {
        "response_text": response.strip() if isinstance(response, str) else "",
        "emotion_label": emotion if isinstance(emotion, str) and emotion in EMOTIONS else "",
        "user_profile": normalized_profile,
        "memory_refs": memories if memories_valid else [],
    }
    return result, profile_complete, memories_valid


def parse_json(text: str) -> tuple[dict[str, Any], bool, bool, bool]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.S | re.I)
    candidate = fenced.group(1) if fenced else cleaned
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        value = None
        for index, char in enumerate(candidate):
            if char == "{":
                try:
                    value, _ = decoder.raw_decode(candidate[index:])
                    break
                except json.JSONDecodeError:
                    continue
    if not isinstance(value, dict):
        return normalize({})[0], False, False, False
    result, profile_complete, memories_valid = normalize(value)
    return result, True, profile_complete, memories_valid


def read_rows(path: Path, limit: int | None):
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if line.strip():
                yield index + 1, json.loads(line)


def main() -> None:
    options = parse_args()
    for path, label in ((options.model_path, "基础模型"), (options.adapter_path, "LoRA adapter"), (options.data_file, "验证集")):
        if not path.exists():
            raise FileNotFoundError(f"{label}不存在: {path}")
    if options.limit is not None and options.limit <= 0:
        raise ValueError("--limit 必须为正整数")
    torch, PeftModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = load_stack()
    tokenizer = AutoTokenizer.from_pretrained(options.model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True) if options.load_in_4bit else None
    kwargs = {"local_files_only": True, "torch_dtype": torch.bfloat16}
    if quant is not None:
        kwargs["quantization_config"] = quant
    model = AutoModelForCausalLM.from_pretrained(options.model_path, **kwargs)
    model = PeftModel.from_pretrained(model, options.adapter_path, is_trainable=False)
    model.eval()
    device = next(model.parameters()).device
    options.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = options.output_dir / "prediction.jsonl"
    totals = {"samples": 0, "json_success": 0, "response_nonempty": 0, "emotion_correct": 0, "profile_complete": 0, "memory_format_correct": 0}
    with prediction_path.open("w", encoding="utf-8", newline="\n") as output:
        for row_number, row in read_rows(options.data_file, options.limit):
            messages = row["messages"]
            target = json.loads(messages[2]["content"])
            prompt = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=options.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            prediction, json_success, profile_complete, memory_format_correct = parse_json(text)
            totals["samples"] += 1
            totals["json_success"] += json_success
            totals["response_nonempty"] += bool(prediction["response_text"])
            totals["emotion_correct"] += prediction["emotion_label"] == target.get("emotion_label")
            totals["profile_complete"] += profile_complete
            totals["memory_format_correct"] += memory_format_correct
            record = {"id": row.get("id", f"val_{row_number:06d}"), **prediction}
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    n = totals["samples"] or 1
    summary = {"samples": totals["samples"], "json_parse_success_rate": totals["json_success"] / n, "response_text_nonempty_rate": totals["response_nonempty"] / n, "emotion_label_accuracy": totals["emotion_correct"] / n, "user_profile_field_complete_rate": totals["profile_complete"] / n, "memory_refs_format_correct_rate": totals["memory_format_correct"] / n, "prediction_file": str(prediction_path)}
    (options.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
