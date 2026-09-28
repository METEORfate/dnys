#!/usr/bin/env python3
"""Evaluate a Qwen LoRA adapter with local proxy metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.output_schema import (
    EMOTIONS,
    PROFILE_LABELS,
    memory_refs_valid,
    memory_refs_value_valid,
    normalize_prediction,
    parse_prediction,
    profile_complete,
)
from core.message_builder import build_messages_from_user_content
from core.prompt_loader import load_system_prompt
DEFAULT_MODEL = ROOT / "participant/model/Qwen2___5-7B-Instruct"
DEFAULT_DATA = ROOT / "training/data/val_sft.jsonl"
DEFAULT_OUTPUT = ROOT / "training/eval_results"
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+|[^\s\u4e00-\u9fffA-Za-z0-9]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="LoRA adapter directory; omit for Base-only evaluation",
    )
    p.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--max-new-tokens", type=int, default=1200)
    p.add_argument("--limit", type=int, default=None, help="只评估前 N 条，用于本地冒烟测试")
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def load_stack():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Base Model 评估需要 torch、transformers 和 bitsandbytes。") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def attach_adapter_if_requested(
    base_model: Any,
    adapter_path: Path | None,
    peft_model_cls: Any | None = None,
) -> tuple[Any, str]:
    """Return the evaluation model and mode without importing PEFT for Base-only."""
    if adapter_path is None:
        return base_model, "base"
    if peft_model_cls is None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("LoRA 评估需要 peft；Base-only 模式不依赖 peft。") from exc
        peft_model_cls = PeftModel
    return peft_model_cls.from_pretrained(base_model, adapter_path, is_trainable=False), "lora"


def validate_paths(
    model_path: Path, data_file: Path, adapter_path: Path | None
) -> None:
    for path, label in ((model_path, "基础模型"), (data_file, "验证集")):
        if not path.exists():
            raise FileNotFoundError(f"{label}不存在: {path}")
    if adapter_path is not None and not adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter不存在: {adapter_path}")


def normalize(raw: Any) -> tuple[dict[str, Any], bool, bool]:
    prediction = normalize_prediction("", raw)
    raw_memories = raw.get("memory_refs") if isinstance(raw, dict) else None
    return prediction, profile_complete(prediction), memory_refs_value_valid(raw_memories)


def parse_json(text: str) -> tuple[dict[str, Any], bool, bool, bool]:
    prediction, json_failed = parse_prediction("", text)
    return prediction, not json_failed, profile_complete(prediction), memory_refs_valid(prediction)


def read_rows(path: Path, limit: int | None):
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if line.strip():
                yield index + 1, json.loads(line)


def text_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def sentence_bleu4(reference: str, hypothesis: str) -> float:
    ref, hyp = text_tokens(reference), text_tokens(hypothesis)
    if not hyp:
        return 0.0
    precisions = []
    for n in range(1, 5):
        candidate = ngrams(hyp, n)
        reference_counts = ngrams(ref, n)
        clipped = sum(min(count, reference_counts[gram]) for gram, count in candidate.items())
        precisions.append((clipped + 1.0) / (sum(candidate.values()) + 1.0))
    brevity = 1.0 if len(hyp) >= len(ref) else math.exp(1.0 - len(ref) / max(1, len(hyp)))
    return brevity * math.exp(sum(math.log(value) for value in precisions) / 4)


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    ref, hyp = text_tokens(reference), text_tokens(hypothesis)
    if not ref or not hyp:
        return 0.0
    previous = [0] * (len(hyp) + 1)
    for ref_token in ref:
        current = [0]
        for j, hyp_token in enumerate(hyp, 1):
            current.append(previous[j - 1] + 1 if ref_token == hyp_token else max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(hyp), lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def exact_set_accuracy(prediction: dict[str, Any], target: dict[str, Any], name: str) -> bool:
    return set(prediction.get("user_profile", {}).get(name, [])) == set(target.get("user_profile", {}).get(name, []))


def classification_diagnostics(predictions: list[str], targets: list[str]) -> dict[str, float]:
    values = []
    for label in sorted(EMOTIONS):
        tp = sum(p == label and t == label for p, t in zip(predictions, targets))
        fp = sum(p == label and t != label for p, t in zip(predictions, targets))
        fn = sum(p != label and t == label for p, t in zip(predictions, targets))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values.append((precision, recall, f1))
    return {"precision": sum(v[0] for v in values) / len(values), "recall": sum(v[1] for v in values) / len(values), "f1": sum(v[2] for v in values) / len(values)}


def compute_bertscore(predictions: list[str], targets: list[str]) -> tuple[dict[str, float] | None, str]:
    try:
        from bert_score import score
        precision, recall, f1 = score(predictions, targets, lang="zh", verbose=False)
        return {"precision": float(precision.mean()), "recall": float(recall.mean()), "f1": float(f1.mean())}, "computed"
    except Exception as exc:
        return None, f"unavailable: {type(exc).__name__}: {exc}"


def main() -> None:
    options = parse_args()
    validate_paths(options.model_path, options.data_file, options.adapter_path)
    if options.limit is not None and options.limit <= 0:
        raise ValueError("--limit 必须为正整数")
    torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = load_stack()
    tokenizer = AutoTokenizer.from_pretrained(options.model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True) if options.load_in_4bit else None
    model_kwargs = {"local_files_only": True, "torch_dtype": torch.bfloat16}
    if quant is not None:
        model_kwargs["quantization_config"] = quant
    base_model = AutoModelForCausalLM.from_pretrained(options.model_path, **model_kwargs)
    model, evaluation_mode = attach_adapter_if_requested(base_model, options.adapter_path)
    model.eval()
    device = next(model.parameters()).device
    options.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = options.output_dir / "prediction.jsonl"
    predictions: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    latencies: list[float] = []
    counters = Counter()
    rows = list(read_rows(options.data_file, options.limit))
    with prediction_path.open("w", encoding="utf-8", newline="\n") as output:
        for row_number, row in rows:
            messages = row["messages"]
            target = json.loads(messages[2]["content"])
            prompt_messages = build_messages_from_user_content(messages[1]["content"])
            prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=options.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            latencies.append((time.perf_counter() - started) * 1000)
            text = tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            prediction, json_ok, profile_ok, memory_ok = parse_json(text)
            counters["json_success"] += json_ok
            counters["response_nonempty"] += bool(prediction["response_text"])
            counters["emotion_correct"] += prediction["emotion_label"] == target.get("emotion_label")
            counters["profile_personality_traits"] += exact_set_accuracy(prediction, target, "personality_traits")
            counters["profile_interests"] += exact_set_accuracy(prediction, target, "interests")
            counters["profile_style"] += exact_set_accuracy(prediction, target, "style")
            counters["profile_complete"] += profile_ok
            counters["memory_format_correct"] += memory_ok
            predictions.append(prediction)
            targets.append(target)
            output.write(json.dumps({"id": row.get("id", f"val_{row_number:06d}"), **prediction}, ensure_ascii=False, separators=(",", ":")) + "\n")
    count = len(predictions)
    if not count:
        raise ValueError("验证集没有有效样本")
    predicted_texts = [row["response_text"] for row in predictions]
    target_texts = [row["response_text"] for row in targets]
    bert_scores, bert_status = compute_bertscore(predicted_texts, target_texts)
    diagnostics = classification_diagnostics([row["emotion_label"] for row in predictions], [row.get("emotion_label", "") for row in targets])
    summary = {
        "metric_status": "local proxy metrics; not official competition score",
        "evaluation_mode": evaluation_mode,
        "base_model_path": str(options.model_path),
        "adapter_path": str(options.adapter_path) if options.adapter_path is not None else None,
        "samples": count,
        "emotion_accuracy": counters["emotion_correct"] / count,
        "emotion_diagnostics": diagnostics,
        "precision": diagnostics["precision"], "recall": diagnostics["recall"], "f1": diagnostics["f1"],
        "bleu_4": sum(sentence_bleu4(target, prediction) for target, prediction in zip(target_texts, predicted_texts)) / count,
        "rouge_l": sum(rouge_l_f1(target, prediction) for target, prediction in zip(target_texts, predicted_texts)) / count,
        "bertscore": bert_scores,
        "bertscore_status": bert_status,
        "profile_personality_traits_exact_set_accuracy": counters["profile_personality_traits"] / count,
        "profile_interests_exact_set_accuracy": counters["profile_interests"] / count,
        "profile_style_exact_set_accuracy": counters["profile_style"] / count,
        "profile_mean_accuracy": sum(counters[f"profile_{name}"] for name in ("personality_traits", "interests", "style")) / (3 * count),
        "json_parse_success_rate": counters["json_success"] / count,
        "response_text_nonempty_rate": counters["response_nonempty"] / count,
        "memory_refs_format_correct_rate": counters["memory_format_correct"] / count,
        "average_end_to_end_latency_ms": sum(latencies) / count,
        "prediction_file": str(prediction_path),
    }
    (options.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
