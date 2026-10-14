#!/usr/bin/env python3
"""Run local Transformers or OpenAI-compatible inference and timing in one pass.

Public CLI contract: python run_inference.py TEST_FILE RESULT_DIR
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "inference_config.json"
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from core.message_builder import build_messages
from core.output_schema import EMOTIONS, MEMORY_RE, PROFILE_LABELS, parse_prediction
from core.prompt_loader import load_system_prompt
from style_postprocessor import StylePostprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地模型单轮推理并计时")
    parser.add_argument("test_file", type=Path, help="具体测试 JSONL 文件")
    parser.add_argument("result_dir", type=Path, help="结果输出目录")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: JSON 解析失败: {exc.msg}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("history"), list):
                raise ValueError(f"{path}:{number}: 必须包含字符串 id 和数组 history")
            rows.append(row)
    if not rows:
        raise ValueError(f"测试文件为空: {path}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("测试文件包含重复 id")
    return rows


def validate_test_file(test_file: Path) -> Path:
    if not test_file.is_file():
        raise ValueError(f"测试文件不存在或不是文件: {test_file}")
    if test_file.suffix.lower() != ".jsonl":
        raise ValueError(f"测试文件必须使用 .jsonl 扩展名: {test_file}")
    return test_file


def prepare_result_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"结果路径已存在但不是目录: {path}")
    path.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    environment_overrides = {
        "EVALUATION_BACKEND": "backend",
        "EVALUATION_BASE_URL": "base_url",
        "EVALUATION_API_MODEL": "api_model",
        "EVALUATION_API_KEY": "api_key",
    }
    for environment_name, config_name in environment_overrides.items():
        if environment_name in os.environ and (
            config_name == "api_key" or os.environ[environment_name]
        ):
            config[config_name] = os.environ[environment_name]
    if os.environ.get("EVALUATION_MODEL_PATH"):
        config["model_path"] = os.environ["EVALUATION_MODEL_PATH"]
    if os.environ.get("EVALUATION_HARDWARE_LABEL"):
        config["hardware_label"] = os.environ["EVALUATION_HARDWARE_LABEL"]
    config.setdefault("backend", "transformers")
    required = {"backend", "generation", "performance_rounds", "hardware_label"}
    if not required <= set(config):
        raise ValueError(f"推理配置缺少字段: {sorted(required - set(config))}")
    if config["backend"] not in {"transformers", "openai_compatible"}:
        raise ValueError("backend 必须为 transformers 或 openai_compatible")
    if config["backend"] == "transformers" and not config.get("model_path"):
        raise ValueError("transformers 模式必须配置 model_path")
    if config["backend"] == "openai_compatible":
        missing = [name for name in ("base_url", "api_model") if not config.get(name)]
        if missing:
            raise ValueError(f"openai_compatible 模式缺少字段: {missing}")
        api_endpoint(config["base_url"])
        timeout = config.get("request_timeout_seconds", 600)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("request_timeout_seconds 必须为正数")
        if not isinstance(config.get("api_options", {}), dict):
            raise ValueError("api_options 必须为 JSON object")
    if not isinstance(config["performance_rounds"], int) or config["performance_rounds"] <= 0:
        raise ValueError("performance_rounds 必须为正整数")
    return config


def model_path(config: dict[str, Any]) -> Path:
    path = Path(config["model_path"])
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(f"本地模型目录不存在: {resolved}")
    return resolved.resolve()


def api_endpoint(base_url: str) -> str:
    if not isinstance(base_url, str):
        raise ValueError("base_url 必须为字符串")
    normalized = base_url.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"base_url 必须是有效的 HTTP(S) URL: {base_url}")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def api_generation_options(generation: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "temperature", "top_p", "stop", "seed", "frequency_penalty",
        "presence_penalty", "min_p", "top_k",
    }
    options = {name: value for name, value in generation.items() if name in allowed}
    if "max_new_tokens" in generation:
        options["max_tokens"] = generation["max_new_tokens"]
    elif "max_tokens" in generation:
        options["max_tokens"] = generation["max_tokens"]
    if generation.get("do_sample") is False:
        options["temperature"] = 0
    return options


def response_content(payload: Any) -> str:
    try:
        choice = payload["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("API 响应缺少 choices[0].message.content") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    raise ValueError("API 响应的 message.content 必须为字符串或文本片段数组")


def request_chat_completion(
    config: dict[str, Any],
    messages: list[dict[str, str]],
    generation: dict[str, Any],
) -> str:
    reserved = {"model", "messages"}
    api_options = dict(config.get("api_options", {}))
    if reserved & set(api_options):
        raise ValueError(f"api_options 不得覆盖字段: {sorted(reserved & set(api_options))}")
    payload = {
        "model": config["api_model"],
        "messages": messages,
        **api_generation_options(generation),
        **api_options,
    }
    headers = {"Content-Type": "application/json"}
    api_key = config.get("api_key", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        api_endpoint(config["base_url"]),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=float(config.get("request_timeout_seconds", 600)),
        ) as response:
            raw_response = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"API 请求失败: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接本地 Base URL: {exc.reason}") from exc
    try:
        return response_content(json.loads(raw_response))
    except json.JSONDecodeError as exc:
        raise ValueError("API 响应不是有效 JSON") from exc


def conversation_text(sample: dict[str, Any]) -> str:
    return build_messages(sample)[1]["content"]


def synchronize(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    test_file = validate_test_file(args.test_file.resolve())
    result_dir = args.result_dir.resolve()
    prepare_result_dir(result_dir)
    config = load_config()
    backend = config["backend"]
    backend_report: dict[str, Any]

    if backend == "transformers":
        local_model = model_path(config)
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoTokenizer,
            )
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
            )
        except ImportError as exc:
            raise RuntimeError("请先安装 participant/requirements.txt") from exc

        trust_remote_code = bool(config.get("trust_remote_code", False))
        model_config = AutoConfig.from_pretrained(
            local_model,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        model_loader = (
            AutoModelForImageTextToText
            if model_config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
            else AutoModelForCausalLM
        )
        tokenizer = AutoTokenizer.from_pretrained(local_model, trust_remote_code=trust_remote_code, local_files_only=True)
        model = model_loader.from_pretrained(
            local_model,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
            device_map=config.get("device_map", "auto"),
            torch_dtype=config.get("torch_dtype", "auto"),
        )
        model.eval()
        backend_report = {
            "version": "transformers_inference_v2",
            "timing_scope": "tokenize+generate+decode+parse+validate; model loading excluded",
            "model_path_name": local_model.name,
            "model_type": model_config.model_type,
            "model_loader": model_loader.__name__,
        }
    else:
        backend_report = {
            "version": "openai_compatible_inference_v2",
            "timing_scope": "HTTP request+server inference+response decode+parse+validate; server startup excluded",
            "api_model": config["api_model"],
            "base_url": config["base_url"],
        }

    samples = read_jsonl(test_file)
    submission_path = result_dir / "submission.jsonl"
    submission_path.write_text("", encoding="utf-8")
    (result_dir / "style_causal_audit.jsonl").write_text("", encoding="utf-8")
    latencies: list[float] = []
    predictions: list[dict[str, Any]] = []
    parse_failure_ids: list[str] = []
    generation = dict(config["generation"])

    system_prompt = load_system_prompt()
    style_postprocessor = StylePostprocessor()
    print("STYLE_CLASSIFIER_LOADED:", style_postprocessor.loaded, flush=True)

    for index, sample in enumerate(samples, 1):
        messages = build_messages(sample)
        if backend == "transformers":
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            synchronize(torch)
            started = time.perf_counter()
            inputs = tokenizer(prompt, return_tensors="pt")
            input_device = next(model.parameters()).device
            inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}
            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generation)
            generated = output_ids[0, inputs["input_ids"].shape[1]:]
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
        else:
            started = time.perf_counter()
            decoded = request_chat_completion(config, messages, generation)
        prediction, parse_failed = parse_prediction(sample["id"], decoded)
        if parse_failed:
            parse_failure_ids.append(sample["id"])
        pre_style_result = deepcopy(prediction)
        if not parse_failed:
            prediction = style_postprocessor.apply(prediction, sample["history"])
        for key in ("id", "response_text", "emotion_label", "memory_refs"):
            assert pre_style_result[key] == prediction[key], f"Non-style field changed: {key}"
        for key in ("interests",):
            assert pre_style_result["user_profile"][key] == prediction["user_profile"][key], f"Non-style profile changed: {key}"
        final_style = prediction["user_profile"]["style"]
        assert isinstance(final_style, list) and len(final_style)==len(set(final_style)) and set(final_style)<=PROFILE_LABELS["style"]
        final_personality = prediction["user_profile"]["personality_traits"]
        assert isinstance(final_personality, list) and len(final_personality)==len(set(final_personality)) and set(final_personality)<=PROFILE_LABELS["personality_traits"]
        if backend == "transformers":
            synchronize(torch)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if index <= config["performance_rounds"]:
            latencies.append(elapsed_ms)
        with (result_dir / "style_causal_audit.jsonl").open("a", encoding="utf-8") as audit:
            audit.write(json.dumps({"pre_style_result": pre_style_result, "final_result": prediction}, ensure_ascii=False) + "\n")
        predictions.append(prediction)
        with submission_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"[{index}/{len(samples)}] {sample['id']}: {elapsed_ms:.3f} ms", flush=True)

    report = {
        **backend_report,
        "backend": backend,
        "complete": len(latencies) == config["performance_rounds"],
        "required_rounds": config["performance_rounds"],
        "rounds": len(latencies),
        "average_latency_ms": round(statistics.fmean(latencies), 6),
        "median_latency_ms": round(statistics.median(latencies), 6),
        "p95_latency_ms": round(percentile95(latencies), 6),
        "min_latency_ms": round(min(latencies), 6),
        "max_latency_ms": round(max(latencies), 6),
        "latencies_ms": [round(value, 6) for value in latencies],
        "hardware_label": config["hardware_label"],
        "test_file_sha256": sha256(test_file),
        "submission_sha256": sha256(submission_path),
        "samples": len(predictions),
        "style_classifier_loaded": style_postprocessor.loaded,
        "personality_classifier_loaded": style_postprocessor.personality_loaded,
        "personality_override_count": style_postprocessor.personality_override_count,
        "personality_fallback_count": style_postprocessor.personality_fallback_count,
        "style_classifier_invalid_output_count": style_postprocessor.invalid_output_count,
        "classifier_average_latency_ms": statistics.fmean(style_postprocessor.predict_latencies_ms) if style_postprocessor.predict_latencies_ms else None,
        "classifier_p50_latency_ms": statistics.median(style_postprocessor.predict_latencies_ms) if style_postprocessor.predict_latencies_ms else None,
        "classifier_p95_latency_ms": percentile95(style_postprocessor.predict_latencies_ms) if style_postprocessor.predict_latencies_ms else None,
        "style_classifier_fallback_count": style_postprocessor.fallback_count,
        "parse_failure_count": len(parse_failure_ids),
        "parse_failure_ids": parse_failure_ids,
    }
    (result_dir / "performance_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {submission_path}")
    print(f"PASS: {result_dir / 'performance_report.json'}")


if __name__ == "__main__":
    main()
