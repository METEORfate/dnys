#!/usr/bin/env python3
"""Train a Qwen2.5 LoRA adapter with explicit assistant-only labels."""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Any

from loss_mask_utils import (
    ExplicitLabelDataCollator,
    build_explicit_labels,
    truncate_for_training,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "participant/model/Qwen2___5-7B-Instruct"
DEFAULT_TRAIN = ROOT / "training/data/train_sft.jsonl"
DEFAULT_VAL = ROOT / "training/data/val_sft.jsonl"
DEFAULT_OUTPUT = ROOT / "training/outputs/qwen2.5-7b-lora"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val-file", type=Path, default=DEFAULT_VAL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--num-train-epochs", type=float, default=2.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在: {path}")


def get_local_rank(environ: Any = os.environ) -> int:
    """Resolve the torchrun local rank, defaulting to the single-GPU rank."""
    try:
        local_rank = int(environ.get("LOCAL_RANK", "0"))
    except (TypeError, ValueError) as exc:
        raise ValueError("LOCAL_RANK 必须是整数") from exc
    if local_rank < 0:
        raise ValueError("LOCAL_RANK 不能为负数")
    return local_rank


def build_model_kwargs(torch: Any, stack: dict[str, Any], options: argparse.Namespace, local_rank: int) -> dict[str, Any]:
    """Build rank-local model loading arguments for quantized training."""
    quantization_config = None
    if options.load_in_4bit:
        quantization_config = stack["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model_kwargs = {
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = {"": local_rank}
    return model_kwargs


def import_training_stack() -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "训练需要 torch、transformers、datasets、peft、trl 和 accelerate；"
            "请在训练环境安装对应依赖。"
        ) from exc
    try:
        from trl import SFTConfig
    except ImportError:
        SFTConfig = None
    return locals()


def build_training_args(stack: dict[str, Any], options: argparse.Namespace) -> Any:
    common = {
        "output_dir": str(options.output_dir),
        "num_train_epochs": options.num_train_epochs,
        "per_device_train_batch_size": options.per_device_train_batch_size,
        "per_device_eval_batch_size": options.per_device_eval_batch_size,
        "gradient_accumulation_steps": options.gradient_accumulation_steps,
        "learning_rate": options.learning_rate,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 200,
        "save_total_limit": 2,
        "eval_steps": 200,
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": options.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "ddp_find_unused_parameters": False,
        "remove_unused_columns": False,
        "report_to": "none",
        "seed": options.seed,
    }
    if stack["SFTConfig"] is None:
        raise RuntimeError("需要可用的 TRL SFTConfig")
    cls = stack["SFTConfig"]
    supported = inspect.signature(cls).parameters
    common = {key: value for key, value in common.items() if key in supported}
    if "eval_strategy" in supported:
        common["eval_strategy"] = "steps"
    elif "evaluation_strategy" in supported:
        common["evaluation_strategy"] = "steps"
    if "max_length" in supported:
        common["max_length"] = options.max_seq_length
    elif "max_seq_length" in supported:
        common["max_seq_length"] = options.max_seq_length
    # Labels are already explicit. Keep TRL's automatic path disabled when the
    # installed version exposes this option; training correctness does not rely
    # on it and no generation markers are required.
    if "assistant_only_loss" in supported:
        common["assistant_only_loss"] = False
    if "packing" in supported:
        common["packing"] = False
    if "eval_packing" in supported:
        common["eval_packing"] = False
    if "dataset_kwargs" in supported:
        common["dataset_kwargs"] = {"skip_prepare_dataset": True}
    return cls(**common)


def prepare_dataset(dataset: Any, tokenizer: Any, max_length: int, split: str) -> Any:
    """Render tokenized features and explicit assistant-only labels.

    Truncation happens before the final rendering. A sample is rejected if its
    assistant target or irreducible context cannot fit; neither case is silently
    truncated or dropped.
    """
    counters = {
        "samples": 0,
        "trimmed": 0,
        "assistant_target_over_budget": 0,
        "context_over_budget": 0,
        "prefix_mismatch": 0,
    }

    def transform(row: dict[str, Any], index: int) -> dict[str, list[int]]:
        counters["samples"] += 1
        try:
            messages, details = truncate_for_training(tokenizer, row["messages"], max_length)
        except Exception as exc:
            counters["prefix_mismatch"] += 1
            raise RuntimeError(f"{split} sample {index} 渲染前缀校验失败: {exc}") from exc
        if details["status"] != "ok" or messages is None:
            status = details["status"]
            counters[status] = counters.get(status, 0) + 1
            raise RuntimeError(
                f"{split} sample {index} 被拒绝（{status}）："
                f"original_tokens={details.get('original_tokens')}, "
                f"assistant_target_tokens={details.get('assistant_target_tokens')}"
            )
        if details.get("deleted_tokens", 0):
            counters["trimmed"] += 1
        try:
            return build_explicit_labels(tokenizer, messages)
        except Exception as exc:
            counters["prefix_mismatch"] += 1
            raise RuntimeError(f"{split} sample {index} labels 构造失败: {exc}") from exc

    prepared = dataset.map(
        transform,
        with_indices=True,
        remove_columns=dataset.column_names,
        desc=f"Render explicit labels for {split}",
    )
    if not len(prepared):
        raise RuntimeError(f"{split} 数据没有可训练样本")
    print(f"{split} explicit-label audit: {counters}, kept={len(prepared)}")
    return prepared


def main() -> None:
    options = parse_args()
    if options.max_seq_length <= 0:
        raise ValueError("--max-seq-length 必须为正整数")
    require_file(options.train_file, "训练数据")
    require_file(options.val_file, "验证数据")
    if not options.model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {options.model_path}")

    stack = import_training_stack()
    torch = stack["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA 训练需要 CUDA；当前未检测到 GPU")
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not torch.distributed.is_available():
        raise RuntimeError("检测到分布式训练进程，但 torch.distributed 不可用")

    tokenizer = stack["AutoTokenizer"].from_pretrained(
        options.model_path, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    model_kwargs = build_model_kwargs(torch, stack, options, local_rank)
    model = stack["AutoModelForCausalLM"].from_pretrained(options.model_path, **model_kwargs)
    model.config.use_cache = False
    if options.load_in_4bit:
        model = stack["prepare_model_for_kbit_training"](model)

    lora_config = stack["LoraConfig"](
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    dataset = stack["load_dataset"](
        "json",
        data_files={"train": str(options.train_file), "validation": str(options.val_file)},
    )
    dataset["train"] = prepare_dataset(dataset["train"], tokenizer, options.max_seq_length, "train")
    dataset["validation"] = prepare_dataset(dataset["validation"], tokenizer, options.max_seq_length, "validation")
    training_args = build_training_args(stack, options)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "processing_class": tokenizer,
        "data_collator": ExplicitLabelDataCollator(tokenizer),
        "peft_config": lora_config,
    }
    supported = inspect.signature(stack["SFTTrainer"]).parameters
    if "tokenizer" in supported and "processing_class" not in supported:
        trainer_kwargs["tokenizer"] = trainer_kwargs.pop("processing_class")
    if "dataset_kwargs" in supported and "dataset_kwargs" not in trainer_kwargs:
        trainer_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
    trainer = stack["SFTTrainer"](**trainer_kwargs)
    trainer.train(resume_from_checkpoint=options.resume_from_checkpoint)
    trainer.save_model(str(options.output_dir))
    tokenizer.save_pretrained(str(options.output_dir))
    print(f"LoRA adapter saved to: {options.output_dir}")


if __name__ == "__main__":
    main()
