#!/usr/bin/env python3
"""Train a Qwen2.5 LoRA adapter with TRL SFTTrainer."""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Any


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


def import_training_stack() -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
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
    cls = stack["SFTConfig"] or stack["TrainingArguments"]
    supported = inspect.signature(cls).parameters
    common = {key: value for key, value in common.items() if key in supported}
    if "eval_strategy" in supported:
        common["eval_strategy"] = "steps"
    elif "evaluation_strategy" in supported:
        common["evaluation_strategy"] = "steps"
    # TRL 1.13.0 uses max_length; older releases used max_seq_length.
    if stack["SFTConfig"] is not None:
        if "max_length" in supported:
            common["max_length"] = options.max_seq_length
        elif "max_seq_length" in supported:
            common["max_seq_length"] = options.max_seq_length
        common["packing"] = True
    return cls(**common)


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
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not torch.distributed.is_available():
        raise RuntimeError("检测到分布式训练进程，但 torch.distributed 不可用")

    tokenizer = stack["AutoTokenizer"].from_pretrained(
        options.model_path, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        "quantization_config": quantization_config,
    }
    if quantization_config is None:
        model_kwargs.pop("quantization_config")
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
    training_args = build_training_args(stack, options)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "processing_class": tokenizer,
        "peft_config": lora_config,
    }
    supported = inspect.signature(stack["SFTTrainer"]).parameters
    if "tokenizer" in supported and "processing_class" not in supported:
        trainer_kwargs["tokenizer"] = trainer_kwargs.pop("processing_class")
    trainer = stack["SFTTrainer"](**trainer_kwargs)
    trainer.train(resume_from_checkpoint=options.resume_from_checkpoint)
    trainer.save_model(str(options.output_dir))
    tokenizer.save_pretrained(str(options.output_dir))
    print(f"LoRA adapter saved to: {options.output_dir}")


if __name__ == "__main__":
    main()
