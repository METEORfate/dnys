# LoRA SFT 训练

安装训练依赖后，在项目根目录启动双卡训练：

```bash
torchrun --standalone --nproc_per_node=2 training/train_lora.py
```

脚本默认读取 `training/data/train_sft.jsonl` 和 `training/data/val_sft.jsonl`，
使用 Qwen2.5-7B-Instruct、PEFT LoRA、TRL `SFTTrainer`、`packing=True` 和
`max_seq_length=4096`。默认启用 NF4 4-bit 加载，适合双 4090；生成的目录只保存
LoRA adapter、tokenizer 和训练状态：`training/outputs/qwen2.5-7b-lora`。

可通过参数覆盖训练设置，例如：

```bash
torchrun --standalone --nproc_per_node=2 training/train_lora.py \
  --num-train-epochs 2 \
  --learning-rate 2e-4 \
  --gradient-accumulation-steps 8 \
  --output-dir training/outputs/experiment-01
```

如需从检查点继续：

```bash
torchrun --standalone --nproc_per_node=2 training/train_lora.py \
  --resume-from-checkpoint training/outputs/experiment-01/checkpoint-200
```
