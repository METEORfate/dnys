# SFT 数据预处理

在项目根目录运行：

```bash
python training/preprocess.py
```

默认读取 `训练-验证-数据集/train/train_public.jsonl` 和
`训练-验证-数据集/val/val_public.jsonl`，生成 `training/data/train_sft.jsonl`
与 `training/data/val_sft.jsonl`。每个 `prediction_point` 生成一条 Qwen
`messages` 样本。输入按 `turn_id` 排序，只包含目标用户轮次及之前的内容。

`ENABLE_MEMORY_REFS` 默认关闭，关闭时目标 JSON 中的 `memory_refs` 为 `[]`；
设为 `1` 时保留原始引用：

```bash
ENABLE_MEMORY_REFS=1 python training/preprocess.py
```

只预览 train 和 val 各自前 2 条转换样本，不创建输出文件：

```bash
python training/preprocess.py --dry-run 2
```

可通过 `--train-input`、`--val-input`、`--output-dir` 覆盖路径。完整转换后分别
打印样本数、输入平均/最大字符数及情绪分布。长度按 system 和 user 消息内容的
Unicode 字符数之和计算，不是模型 token 数。
