# Token 长度分析

使用本地 Qwen2.5-7B-Instruct tokenizer 分析 SFT 数据：

```bash
python training/analyze_token_lengths.py --report training/data/token_length_report.json
```

如果模型不在默认目录，可指定：

```bash
python training/analyze_token_lengths.py --tokenizer-path /path/to/Qwen2.5-7B-Instruct
```

统计定义：`input_tokens` 是 system 和 user 经 Qwen chat template 加 generation
prompt 后的 token 数；`assistant_target_tokens` 是 assistant 目标 JSON 文本本身的
token 数；`total_tokens` 是两者之和。截断比例按 `total_tokens > max_seq_length` 计算。
