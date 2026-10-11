# Experiment Scorecard

每个模型或 checkpoint 实验必须填写一份 scorecard。以下结果均标记为 **local proxy metrics**，不是官方得分。

## 实验信息

| 字段 | 内容 |
|---|---|
| model/checkpoint | |
| base model | Qwen2.5-7B-Instruct |
| validation set | `training/data/val_sft.jsonl` |
| max_seq_length | |
| hardware | |
| evaluation time | |

## Local proxy metrics

| 指标 | 数值 |
|---|---:|
| emotion_accuracy | |
| emotion precision | |
| emotion recall | |
| emotion F1 | |
| BLEU-4 | |
| ROUGE-L | |
| BERTScore precision | |
| BERTScore recall | |
| BERTScore F1 | |
| personality_traits exact-set accuracy | |
| interests exact-set accuracy | |
| style exact-set accuracy | |
| profile mean accuracy | |
| JSON 合法率 | |
| response_text 非空率 | |
| memory_refs 格式正确率 | |
| 平均端到端延迟（ms） | GPU 端填写 |

## 结论

填写内容：

- 是否优于当前基线。
- 主要提升或退化的指标。
- 是否存在 JSON、画像或 memory_refs 格式问题。
- 是否建议进入下一轮实验。

## 解释边界

BLEU、ROUGE-L、BERTScore 的官方综合权重，以及画像多标签 accuracy 的官方实现没有公开。因此本表不能填写官方总分，只能记录 local proxy metrics 和模型间相对变化。
