# V2-E2 FAST96 结果

- 实验：V2-E2 short diversity repair
- 评估样本：96
- 训练数据：21157（覆盖率 100%）
- 训练步数：662
- 训练耗时：144.49 分钟
- Adapter：training/outputs/v2-e2-short-diversity-repair
- evaluation_mode：lora

## 指标

| 指标 | V2-E2 | 相对 V2-E1 |
|---|---:|---:|
| emotion accuracy | 0.666666667 | -0.031250000 |
| profile mean | 0.670138889 | +0.013888889 |
| BLEU-4 | 0.139499100 | +0.004839749 |
| ROUGE-L | 0.272679014 | +0.003205718 |
| BERTScore F1 | 0.732902288 | +0.001709759 |
| response proxy | 0.381693468 | +0.003251742 |

BERTScore 使用本地模型 /root/development/assets/bert-base-multilingual-cased，
um_layers=9、lang=zh、
escale_with_baseline=False。

## 完整性

- prediction 行数：96
- JSON 有效率：1.0
- 非空回复率：1.0
- memory_refs 格式正确率：1.0
- 平均端到端延迟：9003.545 ms

## 判定

FAST_RESULT：**REJECT**。emotion accuracy 为 0.666667，低于快速淘汰阈值 0.687917；即使 response proxy 上升，仍按预设规则淘汰。

blocking_issue：NONE
