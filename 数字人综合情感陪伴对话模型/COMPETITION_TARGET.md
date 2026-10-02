# Competition Target

本文件定义 Gate G0 使用的评分对齐口径。`training/evaluate_lora.py` 输出的结果统一标记为 **local proxy metrics**，不得称为官方得分。官方的具体聚合权重和画像多标签 accuracy 实现没有公开，因此本地结果用于模型比较和诊断。

## 官方评分结构

| 模块 | 官方分值 | 官方指标 | 项目对应模块 | 当前状态 |
|---|---:|---|---|---|
| 情感识别准确性 | 25 | emotion accuracy | `training/evaluate_lora.py` | 已实现 local proxy |
| 回复质量 | 20 | BLEU-4、ROUGE-L、BERTScore 综合 | `training/evaluate_lora.py` | 已实现 local proxy；官方综合权重未知 |
| 用户画像推断 | 15 | 画像维度平均准确率 | `training/evaluate_lora.py` | 已实现三维 exact-set proxy；官方多标签实现未知 |
| 推理性能 | 10 | 平均端到端推理延迟 | `participant/run_inference.py`、评估脚本 | 已记录本地延迟；正式硬件结果待 GPU |
| 方案完整性 | 10 | 专家评审 | 文档、训练和评估流程 | 需人工准备材料 |
| 代码规范性 | 10 | 专家评审 | 项目代码与 `PROJECT_RULES.md` | 需人工检查 |
| 平台运行记录 | 10 | 专家评审 | `sanlogs/`、平台运行记录 | 部分具备，GPU 实验后补全 |

## Local proxy 定义

- `emotion_accuracy`：预测情绪标签与验证集标签完全相等的比例。
- `BLEU-4`：基于本地文本分词和四阶 n-gram 平滑的样本平均值。
- `ROUGE-L`：基于最长公共子序列的样本平均 F1。
- `BERTScore`：调用 `bert-score` 的中文 P/R/F1；依赖或模型不可用时记录 `null` 和原因。
- 画像 accuracy：`personality_traits`、`interests`、`style` 三个集合完全相等的比例；三者平均为 `profile_mean_accuracy`。
- `precision`、`recall`、`f1`：情绪标签的宏平均诊断指标。
- `memory_refs`：只做 `mem_######` schema 和唯一性检查，不作为主要优化目标。

## 未完全复现的官方指标

本地不能从公开资料确定 BLEU、ROUGE-L、BERTScore 的官方综合权重，也不能确定画像多标签 accuracy 的内部实现。因此这些数值只能用于 local proxy 比较，不能换算为官方 70 分中的精确得分。
