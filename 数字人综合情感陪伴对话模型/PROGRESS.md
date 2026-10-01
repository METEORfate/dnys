# Project Progress

## Project Information

项目名称：

数字人综合情感陪伴对话模型

当前分支：

sanbitaria/lora-training

----------------------------------------

## Operating Rules

PROGRESS.md 是 Local Codex 和 GPU Instance Codex 的共享状态文件。

每次行动前必须读取：

1. PROGRESS.md
2. PROJECT_RULES.md
3. 最新 sanlogs

禁止：

- 假设另一端已有文件
- 未确认同步状态直接修改
- patch应用到不存在的文件

----------------------------------------

## Gate Status

G0:
PASS

G1-A:
PASS

G1-B:
PASS

G1-C-train:
PASS

G1-C-val:
PASS

G2-A:
PASS

G2-B:
PASS

G2-C:
BLOCKED

----------------------------------------

## Current Objective

当前目标：

完成 Base Model quality baseline。

当前阻塞：

GPU实例缺少 evaluator 完整文件：

training/evaluate_lora.py
training/test_evaluate_lora.py

下一动作：

同步完整 evaluator 文件。

不要使用 patch，除非目标文件已存在。

----------------------------------------

## Local Codex Responsibility

负责：

- 代码开发
- 单元测试
- 静态检查
- evaluator开发
- 实验设计

----------------------------------------

## GPU Instance Codex Responsibility

负责：

- GPU环境
- 模型加载
- 推理实验
- LoRA训练
- 性能测试

----------------------------------------

## Synchronization Rules

新增文件：

必须完整同步。

修改已有文件：

可以使用 patch。

所有同步：

必须验证 SHA256。

同步完成后：

更新：

SYNC_STATUS

----------------------------------------

## Pending Tasks

1. 创建 evaluator 完整同步包

2. 同步到 GPU 实例

3. 实例执行：

python training/test_evaluate_lora.py

python -m py_compile training/evaluate_lora.py

4. Base Model smoke test:

20 samples

5. Base Model val_dev baseline:

1649 samples

----------------------------------------

## Experiment History

2026-09-25:

完成：

- SFT alignment验证
- assistant-only loss验证
- truncation验证
- baseline participant验证
- validation split同步

2026-09-26:

完成：

- Base-only evaluator设计
- G2-C evaluator修复

当前：

等待 evaluator 同步。

----------------------------------------

## Update Rule

每次行动后必须更新：

1. PROGRESS.md

2. sanlogs/log_TIMESTAMP_role.md

记录：

- 行动
- 修改文件
- 测试结果
- 下一步

2026-09-28: Local Codex completed architecture phase 1 static review; prompt, input formatting, and output normalization differ across participant/training/evaluator. Gate statuses unchanged. See sanlogs/log_20260928_191121_local.md. Next: await confirmation for phase 2.


Prompt status:
FROZEN_V1

memory_refs:
fixed []

当前阶段：
architecture refactor preparation


2026-09-28: Completed competition runtime architecture alignment. Added core prompt/message/schema modules; participant, preprocess, evaluator now use shared runtime. Rebuilt train=18657 and val=2354 SFT data; alignment passed. Commit pending. See sanlogs/log_20260928_192931_local.md.


2026-10-01: Competition Runtime LoRA integration prepared. participant/start.sh now requires participant/model/lora_adapter and starts vLLM with emotion-companion-lora. Commit pending. See sanlogs/log_20261001_182928_local.md.

