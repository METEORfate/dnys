# Prompt A/B 自动化评估（开发分支）

本工具不训练模型、不改动 LoRA 权重。默认加载现有 `configs/system_prompt_v1.txt`；这是正式推理的稳定默认行为。`configs/prompts/prompt_v0.txt` 是原样冻结基线，`prompt_v1.txt` 是**未经 GPU 验证**的初始候选。

## 使用前的检查

本工具以**同一个已经启动的** OpenAI-compatible vLLM 服务为后端，一次加载模型、重复调用不同 system Prompt。模型应已加载固定的 Qwen2.5-7B-Instruct + LoRA，服务模型名应与 `participant/configs/inference_config.json` 的 `api_model` 一致。

在 GPU 服务器上先只读执行：`nvidia-smi`、`python -c "import torch; print(torch.cuda.is_available(),torch.version.cuda)"`，核实模型和 adapter 路径、可用显存。若 NVML 报错，停止加载并先排查容器 GPU 映射/驱动。不要未经批准启动额外计费实例或安装依赖。

## 命令

在工程根目录执行（使用已安装项目依赖的 Python 环境）：

```bash
# 单独指定版本：不改变默认的正式 Prompt
DNYS_PROMPT=v1 python participant/run_inference.py INPUT.jsonl OUTPUT_DIR

# 服务已运行，直接进行固定样本的两个 Prompt 对照
python -m prompt_work.run_ab --prompts v0 v1 --tier smoke --split dev
python -m prompt_work.run_ab --prompts v0 v1 --tier fast --split dev
python -m prompt_work.run_ab --prompts v0 v1 --tier full --split dev

# 保留集只允许最终候选使用，不参与日常迭代
python -m prompt_work.run_ab --prompts v0 v1 --tier full --split holdout

# CPU-only 单元测试
python -m unittest prompt_work.test_workflow -v

# 正式入口（默认 Prompt）
bash /root/participant/start.sh TEST_FILE RESULT_DIR
```

运行前按实际环境设置 `EVALUATION_BASE_URL` / `EVALUATION_API_MODEL`，如需覆盖默认后端设 `EVALUATION_BACKEND=openai_compatible`。Linux 默认的 vLLM 张量并行度现在是 1，可使用 `VLLM_TENSOR_PARALLEL_SIZE` 指定其它数目；该修改尚须在实际单卡 GPU 验证。不得将未验证的 v1 默认推广到正式比赛。

## 数据隔离与公平性

脚本从 `val_public.jsonl` 的原始会话重建目标：只截取 `target_user_turn_id` 及之前的 history，显式排除 `assistant_response_turn_id`，后续轮次和目标标签不进入模型消息。开发与保留划分复用 `choose_dev_conversations(seed=42, dev_ratio=.70)`，所有 Prompt 对相同样本按同一顺序执行，解码参数保持同一个配置。为提高可复现性，默认贪心解码，GPU 服务中的非确定性仍须通过复跑确认。

`smoke` 8 个固定样本；`fast` 64 个固定样本；`full` 是 dev 全量。请勿为追逐保留集成绩反复查看标签/调参。实际划分应与 `training/validation_split_manifest.json` 的 1649/705 条及输入 SHA256 核验一致；该脚本不会覆盖已有训练数据。

## 产物和口径

每次创建独立时间戳目录 `prompt_work/runs/<UTC时间>_<档位>/`，不覆盖队友实验。输出：`comparison.json`（包括 Prompt 文件 SHA256、模型配置[不含 API key]、数据 SHA256）、`paired_cases.jsonl`（逐项情绪改进/退步）、各版本 `prompt.txt`、`predictions.jsonl`、`metrics.json`。逐条记录 raw_text、raw_object、原始合法性、归一化预测、画像后处理预测、标签、耗时、调用失败及 JSON 解析失败。请谨慎处理含对话历史的报告，避免把敏感数据提交到 GitHub。

`model_raw_normalized` 评价模型结果经过 schema 归一化后的指标，`raw_validity_rates` **针对归一化之前** 的模型字段；`after_classifier` 是画像分类器后处理的指标，避免混淆。情绪含 Macro-F1、逐类 P/R/F1 和混淆矩阵；回复含 BLEU-4、ROUGE-L，BERTScore 因可能联网下载模型而默认不执行并标明 null；画像为 exact-set accuracy。结果均为 local proxy，不是官方得分。回复共情和指令边界需人工复核，工具不因某个单项上升而自动推广。

## 当前限制

- 现有正式 `participant/run_inference.py` 默认 Prompt 未变，但需在 GPU 上验收实测。
- A/B 服务必须预先启动；工具不会自动启动计费 GPU。
- 完整原始输出只属于 A/B 开发工具；正式比赛仍产出原有两份文件。
- 当前本地没有可证明的 memory bank，保持 `memory_refs=[]`，不可虚构 ID。
- 尚未完成真实 GPU Smoke、BERTScore 和正式运行入口联调，不能宣称 V1 改善。
