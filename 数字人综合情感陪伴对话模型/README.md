# 参赛者资料说明

本目录中的资料用于模型训练、推理脚本调试及参赛环境准备，可以提供给参赛者。

## 文件说明

- `训练-验证-数据集/`：公开训练集、验证集及对应的数据格式 Schema；
- `participant/`：本地推理脚本、配置文件、依赖文件和提交格式模板；
- `test_inference_data.jsonl`：用于验证推理脚本能否正常运行的样例数据；
- `咪咕仝学平台实例创建demoV1.docx`：咪咕仝学平台实例创建操作说明。

## 快速使用

1. 按操作说明创建咪咕仝学平台实例。
2. 将模型放入 `participant/model`，并按需修改
   `participant/configs/inference_config.json`。
3. 安装依赖：

```bash
python3 -m pip install -r /root/participant/requirements.txt
```

4. 在 `/root` 目录下使用样例数据验证推理脚本：

```bash
bash /root/participant/start.sh /root/test_inference_data.jsonl /root/result
```

运行完成后，`/root/result` 中应生成 `submission.jsonl` 和
`performance_report.json`。详细配置和运行方式见 `participant/README.md`。


## 正式冻结版本 2026-10-14

正式冠军：V2_E1_FINAL_2334 + P0_CURRENT + V2-P1 Style + V2-P2 Personality margin 0.50。部署源码位于 `participant/`；模型和classifier二进制需另行准备。详见 `analysis/final_release_20261014/README.md`、资产manifest和清理建议。未晋级实验仅在 `archive/v2-final-experiments-20261014` 归档，不接入main正式推理。


## 正式定版 2026-10-23

E91_a075 + P0 + P1 Style + P2 Personality Gate0.50; repetition_penalty=1.05。此版本取代上文2026-10-14版本；历史证据保留。详见 `analysis/final_release_20261023/FINAL_SELECTION_REPORT.md`、manifest及提交说明。PROXY_SCORE不是官方总分。
