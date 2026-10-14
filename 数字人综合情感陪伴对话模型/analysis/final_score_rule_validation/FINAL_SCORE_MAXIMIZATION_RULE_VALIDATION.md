# FINAL_SCORE_MAXIMIZATION_RULE_VALIDATION

## 结论

DECISION: REJECT
FINAL_RUNTIME_MODIFIED: NO
ONLY_EMOTION_DIFF: YES
RULE_SET_HASH: 83d04374ad140709de0d4fd6c5c1152b041c328bbdad05b9a541052d9512e3d4
规则文件：/root/development/dnys/数字人综合情感陪伴对话模型/analysis/final_score_maximization/conservative_emotion_rule.py
规则来源文件 SHA256：563585fa3aab29a4645e0acae47ea3b7e2873629601f797e52ddba5baa4d53d3

## 方法

只使用已有完整val_dev 1649条的P0正式栈预测，CPU逐条回放完全冻结的规则。未重新推理、未接触holdout，未修改模型/Prompt/P1/P2。
baseline来源：/root/development/dnys/数字人综合情感陪伴对话模型/analysis/v2_pr2a_valdev/paired_predictions.jsonl
模型栈：V2_E1_FINAL_2334 + P0_CURRENT + P1 Style + P2 Personality margin 0.50。输入和正式runtime hashes均复核一致。
独立规则文件由FAST192实验代码原函数逐字提取，AST完全相同，并先复现FAST192 139/192正确，未增加或修改规则。
BERT均值复用已计算的同配置baseline结果；在1649/1649 response字节完全一致后验证候选BERT必然相同。
延迟：baseline沿用历史HTTP请求耗时；candidate为相同记录加本次CPU规则调用实测。候选avg/p95为回放估计，不是重新运行官方入口benchmark。p95采用经验分位下取样。

## 核心指标

| metric | CURRENT | CANDIDATE | delta |
|---|---:|---:|---:|
| emotion | 0.746513038 | 0.742874469 | -0.003638569 |
| macro_f1 | 0.669628620 | 0.664963956 | -0.004664664 |
| personality | 0.651303820 | 0.651303820 | +0.000000000 |
| interests | 0.805336568 | 0.805336568 | +0.000000000 |
| style | 0.554881747 | 0.554881747 | +0.000000000 |
| profile_mean | 0.670507378 | 0.670507378 | +0.000000000 |
| bleu | 0.140827077 | 0.140827077 | +0.000000000 |
| rouge | 0.289335327 | 0.289335327 | +0.000000000 |
| bert | 0.736532782 | 0.736532782 | +0.000000000 |
| avg_latency_ms | 1508.268958888 | 1508.270016850 | +0.001057962 |
| p95_latency_ms | 3876.899582334 | 3876.901546493 | +0.001964159 |

## Paired emotion

wrong_to_right: 11
right_to_wrong: 17
net: -6
changed_count: 30

## 重点类别

| label | support | CURRENT precision/recall/F1 | CANDIDATE precision/recall/F1 |
|---|---:|---|---|
| anxiety | 514 | 0.874142 / 0.743191 / 0.803365 | 0.857456 / 0.760700 / 0.806186 |
| fear | 92 | 0.611111 / 0.836957 / 0.706422 | 0.635514 / 0.739130 / 0.683417 |
| helplessness | 103 | 0.666667 / 0.796117 / 0.725664 | 0.666667 / 0.796117 / 0.725664 |
| mixed | 120 | 0.578947 / 0.366667 / 0.448980 | 0.578947 / 0.366667 / 0.448980 |
| relaxed | 290 | 0.797386 / 0.841379 / 0.818792 | 0.776025 / 0.848276 / 0.810544 |
| sadness | 93 | 0.752809 / 0.720430 / 0.736264 | 0.752809 / 0.720430 / 0.736264 |
| gratitude | 87 | 0.761364 / 0.770115 / 0.765714 | 0.766234 / 0.678161 / 0.719512 |
| pride | 31 | 0.444444 / 0.903226 / 0.595745 | 0.444444 / 0.903226 / 0.595745 |

## 全部emotion变化

IDs为本轮val_dev行顺序的实验ID，不能与FAST96/FAST192实验相同数字的ID视为同一源样本。row_index为0-based。

| id | row_index | GT | before | after | outcome |
|---|---:|---|---|---|---|
| test_000058_t1 | 57 | relaxed | gratitude | relaxed | wrong→right |
| test_000124_t1 | 123 | relaxed | gratitude | relaxed | wrong→right |
| test_000183_t1 | 182 | mixed | fear | anxiety | both wrong |
| test_000228_t1 | 227 | anxiety | fear | anxiety | wrong→right |
| test_000291_t1 | 290 | gratitude | gratitude | relaxed | right→wrong |
| test_000298_t1 | 297 | gratitude | gratitude | relaxed | right→wrong |
| test_000312_t1 | 311 | anxiety | fear | anxiety | wrong→right |
| test_000332_t1 | 331 | fear | fear | anxiety | right→wrong |
| test_000333_t1 | 332 | fear | fear | anxiety | right→wrong |
| test_000359_t1 | 358 | anxiety | fear | anxiety | wrong→right |
| test_000424_t1 | 423 | fear | fear | anxiety | right→wrong |
| test_000425_t1 | 424 | anxiety | gratitude | relaxed | both wrong |
| test_000448_t1 | 447 | fear | fear | anxiety | right→wrong |
| test_000471_t1 | 470 | gratitude | gratitude | relaxed | right→wrong |
| test_000492_t1 | 491 | fear | fear | anxiety | right→wrong |
| test_000502_t1 | 501 | anxiety | fear | anxiety | wrong→right |
| test_000511_t1 | 510 | fear | fear | anxiety | right→wrong |
| test_000521_t1 | 520 | fear | fear | anxiety | right→wrong |
| test_000604_t1 | 603 | gratitude | gratitude | relaxed | right→wrong |
| test_000689_t1 | 688 | anxiety | fear | anxiety | wrong→right |
| test_000779_t1 | 778 | anxiety | fear | anxiety | wrong→right |
| test_000911_t1 | 910 | anxiety | fear | anxiety | wrong→right |
| test_000958_t1 | 957 | fear | fear | anxiety | right→wrong |
| test_001134_t1 | 1133 | gratitude | gratitude | relaxed | right→wrong |
| test_001139_t1 | 1138 | gratitude | gratitude | relaxed | right→wrong |
| test_001185_t1 | 1184 | anxiety | fear | anxiety | wrong→right |
| test_001255_t1 | 1254 | anxiety | fear | anxiety | wrong→right |
| test_001307_t1 | 1306 | fear | fear | anxiety | right→wrong |
| test_001370_t1 | 1369 | gratitude | gratitude | relaxed | right→wrong |
| test_001394_t1 | 1393 | gratitude | gratitude | relaxed | right→wrong |

## 保护检查

response exact match: 100%
profile exact match: 100%
memory_refs exact match: 100%
id exact match: 100%
CPU规则avg ms: 0.001057962
CPU规则p95 ms: 0.002264045

PROMOTE需accuracy不下降且wrong_to_right>right_to_wrong，本次不满足。STRONG_PROMOTE需accuracy提升>=0.005且wrong_to_right>=10/right_to_wrong<=5，本次也不满足。
FAST192收益未在完整val_dev复现。保持正式冠军不变，不自动修改规则、不运行holdout。

blocking_issue: NONE
