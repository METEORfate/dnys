# FINAL_SCORE_MAXIMIZATION 完整报告

报告日期：2026-10-13（Asia/Shanghai）

## 结论

- BEST_CANDIDATE：rule（仅 emotion 保守规则）。
- 是否值得进入完整 val_dev：YES；尚未执行完整 val_dev 或 holdout。
- 正式冠军保持冻结：V2_E1_FINAL_2334 + P0_CURRENT + P1 Style + P2 Personality margin 0.50。
- FORMAL_RUNTIME_MODIFIED：NO；正式源码、Prompt、配置及 classifier 资产哈希复核通过。
- blocking_issue：NONE（已完成结果）；检查完成后平台 NVML 再次异常，后续 GPU 任务需重新健康检查。
- 本次剩余评估用时：12.49 分钟。
- 早先特征提取/分类头训练曾受 CUDA/NCCL/NVML 中断；恢复时复用了全部缓存和已训练分类头，没有重新训练 Qwen/LoRA。

## 数据与方法

训练 prediction points：18657。内部分类头 train/valid 按标签分层 80/20，seed=62。只优化新分类头，backbone/LoRA 所有参数冻结。
特征：最终 decoder 层的最新内容末 token，拼接前128个 context token 均值；特征输入上限4096。特征使用已有 NF4/bf16 离线模型路径，生成继续使用原正式 vLLM 栈。
FAST96 完全复用原固定集；FAST192=原96+seed62追加的96条不重复 val_dev 样本。train 与所选评估输入 hash 无交集。未使用 holdout。
隐藏分类头未使用 val_dev 训练或选 checkpoint，checkpoint 依据 train 内部 valid accuracy 保存。
规则与 confidence gate 在推理前固定；Hybrid confidence>=0.90 且 probability margin>=0.50。未在结果出来后调整。
FAST192晋级需整体提升，并要求新增96单独仍有收益。这里只提供候选筛选结论，不推算官方总分。

## 分类头训练

| Head | train accuracy | internal valid accuracy | internal macro-F1 | FAST96 accuracy | FAST96 macro-F1 |
|---|---:|---:|---:|---:|---:|
| linear | 0.932328 | 0.576367 | 0.435224 | 0.562500 | 0.366035 |
| mlp | 0.994439 | 0.621383 | 0.505804 | 0.604167 | 0.463515 |

训练/内部验证差距较大。内部 split 是 prediction-point 层面，未保证 conversation 独立，内部验证不得替代外部泛化判断。

## FAST96 完整指标

| Candidate | emotion | macro-F1 | personality | interests | style | profile mean | BLEU | ROUGE-L | BERT | avg ms | p95 ms | result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CURRENT | 0.635417 | 0.526453 | 0.666667 | 0.791667 | 0.572917 | 0.677083 | 0.136300 | 0.264278 | 0.726949 | 1551.365 | 3310.926 | BASELINE |
| linear | 0.562500 | 0.366035 | 0.666667 | 0.791667 | 0.572917 | 0.677083 | 0.136300 | 0.264278 | 0.726949 | 1876.851 | 3636.412 | REJECT |
| mlp | 0.604167 | 0.463515 | 0.666667 | 0.791667 | 0.572917 | 0.677083 | 0.136300 | 0.264278 | 0.726949 | 1876.898 | 3636.460 | REJECT |
| rule | 0.656250 | 0.534821 | 0.666667 | 0.791667 | 0.572917 | 0.677083 | 0.136300 | 0.264278 | 0.726949 | 1551.381 | 3310.942 | ADVANCE_FAST192 |
| hybrid | 0.635417 | 0.506242 | 0.666667 | 0.791667 | 0.572917 | 0.677083 | 0.136300 | 0.264278 | 0.726949 | 1876.973 | 3636.535 | REJECT |
| response_wrapper | 0.635417 | 0.526453 | 0.666667 | 0.791667 | 0.572917 | 0.677083 | 0.137586 | 0.269211 | 0.727623 | 3084.487 | 6194.513 | REJECT |

### FAST96 相对 CURRENT 的 paired delta

| Candidate | emotion delta | macro-F1 delta | profile delta | BLEU delta | ROUGE delta | BERT delta | wrong→right | right→wrong | net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | -0.072917 | -0.160418 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 18 | 25 | -7 |
| mlp | -0.031250 | -0.062938 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 17 | 20 | -3 |
| rule | +0.020833 | +0.008369 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 3 | 1 | +2 |
| hybrid | +0.000000 | -0.020210 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 5 | 5 | +0 |
| response_wrapper | +0.000000 | +0.000000 | +0.000000 | +0.001285 | +0.004933 | +0.000674 | 0 | 0 | +0 |

Response wrapper：【仅回复补充】response_text先直接承接用户当前需要，避免重复已经解释过的道理。未请求建议时不追加方案，未提出新问题时不必追问。其它字段判断规则完全沿用前文。

Wrapper仅提供response_text，其余字段复用baseline。初筛预注册的明显response提升线为BLEU和ROUGE均至少+0.002、BERT不低于-0.0005。Wrapper的BLEU仅+0.001285，因此没有纳入FAST192候选指标。
扩展阶段生成器仍记录了wrapper输出，但未用于FAST192模型选择或调参。

## FAST192 规则候选确认

| Candidate | emotion | macro-F1 | profile mean | BLEU | ROUGE-L | BERT | avg ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CURRENT | 0.697917 | 0.603593 | 0.654514 | 0.133617 | 0.264998 | 0.728205 | 1532.046 | 3247.954 |
| rule | 0.723958 | 0.615161 | 0.654514 | 0.133617 | 0.264998 | 0.728205 | 1532.051 | 3247.959 |

emotion delta：+0.026042；macro-F1 delta：+0.011569。
wrong→right：6；right→wrong：1；net：+5。
新增96：CURRENT=73/96，rule=76/96，delta=+0.031250，修复3条、损伤0条。
Profile、BLEU、ROUGE、BERT delta全部0，response_text、user_profile、memory_refs沿用同条baseline。

## 实际规则（未调整）

1. baseline为gratitude/pride，且最新user明确出现释然/压力缓解短语时，改为relaxed。
   正则：`松了一口气|释然|轻松(?:多了|一些|了些)|没那么堵|心里.*石头.*轻|不用再.*猜`
2. baseline为fear，且最新user包含`万一|会不会|最担心`，同时不包含`医生|体检|确诊|癌|刀|打我|死亡|危险`，改为anxiety。
3. baseline为anxiety，且最新user包含`无能为力|无路可走|怎么选都不对|根本没办法|不管.*都没用`，改为helplessness。

### FAST192全部规则迁移

| ID | GT | CURRENT | rule |
|---|---|---|---|
| test_000008_t1 | anxiety | fear | anxiety |
| test_000030_t1 | anxiety | fear | anxiety |
| test_000041_t1 | fear | fear | anxiety |
| test_000081_t1 | relaxed | gratitude | relaxed |
| test_000129_t1 | anxiety | fear | anxiety |
| test_000139_t1 | mixed | fear | anxiety |
| test_000156_t1 | anxiety | fear | anxiety |
| test_000167_t1 | anxiety | fear | anxiety |

ID为本实验固定row ID，追加样本的原val_dev行号保存在fast192_manifest.json。

## 合法性、耗时与局限

各候选JSON valid/nonempty均为1.0。P1/P2 fallback断言为0。
正式baseline与wrapper延迟为本轮HTTP生成实际计时；wrapper的组合延迟包括baseline与二次请求，平均约3084.49ms，近乎翻倍。
规则额外耗时来自本次规则选择循环实测，FAST192均值差约0.00494ms；尚不是官方入口benchmark，不能由此宣称官方性能收益。
Linear/MLP/Hybrid总延迟是baseline实际计时加特征开销估计；特征开销325.41ms来自前100条训练特征提取计时proxy，不是正式runtime端到端测量。分类头单次推理实测约Linear=0.075ms、MLP=0.122ms。
收益仍需冻结规则后完整val_dev验证；FAST192按emotion扩展，分布不等于官方自然测试分布。规则包含1条明确损伤及1条仍错误的修改，不能直接晋升正式runtime。
任务完成后检测到NVML异常；未重启、未修改环境，结果已完整保存。

## 文件

- result.json
- head_results.json
- head_fast96_metrics.json
- fast96_result.json
- fast192_result.json
- fast192_manifest.json
- generation.jsonl（完整预测，不提交Git）

BEST_CANDIDATE: rule
WORTH_FULL_VALDEV: YES
FORMAL_RUNTIME_MODIFIED: NO
blocking_issue: NONE（已完成结果；后续运行需GPU健康检查）
