# V2-RQ6 Final Result

Result: **REJECT**. Training: 160 steps, 29.083 minutes. All four FAST96 evaluations complete; official P1/P2 profile postprocessing applied; BERTScore calculated offline on CPU using multilingual BERT, layer 9.

## Fixed setup

Start: V2-E1 FINAL_2334. Response-only samples: 7334. Emotion anchors: 1200, six classes x200, seed61. Rank-local microbatches follow RRRE; lr3e-6, gradient accumulation8, max_length4096. Mask audit PASS; zero supervised samples0. No holdout or official runtime changes.

## Metrics

Baseline: {"emotion": 0.6979166666666666, "bleu": 0.13465935079395705, "rouge": 0.26947329601946673, "json_valid": 1.0, "nonempty": 1.0, "personality": 0.6770833333333334, "interests": 0.8333333333333334, "style": 0.5729166666666666, "profile_mean": 0.6944444444444445, "bert_f1": 0.7311925888061523}

| Step | Emotion | Profile mean | BLEU | ROUGE | BERT F1 | Decision |
|---:|---:|---:|---:|---:|---:|---|
| 40 | 0.687500000 | 0.701388889 | 0.126981116 | 0.259168278 | 0.727630615 | REJECT |
| 80 | 0.677083333 | 0.694444444 | 0.127709640 | 0.260001854 | 0.728795350 | REJECT |
| 120 | 0.656250000 | 0.701388889 | 0.134378613 | 0.265090699 | 0.729568005 | REJECT |
| 160 | 0.656250000 | 0.701388889 | 0.139541390 | 0.269801392 | 0.730799377 | REJECT |

Only checkpoint40 passed emotion/profile guardrails, but all response metrics declined. Others failed emotion guardrail. Stop response training line; preserve champion.

## Deltas and emotion diagnostics

### Checkpoint 40

{"delta": {"emotion": -0.01041666666666663, "profile_mean": 0.006944444444444309, "bleu": -0.00767823469456963, "rouge": -0.010305017960105411, "bert_f1": -0.0035619735717773438}, "recall": {"anxiety": {"support": 30, "baseline": 0.7333333333333333, "candidate": 0.6333333333333333}, "mixed": {"support": 7, "baseline": 0.42857142857142855, "candidate": 0.8571428571428571}, "relaxed": {"support": 17, "baseline": 0.7647058823529411, "candidate": 0.7058823529411765}, "fear": {"support": 5, "baseline": 1.0, "candidate": 0.8}, "helplessness": {"support": 6, "baseline": 0.3333333333333333, "candidate": 0.3333333333333333}, "sadness": {"support": 5, "baseline": 0.8, "candidate": 1.0}}, "emotion_diagnostic": {"wrong_to_right": 4, "right_to_wrong": 5, "regressions": [{"row_index": 15, "sample_id": "fast96_row_015", "gt": "anxiety", "baseline": "anxiety", "candidate": "mixed"}, {"row_index": 22, "sample_id": "fast96_row_022", "gt": "anxiety", "baseline": "anxiety", "candidate": "mixed"}, {"row_index": 30, "sample_id": "fast96_row_030", "gt": "anxiety", "baseline": "anxiety", "candidate": "fear"}, {"row_index": 41, "sample_id": "fast96_row_041", "gt": "fear", "baseline": "fear", "candidate": "helplessness"}, {"row_index": 76, "sample_id": "fast96_row_076", "gt": "relaxed", "baseline": "relaxed", "candidate": "mixed"}]}}

### Checkpoint 80

{"delta": {"emotion": -0.02083333333333326, "profile_mean": -2.220446049250313e-16, "bleu": -0.006949710844107337, "rouge": -0.009471442245206296, "bert_f1": -0.0023972392082214355}, "recall": {"anxiety": {"support": 30, "baseline": 0.7333333333333333, "candidate": 0.6666666666666666}, "mixed": {"support": 7, "baseline": 0.42857142857142855, "candidate": 0.8571428571428571}, "relaxed": {"support": 17, "baseline": 0.7647058823529411, "candidate": 0.7058823529411765}, "fear": {"support": 5, "baseline": 1.0, "candidate": 0.8}, "helplessness": {"support": 6, "baseline": 0.3333333333333333, "candidate": 0.3333333333333333}, "sadness": {"support": 5, "baseline": 0.8, "candidate": 1.0}}, "emotion_diagnostic": null}

### Checkpoint 120

{"delta": {"emotion": -0.04166666666666663, "profile_mean": 0.006944444444444309, "bleu": -0.0002807381299499423, "rouge": -0.004382597487415152, "bert_f1": -0.0016245841979980469}, "recall": {"anxiety": {"support": 30, "baseline": 0.7333333333333333, "candidate": 0.6666666666666666}, "mixed": {"support": 7, "baseline": 0.42857142857142855, "candidate": 0.8571428571428571}, "relaxed": {"support": 17, "baseline": 0.7647058823529411, "candidate": 0.7058823529411765}, "fear": {"support": 5, "baseline": 1.0, "candidate": 0.8}, "helplessness": {"support": 6, "baseline": 0.3333333333333333, "candidate": 0.16666666666666666}, "sadness": {"support": 5, "baseline": 0.8, "candidate": 1.0}}, "emotion_diagnostic": null}

### Checkpoint 160

{"delta": {"emotion": -0.04166666666666663, "profile_mean": 0.006944444444444309, "bleu": 0.0048820397011498096, "rouge": 0.00032809612599682403, "bert_f1": -0.00039321184158325195}, "recall": {"anxiety": {"support": 30, "baseline": 0.7333333333333333, "candidate": 0.6333333333333333}, "mixed": {"support": 7, "baseline": 0.42857142857142855, "candidate": 0.8571428571428571}, "relaxed": {"support": 17, "baseline": 0.7647058823529411, "candidate": 0.7058823529411765}, "fear": {"support": 5, "baseline": 1.0, "candidate": 0.8}, "helplessness": {"support": 6, "baseline": 0.3333333333333333, "candidate": 0.3333333333333333}, "sadness": {"support": 5, "baseline": 0.8, "candidate": 0.8}}, "emotion_diagnostic": null}
