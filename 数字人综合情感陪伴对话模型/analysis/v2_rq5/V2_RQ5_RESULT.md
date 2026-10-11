# V2-RQ5 Module-Selective Delta Transplant

Result: **REJECT**. All six FAST96 runs complete; official P1/P2 applied. BERTScore completed on CPU using local multilingual BERT, num_layers=9, lang=zh, rescale_with_baseline=False.

Adapter compatible: YES. Layer count: 28. No training, holdout or official runtime changes.

Baseline: {"emotion": 0.6979166666666666, "bleu": 0.13465935079395705, "rouge": 0.26947329601946673, "json_valid": 1.0, "nonempty": 1.0, "personality": 0.6770833333333334, "interests": 0.8333333333333334, "style": 0.5729166666666666, "profile_mean": 0.6944444444444445, "bert_f1": 0.7311925888061523}

| Candidate | Selected keys | Parameter % | Emotion | Personality | Interests | Style | Profile | BLEU | ROUGE | BERT | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ATTN_ALL | 224/392 | 25.0 | 0.666666667 | 0.677083333 | 0.854166667 | 0.572916667 | 0.701388889 | 0.135723322 | 0.263771282 | 0.730178654 | REJECT |
| MLP_ALL | 168/392 | 75.0 | 0.666666667 | 0.656250000 | 0.854166667 | 0.572916667 | 0.694444444 | 0.139160091 | 0.271390773 | 0.732459486 | REJECT |
| LATE_HALF_ALL | 196/392 | 50.0 | 0.666666667 | 0.666666667 | 0.833333333 | 0.572916667 | 0.690972222 | 0.140338290 | 0.268435168 | 0.730878830 | REJECT |
| LATE_HALF_ATTN | 112/392 | 12.5 | 0.687500000 | 0.687500000 | 0.854166667 | 0.572916667 | 0.704861111 | 0.136643775 | 0.267140511 | 0.730142117 | REJECT |
| LATE_HALF_MLP | 84/392 | 37.5 | 0.687500000 | 0.666666667 | 0.854166667 | 0.572916667 | 0.697916667 | 0.131826070 | 0.262512836 | 0.727277935 | REJECT |
| LATE_QUARTER_ALL | 98/392 | 25.0 | 0.677083333 | 0.677083333 | 0.854166667 | 0.572916667 | 0.701388889 | 0.138063093 | 0.271730083 | 0.731576920 | REJECT |

All unselected keys tensor/dtype exact; all NaN/Inf counts zero. All JSON/nonempty rates 1.0.

## Deltas and regression diagnostics

### ATTN_ALL

{"delta": {"emotion": -0.03125, "profile_mean": 0.006944444444444309, "bleu": 0.0010639716337826066, "rouge": -0.005702014208581074, "bert_f1": -0.00101393461227417}, "guardrail_pass": false, "emotion_diagnostic": null}

### MLP_ALL

{"delta": {"emotion": -0.03125, "profile_mean": -2.220446049250313e-16, "bleu": 0.004500740543814524, "rouge": 0.0019174768530543784, "bert_f1": 0.0012668967247009277}, "guardrail_pass": false, "emotion_diagnostic": null}

### LATE_HALF_ALL

{"delta": {"emotion": -0.03125, "profile_mean": -0.003472222222222321, "bleu": 0.005678939406335903, "rouge": -0.0010381276991293653, "bert_f1": -0.00031375885009765625}, "guardrail_pass": false, "emotion_diagnostic": null}

### LATE_HALF_ATTN

{"delta": {"emotion": -0.01041666666666663, "profile_mean": 0.010416666666666519, "bleu": 0.001984424624433462, "rouge": -0.002332785436198692, "bert_f1": -0.0010504722595214844}, "guardrail_pass": true, "emotion_diagnostic": {"wrong_to_right": 1, "right_to_wrong": 2, "regressions": [{"row_index": 38, "sample_id": "fast96_row_038", "gt": "fear", "baseline": "fear", "candidate": "anxiety"}, {"row_index": 41, "sample_id": "fast96_row_041", "gt": "fear", "baseline": "fear", "candidate": "sadness"}], "sample_id_note": "row identifier used when source id absent"}}

### LATE_HALF_MLP

{"delta": {"emotion": -0.01041666666666663, "profile_mean": 0.003472222222222099, "bleu": -0.0028332811669460334, "rouge": -0.0069604595674407355, "bert_f1": -0.00391465425491333}, "guardrail_pass": true, "emotion_diagnostic": {"wrong_to_right": 0, "right_to_wrong": 1, "regressions": [{"row_index": 81, "sample_id": "fast96_row_081", "gt": "relaxed", "baseline": "relaxed", "candidate": "pride"}], "sample_id_note": "row identifier used when source id absent"}}

### LATE_QUARTER_ALL

{"delta": {"emotion": -0.02083333333333326, "profile_mean": 0.006944444444444309, "bleu": 0.003403742629025014, "rouge": 0.00225678697237941, "bert_f1": 0.00038433074951171875}, "guardrail_pass": false, "emotion_diagnostic": null}

Best eligible candidate: LATE_HALF_ATTN, REJECT. ROUGE/BERT decline; no promoted candidate. Keep frozen champion.
Blocking issue: NONE.
