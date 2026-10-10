# V2-E3 FAST PROFILE BOOST Result

dataset_total: 6000
emotion_anchor_count: 4200
profile_positive_count: 1800
training_steps: 120
training_minutes: ~36

| checkpoint | emotion | profile | personality | interests | style | BLEU | ROUGE | BERTScore | proxy | result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| checkpoint-40 | 0.687500 | 0.673611 | 0.687500 | 0.864583 | 0.468750 | 0.137415 | 0.275432 | 0.732216 | 0.381688 | PROMOTE |
| checkpoint-80 | 0.687500 | 0.684028 | 0.687500 | 0.864583 | 0.500000 | 0.131611 | 0.269078 | 0.731645 | 0.377445 | REJECT (BLEU protection) |
| checkpoint-120 | 0.687500 | 0.666667 | 0.666667 | 0.864583 | 0.468750 | 0.132122 | 0.265706 | 0.730195 | 0.376008 | REJECT (ROUGE protection) |

FAST_WINNER: checkpoint-40
blocking_issue: NONE
