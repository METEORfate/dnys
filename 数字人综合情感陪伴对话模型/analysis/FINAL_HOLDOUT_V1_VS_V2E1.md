# Final Holdout V1 vs V2-E1

samples: 705

| metric | V1 | V2-E1 | delta |
|---|---:|---:|---:|
| emotion_accuracy | 0.7120567376 | 0.7716312057 | +0.0595744681 |
| emotion_f1 | 0.5256771854 | 0.7343320765 | +0.2086548911 |
| bleu_4 | 0.1526327620 | 0.1497540200 | -0.0028787420 |
| rouge_l | 0.3031493371 | 0.2949278653 | -0.0082214718 |
| profile_personality | 0.6411347518 | 0.6340425532 | -0.0070921986 |
| profile_interests | 0.7929078014 | 0.8609929078 | +0.0680851064 |
| profile_style | 0.4468085106 | 0.4808510638 | +0.0340425532 |
| profile_mean | 0.6269503546 | 0.6586288416 | +0.0316784870 |

json_valid_v1/v2: 1.0 / 1.0
response_nonempty_v1/v2: 1.0 / 1.0
memory_refs_v1/v2: 1.0 / 1.0
BERTScore: unavailable (bert_score not installed)

FINAL_MODEL_CANDIDATE: V2_E1
holdout_supports_valdev_result: YES

V2 improves emotion accuracy/F1 and profile mean, with modest BLEU/ROUGE and personality declines.
