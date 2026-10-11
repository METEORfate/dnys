# V2-E3 checkpoint-40 Full val_dev Confirmation

samples: 1649
prediction_lines: 1649
json_parse_success_rate: 1.0

| metric | V2-E1 | checkpoint-40 | delta |
|---|---:|---:|---:|
| emotion_accuracy | 0.7647058824 | 0.7701637355973318 | 0.005457853197331719 |
| emotion_f1 | 0.6980062645 | 0.6967969018519943 | -0.0012093626480057518 |
| personality | 0.6373559733 | 0.6434202546998181 | 0.006064281399818072 |
| interests | 0.8259551243 | 0.8247422680412371 | -0.0012128562587628844 |
| style | 0.4627046695 | 0.44329896907216493 | -0.01940570042783507 |
| profile_mean | 0.6420052557 | 0.6371538306044067 | -0.004851425095593331 |
| BLEU | 0.1467965098 | 0.14614470094148613 | -0.0006518088585138748 |
| ROUGE | 0.296921823 | 0.295330567471937 | -0.0015912555280629959 |

BERTScore: None
BERTScore_delta: None
FULL_RESULT: REJECT
blocking_issue: NONE

Execution note: The unchanged evaluator automatically attempted its default BERTScore Hugging Face request; this request failed and produced no score. The fixed-local-model paired BERTScore was skipped because the profile gate failed. This is an execution deviation from the requested offline BERTScore flow.
