# V2-E7 Emotion Boundary LoRA

Result: REJECT; best: CURRENT
FORMAL_RUNTIME_MODIFIED=NO

Source adapter: /root/development/dnys/数字人综合情感陪伴对话模型/training/outputs/v2-e1-emotion-rebalance-fixed18657-rerun2
Output adapter: /root/development/dnys/数字人综合情感陪伴对话模型/training/outputs/v2-e7-emotion-boundary
Training: 80 steps, 16.12 minutes including loading; loss=0.908703; exact V2-E1 continuation verified on both ranks.
r=16, alpha=32, original 7 modules; lr=1e-5 with existing linear scheduler, batch=1, grad_acc=8, max_length=4096, checkpointing=true, packing=false, assistant-only loss.

## Dataset
2800 unique original train samples = 1960 boundary + 840 V2-E1-distribution anchor; seed=66; no relabeling or synthetic data. Hard selection uses fixed latest-user keyword cues, not confirmed LLM errors. No validation/holdout used for selection.
Mask: 2800 retained, 226 context-trimmed, zero supervised=0; 7 existing CPU tests PASS.
{"anxiety": 636, "relaxed": 379, "fear": 425, "helplessness": 376, "pride": 23, "gratitude": 327, "loneliness": 54, "neutral": 45, "mixed": 318, "shame": 50, "sadness": 62, "disgust": 15, "anger": 54, "joy": 18, "care": 10, "surprise": 8}

## FAST96
One TP2 vLLM server, max_num_seqs=8, P0 and original generation, serial rotating requests; all outputs use P1/P2. Fresh CURRENT=61/96; historical 67/96 not substituted.

|adapter|emotion|macro_f1|personality|interests|style|profile_mean|BLEU|ROUGE|BERT|avg_ms|p95_ms|
|---|---|---|---|---|---|---|---|---|---|---|---|
|p0|0.635417|0.526577|0.666667|0.791667|0.572917|0.677083|0.129899|0.262326|0.727173|1575.911393|3486.434258|
|e7-40|0.625000|0.522674|0.656250|0.791667|0.572917|0.673611|0.132652|0.266556|0.726317|1571.309329|3513.499544|
|e7-80|0.645833|0.531081|0.656250|0.770833|0.572917|0.666667|0.136198|0.269932|0.730039|1534.099684|3259.388286|

|class recall|CURRENT|checkpoint-40|checkpoint-80|
|---|---|---|---|
|anxiety|0.600000|0.633333|0.666667|
|fear|1.000000|0.800000|0.800000|
|helplessness|0.333333|0.500000|0.500000|
|mixed|0.428571|0.428571|0.428571|
|relaxed|0.647059|0.647059|0.705882|
|gratitude|0.400000|0.400000|0.400000|

e7-40: wrong->right=2, right->wrong=3, net=-1; failures=['emotion_delta_min', 'macro_f1_delta_min']
{"emotion": -0.01041666666666663, "macro_f1": -0.0039026197114432337, "personality": -0.01041666666666663, "interests": 0.0, "style": 0.0, "profile_mean": -0.003472222222222321, "bleu": 0.0027524792977244583, "rouge": 0.004229951999901116, "bert": -0.000855863094329834}

e7-80: wrong->right=4, right->wrong=3, net=1; failures=['profile_delta_min']
{"emotion": 0.01041666666666674, "macro_f1": 0.004504366820543337, "personality": -0.01041666666666663, "interests": -0.02083333333333326, "style": 0.0, "profile_mean": -0.01041666666666674, "bleu": 0.006298535530710719, "rouge": 0.00760615016062044, "bert": 0.0028653740882873535}

All groups JSON/nonempty=100%, P1/P2 fallback=0. FAST192 skipped because both failed frozen guards.
checkpoint-80 improves emotion by 1/96 and all response metrics, but profile mean falls by 0.010417 beyond the fixed 0.007 guard. checkpoint-40 loses emotion and macro-F1.
No promotion, holdout or official file modification. Hashes unchanged.
Summary profile key mapping bug fixed; original generated predictions and thresholds unchanged. Non-BERT metrics rechecked against saved predictions, existing BERT scores retained.
blocking_issue: NONE
