# V2-E7.2 Low-strength LoRA Delta Interpolation

FORMAL_RUNTIME_MODIFIED=NO
No training or base merge. Three new alpha values: 0.10, 0.15, 0.20. Reference alpha=0.25 is regenerated in the same server.
FAST192 only for new alpha candidates with emotion/profile/BLEU/ROUGE/BERT >= CURRENT. Macro-F1 is reported, not a gate.

## FAST96
|candidate|emotion|macro_f1|personality|interests|style|profile_mean|BLEU|ROUGE|BERT|
|---|---|---|---|---|---|---|---|---|---|
|CURRENT|0.625000|0.516153|0.666667|0.770833|0.572917|0.670139|0.129829|0.262703|0.726006|
|a010|0.635417|0.520803|0.656250|0.812500|0.572917|0.680556|0.129955|0.263486|0.727263|
|a015|0.635417|0.531122|0.656250|0.760417|0.572917|0.663194|0.127987|0.261022|0.725989|
|a020|0.625000|0.522600|0.656250|0.791667|0.572917|0.673611|0.132583|0.263972|0.725675|
|a025|0.625000|0.524759|0.677083|0.791667|0.572917|0.680556|0.132894|0.262297|0.727609|
a010: wrong->right=1, right->wrong=0; delta={"emotion": 0.01041666666666663, "macro_f1": 0.004649634956238824, "personality": -0.01041666666666663, "interests": 0.04166666666666663, "style": 0.0, "profile_mean": 0.01041666666666663, "bleu": 0.0001258205424776726, "rouge": 0.0007831336148360624, "bert": 0.0012563467025756836}; decision=FAST192_ELIGIBLE
a015: wrong->right=2, right->wrong=1; delta={"emotion": 0.01041666666666663, "macro_f1": 0.01496939077231818, "personality": -0.01041666666666663, "interests": -0.01041666666666674, "style": 0.0, "profile_mean": -0.00694444444444442, "bleu": -0.0018422099633104705, "rouge": -0.0016811775161029718, "bert": -1.6987323760986328e-05}; decision=FAST96_ONLY
a020: wrong->right=1, right->wrong=1; delta={"emotion": 0.0, "macro_f1": 0.006446663499590888, "personality": -0.01041666666666663, "interests": 0.02083333333333326, "style": 0.0, "profile_mean": 0.00347222222222221, "bleu": 0.002754199460426099, "rouge": 0.0012687379343092609, "bert": -0.0003317594528198242}; decision=FAST96_ONLY
a025: wrong->right=2, right->wrong=2; delta={"emotion": 0.0, "macro_f1": 0.008606268600372502, "personality": 0.01041666666666674, "interests": 0.02083333333333326, "style": 0.0, "profile_mean": 0.01041666666666663, "bleu": 0.003064748604404477, "rouge": -0.00040557531304885597, "bert": 0.0016023516654968262}; decision=FAST96_ONLY

## FAST192
|candidate|emotion|macro_f1|personality|interests|style|profile_mean|BLEU|ROUGE|BERT|
|---|---|---|---|---|---|---|---|---|---|
|CURRENT|0.687500|0.596388|0.656250|0.776042|0.526042|0.652778|0.130247|0.265032|0.727577|
|a010|0.692708|0.597757|0.656250|0.802083|0.526042|0.661458|0.128210|0.263800|0.728490|
a010: wrong->right=1, right->wrong=0; delta={"emotion": 0.00520833333333337, "macro_f1": 0.0013698250540356005, "personality": 0.0, "interests": 0.02604166666666674, "style": 0.0, "profile_mean": 0.00868055555555558, "bleu": -0.002036874830704416, "rouge": -0.0012320303480788941, "bert": 0.0009131431579589844}; decision=FAST96_ONLY
Official runtime hashes unchanged. blocking_issue: NONE

## Conclusion
Best observed alpha=0.10; official/current winner remains CURRENT. FAST192 emotion +1/192, profile mean +0.008681, BERT +0.000913, but BLEU -0.002037 and ROUGE -0.001232. Fails the fixed all-response-nondecreasing requirement.
Fresh96: {"p0": {"emotion_correct": 72, "emotion": 0.75, "personality": 0.6458333333333334, "interests": 0.78125, "style": 0.4791666666666667, "profile_mean": 0.6354166666666666, "bleu": 0.13066452086578742, "rouge": 0.2673620319603955}, "a010": {"emotion_correct": 72, "emotion": 0.75, "personality": 0.65625, "interests": 0.7916666666666666, "style": 0.4791666666666667, "profile_mean": 0.642361111111111, "bleu": 0.12646495066190092, "rouge": 0.2641148376494016}, "paired": {"wrong_to_right": 0, "right_to_wrong": 0, "net": 0}}
No new emotion gain in the additional96. No formal promotion or runtime modification.
FORMAL_RUNTIME_MODIFIED=NO
blocking_issue: NONE
