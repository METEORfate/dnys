# V2-E7.1 LoRA Delta Interpolation

FORMAL_RUNTIME_MODIFIED=NO
No training or base merge. Three fixed alpha values: 0.25, 0.50, 0.75.
Only emotion>=CURRENT, profile_mean>=CURRENT, and fixed response protection pass allow FAST192. Profile drop beyond 0.007 is hard reject. Macro-F1 reported, not an additional gate.

## FAST96
|candidate|emotion|macro_f1|personality|interests|style|profile_mean|BLEU|ROUGE|BERT|
|---|---|---|---|---|---|---|---|---|---|
|CURRENT|0.635417|0.521907|0.666667|0.791667|0.572917|0.677083|0.132619|0.262754|0.726882|
|a025|0.645833|0.534976|0.666667|0.781250|0.572917|0.673611|0.133885|0.265805|0.727256|
|a050|0.625000|0.520912|0.656250|0.781250|0.572917|0.670139|0.127260|0.260762|0.725862|
|a075|0.614583|0.514896|0.656250|0.802083|0.572917|0.677083|0.133132|0.264790|0.728450|
a025: wrong->right=1, right->wrong=0; delta={"emotion": 0.01041666666666674, "macro_f1": 0.013068181818181923, "personality": 0.0, "interests": -0.01041666666666663, "style": 0.0, "profile_mean": -0.003472222222222321, "bleu": 0.0012661233728048227, "rouge": 0.0030517100674636866, "bert": 0.0003731250762939453}; decision=FAST96_ONLY
a050: wrong->right=1, right->wrong=2; delta={"emotion": -0.01041666666666663, "macro_f1": -0.0009953179989944383, "personality": -0.01041666666666663, "interests": -0.01041666666666663, "style": 0.0, "profile_mean": -0.006944444444444531, "bleu": -0.005359221796901348, "rouge": -0.001991212802999409, "bert": -0.0010207295417785645}; decision=FAST96_ONLY
a075: wrong->right=1, right->wrong=3; delta={"emotion": -0.02083333333333326, "macro_f1": -0.0070112179487178405, "personality": -0.01041666666666663, "interests": 0.01041666666666674, "style": 0.0, "profile_mean": 0.0, "bleu": 0.0005131593387017741, "rouge": 0.002036446235263223, "bert": 0.0015673041343688965}; decision=FAST96_ONLY

## FAST192
Not run: no eligible candidate.
Official runtime hashes unchanged. blocking_issue: NONE

## Conclusion
Best diagnostic alpha=0.25: emotion +1/96, macro-F1 +0.013068, BLEU +0.001266, ROUGE +0.003052, BERT +0.000373; profile mean -0.003472 (one interests exact-set error). It does not pass the user-required profile>=CURRENT FAST192 gate.
Alpha=0.50/0.75 lose emotion. None cross the hard profile loss>0.007 rule, but none qualify for FAST192.
No FAST192, training, promotion or runtime changes. blocking_issue: NONE
