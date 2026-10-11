# V2-R1 Residual Emotion Audit

## VAL_DEV_TOP_ERRORS
- anxiety: errors=114, support=514, recall=0.7782, train=6340
- mixed: errors=63, support=120, recall=0.4750, train=945
- relaxed: errors=48, support=290, recall=0.8345, train=2819
- helplessness: errors=26, support=103, recall=0.7476, train=1575
- sadness: errors=25, support=93, recall=0.7312, train=1085
- neutral: errors=20, support=39, recall=0.4872, train=654
- shame: errors=18, support=82, recall=0.7805, train=752
- fear: errors=17, support=92, recall=0.8152, train=1388
- loneliness: errors=12, support=77, recall=0.8442, train=876
- gratitude: errors=15, support=87, recall=0.8276, train=695

## HOLDOUT_TOP_ERRORS
- anxiety: errors=43, support=237, recall=0.8186, train=6340
- mixed: errors=36, support=57, recall=0.3684, train=945
- relaxed: errors=21, support=119, recall=0.8235, train=2819
- helplessness: errors=12, support=39, recall=0.6923, train=1575
- sadness: errors=11, support=38, recall=0.7105, train=1085
- loneliness: errors=8, support=38, recall=0.7895, train=876
- neutral: errors=7, support=14, recall=0.5000, train=654
- gratitude: errors=5, support=31, recall=0.8387, train=695
- anger: errors=4, support=36, recall=0.8889, train=855
- shame: errors=4, support=36, recall=0.8889, train=752

## TOP_CONFUSIONS_BOTH
- anxiety -> fear: total=44 (val_dev=30, holdout=14)
- mixed -> anxiety: total=33 (val_dev=18, holdout=15)
- anxiety -> mixed: total=31 (val_dev=22, holdout=9)
- mixed -> relaxed: total=17 (val_dev=11, holdout=6)
- gratitude -> relaxed: total=16 (val_dev=12, holdout=4)
- anxiety -> helplessness: total=16 (val_dev=14, holdout=2)
- helplessness -> anxiety: total=16 (val_dev=10, holdout=6)
- relaxed -> pride: total=16 (val_dev=14, holdout=2)
- relaxed -> gratitude: total=15 (val_dev=11, holdout=4)
- fear -> anxiety: total=14 (val_dev=11, holdout=3)

HIGH_VALUE_CLASSES: anxiety, mixed, relaxed, helplessness, sadness, neutral, shame, fear, loneliness, gratitude, anger, joy
V2_REGRESSED_HIGH_SUPPORT_CLASSES: anxiety, relaxed, sadness, joy
PRIMARY_TARGET: anxiety
SECONDARY_TARGET: mixed
DO_NOT_TOUCH_CLASSES: 

V2_E4_DATA_STRATEGY: continued SFT from V2-E1 FINAL_2334; 70-80% V2-E1 emotion anchors and 20-30% targeted hard examples for the primary/secondary confusion pairs, <=8000 samples, max_steps<=160, lr<=1e-5; protect profile/response with FAST96 gates.
recommended_max_steps: 160
recommended_lr: 1e-5
