# V2-E1 Fixed Emotion Rebalance Plan

{
  "original_samples": 18657,
  "resampled_samples": 18657,
  "emotion_distribution_before": {
    "anxiety": 6340,
    "fear": 1388,
    "mixed": 945,
    "relaxed": 2819,
    "joy": 171,
    "neutral": 654,
    "anger": 855,
    "sadness": 1085,
    "helplessness": 1575,
    "shame": 752,
    "loneliness": 876,
    "care": 79,
    "gratitude": 695,
    "pride": 234,
    "surprise": 64,
    "disgust": 125
  },
  "emotion_distribution_after": {
    "sadness": 1366,
    "anxiety": 3455,
    "mixed": 1286,
    "fear": 1453,
    "shame": 1120,
    "helplessness": 1687,
    "gratitude": 1039,
    "relaxed": 2209,
    "neutral": 991,
    "anger": 1207,
    "loneliness": 1209,
    "pride": 507,
    "surprise": 188,
    "care": 218,
    "disgust": 325,
    "joy": 397
  },
  "distribution": {
    "anger": {
      "original_count": 855,
      "original_ratio": 0.04582730342498794,
      "resampled_count": 1207,
      "resampled_ratio": 0.06469421664790695,
      "delta_count": 352,
      "delta_ratio": 0.018866913222919012
    },
    "anxiety": {
      "original_count": 6340,
      "original_ratio": 0.3398188347537117,
      "resampled_count": 3455,
      "resampled_ratio": 0.18518518518518517,
      "delta_count": -2885,
      "delta_ratio": -0.15463364956852657
    },
    "care": {
      "original_count": 79,
      "original_ratio": 0.004234335638098301,
      "resampled_count": 218,
      "resampled_ratio": 0.011684622393739616,
      "delta_count": 139,
      "delta_ratio": 0.007450286755641314
    },
    "disgust": {
      "original_count": 125,
      "original_ratio": 0.006699898161547944,
      "resampled_count": 325,
      "resampled_ratio": 0.017419735220024657,
      "delta_count": 200,
      "delta_ratio": 0.01071983705847671
    },
    "fear": {
      "original_count": 1388,
      "original_ratio": 0.07439566918582838,
      "resampled_count": 1453,
      "resampled_ratio": 0.0778796162298333,
      "delta_count": 65,
      "delta_ratio": 0.003483947044004931
    },
    "gratitude": {
      "original_count": 695,
      "original_ratio": 0.03725143377820657,
      "resampled_count": 1039,
      "resampled_ratio": 0.055689553518786516,
      "delta_count": 344,
      "delta_ratio": 0.018438119740579944
    },
    "helplessness": {
      "original_count": 1575,
      "original_ratio": 0.0844187168355041,
      "resampled_count": 1687,
      "resampled_ratio": 0.09042182558825106,
      "delta_count": 112,
      "delta_ratio": 0.006003108752746959
    },
    "joy": {
      "original_count": 171,
      "original_ratio": 0.009165460684997588,
      "resampled_count": 397,
      "resampled_ratio": 0.02127887656107627,
      "delta_count": 226,
      "delta_ratio": 0.012113415876078684
    },
    "loneliness": {
      "original_count": 876,
      "original_ratio": 0.046952886316128,
      "resampled_count": 1209,
      "resampled_ratio": 0.06480141501849172,
      "delta_count": 333,
      "delta_ratio": 0.017848528702363725
    },
    "mixed": {
      "original_count": 945,
      "original_ratio": 0.05065123010130246,
      "resampled_count": 1286,
      "resampled_ratio": 0.06892855228600525,
      "delta_count": 341,
      "delta_ratio": 0.018277322184702793
    },
    "neutral": {
      "original_count": 654,
      "original_ratio": 0.03505386718121885,
      "resampled_count": 991,
      "resampled_ratio": 0.05311679262475211,
      "delta_count": 337,
      "delta_ratio": 0.01806292544353326
    },
    "pride": {
      "original_count": 234,
      "original_ratio": 0.012542209358417752,
      "resampled_count": 507,
      "resampled_ratio": 0.027174786943238462,
      "delta_count": 273,
      "delta_ratio": 0.01463257758482071
    },
    "relaxed": {
      "original_count": 2819,
      "original_ratio": 0.15109610333922924,
      "resampled_count": 2209,
      "resampled_ratio": 0.11840060031087528,
      "delta_count": -610,
      "delta_ratio": -0.032695503028353966
    },
    "sadness": {
      "original_count": 1085,
      "original_ratio": 0.05815511604223616,
      "resampled_count": 1366,
      "resampled_ratio": 0.07321648710939593,
      "delta_count": 281,
      "delta_ratio": 0.01506137106715978
    },
    "shame": {
      "original_count": 752,
      "original_ratio": 0.040306587339872434,
      "resampled_count": 1120,
      "resampled_ratio": 0.060031087527469586,
      "delta_count": 368,
      "delta_ratio": 0.01972450018759715
    },
    "surprise": {
      "original_count": 64,
      "original_ratio": 0.0034303478587125475,
      "resampled_count": 188,
      "resampled_ratio": 0.010076646834968108,
      "delta_count": 124,
      "delta_ratio": 0.006646298976255561
    }
  },
  "unique_original_samples": 11323,
  "duplicate_sample_count": 5253,
  "max_repeat_factor": 3,
  "seed": 42,
  "sampling": "sqrt inverse-frequency weighted sampling with replacement, capped per original sample at 3",
  "v1_expected_steps": "same as V1 because dataset length is 18657",
  "v2_e1_expected_steps": "same as V1 because dataset length is 18657",
  "steps_equal": true,
  "val_dev_unchanged": true,
  "val_holdout_unchanged": true,
  "data_leakage": false,
  "overlap_counts": {
    "train_val_dev": 0,
    "train_val_holdout": 0
  },
  "manifest": "/root/development/dnys/数字人综合情感陪伴对话模型/training/data/v2_e1_emotion_rebalance_fixed18657_manifest.jsonl",
  "single_variable_changed": "TRAINING_SAMPLE_DISTRIBUTION"
}