# V2 Hard Case Audit

{
  "hard_case_count": 1402,
  "emotion_error_count": 446,
  "profile_personality_error_count": 631,
  "profile_interests_error_count": 372,
  "profile_style_error_count": 891,
  "confusion_pairs": {
    "('gratitude', 'relaxed')": {
      "count": 54,
      "sample_indices": [
        20,
        65,
        90,
        162,
        186,
        190,
        241,
        257,
        261,
        274,
        290,
        339,
        360,
        379,
        401,
        414,
        470,
        475,
        480,
        500,
        502,
        568,
        583,
        603,
        641,
        673,
        689,
        690,
        735,
        737
      ],
      "assessment_counts": {
        "AMBIGUOUS": 54,
        "CLEAR_GT": 0,
        "POSSIBLE_LABEL_NOISE": 0,
        "MODEL_ERROR": 0
      }
    },
    "('anxiety', 'relaxed')": {
      "count": 19,
      "sample_indices": [
        25,
        246,
        518,
        538,
        556,
        606,
        644,
        740,
        958,
        1004,
        1049,
        1076,
        1146,
        1154,
        1266,
        1478,
        1571,
        1614,
        1642
      ],
      "assessment_counts": {
        "AMBIGUOUS": 19,
        "CLEAR_GT": 0,
        "POSSIBLE_LABEL_NOISE": 0,
        "MODEL_ERROR": 0
      }
    },
    "('mixed', 'relaxed')": {
      "count": 31,
      "sample_indices": [
        111,
        149,
        305,
        317,
        563,
        651,
        684,
        713,
        724,
        773,
        806,
        874,
        983,
        1022,
        1037,
        1088,
        1095,
        1111,
        1120,
        1173,
        1206,
        1252,
        1283,
        1307,
        1320,
        1438,
        1469,
        1536,
        1557,
        1563
      ],
      "assessment_counts": {
        "AMBIGUOUS": 31,
        "CLEAR_GT": 0,
        "POSSIBLE_LABEL_NOISE": 0,
        "MODEL_ERROR": 0
      }
    },
    "('mixed', 'anxiety')": {
      "count": 21,
      "sample_indices": [
        182,
        204,
        217,
        222,
        302,
        382,
        398,
        402,
        630,
        704,
        866,
        895,
        1115,
        1198,
        1365,
        1388,
        1433,
        1462,
        1539,
        1587,
        1612
      ],
      "assessment_counts": {
        "AMBIGUOUS": 21,
        "CLEAR_GT": 0,
        "POSSIBLE_LABEL_NOISE": 0,
        "MODEL_ERROR": 0
      }
    },
    "('fear', 'anxiety')": {
      "count": 26,
      "sample_indices": [
        356,
        400,
        418,
        450,
        481,
        509,
        520,
        525,
        555,
        596,
        676,
        697,
        709,
        790,
        916,
        922,
        956,
        999,
        1019,
        1031,
        1129,
        1264,
        1306,
        1496,
        1573,
        1597
      ],
      "assessment_counts": {
        "AMBIGUOUS": 26,
        "CLEAR_GT": 0,
        "POSSIBLE_LABEL_NOISE": 0,
        "MODEL_ERROR": 0
      }
    }
  },
  "profile_leak_audit": {
    "personality_traits": {
      "sensitive": {
        "gt_positive": 337,
        "prediction_positive": 8,
        "false_negative": 334,
        "strong_evidence_rate": 0.0
      },
      "conservative": {
        "gt_positive": 206,
        "prediction_positive": 39,
        "false_negative": 178,
        "strong_evidence_rate": 0.0
      },
      "high_conscientiousness": {
        "gt_positive": 148,
        "prediction_positive": 55,
        "false_negative": 119,
        "strong_evidence_rate": 0.0
      }
    },
    "interests": {
      "career_development": {
        "gt_positive": 99,
        "prediction_positive": 7,
        "false_negative": 95,
        "strong_evidence_rate": 0.0
      },
      "study_exam": {
        "gt_positive": 73,
        "prediction_positive": 38,
        "false_negative": 60,
        "strong_evidence_rate": 0.0
      },
      "pets": {
        "gt_positive": 60,
        "prediction_positive": 28,
        "false_negative": 37,
        "strong_evidence_rate": 0.0
      }
    },
    "style": {
      "rational": {
        "gt_positive": 400,
        "prediction_positive": 97,
        "false_negative": 370,
        "strong_evidence_rate": 0.0
      },
      "direct": {
        "gt_positive": 196,
        "prediction_positive": 80,
        "false_negative": 183,
        "strong_evidence_rate": 0.0
      },
      "detailed": {
        "gt_positive": 94,
        "prediction_positive": 6,
        "false_negative": 92,
        "strong_evidence_rate": 0.0
      }
    }
  },
  "duplicate_groups": 508,
  "label_conflict_groups": 0,
  "emotion_conflict_groups": 0,
  "style_conflict_groups": 0,
  "personality_conflict_groups": 0,
  "top_data_quality_problems": [
    {
      "problem": "emotion class imbalance/confusion",
      "affected_samples": 446,
      "likely_root_cause": "CLASS_IMBALANCE"
    },
    {
      "problem": "profile labels underpredicted",
      "affected_samples": 891,
      "likely_root_cause": "MODEL_UNDERPREDICTION"
    },
    {
      "problem": "possible label ambiguity in confusion pairs",
      "affected_samples": 151,
      "likely_root_cause": "LABEL_AMBIGUITY"
    }
  ],
  "v2_e2_candidates": [
    {
      "name": "confusion-aware oversampling",
      "target_metric": "emotion_macro_f1",
      "affected_samples": "confusion-pair training rows",
      "retrain": true,
      "risk": "overfit boundary cases"
    },
    {
      "name": "profile-positive oversampling",
      "target_metric": "style/profile exact-set",
      "affected_samples": "positive profile rows",
      "retrain": true,
      "risk": "false positive increase"
    },
    {
      "name": "label-noise filtering",
      "target_metric": "label consistency",
      "affected_samples": 0,
      "retrain": true,
      "risk": "remove valid rare examples"
    }
  ],
  "best_cpu_only_next_action": "review sampled confusion/profile cases and choose one isolated V2-E2 data change",
  "runtime_baseline_preserved": true
}