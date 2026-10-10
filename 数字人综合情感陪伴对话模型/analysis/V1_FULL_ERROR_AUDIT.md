# V1 Full Error Audit

{
  "samples": 1649,
  "emotion_accuracy": 0.7295330503335354,
  "emotion_macro_precision": 0.7017214432640283,
  "emotion_macro_recall": 0.5548517578220514,
  "emotion_macro_f1": 0.5950153254042316,
  "emotion_by_label": {
    "": {
      "support": 0,
      "predicted_count": 1,
      "precision": 0.0,
      "recall": 0,
      "f1": 0.0,
      "false_positive": 1,
      "false_negative": 0
    },
    "anger": {
      "support": 79,
      "predicted_count": 91,
      "precision": 0.7472527472527473,
      "recall": 0.8607594936708861,
      "f1": 0.8,
      "false_positive": 23,
      "false_negative": 11
    },
    "anxiety": {
      "support": 514,
      "predicted_count": 548,
      "precision": 0.801094890510949,
      "recall": 0.8540856031128404,
      "f1": 0.8267419962335216,
      "false_positive": 109,
      "false_negative": 75
    },
    "care": {
      "support": 7,
      "predicted_count": 2,
      "precision": 0.5,
      "recall": 0.14285714285714285,
      "f1": 0.2222222222222222,
      "false_positive": 1,
      "false_negative": 6
    },
    "disgust": {
      "support": 15,
      "predicted_count": 10,
      "precision": 0.9,
      "recall": 0.6,
      "f1": 0.72,
      "false_positive": 1,
      "false_negative": 6
    },
    "fear": {
      "support": 92,
      "predicted_count": 82,
      "precision": 0.7560975609756098,
      "recall": 0.6739130434782609,
      "f1": 0.7126436781609196,
      "false_positive": 20,
      "false_negative": 30
    },
    "gratitude": {
      "support": 87,
      "predicted_count": 33,
      "precision": 0.9090909090909091,
      "recall": 0.3448275862068966,
      "f1": 0.5,
      "false_positive": 3,
      "false_negative": 57
    },
    "helplessness": {
      "support": 103,
      "predicted_count": 108,
      "precision": 0.6574074074074074,
      "recall": 0.6893203883495146,
      "f1": 0.6729857819905213,
      "false_positive": 37,
      "false_negative": 32
    },
    "joy": {
      "support": 13,
      "predicted_count": 8,
      "precision": 0.75,
      "recall": 0.46153846153846156,
      "f1": 0.5714285714285714,
      "false_positive": 2,
      "false_negative": 7
    },
    "loneliness": {
      "support": 77,
      "predicted_count": 48,
      "precision": 0.9375,
      "recall": 0.5844155844155844,
      "f1": 0.72,
      "false_positive": 3,
      "false_negative": 32
    },
    "mixed": {
      "support": 120,
      "predicted_count": 46,
      "precision": 0.6956521739130435,
      "recall": 0.26666666666666666,
      "f1": 0.3855421686746988,
      "false_positive": 14,
      "false_negative": 88
    },
    "neutral": {
      "support": 39,
      "predicted_count": 28,
      "precision": 0.42857142857142855,
      "recall": 0.3076923076923077,
      "f1": 0.3582089552238806,
      "false_positive": 16,
      "false_negative": 27
    },
    "pride": {
      "support": 31,
      "predicted_count": 22,
      "precision": 0.7727272727272727,
      "recall": 0.5483870967741935,
      "f1": 0.6415094339622641,
      "false_positive": 5,
      "false_negative": 14
    },
    "relaxed": {
      "support": 290,
      "predicted_count": 421,
      "precision": 0.667458432304038,
      "recall": 0.9689655172413794,
      "f1": 0.790436005625879,
      "false_positive": 140,
      "false_negative": 9
    },
    "sadness": {
      "support": 93,
      "predicted_count": 139,
      "precision": 0.5467625899280576,
      "recall": 0.8172043010752689,
      "f1": 0.6551724137931034,
      "false_positive": 63,
      "false_negative": 17
    },
    "shame": {
      "support": 82,
      "predicted_count": 57,
      "precision": 0.8596491228070176,
      "recall": 0.5975609756097561,
      "f1": 0.7050359712230215,
      "false_positive": 8,
      "false_negative": 33
    },
    "surprise": {
      "support": 7,
      "predicted_count": 5,
      "precision": 1.0,
      "recall": 0.7142857142857143,
      "f1": 0.8333333333333334,
      "false_positive": 0,
      "false_negative": 2
    }
  },
  "confusion_matrix": {
    "": {
      "": 0,
      "anger": 0,
      "anxiety": 0,
      "care": 0,
      "disgust": 0,
      "fear": 0,
      "gratitude": 0,
      "helplessness": 0,
      "joy": 0,
      "loneliness": 0,
      "mixed": 0,
      "neutral": 0,
      "pride": 0,
      "relaxed": 0,
      "sadness": 0,
      "shame": 0,
      "surprise": 0
    },
    "anger": {
      "": 0,
      "anger": 68,
      "anxiety": 5,
      "care": 0,
      "disgust": 0,
      "fear": 1,
      "gratitude": 0,
      "helplessness": 2,
      "joy": 0,
      "loneliness": 0,
      "mixed": 0,
      "neutral": 1,
      "pride": 0,
      "relaxed": 1,
      "sadness": 1,
      "shame": 0,
      "surprise": 0
    },
    "anxiety": {
      "": 0,
      "anger": 5,
      "anxiety": 439,
      "care": 0,
      "disgust": 0,
      "fear": 15,
      "gratitude": 1,
      "helplessness": 12,
      "joy": 0,
      "loneliness": 0,
      "mixed": 9,
      "neutral": 3,
      "pride": 1,
      "relaxed": 19,
      "sadness": 7,
      "shame": 3,
      "surprise": 0
    },
    "care": {
      "": 0,
      "anger": 0,
      "anxiety": 5,
      "care": 1,
      "disgust": 0,
      "fear": 0,
      "gratitude": 0,
      "helplessness": 0,
      "joy": 0,
      "loneliness": 0,
      "mixed": 0,
      "neutral": 1,
      "pride": 0,
      "relaxed": 0,
      "sadness": 0,
      "shame": 0,
      "surprise": 0
    },
    "disgust": {
      "": 0,
      "anger": 4,
      "anxiety": 1,
      "care": 0,
      "disgust": 9,
      "fear": 0,
      "gratitude": 0,
      "helplessness": 0,
      "joy": 0,
      "loneliness": 0,
      "mixed": 0,
      "neutral": 0,
      "pride": 0,
      "relaxed": 0,
      "sadness": 1,
      "shame": 0,
      "surprise": 0
    },
    "fear": {
      "": 0,
      "anger": 0,
      "anxiety": 26,
      "care": 0,
      "disgust": 0,
      "fear": 62,
      "gratitude": 0,
      "helplessness": 1,
      "joy": 0,
      "loneliness": 0,
      "mixed": 0,
      "neutral": 0,
      "pride": 0,
      "relaxed": 0,
      "sadness": 1,
      "shame": 2,
      "surprise": 0
    },
    "gratitude": {
      "": 1,
      "anger": 0,
      "anxiety": 1,
      "care": 0,
      "disgust": 0,
      "fear": 0,
      "gratitude": 30,
      "helplessness": 0,
      "joy": 0,
      "loneliness": 0,
      "mixed": 0,
      "neutral": 0,
      "pride": 0,
      "relaxed": 54,
      "sadness": 1,
      "shame": 0,
      "surprise": 0
    },
    "helplessness": {
      "": 0,
      "anger": 3,
      "anxiety": 12,
      "care": 0,
      "disgust": 0,
      "fear": 1,
      "gratitude": 0,
      "helplessness": 71,
      "joy": 0,
      "loneliness": 2,
      "mixed": 0,
      "neutral": 1,
      "pride": 0,
      "relaxed": 0,
      "sadness": 13,
      "shame": 0,
      "surprise": 0
    },
    "joy": {
      "": 0,
      "anger": 0,
      "anxiety": 0,
      "care": 0,
      "disgust": 0,
      "fear": 0,
      "gratitude": 0,
      "helplessness": 0,
      "joy": 6,
      "loneliness": 0,
      "mixed": 1,
      "neutral": 0,
      "pride": 0,
      "relaxed": 6,
      "sadness": 0,
      "shame": 0,
      "surprise": 0
    },
    "loneliness": {
      "": 0,
      "anger": 0,
      "anxiety": 9,
      "care": 0,
      "disgust": 0,
      "fear": 0,
      "gratitude": 0,
      "helplessness": 4,
      "joy": 0,
      "loneliness": 45,
      "mixed": 0,
      "neutral": 1,
      "pride": 0,
      "relaxed": 1,
      "sadness": 17,
      "shame": 0,
      "surprise": 0
    },
    "mixed": {
      "": 0,
      "anger": 6,
      "anxiety": 21,
      "care": 1,
      "disgust": 1,
      "fear": 0,
      "gratitude": 1,
      "helplessness": 6,
      "joy": 1,
      "loneliness": 0,
      "mixed": 32,
      "neutral": 5,
      "pride": 3,
      "relaxed": 31,
      "sadness": 10,
      "shame": 2,
      "surprise": 0
    },
    "neutral": {
      "": 0,
      "anger": 4,
      "anxiety": 7,
      "care": 0,
      "disgust": 0,
      "fear": 0,
      "gratitude": 1,
      "helplessness": 1,
      "joy": 0,
      "loneliness": 0,
      "mixed": 0,
      "neutral": 12,
      "pride": 0,
      "relaxed": 13,
      "sadness": 1,
      "shame": 0,
      "surprise": 0
    },
    "pride": {
      "": 0,
      "anger": 0,
      "anxiety": 0,
      "care": 0,
      "disgust": 0,
      "fear": 0,
      "gratitude": 0,
      "helplessness": 0,
      "joy": 0,
      "loneliness": 0,
      "mixed": 1,
      "neutral": 1,
      "pride": 17,
      "relaxed": 12,
      "sadness": 0,
      "shame": 0,
      "surprise": 0
    },
    "relaxed": {
      "": 0,
      "anger": 0,
      "anxiety": 4,
      "care": 0,
      "disgust": 0,
      "fear": 0,
      "gratitude": 0,
      "helplessness": 0,
      "joy": 1,
      "loneliness": 0,
      "mixed": 0,
      "neutral": 3,
      "pride": 1,
      "relaxed": 281,
      "sadness": 0,
      "shame": 0,
      "surprise": 0
    },
    "sadness": {
      "": 0,
      "anger": 1,
      "anxiety": 2,
      "care": 0,
      "disgust": 0,
      "fear": 2,
      "gratitude": 0,
      "helplessness": 9,
      "joy": 0,
      "loneliness": 0,
      "mixed": 0,
      "neutral": 0,
      "pride": 0,
      "relaxed": 2,
      "sadness": 76,
      "shame": 1,
      "surprise": 0
    },
    "shame": {
      "": 0,
      "anger": 0,
      "anxiety": 16,
      "care": 0,
      "disgust": 0,
      "fear": 1,
      "gratitude": 0,
      "helplessness": 2,
      "joy": 0,
      "loneliness": 1,
      "mixed": 2,
      "neutral": 0,
      "pride": 0,
      "relaxed": 1,
      "sadness": 10,
      "shame": 49,
      "surprise": 0
    },
    "surprise": {
      "": 0,
      "anger": 0,
      "anxiety": 0,
      "care": 0,
      "disgust": 0,
      "fear": 0,
      "gratitude": 0,
      "helplessness": 0,
      "joy": 0,
      "loneliness": 0,
      "mixed": 1,
      "neutral": 0,
      "pride": 0,
      "relaxed": 0,
      "sadness": 1,
      "shame": 0,
      "surprise": 5
    }
  },
  "top_confusions": [
    {
      "true_label": "gratitude",
      "predicted_label": "relaxed",
      "count": 54,
      "true_label_ratio": 0.6206896551724138
    },
    {
      "true_label": "mixed",
      "predicted_label": "relaxed",
      "count": 31,
      "true_label_ratio": 0.25833333333333336
    },
    {
      "true_label": "fear",
      "predicted_label": "anxiety",
      "count": 26,
      "true_label_ratio": 0.2826086956521739
    },
    {
      "true_label": "mixed",
      "predicted_label": "anxiety",
      "count": 21,
      "true_label_ratio": 0.175
    },
    {
      "true_label": "anxiety",
      "predicted_label": "relaxed",
      "count": 19,
      "true_label_ratio": 0.03696498054474708
    },
    {
      "true_label": "loneliness",
      "predicted_label": "sadness",
      "count": 17,
      "true_label_ratio": 0.22077922077922077
    },
    {
      "true_label": "shame",
      "predicted_label": "anxiety",
      "count": 16,
      "true_label_ratio": 0.1951219512195122
    },
    {
      "true_label": "anxiety",
      "predicted_label": "fear",
      "count": 15,
      "true_label_ratio": 0.029182879377431907
    },
    {
      "true_label": "helplessness",
      "predicted_label": "sadness",
      "count": 13,
      "true_label_ratio": 0.1262135922330097
    },
    {
      "true_label": "neutral",
      "predicted_label": "relaxed",
      "count": 13,
      "true_label_ratio": 0.3333333333333333
    },
    {
      "true_label": "anxiety",
      "predicted_label": "helplessness",
      "count": 12,
      "true_label_ratio": 0.023346303501945526
    },
    {
      "true_label": "helplessness",
      "predicted_label": "anxiety",
      "count": 12,
      "true_label_ratio": 0.11650485436893204
    },
    {
      "true_label": "pride",
      "predicted_label": "relaxed",
      "count": 12,
      "true_label_ratio": 0.3870967741935484
    },
    {
      "true_label": "mixed",
      "predicted_label": "sadness",
      "count": 10,
      "true_label_ratio": 0.08333333333333333
    },
    {
      "true_label": "shame",
      "predicted_label": "sadness",
      "count": 10,
      "true_label_ratio": 0.12195121951219512
    }
  ],
  "head_mid_tail": {
    "HEAD": {
      "labels": [
        "anxiety",
        "helplessness",
        "mixed",
        "relaxed"
      ],
      "support": 1027,
      "group_accuracy": 0.801363193768257,
      "group_macro_f1": 0.6689264881311552
    },
    "MID": {
      "labels": [
        "anger",
        "fear",
        "gratitude",
        "loneliness",
        "neutral",
        "pride",
        "sadness",
        "shame"
      ],
      "support": 580,
      "group_accuracy": 0.6189655172413793,
      "group_macro_f1": 0.6365713065453986
    },
    "TAIL": {
      "labels": [
        "",
        "care",
        "disgust",
        "joy",
        "surprise"
      ],
      "support": 42,
      "group_accuracy": 0.5,
      "group_macro_f1": 0.4693968253968254
    }
  },
  "profile": {
    "personality_traits": {
      "exact_set_accuracy": 0.6173438447543966,
      "gt_mean_count": 0.5081867798665858,
      "pred_mean_count": 0.08489993935718618,
      "gt_empty_rate": 0.6070345664038811,
      "pred_empty_rate": 0.9187386294724075,
      "labels": {
        "agreeable": {
          "support": 31,
          "predicted_positive": 8,
          "true_positive": 6,
          "false_positive": 2,
          "false_negative": 25,
          "precision": 0.75,
          "recall": 0.1935483870967742,
          "f1": 0.3076923076923077
        },
        "assertive": {
          "support": 59,
          "predicted_positive": 21,
          "true_positive": 10,
          "false_positive": 11,
          "false_negative": 49,
          "precision": 0.47619047619047616,
          "recall": 0.1694915254237288,
          "f1": 0.25
        },
        "casual": {
          "support": 4,
          "predicted_positive": 0,
          "true_positive": 0,
          "false_positive": 0,
          "false_negative": 4,
          "precision": 0,
          "recall": 0.0,
          "f1": 0.0
        },
        "conservative": {
          "support": 206,
          "predicted_positive": 39,
          "true_positive": 28,
          "false_positive": 11,
          "false_negative": 178,
          "precision": 0.717948717948718,
          "recall": 0.13592233009708737,
          "f1": 0.22857142857142856
        },
        "emotionally_stable": {
          "support": 8,
          "predicted_positive": 1,
          "true_positive": 0,
          "false_positive": 1,
          "false_negative": 8,
          "precision": 0.0,
          "recall": 0.0,
          "f1": 0.0
        },
        "extroverted": {
          "support": 2,
          "predicted_positive": 0,
          "true_positive": 0,
          "false_positive": 0,
          "false_negative": 2,
          "precision": 0,
          "recall": 0.0,
          "f1": 0.0
        },
        "high_conscientiousness": {
          "support": 148,
          "predicted_positive": 55,
          "true_positive": 29,
          "false_positive": 26,
          "false_negative": 119,
          "precision": 0.5272727272727272,
          "recall": 0.19594594594594594,
          "f1": 0.2857142857142857
        },
        "introverted": {
          "support": 39,
          "predicted_positive": 8,
          "true_positive": 3,
          "false_positive": 5,
          "false_negative": 36,
          "precision": 0.375,
          "recall": 0.07692307692307693,
          "f1": 0.1276595744680851
        },
        "open": {
          "support": 4,
          "predicted_positive": 0,
          "true_positive": 0,
          "false_positive": 0,
          "false_negative": 4,
          "precision": 0,
          "recall": 0.0,
          "f1": 0.0
        },
        "sensitive": {
          "support": 337,
          "predicted_positive": 8,
          "true_positive": 3,
          "false_positive": 5,
          "false_negative": 334,
          "precision": 0.375,
          "recall": 0.008902077151335312,
          "f1": 0.017391304347826087
        }
      },
      "top_fp": [
        "high_conscientiousness",
        "assertive",
        "conservative",
        "introverted",
        "sensitive",
        "agreeable",
        "emotionally_stable",
        "casual",
        "extroverted",
        "open"
      ],
      "top_fn": [
        "sensitive",
        "conservative",
        "high_conscientiousness",
        "assertive",
        "introverted",
        "agreeable",
        "emotionally_stable",
        "casual",
        "open",
        "extroverted"
      ]
    },
    "interests": {
      "exact_set_accuracy": 0.774408732565191,
      "gt_mean_count": 0.28805336567616735,
      "pred_mean_count": 0.16676773802304426,
      "gt_empty_rate": 0.7343844754396605,
      "pred_empty_rate": 0.8332322619769558,
      "labels": {
        "art_design": {
          "support": 28,
          "predicted_positive": 17,
          "true_positive": 11,
          "false_positive": 6,
          "false_negative": 17,
          "precision": 0.6470588235294118,
          "recall": 0.39285714285714285,
          "f1": 0.4888888888888889
        },
        "career_development": {
          "support": 99,
          "predicted_positive": 7,
          "true_positive": 4,
          "false_positive": 3,
          "false_negative": 95,
          "precision": 0.5714285714285714,
          "recall": 0.04040404040404041,
          "f1": 0.07547169811320754
        },
        "film_animation": {
          "support": 13,
          "predicted_positive": 6,
          "true_positive": 1,
          "false_positive": 5,
          "false_negative": 12,
          "precision": 0.16666666666666666,
          "recall": 0.07692307692307693,
          "f1": 0.10526315789473684
        },
        "games": {
          "support": 49,
          "predicted_positive": 28,
          "true_positive": 21,
          "false_positive": 7,
          "false_negative": 28,
          "precision": 0.75,
          "recall": 0.42857142857142855,
          "f1": 0.5454545454545454
        },
        "music": {
          "support": 19,
          "predicted_positive": 14,
          "true_positive": 11,
          "false_positive": 3,
          "false_negative": 8,
          "precision": 0.7857142857142857,
          "recall": 0.5789473684210527,
          "f1": 0.6666666666666666
        },
        "pets": {
          "support": 60,
          "predicted_positive": 28,
          "true_positive": 23,
          "false_positive": 5,
          "false_negative": 37,
          "precision": 0.8214285714285714,
          "recall": 0.38333333333333336,
          "f1": 0.5227272727272727
        },
        "programming_technology": {
          "support": 1,
          "predicted_positive": 55,
          "true_positive": 0,
          "false_positive": 55,
          "false_negative": 1,
          "precision": 0.0,
          "recall": 0.0,
          "f1": 0.0
        },
        "reading_writing": {
          "support": 36,
          "predicted_positive": 9,
          "true_positive": 7,
          "false_positive": 2,
          "false_negative": 29,
          "precision": 0.7777777777777778,
          "recall": 0.19444444444444445,
          "f1": 0.3111111111111111
        },
        "social": {
          "support": 14,
          "predicted_positive": 3,
          "true_positive": 0,
          "false_positive": 3,
          "false_negative": 14,
          "precision": 0.0,
          "recall": 0.0,
          "f1": 0.0
        },
        "sports_fitness": {
          "support": 68,
          "predicted_positive": 68,
          "true_positive": 51,
          "false_positive": 17,
          "false_negative": 17,
          "precision": 0.75,
          "recall": 0.75,
          "f1": 0.75
        },
        "study_exam": {
          "support": 73,
          "predicted_positive": 38,
          "true_positive": 13,
          "false_positive": 25,
          "false_negative": 60,
          "precision": 0.34210526315789475,
          "recall": 0.1780821917808219,
          "f1": 0.23423423423423423
        },
        "travel_outdoor": {
          "support": 15,
          "predicted_positive": 2,
          "true_positive": 1,
          "false_positive": 1,
          "false_negative": 14,
          "precision": 0.5,
          "recall": 0.06666666666666667,
          "f1": 0.11764705882352941
        }
      },
      "top_fp": [
        "programming_technology",
        "study_exam",
        "sports_fitness",
        "games",
        "art_design",
        "film_animation",
        "pets",
        "career_development",
        "music",
        "social"
      ],
      "top_fn": [
        "career_development",
        "study_exam",
        "pets",
        "reading_writing",
        "games",
        "art_design",
        "sports_fitness",
        "social",
        "travel_outdoor",
        "film_animation"
      ]
    },
    "style": {
      "exact_set_accuracy": 0.45967252880533654,
      "gt_mean_count": 0.5524560339599758,
      "pred_mean_count": 0.16070345664038813,
      "gt_empty_rate": 0.5288053365676167,
      "pred_empty_rate": 0.8392965433596119,
      "labels": {
        "brief": {
          "support": 84,
          "predicted_positive": 9,
          "true_positive": 2,
          "false_positive": 7,
          "false_negative": 82,
          "precision": 0.2222222222222222,
          "recall": 0.023809523809523808,
          "f1": 0.043010752688172046
        },
        "colloquial": {
          "support": 33,
          "predicted_positive": 38,
          "true_positive": 3,
          "false_positive": 35,
          "false_negative": 30,
          "precision": 0.07894736842105263,
          "recall": 0.09090909090909091,
          "f1": 0.08450704225352113
        },
        "detailed": {
          "support": 94,
          "predicted_positive": 6,
          "true_positive": 2,
          "false_positive": 4,
          "false_negative": 92,
          "precision": 0.3333333333333333,
          "recall": 0.02127659574468085,
          "f1": 0.04
        },
        "direct": {
          "support": 196,
          "predicted_positive": 80,
          "true_positive": 13,
          "false_positive": 67,
          "false_negative": 183,
          "precision": 0.1625,
          "recall": 0.0663265306122449,
          "f1": 0.09420289855072464
        },
        "formal": {
          "support": 17,
          "predicted_positive": 0,
          "true_positive": 0,
          "false_positive": 0,
          "false_negative": 17,
          "precision": 0,
          "recall": 0.0,
          "f1": 0.0
        },
        "high_emotional_expression": {
          "support": 43,
          "predicted_positive": 25,
          "true_positive": 1,
          "false_positive": 24,
          "false_negative": 42,
          "precision": 0.04,
          "recall": 0.023255813953488372,
          "f1": 0.029411764705882353
        },
        "humorous": {
          "support": 7,
          "predicted_positive": 0,
          "true_positive": 0,
          "false_positive": 0,
          "false_negative": 7,
          "precision": 0,
          "recall": 0.0,
          "f1": 0.0
        },
        "indirect": {
          "support": 37,
          "predicted_positive": 10,
          "true_positive": 1,
          "false_positive": 9,
          "false_negative": 36,
          "precision": 0.1,
          "recall": 0.02702702702702703,
          "f1": 0.0425531914893617
        },
        "rational": {
          "support": 400,
          "predicted_positive": 97,
          "true_positive": 30,
          "false_positive": 67,
          "false_negative": 370,
          "precision": 0.30927835051546393,
          "recall": 0.075,
          "f1": 0.12072434607645875
        }
      },
      "top_fp": [
        "direct",
        "rational",
        "colloquial",
        "high_emotional_expression",
        "indirect",
        "brief",
        "detailed",
        "formal",
        "humorous"
      ],
      "top_fn": [
        "rational",
        "direct",
        "detailed",
        "brief",
        "high_emotional_expression",
        "indirect",
        "colloquial",
        "formal",
        "humorous"
      ]
    }
  },
  "length": {
    "reference_mean": 157.63735597331717,
    "reference_median": 99,
    "prediction_mean": 152.05093996361433,
    "prediction_median": 98,
    "mean_length_ratio": 1.1437329598328352,
    "ratio_groups": {
      "too_short": {
        "count": 167,
        "rate": 0.1012734990903578,
        "bleu_mean": 0,
        "rouge_mean": 0.008699551796447186
      },
      "roughly_matched": {
        "count": 1244,
        "rate": 0.7543966040024257,
        "bleu_mean": 0.00027205466323166044,
        "rouge_mean": 0.02571143115377153
      },
      "too_long": {
        "count": 238,
        "rate": 0.14432989690721648,
        "bleu_mean": 0,
        "rouge_mean": 0.014845607183661293
      }
    }
  },
  "bleu_lowest_10pct": {
    "count": 165,
    "mean": 0.03570103753205677
  },
  "rouge_lowest_10pct": {
    "count": 165,
    "mean": 0.1589946761228808
  },
  "bleu_mean": 0.15308102246542069,
  "rouge_mean": 0.3027020980927125,
  "cross_metric": {
    "emotion_correct": {
      "count": 1203,
      "bleu_mean": 0.0002813266841730553,
      "rouge_mean": 0.021750721252112745
    },
    "emotion_wrong": {
      "count": 446,
      "bleu_mean": 0,
      "rouge_mean": 0.024226417822238156
    },
    "profile_exact_correct": {
      "count": 494,
      "bleu_mean": 0,
      "rouge_mean": 0.022641445300176408
    },
    "profile_exact_wrong": {
      "count": 1155,
      "bleu_mean": 0.0002930181827360914,
      "rouge_mean": 0.022325736828331347
    }
  },
  "row_order_verified": true,
  "bottlenecks": [
    {
      "rank": 1,
      "metric": "emotion_macro_f1",
      "evidence": 0.5950153254042316,
      "likely_cause": "emotion confusion",
      "recommended_intervention": "DATA_BALANCING"
    },
    {
      "rank": 2,
      "metric": "style_exact_set_accuracy",
      "evidence": 0.45967252880533654,
      "likely_cause": "sparse multi-label exact-set strictness",
      "recommended_intervention": "SFT_DATA"
    },
    {
      "rank": 3,
      "metric": "response_length_ratio",
      "evidence": 1.1437329598328352,
      "likely_cause": "length mismatch",
      "recommended_intervention": "DECODING"
    },
    {
      "rank": 4,
      "metric": "interests_exact_set_accuracy",
      "evidence": 0.774408732565191,
      "likely_cause": "multi-label omission",
      "recommended_intervention": "SFT_DATA"
    },
    {
      "rank": 5,
      "metric": "bleu_4",
      "evidence": 0.0002052371140449882,
      "likely_cause": "lexical divergence",
      "recommended_intervention": "SFT_DATA"
    }
  ],
  "recommended_first_v2_experiment": {
    "experiment_name": "V2 style SFT data balancing",
    "target_metric": "style_exact_set_accuracy",
    "single_variable_changed": "rebalance style-labeled SFT examples only",
    "why_this_first": "style is weakest profile dimension",
    "expected_gain": "improve style exact-set accuracy",
    "risk": "distribution shift",
    "what_must_remain_fixed": "Prompt, model, LoRA, loss, decoding, evaluator, validation set"
  }
}