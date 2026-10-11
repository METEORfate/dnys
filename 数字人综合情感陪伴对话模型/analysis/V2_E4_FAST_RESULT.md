{
  "experiment": "V2-E4 FAST REGRESSION RECOVERY",
  "dataset": {
    "dataset_total": 6000,
    "anchor_count": 4800,
    "recovery_count": 1200,
    "recovered_anxiety": 600,
    "recovered_relaxed": 300,
    "recovered_sadness": 150,
    "recovered_mixed": 150,
    "mixed_anchor": 313,
    "recovery_samples_previously_missing_from_v2e1": 1200,
    "unique_original": 5461,
    "max_repeat": 2,
    "train_only_verified": true
  },
  "training_steps_completed": 80,
  "training_minutes": 15,
  "checkpoint_status": {
    "checkpoint-20": "MISSING",
    "checkpoint-40": "MISSING",
    "checkpoint-60": "MISSING",
    "checkpoint-80": "AVAILABLE"
  },
  "checkpoint-80": {
    "emotion": 0.65625,
    "anxiety_recall": 0.7333333333333333,
    "mixed_recall": 0.42857142857142855,
    "relaxed_recall": 0.7647058823529411,
    "sadness_recall": 0.8,
    "profile": 0.6736111111111112,
    "BLEU": 0.1319690546296319,
    "ROUGE": 0.2660195405744552,
    "BERTScore": null,
    "result": "V2_E1"
  },
  "FAST_WINNER": "V2-E1",
  "FAST_RESULT": "V2_E1",
  "blocking_issue": "Intermediate checkpoint saves failed with KeyError TrainerControl; checkpoint-80 completed but failed final save exit status after 80 steps. Emotion 0.65625 is below baseline 0.697917, so rejected before BERTScore."
}
