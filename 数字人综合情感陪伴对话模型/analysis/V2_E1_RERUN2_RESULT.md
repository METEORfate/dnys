# V2-E1 RERUN2 Result

- samples: 1649
- emotion_accuracy: 0.7647058824 (V1 0.7295330503, delta +0.0351728319)
- emotion_f1: 0.6980062645 (V1 0.6322037832, delta +0.0658024813)
- bleu_4: 0.1467965098 (delta -0.0062845127)
- rouge_l: 0.2969218230 (delta -0.0057802751)
- profile_personality: 0.6373559733 (delta +0.0200121286)
- profile_interests: 0.8259551243 (delta +0.0515461488)
- profile_style: 0.4627046695 (delta +0.0030321410)
- profile_mean: 0.6420052557 (delta +0.0248635537)
- json_parse: 1.0
- response_nonempty: 1.0
- memory_refs: 1.0
- latency_ms: 8436.47

Known confusion counts (V1 -> rerun2): gratitude->relaxed 54 -> 12; fear->anxiety 26 -> 11; mixed->relaxed 31 -> 11; mixed->anxiety 21 -> 18; anxiety->relaxed 19 -> 8.

Result: TRADEOFF. Emotion and profile metrics improved, while BLEU/ROUGE declined modestly.
