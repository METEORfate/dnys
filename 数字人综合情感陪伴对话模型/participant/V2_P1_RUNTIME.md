# V2-P1 style runtime integration

The generation model remains V2-E1 FINAL_2334. The postprocessor changes only user_profile.style after successful JSON parsing. Every other output field is asserted unchanged. Exceptions preserve the original style; counters and classifier latency appear in performance_report.json. style_causal_audit.jsonl records before/after results locally and must not be committed.

Required local assets (not in Git): model/Qwen2___5-7B-Instruct, model/lora_adapter, profile_classifier/vectorizer.joblib, profile_classifier/classifiers.joblib, profile_classifier/config.json.

Create an isolated client environment under participant/python_env using the serving environment Python with --system-site-packages, then install requirements_profile_classifier.txt into that environment. start.sh defaults to python_env/bin/python; PYTHON_BIN may select another interpreter with the same classifier dependencies. Existing conda_env/vLLM activation and generation parameters are unchanged. Do not upgrade the serving stack.

Training tooling: training/train_profile_classifier.py uses train_sft.jsonl and val_dev_sft.jsonl. It extracts USER_HISTORY from serialized conversation, reads enums from core/output_schema.py, and compares exact-set LinearSVC with independent multilabel LogisticRegression. Saved weights and environments are local assets.

Verified final promotion: smoke 3/3 and benchmark 100/100 schema/causal preservation; avg 2093.357205 ms, p50 1624.621188 ms, p95 5173.144773 ms, fallback 0. Holdout hybrid profile_mean 0.6827423168, delta +0.0241134752. See analysis/V2_P1_FINAL_PROMOTION.md.
