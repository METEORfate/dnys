# V2-E7 Emotion Boundary LoRA Experiment

Result: ABORTED_INFRA
FORMAL_RUNTIME_MODIFIED=NO

Source adapter: /root/development/dnys/数字人综合情感陪伴对话模型/training/outputs/v2-e1-emotion-rebalance-fixed18657-rerun2
Planned output: /root/development/dnys/数字人综合情感陪伴对话模型/training/outputs/v2-e7-emotion-boundary
P0/P1/P2/model/LoRA official files unchanged; hashes checked.

## Dataset
2800 unique original train samples = 1960 boundary + 840 V2-E1-distribution anchor. Seed=66. No generated or relabeled samples; no validation or holdout used in training selection.
Hard selection: fixed keyword cue groups on latest user text, prioritized co-occurring boundary cues within GT classes; not verified LLM training errors.

|emotion|count|
|---|---|
|anger|54|
|anxiety|636|
|care|10|
|disgust|15|
|fear|425|
|gratitude|327|
|helplessness|376|
|joy|18|
|loneliness|54|
|mixed|318|
|neutral|45|
|pride|23|
|relaxed|379|
|sadness|62|
|shame|50|
|surprise|8|

## Loss mask audit
Full assistant-only CE, packing=false, max_length=4096. 2800 kept; 226 context-trimmed; zero-supervised=0; target-over-budget=0; context-over-budget=0; prefix-mismatch=0.
Syntax and existing CPU mask/truncation unit tests PASS.

## Training and evaluation
Planned continued V2-E1 LoRA: r=16, alpha=32, existing 7 modules; lr=1e-5, batch=1, grad_acc=8, gradient checkpointing; max_steps=80, save_steps=40.
Training steps=0; training minutes=0. No checkpoint created. FAST96/FAST192 metrics unavailable.

## Blocking issue
Dual rank preflight: both ranks RuntimeError: No CUDA GPUs are available. nvidia-smi: Failed to initialize NVML: Unknown Error.
GPU_PLATFORM_UNSTABLE. No environment modification, automatic restart or promotion performed.
Prepared dataset/features can be reused after platform recovery.
