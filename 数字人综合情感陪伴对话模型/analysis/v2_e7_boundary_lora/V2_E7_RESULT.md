# V2-E7 Emotion Boundary LoRA Experiment

Result: INCOMPLETE_INFRA (training completed, evaluation pending)
FORMAL_RUNTIME_MODIFIED=NO

Source adapter: /root/development/dnys/数字人综合情感陪伴对话模型/training/outputs/v2-e1-emotion-rebalance-fixed18657-rerun2
Output adapter: /root/development/dnys/数字人综合情感陪伴对话模型/training/outputs/v2-e7-emotion-boundary

## Dataset
2800 unique original train samples = 1960 boundary + 840 V2-E1-distribution anchor. Seed=66. No generated or relabeled data; no validation/holdout used in training selection.
Hard examples selected using fixed latest-user keyword boundary cues within GT classes, not confirmed LLM errors.

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

## Training
80 steps completed; elapsed including loading 16.12 minutes; trainer optimization runtime 889.9513 seconds; train loss 0.908703.
Continued V2-E1 weights exact verified on both ranks before training. r=16, alpha=32, 7 original target modules. lr=1e-5 with existing linear scheduler; batch=1, grad_acc=8, world_size=2, gradient checkpointing, max_length=4096, packing=false, full assistant-only loss.
Mask audit: 2800 retained, 226 context-trimmed, zero supervised samples=0; existing 7 CPU mask/truncation tests PASS.

## Checkpoints
checkpoint-40: /root/development/dnys/数字人综合情感陪伴对话模型/training/outputs/v2-e7-emotion-boundary/checkpoint-40
Compatible keys/shapes/rank/alpha/targets; all finite; 392/392 tensors changed vs V2-E1; sha256=b7d79d499edfcc7cd11225b5a533a347b926379cc470ccd5c03dbac0cb87b747
checkpoint-80: /root/development/dnys/数字人综合情感陪伴对话模型/training/outputs/v2-e7-emotion-boundary/checkpoint-80
Compatible keys/shapes/rank/alpha/targets; all finite; 392/392 tensors changed vs V2-E1; sha256=471dd1623e1a12d3bd2782456934cb25216c63dad01d418fed91e0648de6ac6a

## Evaluation
FAST96 completed=0. FAST192 not run. No quality verdict or promotion.
All metrics pending; no fabricated baseline/checkpoint metrics.

## Blocking issue
GPU_PLATFORM_UNSTABLE_AFTER_TRAINING. nvidia-smi: Failed to initialize NVML: Unknown Error. Both training/runtime environments: cuda_available=False, device_count=0.
No CUDA/software/environment modifications. Official runtime hashes unchanged.
Recovery requires only rerunning evaluate.py after GPU recovery, using the saved checkpoints; do not rerun training.
