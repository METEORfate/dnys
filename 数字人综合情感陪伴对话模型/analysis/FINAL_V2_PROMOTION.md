# Final V2 Promotion

V1 backup: /root/development/assets/lora-v1-runtime-final-backup (real copy, verified).
V2 adapter: /root/participant/model/lora_adapter.
Source SHA256 verified for adapter_config.json and adapter_model.safetensors.
Base remains a real directory; runtime code and Prompt unchanged.
Final smoke: BLOCKED. CUDA unavailable, device count 0; NVML Unknown Error.
FINAL_SUBMISSION_READY: NO (final smoke pending).
Rollback ready: YES.

## Retry
GPU precheck: CUDA True, 2 RTX4090. GPU/NVML became unavailable during vLLM worker initialization. Worker ranks 0/1 out of bounds; engine startup failed. Final smoke did not generate predictions. FINAL_SUBMISSION_READY: NO. Log: /tmp/competition-vllm.MDmoZ3.log. Adapter unchanged.
