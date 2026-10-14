# Frozen official release 2026-10-14

Champion: V2_E1_FINAL_2334 + P0_CURRENT + V2-P1 Style + V2-P2 Personality margin 0.50.
Only personality/style postprocessing may change those fields; emotion, response, interests and memory use V2-E1. No PR/E7/ARCH/emotion-rule integration.

Deploy source from participant/ together with the separately provisioned Base, LoRA, classifier and Python environment. Entry: `bash /root/participant/start.sh TEST_FILE RESULT_DIR`.

See FORMAL_ASSET_MANIFEST.json for exact binary/source hashes and installed environment versions. A source tag does not back up model/LoRA/classifier binaries. These remain on the server; a verified off-instance asset backup is a separate task.

Experiment archive branch: archive/v2-final-experiments-20261014. Failed/unpromoted experiments remain isolated from main inference.

Offline CPU validation: `python analysis/final_release_20261014/run_cpu_validation.py --tokenizer-path /root/participant/model/Qwen2___5-7B-Instruct --train-data TRAIN_SFT_PATH`. No GPU or downloads. Initial tests required fixture/asset corrections documented in VALIDATION.json.

Historical Git already tracks public raw datasets and a project zip; this release adds/changes none of those. Local dataset edits are left untouched.

CLEANUP_PLAN.json is advice only; logs/reports/validation/platform evidence are protected. No files were deleted.
