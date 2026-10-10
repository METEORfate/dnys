# V2-P1 Final Promotion

模型：V2_E1_FINAL_2334 + V2_P1_STYLE_CLASSIFIER

正式路径：/root/participant

替换：start.sh、run_inference.py。新增：style_postprocessor.py、profile_classifier/、隔离python_env/。

备份：/root/development/runtime_backups/pre_v2_p1/

Smoke 3/3 schema/causal通过；benchmark 100/100 schema/causal通过。

延迟 avg=2093.357205 ms；p50=1624.621188；p95=5173.144773。fallback=0。

Base/LoRA/Prompt/decoding/core未变，资产真实文件，无开发目录运行依赖。未训练，未提交模型权重。

FINAL_RUNTIME_READY=True
