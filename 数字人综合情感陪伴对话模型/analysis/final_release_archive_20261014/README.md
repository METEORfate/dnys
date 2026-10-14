# V2 experimental archive

This branch retains unpublished commit history and non-promoted experimental tooling. Do not connect these experiments to the official inference path.
Source commit: 8fe47dc725ea2616a69a3168c43a880707bb18dd

Commands below are entry-point references, not instructions to restart finished experiments; many require local assets and original CLI arguments. See each setup/result and server_setup for the actual fixed command and prompt hashes. JSONL datasets and full predictions remain local; manifests preserve row/source IDs.

- `/root/participant/python_env/bin/python analysis/evaluate_emotion_second_pass.py`
- `/root/participant/python_env/bin/python analysis/fast_evaluate_lora_decode.py`
- `/root/participant/python_env/bin/python analysis/final_score_maximization/evaluate_candidates.py`
- `/root/participant/python_env/bin/python analysis/final_score_maximization/run_pipeline.py`
- `/root/participant/python_env/bin/python analysis/v2_arch1/run_arch1.py`
- `/root/participant/python_env/bin/python analysis/v2_arch2/run_arch2.py`
- `/root/participant/python_env/bin/python analysis/v2_arch2_offline/run_offline.py`
- `/root/participant/python_env/bin/python analysis/v2_e7_1_interpolation/evaluate.py`
- `/root/participant/python_env/bin/python analysis/v2_e7_2_interpolation/evaluate.py`
- `/root/participant/python_env/bin/python analysis/v2_e7_boundary_lora/evaluate.py`
- `/root/participant/python_env/bin/python analysis/v2_pr1/run_screen.py`
- `/root/participant/python_env/bin/python analysis/v2_pr1b/run_paired.py`
- `/root/participant/python_env/bin/python analysis/v2_pr1b/run_paired_seqs8.py`
- `/root/participant/python_env/bin/python analysis/v2_pr2/run_pr2.py`
- `/root/participant/python_env/bin/python analysis/v2_pr2a_valdev/run_full.py`
- `/root/participant/python_env/bin/python analysis/v2_pr3_minimal_prompt/run_pr3.py`
- `/root/participant/python_env/bin/python analysis/v2_rq1/run_retrieval.py`
- `/root/participant/python_env/bin/python analysis/v2_rq2/run_length_calibration.py`
- `/root/participant/python_env/bin/python analysis/v2_rq3/run_pipeline.py`
- `/root/participant/python_env/bin/python analysis/v2_rq4/run_interpolation.py`
- `/root/participant/python_env/bin/python analysis/v2_rq5/run_module_transplant.py`
- `/root/participant/python_env/bin/python analysis/v2_rq6/run_pipeline.py`

Log summaries retain SHA256/event counts/final numeric loss records; original logs are preserved on the server.
