# V1 Error Audit

- prediction_row_mapping_verified: YES
- verification_method: CODE_PROOF
- prediction_path: `training/eval_results/v1_valdev_reconstructed/prediction.jsonl`
- val_dev_path: `training/data/val_dev_sft.jsonl`
- input_lines: 1649
- prediction_lines: 1649
- input_order_preserved_by_code: YES
- shuffle_present: NO
- filter_or_drop_present: NO（仅跳过空物理行；数据实际无空行）
- async_reordering_present: NO
- mapping_checks: N/A（prediction 不保存输入/reference 字段）
- mapping_matches: N/A
- row_manifest: `analysis/v1_valdev_row_manifest.jsonl`

代码审计确认 `read_rows` 按原始 JSONL 行序读取，主循环直接顺序遍历；无 DataLoader、shuffle、异步重排或异常跳过分支，每条输入写出一条 prediction。预测中的 id 为空，因此映射依赖代码证明和行数一致性。
