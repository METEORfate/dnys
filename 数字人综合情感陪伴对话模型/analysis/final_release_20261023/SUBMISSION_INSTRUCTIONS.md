# 最终提交说明

选定版本：E91_a075 + P0 + P1 Style + P2 Personality Gate0.50; repetition_penalty=1.05。正式运行入口：

```bash
bash /root/participant/start.sh TEST_FILE RESULT_DIR
```

最终源码标签：v2-final-20261023。模型、tokenizer、Prompt、Schema、P1/P2及配置SHA256以FINAL_RELEASE_MANIFEST.json为准。输出为submission.jsonl和performance_report.json；前100条用于计时。不得将VAL_DEV输出当作赛事未知测试集提交结果。

Git仅提交正式源码、配置和最终报告。release_runtime_assets.tar.gz是运行源码、选中LoRA及P1/P2资产包，**不包含15GB基座模型与Python/Conda环境**；服务器/root/participant已具备这些依赖，基座完整SHA见manifest。离开该实例部署时必须另行配置相同基座和依赖，不能把此增量包当作独立完整镜像。

人工最后一步：登录实际参赛账号，按赛事平台要求提交当前已验收实例的镜像或规定格式的参赛文件；核对官方入口、标签与manifest，保存平台受理回执。当前任务没有可核验的已登录提交会话、提交对象/格式与受理回执，因此不能声称平台已经接受。源码推送成功也不等于比赛平台提交成功。

完整模型与现有环境在当前实例内；打包或导出镜像时须同时包含/root/participant及其Python环境实际指向的Conda依赖。未知官方规则与专家分保持UNKNOWN。
