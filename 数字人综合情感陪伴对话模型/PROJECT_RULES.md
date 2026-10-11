# Project Rules

## Execution roles

- `local`: 代码开发、静态检查、数据格式检查、评估脚本设计和文档维护。
- `gpu`: 依赖安装、模型加载、LoRA 训练、验证生成、性能测试和部署回归。

GPU 实验结果必须回写到 `sanlogs/`，不得提交模型权重、LoRA adapter 或训练数据。

## Change logging

每次代码或配置修改完成后，在项目根目录创建：

```text
sanlogs/log_YYYYMMDD_HHMMSS_local.md
```

日志必须包含：时间、角色、本次操作、修改原因、涉及文件、测试方式、测试结果和后续建议。
