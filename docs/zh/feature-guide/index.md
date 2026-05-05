# 功能指南

面向用户任务的端到端使用指南，以 GPU 为基线。所有配置示例使用 YAML 格式。

| 页面 | 说明 |
|------|------|
| [数据准备](data_preparation.md) | 标准数据格式、数据集配置文件、混合多数据集 |
| [监督微调（SFT）](sft.md) | 全参数微调、LoRA、Freeze 三种模式的配置与使用 |
| [偏好对齐（DPO）](dpo.md) | 当前实现状态、数据格式和后续接入位置 |
| [推理与部署](inference.md) | CLI 对话模式、推理配置、LoRA 模型使用 |
| [模型导出](model_export.md) | LoRA 合并导出、Hub 推送 |
| [模型保存与恢复](model_saving.md) | 最终模型保存、checkpoint、断点恢复、DCP/DeepSpeed 保存格式 |
| [分布式训练](distributed_training.md) | FSDP2、DeepSpeed、Context Parallel、DDP |
| [自定义算子](custom_kernels.md) | Kernel 配置、GPU 算子、硬件后端扩展 |
