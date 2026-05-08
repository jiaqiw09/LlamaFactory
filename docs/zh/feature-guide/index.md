# 功能指南

面向用户任务的端到端使用指南，以 GPU 为基线后端。所有示例使用 YAML 配置，参数细节链接到 [参数参考](../parameter-reference/index.md)。

## 数据 → 训练

| 页面 | 说明 |
|------|------|
| [数据准备](data_preparation.md) | 标准 Messages 格式、四种入参形态、多数据集混合、自定义 converter |
| [SFT](sft.md) | 全参 / LoRA / Freeze 三种模式 |
| [DPO](dpo.md) | 当前实现状态与数据格式 |

## 训练后

| 页面 | 说明 |
|------|------|
| [模型保存与恢复](model_saving.md) | 最终模型保存、checkpoint、断点续训、保留策略 |
| [模型导出](model_export.md) | LoRA 合并并导出为可部署的 HF 目录 |
| [推理与部署](inference.md) | CLI 对话、批量推理、LoRA 直接加载 |

## 性能与硬件

| 页面 | 说明 |
|------|------|
| [分布式训练](distributed_training.md) | FSDP2、DeepSpeed、Context Parallel、DDP 自动回退 |
| [自定义算子](custom_kernels.md) | `kernel_config` 启用方式、与硬件后端的关系 |
