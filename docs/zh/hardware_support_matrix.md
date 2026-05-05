# 硬件支持矩阵

本页汇总 v1 功能在各硬件后端上的支持状态。GPU（CUDA）是核心文档的默认路径；其他后端只记录差异，详见 [多后端支持](multi-backend/index.md)。

| 功能 | GPU（CUDA） | NPU（Ascend） | 说明 |
|------|:----------:|:------------:|------|
| SFT 全参数训练 | 支持 | 支持 | 通过 `llamafactory-cli sft` 或 `llamafactory-cli train` |
| LoRA SFT | 支持 | 支持 | 通过 `peft_config.name: lora` |
| Freeze SFT | 支持 | 支持 | 通过 `peft_config.name: freeze` |
| QLoRA | 支持 | 不支持 | 当前基于 bitsandbytes 4-bit 量化 |
| DPO | 未实现 | 未实现 | v1 入口保留，训练器尚未接入 |
| RM | 未实现 | 未实现 | v1 入口保留，训练器尚未接入 |
| FSDP2 | 支持 | 支持 | 通过 `dist_config.name: fsdp2` |
| DeepSpeed | 支持 | 不支持 | 通过 `dist_config.name: deepspeed` |
| Context Parallel | 支持 | 支持 | 当前 `cp_mode` 为 `ulysses` |
| 自定义融合算子 | 标准库算子 | 支持 | NPU kernel 列表见 [NPU 后端](multi-backend/npu/index.md) |
| CLI 推理 | 支持 | 支持 | 当前 `sample_backend: hf` |

> **注意**：表格只表达 v1 代码路径中的支持状态；具体可运行性仍取决于 PyTorch、驱动、Transformers、torch-npu、bitsandbytes 等运行环境。
