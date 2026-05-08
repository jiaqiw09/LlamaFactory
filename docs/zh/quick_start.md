# 快速开始

最短可运行链路：装好依赖 → 准备数据 → 跑一条命令。

## 训练方法支持矩阵

| 方法 | 全参 | Freeze | LoRA | QLoRA |
|------|:----:|:------:|:----:|:-----:|
| 监督微调（SFT） | 支持 | 支持 | 支持 | 支持 |
| 奖励建模（RM） | 未实现 | 未实现 | 未实现 | 未实现 |
| 偏好对齐（DPO） | 未实现 | 未实现 | 未实现 | 未实现 |

## 软件依赖

| 必需 | 至少 | 推荐 |
|------|------|------|
| python | 3.11 | 3.12 |
| torch | 2.7.1 | 2.7.1 |
| torchvision | 0.22.1 | 0.22.1 |
| transformers | 5.0.0 | 5.0.0 |
| datasets | 3.2.0 | 4.0.0 |
| peft | 0.18.1 | 0.18.1 |

| 可选 | 至少 | 推荐 |
|------|------|------|
| CUDA（NVIDIA GPU） | 11.6 | 12.2 |
| deepspeed | 0.18.4 | 0.18.4 |
| flash-attn（NVIDIA GPU） | 2.5.6 | 2.7.2 |

其它硬件后端依赖见 [多后端支持](multi-backend/index.md)。

## 安装

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
```

## 数据

数据集格式与配置文件写法见 [数据准备](feature-guide/data_preparation.md)；本仓库 `data/v1_sft_demo.yaml` 是一份开箱即跑的最小配置，下面的命令可以直接用它。

## 第一次运行

启用 v1 入口：

```bash
export USE_V1=1
```

跑一次 Qwen3-0.6B 全参 SFT（FSDP2 后端）：

```bash
llamafactory-cli sft examples/v1/train_full/train_full_fsdp2.yaml
```

`sft` 与 `train` 等价：

```bash
llamafactory-cli train examples/v1/train_full/train_full_fsdp2.yaml
```

多 GPU 自动通过 `torchrun` 启动；要想单卡也强制走分布式路径，设 `FORCE_TORCHRUN=1`。

## 下一步

- 数据格式与多数据集混合 → [数据准备](feature-guide/data_preparation.md)
- 全参 / LoRA / Freeze 三种 SFT 模式 → [SFT](feature-guide/sft.md)
- FSDP2 / DeepSpeed / Context Parallel → [分布式训练](feature-guide/distributed_training.md)
- 推理与对话 → [推理与部署](feature-guide/inference.md)
- 框架原理与扩展点 → [开发者指南](developer-guide/architecture_overview.md)
