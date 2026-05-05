# 快速开始

## 训练方法

| 方法 | 全参数训练 | 部分参数训练 | LoRA | QLoRA |
|:---:|:---:|:---:|:---:|:---:|
| 指令监督微调 | 支持 | 支持 | 支持 | 支持 |
| 奖励模型训练 | 未实现 | 未实现 | 未实现 | 未实现 |
| DPO 训练 | 未实现 | 未实现 | 未实现 | 未实现 |

## 软件依赖

| 必需项 | 至少 | 推荐 |
|:---:|---|---|
| python | 3.11 | 3.12 |
| torch | 2.7.1 | 2.7.1 |
| torchvision | 0.22.1 | 0.22.1 |
| transformers | 5.0.0 | 5.0.0 |
| datasets | 3.2.0 | 4.0.0 |
| peft | 0.18.1 | 0.18.1 |

| 可选项 | 至少 | 推荐 |
|:---:|---|---|
| CUDA（NVIDIA GPU） | 11.6 | 12.2 |
| deepspeed | 0.18.4 | 0.18.4 |
| flash-attn（NVIDIA GPU） | 2.5.6 | 2.7.2 |

> 其他硬件后端依赖见 [多后端支持](multi-backend/index.md)。

## 安装

> [!IMPORTANT]
> 此步骤为必需。

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
```

## 数据准备

关于数据集文件的格式，请参考 [数据准备](feature-guide/data_preparation.md)。你可以使用 HuggingFace / ModelScope 上的数据集或加载本地数据集。

> [!NOTE]
> 使用自定义数据集或自定义数据集格式时，请参照 [数据准备](feature-guide/data_preparation.md) 进行配置，如有必要，请重新实现自定义数据集的数据处理逻辑，包括对应的 `converter`。

您也可以使用 **[Easy Dataset](https://github.com/ConardLi/easy-dataset)**、**[DataFlow](https://github.com/OpenDCAI/DataFlow)** 和 **[GraphGen](https://github.com/open-sciencelab/GraphGen)** 构建用于微调的合成数据。

## 快速开始

下面的命令展示了对 Qwen3-0.6B 模型使用 FSDP2 进行全参微调，两行命令等价。

```bash
export USE_V1=1
llamafactory-cli sft examples/v1/train_full/train_full_fsdp2.yaml
llamafactory-cli train examples/v1/train_full/train_full_fsdp2.yaml
```

高级用法请参考 [开发指南](developer-guide/architecture_overview.md)（包括多卡多机微调、分布式、LoRA、量化、以及各种加速特性等）。
