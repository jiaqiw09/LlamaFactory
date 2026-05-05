# 数据准备

## 总览

v1 采用统一的 **Messages 格式**作为标准数据格式，所有数据最终都会被转换为标准的对话消息列表。通过内置的 `DataConverterPlugin`，Alpaca、ShareGPT、Pair 等格式可自动转换；自定义格式可通过注册新的 Converter 实现。

与 v0 相比，v1 通过 `DataEngine` + Plugin 机制提供了统一的数据处理流程，具有更好的可扩展性和一致性。

## 基本用法

### 在训练配置文件中配置数据集

<details open>
<summary>方式 1：使用 HF Hub Repo ID</summary>

直接指定 HF Hub 上的数据集 Repo ID，`DataEngine` 会自动从 Hub 下载并加载。

```yaml
train_dataset: llamafactory/v1-sft-demo
```

</details>

<details>
<summary>方式 2：使用 HF Hub 上的 YAML 配置文件</summary>

`train_dataset` 字段指定 HF Hub 上的 `dataset_info.yaml` 路径，`DataEngine` 会自动下载该配置文件并根据其中的配置加载数据集。

```yaml
train_dataset: llamafactory/v1-sft-demo/dataset_info.yaml
```

</details>

<details>
<summary>方式 3：使用本地数据集文件路径</summary>

`train_dataset` 字段指定本地的数据集文件路径（`.json`、`.jsonl` 等）。直接指定数据集文件路径时，要求该数据文件为标准 Messages 格式。

```yaml
train_dataset: ~/data/v1_sft_demo.jsonl
```

</details>

<details>
<summary>方式 4：使用本地 YAML 配置文件路径</summary>

`train_dataset` 字段指定本地的 `dataset_info.yaml` 配置文件路径，`DataEngine` 会根据该配置加载数据集。

```yaml
train_dataset: ~/data/dataset_info.yaml
```

</details>

## 标准数据格式

v1 使用统一的 **Messages 格式**作为标准数据格式。每个样本都是一个包含 `messages` 字段的 JSON 对象。

针对 Alpaca、ShareGPT、Pair 等格式的数据，可以通过内置的 Converter 自动转换。对于其他自定义格式的数据，可通过注册自定义 Converter 来实现格式标准化，详见 [DataConverterPlugin](../developer-guide/plugins/data_plugins.md)。

### SFT（监督微调）样本格式

```json
{
  "messages": [
    {
      "role": "system",
      "content": [{"type": "text", "value": "You are a helpful assistant."}],
      "loss_weight": 0.0
    },
    {
      "role": "user",
      "content": [{"type": "text", "value": "Hello, who are you?"}],
      "loss_weight": 0.0
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "value": "I am an AI assistant."}],
      "loss_weight": 1.0
    }
  ]
}
```

字段说明：

- **messages**: 消息列表，包含一轮或多轮对话
  - **role**: 消息角色，可选值：`"system"`、`"user"`、`"assistant"`、`"tool"`
  - **content**: 内容列表，每个元素包含：
    - **type**: 内容类型，可选值：`"text"`、`"image_url"`、`"audio_url"`、`"video_url"`、`"tools"`、`"tool_call"`、`"reasoning"`
    - **value**: 具体内容（字符串）
  - **loss_weight**: 损失权重（浮点数），`0.0` 不计算损失，`1.0` 完全计算损失
- **_dataset_name** (可选): 数据集名称，由 DataEngine 自动添加
- **extra_info** (可选): 额外信息字段

### DPO（偏好对齐）样本格式

```json
{
  "chosen_messages": [
    {
      "role": "user",
      "content": [{"type": "text", "value": "用户提问"}],
      "loss_weight": 0.0
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "value": "更优的回答"}],
      "loss_weight": 1.0
    }
  ],
  "rejected_messages": [
    {
      "role": "user",
      "content": [{"type": "text", "value": "用户提问"}],
      "loss_weight": 0.0
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "value": "较差的回答"}],
      "loss_weight": 1.0
    }
  ]
}
```

### 多模态支持

在 `content` 列表中添加非文本类型的内容：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "value": "这张图片里有什么？"},
        {"type": "image_url", "value": "path/to/image.jpg"}
      ],
      "loss_weight": 0.0
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "value": "图片中有一只猫。"}],
      "loss_weight": 1.0
    }
  ]
}
```

## 数据集配置文件

### dataset_info.yaml 格式

`dataset_info.yaml` 支持同时配置多个数据集，数据集默认会混合并打乱顺序。

```yaml
identity:
  path: ~/data/identity.json
  source: local
  converter: alpaca

alpaca_en_demo:
  path: ~/data/alpaca_en_demo.json
  source: local
  converter: alpaca
  size: 500
  weight: 0.5
  split: train
  streaming: false

hf_dataset:
  path: llamafactory/v1-sft-demo
  source: hf_hub
  streaming: false

standard:
  path: ~/data/v1_sft_demo.jsonl
  source: local

custom_dataset:
  path: custom_data.json
  source: local
  converter: custom_converter
  weight: 1.0
```

配置字段说明详见 [DatasetInfo 参数参考](../parameter-reference/dataset_info.md)。

## 完整示例

### 基础使用

```yaml
model: Qwen/Qwen3-0.6B
template: qwen3_nothink
train_dataset: data/v1_sft_demo.yaml

micro_batch_size: 1
global_batch_size: 8
cutoff_len: 2048
learning_rate: 5e-5
num_train_epochs: 3
bf16: true
```

### 混合多数据集

**配置文件：`data/mixed_datasets.yaml`**

```yaml
dataset_1:
  path: alpaca_en_demo.json
  source: local
  converter: alpaca
  weight: 1.0

dataset_2:
  path: identity.json
  source: local
  converter: alpaca
  weight: 2.0

dataset_3:
  path: llamafactory/v1-sft-demo
  source: hf_hub
  weight: 1.5
```

```yaml
model: Qwen/Qwen3-0.6B
template: qwen3_nothink
train_dataset: data/mixed_datasets.yaml

micro_batch_size: 2
global_batch_size: 16
cutoff_len: 2048
learning_rate: 1e-4
num_train_epochs: 3
```

### 多模态数据

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "value": "Who are they?"},
          {"type": "image_url", "value": "mllm_demo_data/1.jpg"}
        ],
        "loss_weight": 0.0
      },
      {
        "role": "assistant",
        "content": [{"type": "text", "value": "They're Kane and Gretzka from Bayern Munich."}],
        "loss_weight": 1.0
      }
    ]
  }
]
```

> **注意**：
> 1. 所有数据最终都会转换为标准的 Messages 格式
> 2. 通过 `converter` 字段指定转换器，支持 `alpaca`、`sharegpt`、`pair`，不指定则假定数据为标准格式
> 3. 通过 `weight` 和 `size` 参数可以灵活控制数据分布
> 4. 更多技术细节请参考 [DataEngine](../developer-guide/core/data_engine.md) 和 [Data Plugins](../developer-guide/plugins/data_plugins.md)
