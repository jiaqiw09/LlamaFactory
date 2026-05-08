# 数据准备

v1 把所有数据统一成 **Messages 格式**：每个样本是一份消息列表，role 字段标识角色，`loss_weight` 决定哪些 token 参与训练。

实际加载流程由 `DataEngine` 根据 `train_dataset` 字段的取值形态分派；参数细节见 [DataArguments](../parameter-reference/data_arguments.md) 与 [DatasetInfo](../parameter-reference/dataset_info.md)。

## `train_dataset` 的四种取值

| 形态 | 含义 | 例子 |
|------|------|------|
| 本地 YAML 文件 | 多数据集配置，最常用 | `data/v1_sft_demo.yaml` |
| HF Hub 上的 YAML | YAML 在 dataset repo 里 | `llamafactory/v1-sft-demo/dataset_info.yaml` |
| 本地数据文件或目录 | 数据本身已经是标准 Messages 格式 | `~/data/v1_sft_demo.jsonl` |
| HF Hub 数据集 ID | 默认从 Hub 拉数据 | `llamafactory/v1-sft-demo` |

YAML 里的字段定义见 [DatasetInfo](../parameter-reference/dataset_info.md)。直接给数据文件 / Hub ID 时，框架会自动包装成 `{"default": {"path": ..., "source": ...}}`，但要求文件本身已经是标准 Messages 格式。

## 标准 Messages 格式

### SFT 样本

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

字段：

- `role`：`system` / `user` / `assistant` / `tool`
- `content`：`{"type": ..., "value": ...}` 列表，`type` 取 `text` / `image_url` / `audio_url` / `video_url` / `tools` / `tool_call` / `reasoning`
- `loss_weight`：`0.0` 不计算损失，`1.0` 完全计算
- `_dataset_name`（可选）：由 `DataEngine` 自动注入
- `extra_info`（可选）：原样透传到 `ModelInput`，便于训练循环按数据集打标

### DPO 偏好对样本

```json
{
  "chosen_messages": [
    {"role": "user", "content": [{"type": "text", "value": "提问"}], "loss_weight": 0.0},
    {"role": "assistant", "content": [{"type": "text", "value": "更优回答"}], "loss_weight": 1.0}
  ],
  "rejected_messages": [
    {"role": "user", "content": [{"type": "text", "value": "提问"}], "loss_weight": 0.0},
    {"role": "assistant", "content": [{"type": "text", "value": "较差回答"}], "loss_weight": 1.0}
  ]
}
```

DPO 训练入口尚未接入，见 [DPO](dpo.md)；但数据层已经支持这套结构。

### 多模态

`content` 列表里直接掺入非文本类型即可：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "value": "这张图里有什么？"},
        {"type": "image_url", "value": "path/to/image.jpg"}
      ],
      "loss_weight": 0.0
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "value": "图里有一只猫。"}],
      "loss_weight": 1.0
    }
  ]
}
```

## 已有格式的转换

不想手写 Messages 时，可以让 `converter` 字段把常见格式自动转成标准格式：

| converter | 输入示例 | 适用 |
|-----------|---------|------|
| `alpaca` | `{instruction, input, output, system?}` | 单轮指令 |
| `sharegpt` | `{conversations: [{from, value}], tools?}` | 多轮对话 + tool call |
| `pair` | `{chosen: [...], rejected: [...]}` | DPO 偏好对 |

每种转换器的字段细节见 [data_plugins](../developer-guide/plugins/data_plugins.md)。

## YAML 配置文件

一个 YAML 可以同时声明多个数据集；最终 `DataEngine` 会把它们按顺序拼成全局索引。

```yaml
identity:
  path: ~/data/identity.json
  source: local
  converter: alpaca

alpaca_en_demo:
  path: ~/data/alpaca_en_demo.json
  source: local
  converter: alpaca
  size: 500          # 限制为 500 条
  weight: 0.5        # 再缩到一半
  split: train
  streaming: false

hf_dataset:
  path: llamafactory/v1-sft-demo
  source: hf_hub

standard:
  path: ~/data/v1_sft_demo.jsonl
  source: local       # 不指定 converter，假定数据已是 Messages 格式
```

`size` 与 `weight` 同时设置时先 `size` 再 `weight`，行为细节见 [DatasetInfo](../parameter-reference/dataset_info.md)。

> **注**：streaming 必须全开或全关。一个 YAML 里同时混合 `streaming: true` 与 `streaming: false` 会在加载阶段直接报错。

## 整合到训练配置

最简单的写法——把数据 YAML 路径塞给 `train_dataset`：

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

混合数据集，给 YAML 里每个条目设 `weight`：

```yaml
# data/mixed.yaml
math:
  path: data/math.json
  source: local
  converter: alpaca
  weight: 1.5
chat:
  path: llamafactory/v1-sft-demo
  source: hf_hub
  weight: 1.0
```

```yaml
# 训练配置
train_dataset: data/mixed.yaml
```

## 自定义 converter

注册新的 `DataConverterPlugin` 后，YAML 里直接写 `converter: 你的名字` 即可。完整接口与例子见 [data_plugins](../developer-guide/plugins/data_plugins.md)。
