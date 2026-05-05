# 数据插件

数据相关的插件集合，包括数据格式转换（DataConverterPlugin）和数据加载（DataLoaderPlugin）。

## DataConverterPlugin

DataConverterPlugin 负责将非标准格式的数据集转换为 v1 的标准 Messages 格式。当前已内置 `alpaca`、`sharegpt` 和 `pair` 三种转换器。

### Alpaca Converter

Alpaca 格式是一种常见的指令微调数据格式：

```json
{
  "system": "You are a helpful assistant.",
  "instruction": "Describe a process of making crepes.",
  "input": "",
  "output": "Making crepes is an easy and delicious process..."
}
```

转换逻辑：

- 若存在 `system` 字段，生成一条系统消息（`loss_weight = 0.0`）
- 若存在 `instruction` 或 `input` 字段，合并为一条用户消息（`loss_weight = 0.0`）
- 若存在 `output` 字段，生成一条助手回复消息（`loss_weight = 1.0`）

转换示例：

**输入（Alpaca 格式）：**

```json
{
  "instruction": "What is the capital of France?",
  "input": "",
  "output": "The capital of France is Paris."
}
```

**输出（标准格式）：**

```json
{
  "messages": [
    {
      "role": "user",
      "content": [{"type": "text", "value": "What is the capital of France?"}],
      "loss_weight": 0.0
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "value": "The capital of France is Paris."}],
      "loss_weight": 1.0
    }
  ]
}
```

### ShareGPT Converter

将 ShareGPT 格式的多轮对话转换为标准 Messages 格式。角色标签映射：

| ShareGPT 标签 | 标准角色 |
|--------------|---------|
| `system` | `system` |
| `human` | `user` |
| `gpt` | `assistant` |
| `observation` | `tool` |
| `function_call` | `assistant`（解析为 `tool_call` 类型内容） |

### Pair Converter

将 OpenAI 风格的偏好对（chosen/rejected）转换为 DPO 标准格式（`chosen_messages` / `rejected_messages`）。

### 自定义转换器

添加自定义转换器：

```python
# src/llamafactory/v1/plugins/data_plugins/converter.py

from ...utils.types import SFTSample

def custom_converter(raw_sample: dict) -> SFTSample:
    messages = []
    user_text = raw_sample["question"]
    messages.append({
        "role": "user",
        "content": [{"type": "text", "value": user_text}],
        "loss_weight": 0.0,
    })
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "value": raw_sample["answer"]}],
        "loss_weight": 1.0,
    })
    return {"messages": messages}

CONVERTERS = {
    "alpaca": alpaca_converter,
    "sharegpt": sharegpt_converter,
    "pair": pair_converter,
    "custom": custom_converter,
}
```

在 YAML 配置中指定转换器名称：

```yaml
my_dataset:
  path: custom_data.json
  source: local
  converter: custom
```

## DataLoaderPlugin

DataLoaderPlugin 负责从本地文件加载数据集，继承自 `BasePlugin`，当前注册了 `"local"` 一个插件。

### 接口定义

```python
class DataLoaderPlugin(BasePlugin):
    def load(self, dataset_info: DatasetInfo) -> HFDataset:
        path = dataset_info["path"]
        split = dataset_info.get("split", "train")
        streaming = dataset_info.get("streaming", False)
        return super().__call__(path, split, streaming)
```

### Local Loader

注册为 `DataLoaderPlugin("local")`，支持从本地文件或目录加载数据集：

```python
@DataLoaderPlugin("local").register()
def load_data_from_file(filepath: str, split: str, streaming: bool) -> HFDataset:
    ...
```

支持的文件格式：`.json`、`.jsonl`、`.csv`、`.parquet`、`.arrow`、`.txt`。

文件类型通过扩展名自动推断（`.jsonl` 映射为 `json`，`.txt` 映射为 `text`），内部调用 `datasets.load_dataset()` 加载。若 `streaming=True`，结果转换为迭代式数据集。

## 索引调整函数

### adjust_data_index

位于 `src/llamafactory/v1/plugins/data_plugins/loader.py`，根据 `size` 和 `weight` 调整数据索引分布。

```python
def adjust_data_index(
    data_index: list[tuple[str, int]], size: int | None, weight: float | None
) -> list[tuple[str, int]]:
    ...
```

- **size**：限制使用的样本数量，通过 `random.choices` 采样
- **weight**：调整数据集在混合训练中的采样频率，按权重倍数复制索引

在 `DataEngine._build_data_index()` 中被自动调用。

### select_data_sample

位于 `src/llamafactory/v1/plugins/data_plugins/loader.py`，根据索引选择数据集样本。

```python
def select_data_sample(
    data_index: list[tuple[str, int]], index: slice | list[int] | Any
) -> tuple[str, int] | list[tuple[str, int]]:
    ...
```

支持切片（`slice`）和索引列表（`list[int]`）两种访问方式。在 `DataEngine.__getitem__()` 中被自动调用。
