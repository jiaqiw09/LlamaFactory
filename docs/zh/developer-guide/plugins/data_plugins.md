# 数据插件

数据加载与格式转换都通过插件实现，`DataEngine` 只负责按数据集配置分派。两个插件类：

- `DataLoaderPlugin`：把数据读进 `HFDataset`
- `DataConverterPlugin`：把单条样本转成 v1 标准格式

代码位置：`src/llamafactory/v1/plugins/data_plugins/`。

## DataConverterPlugin

输入是数据集原生 schema，输出是 v1 标准 `Sample`。`DataEngine.__getitem__` 取到原始样本后，按 `dataset_info.converter` 字段调用对应转换器。

```python
class DataConverterPlugin(BasePlugin):
    def __call__(self, raw_sample: dict[str, Any]) -> Sample:
        return super().__call__(raw_sample)
```

`Sample` 是 SFT 或 DPO 之一：

| 类型 | 必需字段 |
|------|----------|
| `SFTSample` | `messages: list[Message]` |
| `DPOSample` | `chosen_messages: list[Message]`、`rejected_messages: list[Message]` |

每条 `Message` 携带 `loss_weight`，转换器决定哪些 token 参与训练。

### Alpaca

输入：

```json
{
  "system": "You are a helpful assistant.",
  "instruction": "Describe a process of making crepes.",
  "input": "",
  "output": "..."
}
```

转换规则：

- 有 `system` → 系统消息，`loss_weight=0.0`
- `instruction` 与 `input` 拼接 → 用户消息，`loss_weight=0.0`
- `output` → assistant 消息，`loss_weight=1.0`

### ShareGPT

把 `conversations` 里的消息按 `from` 标签翻译：

| `from` | role | 备注 |
|--------|------|------|
| `system` | `system` | — |
| `human` | `user` | — |
| `gpt` | `assistant` | `loss_weight=1.0` |
| `observation` | `tool` | — |
| `function_call` | `assistant` | `value` 解析为 JSON tool call，`content.type="tool_call"` |

`tools` 字段如果存在会再被 JSON 序列化一次保留到 `Sample.tools`。

### Pair

OpenAI 风格的偏好对：`{chosen: [...], rejected: [...]}` → `{chosen_messages, rejected_messages}`。

assistant 消息 `loss_weight=1.0`，其它 `0.0`。`tool` role 的 `content` 当作 JSON tool call 解析。

### 注册新转换器

```python
from llamafactory.v1.plugins.data_plugins.converter import DataConverterPlugin
from llamafactory.v1.utils.types import SFTSample


@DataConverterPlugin("qa").register()
def qa_converter(raw_sample: dict) -> SFTSample:
    return {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": raw_sample["question"]}],
                "loss_weight": 0.0,
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "value": raw_sample["answer"]}],
                "loss_weight": 1.0,
            },
        ],
    }
```

YAML 配置里使用：

```yaml
my_dataset:
  path: data/qa.json
  source: local
  converter: qa
```

新转换器只要写在 `converter.py`（或被它 import 的文件）里即可，import 时装饰器自动注册到 `DataConverterPlugin._registry`。

## DataLoaderPlugin

```python
class DataLoaderPlugin(BasePlugin):
    def load(self, dataset_info: DatasetInfo) -> HFDataset:
        path = dataset_info["path"]
        split = dataset_info.get("split", "train")
        streaming = dataset_info.get("streaming", False)
        return super().__call__(path, split, streaming)
```

只有 `dataset_info.source != "hf_hub"` 时走插件；`hf_hub` 走 `datasets.load_dataset` 直接加载。

### 内置 local loader

```python
@DataLoaderPlugin("local").register()
def load_data_from_file(filepath, split, streaming) -> HFDataset:
    ...
```

按文件后缀映射 builder：

| 后缀 | builder |
|------|---------|
| `.arrow` | `arrow` |
| `.csv` | `csv` |
| `.json` / `.jsonl` | `json` |
| `.parquet` | `parquet` |
| `.txt` | `text` |

目录路径取 `os.listdir` 第一个文件的后缀作为整体类型。`streaming=True` 时调用 `to_iterable_dataset()` 转成迭代式数据集（本地文件路径下这样比直接 `streaming=True` 更快）。

### 注册新 loader

```python
@DataLoaderPlugin("s3").register()
def load_data_from_s3(filepath, split, streaming) -> HFDataset:
    ...
```

YAML 里 `source: s3` 时自动匹配。

## 索引调整辅助函数

`loader.py` 提供两个独立函数，被 `DataEngine` 在构建索引时调用：

### adjust_data_index

```python
def adjust_data_index(data_index, size: int | None, weight: float | None):
    if size is not None:
        data_index = random.choices(data_index, k=size)
    if weight is not None:
        data_index = random.choices(data_index, k=int(len(data_index) * weight))
    return data_index
```

- `size`：把当前数据集采样到指定条数（可以放大也可以缩小）
- `weight`：按比例缩放当前长度
- 两个同时设置：先 `size` 再 `weight`

### select_data_sample

```python
def select_data_sample(data_index, index: slice | list[int] | Any):
    ...
```

`DataEngine.__getitem__` 接到切片或索引列表时调用，返回 `(name, idx)` 列表，再去原始数据集里取样本。
