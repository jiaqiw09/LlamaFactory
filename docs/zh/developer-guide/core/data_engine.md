# DataEngine

DataEngine 是 LLaMA-Factory v1 数据处理的核心类，继承自 PyTorch 的 `Dataset`，负责数据集的加载、索引构建和格式转换。数据加载和格式转换通过插件机制实现，DataEngine 本身只负责协调这些步骤。

DataEngine 接受一个唯一入参：`dataset_path: str`。

## 接口定义

```python
class DataEngine(Dataset):
    attr:
        path (str): 数据集路径
        datasets (dict[str, HFDataset]): 数据集名称到数据对象的映射
        dataset_infos (dict[str, DatasetInfo]): 数据集名称到元信息的映射
        data_index (list[tuple[str, int]]): 数据索引列表
        streaming (bool): 是否为流式数据集

    def __init__(self, dataset_path: str) -> None:
        """初始化时自动执行：
            1. _get_dataset_info — 解析数据集元信息
            2. _load_dataset — 根据配置加载数据集
            3. _build_data_index — 构建统一的索引列表
        """
```

`dataset_path` 支持以下格式：

- 本地 YAML 配置文件路径（`dataset_info.yaml`）
- HuggingFace Hub 上的 YAML 配置文件路径（如 `repo_id/dataset_info.yaml`）
- 本地数据集文件路径（`.json`、`.jsonl` 等，需为标准格式）
- HuggingFace Hub 数据集 repo id

## 核心方法

### _get_dataset_info

根据 `dataset_path` 判断数据源类型并加载数据集配置，在实例化时自动调用。

| `dataset_path` 格式 | 判定 | 处理方式 |
|---------------------|------|---------|
| 以 `.yaml` 结尾且为本地文件 | 本地 YAML 配置 | `OmegaConf.load(path)` |
| 以 `.yaml` 结尾但非本地文件 | HF Hub YAML 配置 | `hf_hub_download` + `OmegaConf.load` |
| 本地路径存在 | 本地数据文件/目录 | `{"default": {"path": path, "source": "local"}}` |
| 其他 | HF Hub 数据集 | `{"default": {"path": path}}` |

### _load_dataset

遍历所有数据源，根据 `source` 字段选择加载方式，在实例化时自动调用。

```python
for dataset_name, dataset_info in self.dataset_infos.items():
    split = dataset_info.get("split", "train")
    if dataset_info.get("source", "hf_hub") == "hf_hub":
        self.datasets[dataset_name] = load_dataset(dataset_info["path"], split=split, streaming=self.streaming)
    else:
        self.datasets[dataset_name] = DataLoaderPlugin(dataset_info["source"]).load(dataset_info)
```

### _build_data_index

为每个数据集创建索引列表 `[(dataset_name, sample_index), ...]`，在实例化时自动调用。当配置了 `size` 或 `weight` 时，调用 `adjust_data_index()` 调整索引分布。

### _convert_data_sample

将原始数据转换为标准格式，`DataConverterPlugin` 插件在此处被调用。若 `converter` 为空则假定数据集为标准格式。由 `__getitem__` 调用。

```python
def _convert_data_sample(self, raw_sample: dict, dataset_name: str) -> Sample:
    converter = self.dataset_infos[dataset_name].get("converter")
    if converter is not None:
        return {"_dataset_name": dataset_name, **DataConverterPlugin(converter)(raw_sample)}
    else:
        return {"_dataset_name": dataset_name, **raw_sample}
```

## 初始化示例

```python
from llamafactory.v1.core.data_engine import DataEngine

data_engine = DataEngine(dataset_path="data/v1_sft_demo.yaml")
sample = data_engine[0]
```

## 数据访问

实例化后的 DataEngine 支持整数索引、列表索引和切片访问：

```python
sample = data_engine[0]
samples = data_engine[0:10]
samples = data_engine[[0, 5, 10]]
```

流式数据集不支持索引访问，调用 `__getitem__` 会抛出 `ValueError`。
