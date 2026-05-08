# DataEngine

`DataEngine` 是 v1 数据访问的统一入口。它继承 `torch.utils.data.Dataset`，对外只暴露 `__getitem__` 和 `__len__`，内部把"读什么、怎么读、怎么转换"三件事拆成可替换的步骤。

代码位置：`src/llamafactory/v1/core/data_engine.py`。

## 接口

```python
class DataEngine(Dataset):
    def __init__(self, dataset_path: str) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index) -> Sample | list[Sample]: ...
```

属性：

| 属性 | 含义 |
|------|------|
| `path` | 入参 `dataset_path` 原值 |
| `dataset_infos` | `dict[name, DatasetInfo]`，每个数据源的配置 |
| `datasets` | `dict[name, HFDataset]`，已加载的数据 |
| `data_index` | `list[(name, sample_index)]`，全局采样索引 |
| `streaming` | 是否流式 |

## 初始化流程

```text
__init__(dataset_path)
  1. _get_dataset_info()   解析 dataset_path 形态，填充 dataset_infos
  2. _load_dataset()       按 source 字段把每个数据源装载进 datasets
  3. _build_data_index()   拉平成 (name, idx) 索引，必要时按 size/weight 重采样
```

### _get_dataset_info

按 `dataset_path` 字符串形态分派：

| 形态 | 加载方式 | dataset_infos 内容 |
|------|---------|---------------------|
| 本地 `.yaml` 文件 | `OmegaConf.load(path)` | YAML 内全部条目 |
| HF Hub 上的 `.yaml` | `hf_hub_download` 后 `OmegaConf.load` | YAML 内全部条目 |
| 本地存在的路径（非 YAML） | 单条目 | `{"default": {"path": ..., "source": "local"}}` |
| 其它（视为 HF Hub repo id） | 单条目 | `{"default": {"path": ...}}`（默认 `source=hf_hub`） |

YAML 形态的字段定义见 [DatasetInfo](../../parameter-reference/dataset_info.md)。

### _load_dataset

streaming 必须全开或全关：若部分条目 `streaming=true`、其它不是，会直接 `ValueError`。

```python
if dataset_info.get("source", "hf_hub") == "hf_hub":
    datasets[name] = load_dataset(path, split=split, streaming=streaming)
else:
    datasets[name] = DataLoaderPlugin(source).load(dataset_info)
```

`source != "hf_hub"` 时走 [DataLoaderPlugin](../plugins/data_plugins.md)；目前内置 `local` 一种实现。

### _build_data_index

非流式：`[(name, 0), (name, 1), ...]`，长度等于该数据集真实样本数。
流式：固定填 1000 个 `(name, -1)` 占位（仅用于让 `__len__` 有值，索引不真实使用）。

`size` 或 `weight` 字段任一非空时，调用 `adjust_data_index`：

- `size`：`random.choices(data_index, k=size)`，可上采样也可下采样
- `weight`：`random.choices(data_index, k=int(len(data_index) * weight))`，按比例放缩

最后所有数据集的索引拼接成 `self.data_index`，供后续 sampler 索引。

## 取样

```python
def __getitem__(self, index):
    if self.streaming:
        raise ValueError("Streaming dataset does not support index access.")

    if isinstance(index, int):
        name, sample_idx = self.data_index[index]
        return self._convert_data_sample(self.datasets[name][sample_idx], name)
    else:
        # 切片或 list[int]：走 select_data_sample
        ...
```

`_convert_data_sample` 依据 `dataset_infos[name]["converter"]` 决定走哪个 [DataConverterPlugin](../plugins/data_plugins.md)；不指定 converter 时假定数据已经是统一格式，直接附加 `_dataset_name` 字段返回。

## 注意事项

- `__iter__` 当前抛 `NotImplementedError`：流式访问尚未实现，外部全部走 `__getitem__`
- `_dataset_name` 字段会被注入到样本里，便于训练循环按数据集打标
- DataEngine 与模型/分词器无关，转 token 的工作交给 `Renderer`，由 `BatchGenerator` 在 `collate_fn` 里调用

## 示例

```python
from llamafactory.v1.core.data_engine import DataEngine

engine = DataEngine("data/v1_sft_demo.yaml")
print(len(engine))     # 总样本数（已按 size / weight 调整）
print(engine[0])       # 一条 Sample
print(engine[0:4])     # list[Sample]
```
