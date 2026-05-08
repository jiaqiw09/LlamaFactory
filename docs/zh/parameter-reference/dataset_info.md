# DatasetInfo

数据集 YAML 配置文件的字段格式。一个 YAML 文件下可以放多个数据集条目，键名作为数据集名称，值为下表字段。多条目会被 `DataEngine` 拼成统一索引。

## 字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | `str` | — | 必填，本地路径或 HF Hub repo ID |
| `source` | `str` | `"hf_hub"` | `"hf_hub"` / `"local"` |
| `split` | `str` | `"train"` | 数据集 split |
| `converter` | `str` | `None` | 格式转换器，`alpaca` / `sharegpt` / `pair` |
| `size` | `int` | `None` | 采样数量，使用 `random.choices` 截断/扩展 |
| `weight` | `float` | `None` | 采样权重，影响最终样本量 |
| `streaming` | `bool` | `false` | 流式加载 |

字段细节：

- `source`：`"hf_hub"` 走 `datasets.load_dataset(path)`，`"local"` 走 `DataLoaderPlugin("local")`，按文件后缀自动识别 `arrow` / `csv` / `json` / `jsonl` / `parquet` / `txt`
- `converter`：不指定时假定数据已经是统一 Messages 格式（`messages` 字段，或 DPO 的 `chosen_messages` / `rejected_messages`）
- `size` 与 `weight` 同时设置：先按 `size` 重采样，再按 `weight` 缩放

## 示例

```yaml
sft_dataset:
  path: data/alpaca_en.json
  source: local
  converter: alpaca
```

```yaml
math_data:
  path: data/math.json
  source: local
  converter: alpaca
  weight: 0.7
chat_data:
  path: llamafactory/v1-sft-demo
  source: hf_hub
  converter: sharegpt
  weight: 0.3
```

转换器输出的样本结构与新增转换器扩展点见 [data_plugins](../developer-guide/plugins/data_plugins.md)。
