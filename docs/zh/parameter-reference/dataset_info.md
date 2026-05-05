# DatasetInfo

数据集 YAML 配置文件格式，描述数据来源、转换器和采样策略。

## 参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | `str` | 必需 | 数据集路径（本地文件/目录 或 HF Hub repo ID） |
| `source` | `str` | `"hf_hub"` | `"hf_hub"` / `"local"`。`"hf_hub"` 使用 `datasets.load_dataset` 加载，`"local"` 使用 `DataLoaderPlugin` 加载 |
| `split` | `str` | `"train"` | 数据集 split |
| `converter` | `str` | `None` | 格式转换器：`alpaca` / `sharegpt` / `pair`。不指定则假定数据为标准 Messages 格式 |
| `size` | `int` | `None` | 采样数量，通过 `random.choices` 限制使用的样本数 |
| `weight` | `float` | `None` | 数据集权重（多数据集混合时控制采样频率） |
| `streaming` | `bool` | `False` | 流式加载 |

## 示例

```yaml
sft_dataset:
  path: data/alpaca_en.json
  source: local
  converter: alpaca

# 多数据集混合
dataset_a:
  path: data/math.json
  source: local
  converter: alpaca
  weight: 0.7
dataset_b:
  path: llamafactory/v1-sft-demo
  source: hf_hub
  converter: sharegpt
  weight: 0.3
```
