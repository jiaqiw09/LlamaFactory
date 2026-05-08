# DataArguments

数据集路径配置。两个字段都接收同一种字符串，由 `DataEngine` 在加载时根据字符串形态分派到不同的加载方式。

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `train_dataset` | `str \| None` | `None` | 训练集路径 |
| `eval_dataset` | `str \| None` | `None` | 验证集路径 |

## 取值形式

`train_dataset` / `eval_dataset` 接受四种字符串形态：

- 本地 YAML 文件路径，例如 `data/v1_sft_demo.yaml`
- HF Hub 上的 YAML 文件 ID，例如 `username/repo/file.yaml`
- 本地数据集目录路径
- HF Hub 数据集 ID

YAML 形态的具体字段见 [DatasetInfo](dataset_info.md)。完整的数据准备流程见 [data_preparation](../feature-guide/data_preparation.md)。

## 示例

```yaml
train_dataset: data/v1_sft_demo.yaml
eval_dataset: data/v1_sft_eval.yaml
```
