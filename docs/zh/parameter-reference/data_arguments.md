# DataArguments

训练/验证数据集路径配置。

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `train_dataset` | `str \| None` | `None` | 训练集路径。支持本地 YAML 文件、HF Hub YAML、本地数据目录、HF Hub 数据集 ID |
| `eval_dataset` | `str \| None` | `None` | 验证集路径，格式同上 |

## 示例

```yaml
train_dataset: data/v1_sft_demo.yaml
```
