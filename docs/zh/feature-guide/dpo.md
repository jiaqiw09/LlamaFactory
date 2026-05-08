# 偏好对齐（DPO）

> **现状**：DPO 训练入口尚未实现。`llamafactory-cli dpo` 走到 `launcher` 后会直接抛 `NotImplementedError`。本页只覆盖**已经可用**的部分（数据层）。

## 已经可用

- 数据层定义了 DPO 标准样本：`chosen_messages` + `rejected_messages`
- `pair` converter 已实现，可把常见的偏好对数据格式转成标准结构
- CLI 入口 `dpo` 命名已保留，待训练器接入

## 数据格式

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

如果原始数据是 OpenAI 风格（`{chosen: [...], rejected: [...]}`），在 `dataset_info.yaml` 里指定 `converter: pair` 即可自动转换。详见 [数据准备](data_preparation.md) 与 [DatasetInfo](../parameter-reference/dataset_info.md)。

## 后续接入

DPO 训练器接入后，本页会补齐：

- 最小可运行命令与 YAML
- 全参 / PEFT 配置示例
- DPO loss 相关参数
- Checkpoint、导出、推理与 SFT 的衔接方式

实现进度跟踪 `src/llamafactory/v1/trainers/dpo_trainer.py`。
