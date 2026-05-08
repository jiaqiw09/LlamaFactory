# TrainingArguments

训练超参数、批次/精度、Checkpoint 与分布式插件配置。

## 训练规模

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_dir` | `str` | `outputs/<uuid>` | 输出目录，未显式指定时每次启动随机生成一个子目录 |
| `micro_batch_size` | `int` | `1` | 单卡单步 micro batch size |
| `global_batch_size` | `int \| None` | `None` | 全局 batch size，必须能被 `dp_size × micro_batch_size` 整除；为 `None` 时取 `dp_size × micro_batch_size`（即不做梯度累积） |
| `cutoff_len` | `int` | `2048` | 最大序列长度，超出截断 |
| `learning_rate` | `float` | `1e-4` | 学习率 |
| `num_train_epochs` | `int` | `3` | 训练轮数；当 `max_steps` 设置后失效 |
| `max_steps` | `int \| None` | `None` | 最大优化步数，设置后覆盖 `num_train_epochs` |
| `max_grad_norm` | `float` | `1.0` | 梯度裁剪阈值 |
| `seed` | `int` | `42` | 随机种子 |

> **注**：梯度累积步数由 `global_batch_size / (dp_size × micro_batch_size)` 自动推导，无需单独配置。

## 精度与显存

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `bf16` | `bool` | `true` | 启用 bf16 混合精度训练 |
| `enable_activation_checkpointing` | `bool` | `true` | 启用激活值重算（梯度检查点）以节省显存 |

## 批次策略

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `batching_strategy` | `BatchingStrategy` | `normal` | 批次组装策略 |
| `batching_workers` | `int` | `16` | 数据加载线程数 |

`batching_strategy` 取值：

| 值 | 状态 | 说明 |
|----|------|------|
| `normal` | 已实现 | 等长 padding，每个 micro batch 形状一致 |
| `padding_free` | 预留 | 序列拼接，无 padding |
| `dynamic_batching` | 预留 | 动态 batch size |
| `dynamic_padding_free` | 预留 | 动态 batch size + 序列拼接 |

机制详见 [BatchGenerator](../developer-guide/core/batch_generator.md)。

## Checkpoint

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_steps` | `int \| None` | `None` | 每 N 个优化步保存一次中间 checkpoint |
| `save_epochs` | `float \| None` | `None` | 每 N 个 epoch 保存一次中间 checkpoint |
| `save_ckpt_as_hf` | `bool` | `false` | 中间 checkpoint 同时保存 HF 格式（显存占用约翻倍） |
| `save_total_limit` | `int \| None` | `None` | 最多保留的中间 checkpoint 数，超出删除最旧 |
| `resume_from_checkpoint` | `str \| None` | `None` | 恢复训练的 checkpoint 路径，`"auto"` 自动查找 `output_dir` 下的最新 |

完整保存语义（中间 vs 最终、分布式格式 vs HF 格式）见 [model_saving](../feature-guide/model_saving.md)。

## 日志

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `logging_steps` | `int` | `1` | 每 N 个优化步打印一次训练指标 |

## 插件类配置

| 参数 | 子配置页 | 常用 `name` |
|------|----------|-------------|
| `dist_config` | [DistConfig](dist_config.md) | `fsdp2` / `deepspeed` |
| `optim_config` | — | 暂无插件实现，预留字段 |
| `lr_scheduler_config` | — | 暂无插件实现，预留字段 |

## 示例

```yaml
output_dir: outputs/qwen3_lora_sft
micro_batch_size: 1
global_batch_size: 8
cutoff_len: 2048
learning_rate: 1e-4
num_train_epochs: 3
bf16: true
batching_strategy: normal
save_steps: 500
save_total_limit: 3
dist_config:
  name: fsdp2
```
