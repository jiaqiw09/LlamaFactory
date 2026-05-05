# TrainingArguments

训练超参数、checkpoint 及分布式配置。

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_dir` | `str` | `outputs/<uuid>` | 输出目录 |
| `micro_batch_size` | `int` | `1` | 每卡 micro batch size |
| `global_batch_size` | `int \| None` | `None` | 全局 batch size，默认 `dp_size × micro_batch_size` |
| `cutoff_len` | `int` | `2048` | 最大序列长度 |
| `learning_rate` | `float` | `1e-4` | 学习率 |
| `num_train_epochs` | `int` | `3` | 训练 epoch 数，被 `max_steps` 覆盖 |
| `max_steps` | `int \| None` | `None` | 最大训练步数，设置后忽略 `num_train_epochs` |
| `max_grad_norm` | `float` | `1.0` | 梯度裁剪阈值 |
| `bf16` | `bool` | `True` | bf16 混合精度 |
| `seed` | `int` | `42` | 随机种子 |
| `logging_steps` | `int` | `1` | 日志输出间隔（step） |
| `batching_strategy` | `BatchingStrategy` | `NORMAL` | 批次策略：`NORMAL`（已实现）/ `PADDING_FREE` / `DYNAMIC_BATCHING` / `DYNAMIC_PADDING_FREE`（预留） |
| `batching_workers` | `int` | `16` | 数据加载 worker 数 |
| `enable_activation_checkpointing` | `bool` | `True` | 梯度 checkpointing |
| `dist_config` | `PluginConfig \| None` | `None` | 分布式配置，见 [DistConfig](dist_config.md) |
| `optim_config` | `PluginConfig \| None` | `None` | 优化器配置（暂无插件实现） |
| `lr_scheduler_config` | `PluginConfig \| None` | `None` | LR 调度器配置（暂无插件实现） |
| `resume_from_checkpoint` | `str \| None` | `None` | 恢复 checkpoint 路径，`"auto"` 自动查找最新 |
| `save_steps` | `int \| None` | `None` | 每 N steps 保存 checkpoint |
| `save_epochs` | `float \| None` | `None` | 每 N epochs 保存 checkpoint |
| `save_ckpt_as_hf` | `bool` | `False` | 额外保存 HF 格式 checkpoint（加倍内存） |
| `save_total_limit` | `int \| None` | `None` | 最多保留 checkpoint 数，超出自动删除最旧 |

## 示例

```yaml
output_dir: outputs/my_model
micro_batch_size: 1
global_batch_size: 8
cutoff_len: 2048
learning_rate: 1e-4
num_train_epochs: 3
bf16: true
save_steps: 500
save_total_limit: 3
```
