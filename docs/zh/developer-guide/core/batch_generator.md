# BatchGenerator

BatchGenerator 负责从数据集生成训练批次，支持状态保存和恢复，用于断点续训。

## 设计动机

训练循环需要将原始样本（Messages 格式）转换为模型可输入的张量批次，同时支持梯度累积（micro batch → global batch）和分布式数据并行采样。BatchGenerator 封装了这一完整流程，并通过 `StatefulDataLoader` 实现可恢复的数据加载。

## 核心类

### BatchGenerator

位于 `src/llamafactory/v1/core/utils/batching.py`，继承自 `Iterator`。

```python
class BatchGenerator(Iterator):
    def __init__(self, dataset, renderer, micro_batch_size, global_batch_size,
                 cutoff_len, batching_workers, batching_strategy, pin_memory, drop_last, seed):
        ...
```

**初始化流程**：

1. 计算梯度累积步数：`num_micro_batch = global_batch_size / (dp_size * micro_batch_size)`
2. 创建 `StatefulDistributedSampler`（按 DP 维度分片数据）
3. 创建 `StatefulDataLoader`（使用 `renderer.process_samples` 作为 collate_fn）
4. 初始化 `StatefulBuffer`（批次缓冲区）

**迭代流程**：

```text
__next__() → _fill_buffer() → _generate_batch()
                ↓                    ↓
          从 DataLoader 取样本    从 Buffer 取 batch_size 个样本
          放入 StatefulBuffer     pad_and_truncate → default_collate
```

### StatefulBuffer

位于 `src/llamafactory/v1/utils/objects.py`，支持 `state_dict` / `load_state_dict` 的环形缓冲区。

### StatefulDataLoader / StatefulDistributedSampler

来自 `torchdata` 库，支持 `state_dict` / `load_state_dict`，用于断点续训时恢复 DataLoader 状态。

## 批次策略

通过 `BatchingStrategy` 枚举控制：

| 策略 | 说明 | 状态 |
|------|------|------|
| `NORMAL` | 标准 padding + truncate | 已实现 |
| `PADDING_FREE` | 无 padding | 预留 |
| `DYNAMIC_BATCHING` | 动态批次大小 | 预留 |
| `DYNAMIC_PADDING_FREE` | 动态 + 无 padding | 预留 |

非 NORMAL 策略通过 `BatchingPlugin` 扩展，当前均未实现。

## 状态保存与恢复

```python
state = batch_generator.state_dict()
# 包含：buffer 状态 + data_provider 状态

batch_generator.load_state_dict(state)
# 恢复后 _is_resuming = True，不清空 buffer
```

在 `BaseTrainer` 的 checkpoint 恢复流程中自动调用。
