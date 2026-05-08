# BatchGenerator

`BatchGenerator` 把数据集变成训练循环消费的批次序列，并且这个迭代器是**有状态的**——支持 `state_dict` / `load_state_dict`，与 checkpoint 一起恢复，做到精确断点续训。

代码位置：

- `core/utils/batching.py`：`BatchGenerator`、`default_collate_fn`
- `utils/objects.py`：`StatefulBuffer`
- `plugins/trainer_plugins/batching.py`：非 `NORMAL` 策略的预留接口

## 设计

```text
DataEngine ──► StatefulDistributedSampler ──► StatefulDataLoader
                       │                              │ collate_fn = renderer.process_samples
                       │                              ▼
                       │                        list[ModelInput]
                       │                              ▼
                       └─────────────────────► StatefulBuffer
                                                      ▼
                                       default_collate_fn
                                                      ▼
                                       list[BatchInput]
                                       (一个 global step 的全部 micro batch)
```

每次 `__next__` 返回一个 `list[BatchInput]`，长度等于 `num_micro_batch`。

## 推导 num_micro_batch

```python
dp_size = DistributedInterface().get_world_size(Dim.DP)
if global_batch_size is None:
    global_batch_size = dp_size * micro_batch_size
    num_micro_batch = 1
elif global_batch_size % (dp_size * micro_batch_size) == 0:
    num_micro_batch = global_batch_size // dp_size // micro_batch_size
else:
    raise ValueError("global_batch_size must be divisible by dp_size * micro_batch_size")
```

`num_micro_batch` 即梯度累积步数，由 `BaseTrainer` 拿到后驱动一个完整的 optimizer step。

## sampler 与 dataloader

```python
sampler = StatefulDistributedSampler(
    dataset,
    num_replicas=dp_size,
    rank=DistributedInterface().get_rank(Dim.DP),
    shuffle=True,
    seed=seed,
    drop_last=True,
)

dataloader = StatefulDataLoader(
    dataset,
    batch_size=micro_batch_size * num_micro_batch,    # 一次取够整个 step
    sampler=sampler,
    num_workers=batching_workers,
    collate_fn=renderer.process_samples,              # 渲染发生在 worker
    pin_memory=True,
    pin_memory_device=current_device.type,
    drop_last=True,
    generator=Generator().manual_seed(seed),
)
```

来自 `torchdata` 的 `Stateful*` 版本会把"已经派发过的样本索引"写进 `state_dict`，恢复后从下一个样本继续。

> **注**：`drop_last=False` 没有支持，构造时会直接 `ValueError`。流式 / `len() == -1` 的数据集同样不支持，目前 `_init_data_provider` 抛 `NotImplementedError`。

## StatefulBuffer

```python
class BatchGenerator:
    def __next__(self):
        self._fill_buffer()              # 从 dataloader 拿样本进 buffer
        batch = self._generate_batch()   # 从 buffer 拼一组 micro batch
        if batch is None:
            raise StopIteration
        return batch
```

`StatefulBuffer`（`utils/objects.py`）是一个可序列化的样本缓冲队列，`state_dict` 存"还没消费完的样本"，与 dataloader state 一起回放。

## 拼批：default_collate_fn

`NORMAL` 策略下：

1. buffer 里抽 `micro_batch_size * num_micro_batch` 条样本
2. 切成 `num_micro_batch` 段
3. 每段先 `pad_and_truncate(samples, cutoff_len)`，再 `default_collate` 成张量

返回 `list[BatchInput]`，长度 `num_micro_batch`。

## 批次策略

| 策略 | 状态 | 处理位置 |
|------|------|---------|
| `NORMAL` | 已实现 | 在 `BatchGenerator` 内直接用 `default_collate_fn` |
| `PADDING_FREE` / `DYNAMIC_BATCHING` / `DYNAMIC_PADDING_FREE` | 预留 | 由 `BatchingPlugin(name)` 提供 `compute_length` / `fill_buffer` / `generate_batch` |

非 `NORMAL` 路径目前在 `_init_data_provider` 里直接 `NotImplementedError`，等具体策略落地。

## 状态保存与恢复

```python
state = batch_generator.state_dict()   # {"buffer": ..., "data_provider": ...}

batch_generator.load_state_dict(state)
# 内部置 _is_resuming=True，下一次 __iter__ 不清空 buffer
```

`TrainingCheckpointCoordinator` 在保存 / 恢复 checkpoint 时按 rank 分别落盘 / 读取这个 state，每个 rank 独立写自己的 dataloader 状态。

## set_epoch

每个 epoch 开始前 `BaseTrainer` 会调 `batch_generator.set_epoch(epoch)`，最终落到 sampler 的 `set_epoch`，shuffle 种子按 `(seed, epoch)` 派生。
