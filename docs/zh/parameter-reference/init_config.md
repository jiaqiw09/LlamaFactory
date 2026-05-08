# InitConfig

`ModelArguments.init_config` 的子配置，控制模型权重的初始加载位置。`name` 字段是唯一开关，决定模型在哪个 device 上被构造。

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | — | 见下方策略表 |

## 策略

| 策略 | 加载位置 | 适用场景 |
|------|----------|---------|
| `init_on_default` | 当前设备（每张卡各加载一份完整权重） | 单卡或小模型，是默认行为 |
| `init_on_meta` | meta device（空壳，无实际权重） | FSDP2 大模型，由 `materialize_and_load` 后续从 HF checkpoint 或 DCP 加载 |
| `init_on_rank0` | Rank 0 在 CPU 上加载完整权重，其余 rank 在 meta | FSDP2 大模型，避免每张卡重复读盘；FSDP shard 时再从 rank 0 broadcast |

`init_on_meta` 与 `init_on_rank0` 都依赖 FSDP2 在 `shard_model` 阶段的特殊路径来填充实际权重，不能与非 FSDP2 的分布式后端搭配。

## 示例

```yaml
init_config:
  name: init_on_meta
```

完整加载流程见 [ModelEngine](../developer-guide/core/model_engine.md)。
