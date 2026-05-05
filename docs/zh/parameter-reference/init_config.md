# InitConfig

`ModelArguments.init_config` 的子配置，控制模型权重加载策略。

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | - | 初始化策略名，见下表 |

| 策略名 | 说明 | 适用场景 |
|--------|------|---------|
| `init_on_default` | 每张卡分别加载完整模型到当前设备 | 默认行为，单卡或小模型 |
| `init_on_meta` | 在 meta device 上创建空壳模型 | FSDP2 大模型，由框架后续实际加载权重 |
| `init_on_rank0` | Rank 0 加载到 CPU，其余 rank 在 meta device | FSDP2 大模型，减少多卡重复加载开销 |

## 示例

```yaml
init_config:
  name: init_on_meta
```

`init_on_meta` 和 `init_on_rank0` 通常与 FSDP2 配合使用，`init_on_default` 为默认行为，无需显式配置。
