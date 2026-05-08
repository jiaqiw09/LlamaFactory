# DistConfig

`TrainingArguments.dist_config` 的子配置。未设置时多 GPU 自动回退到 DDP，单卡时不启用任何分布式后端。

`name` 决定使用哪个分布式插件，其它字段分两类：**通用** 字段用于初始化进程组与 device mesh，由 `DistributedInterface` 读取；**插件专属** 字段只在对应分支生效。

## 通用字段（所有 `name`）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | — | `fsdp2` / `deepspeed` |
| `mp_replicate_size` | `int` | `1` | 模型并行 replicate 维度 |
| `mp_shard_size` | `int` | 自动 | 模型并行 shard 维度，默认 `world_size / mp_replicate_size` |
| `dp_size` | `int` | 自动 | 数据并行维度，默认 `world_size / cp_size` |
| `cp_size` | `int` | `1` | Context Parallel 维度，> 1 时启用序列并行 |
| `cp_mode` | `str` | `"ulysses"` | Context Parallel 实现，目前仅 `ulysses` |
| `timeout` | `int` | `18000` | 进程组通信超时（秒） |

> **注**：`mp_replicate_size × mp_shard_size` 与 `dp_size × cp_size` 都必须等于 `world_size`，否则 `DistributedInterface` 启动时报错。

## FSDP2

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `reshard_after_forward` | `bool` | `true` | Forward 后重新分片释放显存 |
| `offload_params` | `bool` | `false` | 参数 CPU offload |
| `pin_memory` | `bool` | `true` | CPU offload 是否锁页 |
| `dcp_path` | `str` | `None` | 初始权重的 DCP（分布式 checkpoint）路径，仅 `init_on_meta` 路径下使用，与 `resume_from_checkpoint` 不是一回事 |

```yaml
dist_config:
  name: fsdp2
  reshard_after_forward: true
  offload_params: false
```

## DeepSpeed

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config_file` | `str` | — | 必填，DeepSpeed JSON 配置文件路径 |

DeepSpeed 通过 `accelerate` 的 `DeepSpeedPlugin` 集成，混合精度从 `config_file` 自动推断；`micro_batch_size` 中的 `"auto"` 由训练参数填充。

```yaml
dist_config:
  name: deepspeed
  config_file: examples/deepspeed/ds_z3_config.json
```

完整使用流程见 [distributed_training](../feature-guide/distributed_training.md)。
