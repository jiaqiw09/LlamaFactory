# DistConfig

`TrainingArguments.dist_config` 的子配置。不指定时多 GPU 自动回退 DDP。

## 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | - | `fsdp2` / `deepspeed` |
| `timeout` | `int` | `18000` | 通信超时（秒） |

## FSDP2

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mp_replicate_size` | `int` | `1` | Model replicate 维度 |
| `mp_shard_size` | `int` | 自动 | Model shard 维度 |
| `dp_size` | `int` | 自动 | 数据并行维度 |
| `cp_size` | `int` | `1` | Context Parallel 维度（>1 触发 Ulysses 序列并行） |
| `cp_mode` | `str` | `ulysses` | Context Parallel 实现模式，当前支持 `ulysses` |
| `reshard_after_forward` | `bool` | `True` | Forward 后释放分片参数 |
| `offload_params` | `bool` | `False` | 参数 offload 到 CPU |
| `pin_memory` | `bool` | `True` | CPU offload 内存锁定 |
| `dcp_path` | `str` | `None` | FSDP2 初始化时用于分片权重加载的 DCP 路径，不等同于 `resume_from_checkpoint` |

```yaml
dist_config:
  name: fsdp2
  reshard_after_forward: true
```

## DeepSpeed

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config_file` | `str` | 必需 | DeepSpeed 配置文件路径 |

```yaml
dist_config:
  name: deepspeed
  config_file: examples/deepspeed/ds_z3_config.json
```
