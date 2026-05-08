# 分布式训练

通过 `dist_config.name` 选择分布式后端：当前支持 `fsdp2` 与 `deepspeed`。多 GPU 时由 `launcher` 自动 `torchrun`，单卡可强制开启分布式。完整字段见 [DistConfig](../parameter-reference/dist_config.md)。

## 自动启动

```bash
export USE_V1=1
llamafactory-cli sft config.yaml
```

判定逻辑（见 `launcher.launch`）：

- 已设置 `RANK` 之类的环境变量 → 直接进训练流程
- `FORCE_TORCHRUN=1` → 强制用 torchrun re-exec
- 否则当 `device_count > 1`（且未启用 Ray / KTransformers）时 → 自动 torchrun

| 环境变量 | 作用 | 默认 |
|---------|------|------|
| `FORCE_TORCHRUN` | 单卡也强制走 torchrun | 未设置 |
| `NNODES` | 节点数 | `1` |
| `NODE_RANK` | 节点 rank | `0` |
| `MASTER_ADDR` | 通信地址 | `127.0.0.1` |
| `MASTER_PORT` | 通信端口 | 自动选可用端口 |
| `NPROC_PER_NODE` | 每节点进程数 | 当前设备数 |
| `RDZV_ID` / `MIN_NNODES` / `MAX_NNODES` / `MAX_RESTARTS` | 弹性训练参数 | — |

## FSDP2

```yaml
dist_config:
  name: fsdp2
  reshard_after_forward: true
  offload_params: false
```

要点：

- 默认 `bf16: true` → 参数 bf16，reduce fp32
- `offload_params: true` 把参数 offload 到 CPU，搭配 `pin_memory: true` 可降低显存压力
- `init_on_meta` / `init_on_rank0` 是 FSDP2 专属优化，参考 [InitConfig](../parameter-reference/init_config.md)
- 大模型 LoRA 推荐 `init_on_rank0`，可省去每张卡重复加载

完整字段见 [DistConfig → FSDP2](../parameter-reference/dist_config.md#fsdp2)，机制细节见 [trainer_plugins → fsdp2](../developer-guide/plugins/trainer_plugins.md#fsdp2)。

## DeepSpeed

```yaml
dist_config:
  name: deepspeed
  config_file: examples/deepspeed/ds_z3_config.json
```

DeepSpeed 通过 HuggingFace Accelerate 集成，YAML 里只指定 ZeRO JSON 路径，混合精度从 JSON 中推断。`micro_batch_size` 中的 `"auto"` 会被 v1 自动填回。

## Context Parallel（Ulysses）

把长序列拆到多张卡上，搭配 FSDP2 使用：

```yaml
dist_config:
  name: fsdp2
  cp_mode: ulysses
  cp_size: 2
```

约束：

- attention 实现强制改 `flash_attention_2`
- `num_attention_heads % cp_size == 0`
- `num_key_value_heads % cp_size == 0` 或 `cp_size % num_key_value_heads == 0`
- 不支持 qwen3.5（attention 实现不同）

机制见 [model_plugins → SequenceParallel](../developer-guide/plugins/model_plugins.md#sequenceparallelplugin)。

## DDP（自动回退）

不写 `dist_config` 时：

- 单卡：本地训练
- 多卡：`BaseTrainer._shard_model` 自动用 `DistributedDataParallel(model)` 包一层

DDP 路径不支持 FSDP / DeepSpeed 专属功能（如分片初始化、ZeRO offload）。

## 多机

把同一份 YAML 复制到所有节点，按节点设置环境变量：

```bash
# 节点 0
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
  llamafactory-cli sft config.yaml

# 节点 1
NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
  llamafactory-cli sft config.yaml
```

容错（弹性训练）需额外设 `RDZV_ID` 与 `MIN_NNODES` / `MAX_NNODES`。

## 现成示例

| 文件 | 场景 |
|------|------|
| `examples/v1/train_full/train_full_fsdp2.yaml` | FSDP2 全参 |
| `examples/v1/train_full/train_full_deepspeed.yaml` | DeepSpeed ZeRO-3 |
| `examples/v1/train_full/train_full_ulysses_cp.yaml` | FSDP2 + Ulysses CP |
| `examples/v1/train_lora/train_lora_sft_rank0.yaml` | FSDP2 + LoRA + `init_on_rank0` |

> **硬件支持**：FSDP2 与 Context Parallel 在 GPU 上验证。DeepSpeed 仅 GPU。其它后端的支持情况见 [硬件支持矩阵](../hardware_support_matrix.md) 与 [多后端支持](../multi-backend/index.md)。
