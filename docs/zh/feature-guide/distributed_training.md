# 分布式训练

通过 `dist_config` 配置分布式策略，多 GPU 自动通过 `torchrun` 启动。完整参数说明见 [DistConfig](../parameter-reference/dist_config.md)。

## 自动启动

```bash
export USE_V1=1
llamafactory-cli sft config.yaml
```

| 环境变量 | 说明 | 默认 |
|---------|------|------|
| `FORCE_TORCHRUN` | 单卡也强制分布式 | - |
| `NNODES` | 节点数 | 1 |
| `NODE_RANK` | 节点 rank | 0 |
| `MASTER_ADDR` | 通信地址 | 127.0.0.1 |
| `MASTER_PORT` | 通信端口 | 自动 |

## FSDP2

```yaml
dist_config:
  name: fsdp2
  reshard_after_forward: true
```

更多 FSDP2 参数（`mp_replicate_size`、`cp_size`、`offload_params` 等）见 [DistConfig](../parameter-reference/dist_config.md)。

模型初始化模式（`init_config.name`）：`init_on_default`（默认）、`init_on_meta`（省内存）、`init_on_rank0`（省带宽）。

## DeepSpeed

```yaml
dist_config:
  name: deepspeed
  config_file: examples/deepspeed/ds_z3_config.json
```

DeepSpeed 通过 HuggingFace Accelerate 集成，需提供 ZeRO 配置文件。更多说明见 [DistConfig → DeepSpeed](../parameter-reference/dist_config.md#deepspeed)。

## Context Parallel

```yaml
dist_config:
  name: fsdp2
  cp_size: 2
```

基于 Ulysses 算法，将长序列拆分到多个设备。需要 `flash_attention_2`，不支持 qwen3_5。

## 不指定 dist_config

多 GPU 时自动回退 DDP。

> **硬件支持**：FSDP2 和 Context Parallel 在 GPU 上支持。DeepSpeed 仅 GPU。其他硬件后端的支持情况见 [硬件支持矩阵](../hardware_support_matrix.md) 和 [多后端支持](../multi-backend/index.md)。
