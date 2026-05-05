# Accelerator 层

Accelerator 层提供硬件抽象和分布式通信原语，位于 `src/llamafactory/v1/accelerator/`。

## 设计动机

v1 需要支持多种并行维度（模型并行、数据并行、上下文并行）。Accelerator 层将 PyTorch 原生的分布式 API 封装为统一的 `DistributedInterface` 单例，上层代码无需关心底层通信细节。

## 核心组件

### DistributedInterface（单例）

`DistributedInterface` 是全局唯一的分布式接口，负责：

1. **初始化进程组**：`init_process_group` + DeviceMesh 创建
2. **管理并行拓扑**：`DistributedStrategy` 描述四维并行维度
3. **提供通信原语**：`all_gather`、`all_reduce`、`broadcast`、`barrier`

```python
dist = DistributedInterface(config=dist_config)
dist.get_rank(Dim.DP)
dist.get_world_size(Dim.CP)
dist.all_reduce(loss, dim=Dim.DP)
```

### DistributedStrategy

描述四维并行拓扑：

```text
Model Mesh: mp_replicate × mp_shard  → 模型权重分片
Data Mesh:  dp × cp                   → 数据 & 序列并行
```

约束：`mp_replicate_size * mp_shard_size = world_size`，`dp_size * cp_size = world_size`。

### Dim 枚举

| 维度 | 说明 |
|------|------|
| `MP_REPLICATE` | 模型并行复制维度 |
| `MP_SHARD` | 模型并行分片维度 |
| `DP` | 数据并行维度 |
| `CP` | 上下文并行维度 |

## helper 模块

`accelerator/helper.py` 提供底层设备检测和通信封装：

- `get_current_accelerator()` — 检测当前加速器类型
- `is_distributed()` — 是否在分布式环境中
- `get_rank()` / `get_world_size()` / `get_local_rank()` — 进程信息
- `all_gather()` / `all_reduce()` / `broadcast()` — 通信原语
- `DeviceType` 枚举 — `CUDA`、`NPU`、`CPU`、`META`

## 扩展点

添加新硬件后端支持时，需在 `helper.py` 中扩展 `get_current_accelerator()` 和 `DeviceType`，并在 `DistributedInterface` 中确保 `init_device_mesh` 使用正确的设备类型。各硬件后端的差异见 [多后端支持](../../multi-backend/index.md)。
