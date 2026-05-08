# Accelerator

`accelerator/` 是硬件与分布式抽象层。设计目标是让上层代码不必区分 CUDA / NPU / 单卡 / 多卡：通信原语、设备类型、并行拓扑都通过统一接口访问。

代码位置：

- `accelerator/helper.py`：底层设备探测、通信原语
- `accelerator/interface.py`：`DistributedInterface` 单例 + `DistributedStrategy` 拓扑

## 设备类型

`helper.DeviceType` 枚举所有支持的设备：

```python
class DeviceType(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    META = "meta"
    MPS = "mps"
    NPU = "npu"
    XPU = "xpu"
```

`get_current_accelerator()`（`@lru_cache`）调用 `torch.accelerator.current_accelerator()`，返回当前可用设备；不可用则回退 `cpu`。`get_process_group_backend()` 据此选择 `nccl` / `hccl` / `gloo`。

## DistributedInterface

```python
class DistributedInterface:
    _instance: Optional["DistributedInterface"] = None  # 单例
```

- 单例：第一次构造接受 `dist_config`，之后任意位置 `DistributedInterface()` 直接拿到同一实例
- `__init__` 里完成：`set_device_index`、初始化进程组（`init_process_group(timeout=...)`）、构造两个 `DeviceMesh`
- 通信原语包装在实例方法里：`all_reduce`、`all_gather`、`broadcast`、`sync`、`get_rank(dim)`、`get_world_size(dim)`、`get_group(dim)`、`get_device_mesh(dim)`

非分布式（即未设置 `RANK` 环境变量）时，两个 mesh 都为 `None`，通信原语在单进程下为空操作。

## DistributedStrategy

`DistributedStrategy` 把 `world_size` 拆成两个 2D mesh：

```text
Model Mesh:  mp_replicate × mp_shard       维度名 MP_REPLICATE / MP_SHARD
Data  Mesh:  dp           × cp             维度名 DP / CP
```

`__post_init__` 校验：

- `mp_replicate_size × mp_shard_size == world_size`
- `dp_size × cp_size == world_size`

任一不满足直接 `ValueError`。

`Dim` 枚举给上层代码用：

```python
DistributedInterface().get_world_size(Dim.DP)
DistributedInterface().all_reduce(grad_norm**2, op=ReduceOp.SUM, dim=Dim.CP)
```

## 一般用法

读取并行维度时一律走 `DistributedInterface`，不要直接读环境变量。

```python
dist = DistributedInterface()
rank = dist.get_rank()
dp_rank = dist.get_rank(Dim.DP)
dp_size = dist.get_world_size(Dim.DP)

dist.all_reduce(loss)             # 默认 mean，全局
dist.all_reduce(x, dim=Dim.CP)    # 仅在 cp 维度上 reduce
```

## 加新硬件后端

1. 在 `helper.DeviceType` 增加枚举值
2. 在 `helper.get_process_group_backend()` 加分支
3. 确认 `torch.accelerator` 能识别该设备
4. 检查需要替换 forward 的算子，在 `kernels/ops/` 下注册（见 [custom-kernels](../plugins/custom-kernels/overview.md)）
