# MindSpeed EP/EFSDP + LlamaFactory CP 架构正交网格设计说明

在将 MindSpeed 的专家并行（EP）和专家分片（EFSDP）引擎与 LlamaFactory 的上下文并行（CP）结合时，如果网络通信拓扑构建不当，很容易在 Ascend NPU 的 HCCL 底层触发诸如 `error code 4`（`hcclCommInitRootInfoConfig` 失败）或 `error code 7` 的集合通信错误。

本方案通过重构 `DistributedInterface` 的 Mesh 缓存机制，彻底根除了该问题。

## 问题根因：手动通信域构建的灾难

在 PyTorch 和 HCCL 环境中，跨多个并行维度（如 DP、CP、EP、EFSDP）初始化 Process Group 是一件极具挑战性的事。
以往在 `LlamaFactory` 中，为了让 EP 和 CP 正交，代码采用嵌套的 `for` 循环搭配 rank 计算公式（如 `(edp_rank * ep_size + ep_rank) * cp_size + cp_rank`）手动调用 `torch.distributed.new_group` 来建组。

**这会导致以下严重问题：**
1. **初始化时序错乱**：在 NPU 环境下，多个进程按照跨步（stride）创建通信域时，极易引发底层 communicator id 的碰撞或惰性初始化顺序的不一致。
2. **DeviceMesh 重建冲突**：在外层 `new_group` 之后，`MindSpeedFSDP2Engine` 又在本地依据这些零散的 rank 列表二次创建 `DeviceMesh` 对象，这进一步破坏了分布式状态的一致性，导致首次 `all_gather` 操作立刻崩溃。

## 解决方案：原子化全局 `init_device_mesh`

为了保证 `MindSpeed` 接收到的 EP/EFSDP 网格绝对安全可靠，我们在 `llamafactory/v1/accelerator/interface.py` 中引入了**正交的全局多维网格初始化策略**：

1. **一键生成 4D 网格**：
   抛弃所有的 `new_group` 循环，直接调用 PyTorch 官方扩展：
   ```python
   expert_mesh = init_device_mesh(
       device_type="npu",
       mesh_shape=(edp_size, ep_fsdp_size, ep_size, cp_size),
       mesh_dim_names=("edp", "efsdp", "ep", "expert_cp"),
   )
   ```
   *注：由于 `mesh_shape` 中维度的排列顺序天然等价于原始复杂的 rank 公式，这使得 C++ 底层能一次性、原子化地构建出完全正交、且跨进程时序绝对安全的通信拓扑！*

2. **全局缓存与直接借用**：
   - LlamaFactory 提取出 `expert_mesh["ep"]` 和 `expert_mesh["efsdp"]` 作为缓存。
   - 必须**无条件**将 `expert_mesh["efsdp"]` 加入到 `DistributedInterface` 的 `_extra_groups` 中（即使 `ep_fsdp_size == 1`），以确保后续引擎调用 `get_world_size(Dim.EFSDP)` 计算梯度除数（gradient divide factor）时，能正确返回 1 而不是触发 KeyError 崩溃。
   - `MindSpeedFSDP2Engine` 在包装模型时（`prepare_model_ep`），直接通过 `get_expert_meshes()` 接口提取这些缓存好的纯净 Mesh。
   - 随后直接透传给 MindSpeed 的 `expert_parallelize_modules` 和 `expert_fully_shard_modules`。

## 调用链路

```text
[1] DistributedInterface.__init__() 
    └── _init_extra_groups()
        └── 调用 init_device_mesh(shape=(edp, efsdp, ep, cp))
        └── 缓存 EP Mesh 和 EFSDP Mesh

[2] MindSpeedFSDP2Engine.prepare_model_ep()
    └── self.dist_interface.get_expert_meshes()
    └── 将 EP Mesh 喂给 mindspeed.expert_parallelize_modules
    └── 将 EFSDP Mesh 喂给 mindspeed.expert_fully_shard_modules
```

## 效果
- **CP 与 EP 彻底解耦**：MindSpeed 引擎在处理 EP 和 EFSDP 集合通信时，不再感知外层 CP 带来的跨步碎片，它面对的是一个标准的连续网格。
- **HCCL 报错清零**：由于底层所有的 ProcessGroup 都是通过全局 `init_device_mesh` 原子化创建的，NPU 的集合通信（如首次 `all_gather_into_tensor`）将不再出现建链失败的问题。
