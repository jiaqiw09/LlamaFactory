# LlamaFactory / VeOmni 专家并行 (Expert Parallelism) 架构解析

在 LlamaFactory 的 FSDP2 架构中，Expert Parallelism (EP) 的实现是一套非常优雅的分布式解耦设计。它通过利用 PyTorch 最新的 `torch.distributed.tensor.parallel` 接口和底层设备网格（Device Mesh），实现了对标准 HuggingFace 模型的无缝并行化。由于采用的是 2D 并行机制（EP + ZeRO-3），其可以在显存受限的大规模集群上高效训练含上百个专家的 MoE 模型。

以下是代码内的完整运行链路及核心实现拆解：

---

## 1. 拓扑初始化：多维度 Device Mesh 的构建
**代码位置**：`llamafactory/v1/accelerator/interface.py`

在初始化阶段，系统会根据环境变量或传入的 `dp_size` 和 `ep_size`，将全局 GPU/NPU 集群切分成多维度的通讯网格。对于纯 Dense 模型，只有传统的 Data Parallel (DP) 或 Model Parallel (MP) Mesh，而对于 MoE 模型（`ep_size > 1`）则会额外切分出一个专属的三维稀疏网格 (Sparse Mesh)`(cp_size, efsdp_size, ep_size)`：
- **`EP Mesh` (Expert Parallelism)**：该维度专职处理 Token 的 All-to-All 路由。网格内的各卡会共同分担模型所有 Expert 的计算。
- **`EFSDP Mesh` (Expert FSDP)**：该维度负责解决单张卡上的 Local Expert 参数太大的问题。它会在剩余的数据并行维度上继续运用 FSDP (ZeRO-3) 对 Local Expert 进行显存切片。

---

## 2. 算子动刀：EP Adapters (Monkey Patching)
**代码位置**：`llamafactory/v1/plugins/trainer_plugins/distributed/ep_adapters/`

由于 HuggingFace 标准的 MoE 模型（例如 Qwen2-MoE、Qwen3-MoE）在单卡设计时，内部是一个遍历所有专家的 `for` 循环计算，这天然会破坏分布式并行的独立计算并导致重算。因此系统通过 EP Adapter 做了两级修改，也就是常说的 Monkey Patching（运行态替换）：

1. **重写 Transformer Block (`_ep_moe_block_forward`)**：
   - 将原生的 `for idx, expert in enumerate(experts)` 这种慢速循环分支，替换为了**向量化的按组切分操作**。
   - 截断流量入口：模型先通过顶层的 Router 拿到分类 logits，得到 `selected_experts` 和权重，通过直方图 `torch.histc` 算出每个专家分到了几个 Token (`num_tokens_per_expert`)。
   - 输入重排序：利用 `torch.argsort` 对待输入的 Token 根据其派发的专家下标进行全局 `Permute` 排序，确保在显存里同一专家的 Token 都是物理连续的。

2. **重写 Experts 前向逻辑 (`_ep_experts_forward`)**：
   - 经过重排序的 Tokens 被送入真正的多层感知机。
   - 去掉原先加载几百个权重的逻辑，变为**只计算 Local Experts (即被当前进程认领的专家池)**。
   - 仅利用 `F.linear` 等切块运算 (chunk) 实现并行打分。

---

## 3. 并行魔法：ExpertParallel 挂载通信钩子
**代码位置**：`llamafactory/v1/plugins/trainer_plugins/distributed/expert_parallel.py`

在 Adapter 中，最核心的调用是 `parallelize_module(experts, ep_mesh, ExpertParallel())`。利用 PyTorch 2.0+ 的 DTensor 和 ParallelStyle 特性，该模块在完全隐式的前提下把节点塞入了分布式 Hook：

- **物理显存分片 (`_partition_fn`)**: 
  - 将原生诸如 `gate_up_proj` 这种 `[num_experts, in_dim, out_dim]` 维度的大权重张量，顺着 `dim=0`（即 Expert 堆叠维度）强行物理切分（`Shard(0)`）。例如 16 个专家跑在 16 卡 EP 上，这步操作执行后单机就只保留了 `1` 个专家的真实权重参数区。
- **前向分发分桶 (`_token_dispatch`)**: 
  - 通过 `all_to_all_single_autograd` 发送。在计算真实激活函数前，所有的 Token 根据 `output_splits` 与 `input_splits` 在网络通信层级内被打散，当前卡只留下去自己 Local Expert 的数据，其他数据均以全对全交换流向对应卡。
- **后向收集 (`_token_combine`)**: 
  - 计算完后，再次反向套用 `all_to_all_single_autograd` 将算完的隐层状态原路退回发起节点。
  - 根据上文传入的 `combine_permutation` 再次使用 `index_select` 洗牌，把被按字典序聚集打乱的时序还原为语言模型原始序列。

---

## 4. 显存极致压缩：与 FSDP2 的二维融合
**代码位置**：`llamafactory/v1/plugins/trainer_plugins/distributed/fsdp2.py`

如果仅有 EP，当面临拥有巨型单体 Expert 或 Expert 整体量相比切分卡度较小的情况时还是会 OOM（比如用 128 卡跑 64 专家的参数，每名专家 1 张卡，还有 64 张卡吃灰）。所以在 FSDP2 初始化环节完成了二维收敛：

```python
# 第一维：被 Adapter 和 ParallelStyle 切分开通信的 Expert 层，利用 EFSDP 再次显存降维 (ZeRO-3)
fully_shard(experts, mesh=self.efsdp_mesh, ...)

# 由于 Loss 聚合是在外层的 world_size（Global Batch Size），而反向传播在组内 efsdp_size 收集
# 必须显式重置降低比例以纠正 Loss Scale 和更新步长：
experts.set_gradient_divide_factor(float(self.world_size))

# 第二维：外层模块（非 MoE 比如 self-attention）用标准的全局 DP 网格进行 FSDP 切分
fully_shard(module, mesh=self.fsdp_mesh, ...)
```

这种两层嵌套的设计彻底把非门控网络与门控网络的运算池隔离出来，用最小的通信开销拿到了最大的吞吐量，是一套企业级的先进多模态大模型并行方案规范。
