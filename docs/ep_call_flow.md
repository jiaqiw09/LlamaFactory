# FSDP2 + Expert Parallelism 调用逻辑

## 文件结构

```
distributed/
├── fsdp2.py                    # FSDP2Engine 入口
├── expert_parallel.py          # ExpertParallel (ParallelStyle) + TokenPermute
└── ep_adapters/
    ├── __init__.py             # adapter 注册与获取
    ├── base.py                 # BaseEPAdapter 抽象基类
    └── qwen3_moe.py            # 唯一的 adapter，适配所有新版 HF MoE 模型
```

---

## 一、初始化阶段

### 1.1 入口：`FSDP2Engine.shard_model(model)`

```
shard_model(model)
│
├── 1. apply_ep(model)          ← 先做 EP 分片 + forward 替换
│
├── 2. prepare_model(model)     ← 再做 FSDP2 分片
│
└── 3. materialize_and_load()   ← 加载权重（meta device 场景）
```

### 1.2 `apply_ep(model)` 详细流程

```
apply_ep(model)
│
├── get_ep_adapter(model, "auto")
│   │
│   ├── _infer_adapter_name(model) → "qwen3_moe"
│   │   （新版 HF transformers 所有 MoE 模型结构统一，一个 adapter 覆盖全部）
│   │
│   └── return Qwen3MoeEPAdapter()
│
└── adapter.prepare_and_apply_ep(model, ep_mesh)
    │
    ├── 遍历 model.named_modules()
    │   │
    │   └── 对每个 module 调用 _find_moe_block(module)
    │       │
    │       └── 查找 module.mlp，检查是否同时有 .gate 和 .experts.gate_up_proj
    │           找到 → 返回 moe_block (即 SparseMoeBlock)
    │
    ├── 对每个 moe_block：
    │   │
    │   ├── _has_stacked_params(experts) 验证
    │   │   └── gate_up_proj 是 nn.Parameter 且 ndim == 3 ✓
    │   │
    │   ├── ① patch experts.forward → _ep_experts_forward
    │   │
    │   ├── ② patch moe_block.forward → _ep_moe_block_forward
    │   │
    │   └── ③ parallelize_module(experts, ep_mesh, ExpertParallel())
    │       │
    │       └── ExpertParallel._apply(experts, ep_mesh)
    │           │
    │           └── distribute_module(experts, ep_mesh, ...)
    │               │
    │               ├── partition_fn: 对 experts 的每个参数执行 Shard(0)
    │               │   gate_up_proj: [E, 2*I, H] → 按 expert 维度分片到各 EP rank
    │               │   down_proj:    [E, H, I]   → 按 expert 维度分片到各 EP rank
    │               │
    │               ├── 注册 input_fn  → _token_dispatch  (forward 前触发)
    │               │
    │               └── 注册 output_fn → _token_combine   (forward 后触发)
    │
    └── return patched 数量
```

### 1.3 `prepare_model(model)` — FSDP2 分片

```
prepare_model(model)
│
├── 遍历 model.named_modules()
│   │
│   └── 对每个 transformer layer：
│       │
│       ├── adapter.get_expert_module(layer) → layer.mlp.experts
│       │
│       ├── 如果有 experts 且 efsdp_mesh 存在：
│       │   └── fully_shard(experts, mesh=efsdp_mesh)   ← expert 用 EFSDP mesh
│       │       （EFSDP = EP 内部的 FSDP，只在同 EP group 内 all-gather/reduce-scatter）
│       │
│       └── fully_shard(layer, mesh=fsdp_mesh)          ← 整层用全局 FSDP mesh
│
└── fully_shard(model, mesh=fsdp_mesh)                  ← 根模型
```

**Mesh 关系示意**（以 8 GPU, ep_size=4 为例）：

```
全局 8 GPU:  [0, 1, 2, 3, 4, 5, 6, 7]

fsdp_mesh:   [0, 1, 2, 3, 4, 5, 6, 7]     ← 全局 FSDP

ep_mesh:     [0, 1, 2, 3] | [4, 5, 6, 7]   ← EP group (每组 4 个 expert rank)

efsdp_mesh:  [0, 4] | [1, 5] | [2, 6] | [3, 7]  ← 跨 EP group 的 FSDP
             （同 expert 子集的 rank 之间做 reduce-scatter）
```

---

## 二、训练阶段 — Forward Pass

当一个 token batch 流过 MoE 层时：

```
hidden_states: [batch_size, seq_len, hidden_dim]
        │
        ▼
┌─────────────────────────────────────────────────────┐
│           _ep_moe_block_forward (patched)            │
│                                                      │
│  ① Router                                            │
│     router_logits, routing_weights, selected_experts │
│         = self.gate(hidden_states)                   │
│                                                      │
│  ② 统计每个 expert 分到多少 token                      │
│     num_tokens_per_expert = histc(selected_experts)  │
│     → [E] 例如 [5, 3, 0, 8, 2, 6, 1, 7]             │
│                                                      │
│  ③ 按 expert 排序 token                               │
│     sorted_order = argsort(flat_expert_indices)      │
│     routed_input = hidden_states[sorted_token_idx]   │
│                                                      │
│  ④ 调用 self.experts(routed_input, num_tokens_per_expert)
│     │                                                │
│     ▼                                                │
│  ┌──────────────────────────────────────────────┐    │
│  │  ExpertParallel hooks (distribute_module)     │    │
│  │                                               │    │
│  │  input_fn: _token_dispatch                    │    │
│  │  ┌────────────────────────────────────────┐   │    │
│  │  │ DefaultTokenPermuteBackend.permute()    │   │    │
│  │  │                                         │   │    │
│  │  │ a) all_to_all: 交换 token 计数           │   │    │
│  │  │    每个 rank 告诉其他 rank：              │   │    │
│  │  │    "我有 N 个 token 要发给你的 expert"    │   │    │
│  │  │                                         │   │    │
│  │  │ b) all_to_all: 交换实际 token 数据       │   │    │
│  │  │    Rank0 的 Expert2 的 token → Rank2     │   │    │
│  │  │    Rank2 的 Expert0 的 token → Rank0     │   │    │
│  │  │                                         │   │    │
│  │  │ c) 重排序：按 local expert 分组           │   │    │
│  │  │    收到的 token 来自不同 rank，           │   │    │
│  │  │    重排成 [expert0的token, expert1的...]  │   │    │
│  │  │                                         │   │    │
│  │  │ → 返回 (local_tokens, local_counts,     │   │    │
│  │  │         state{splits, permutation})     │   │    │
│  │  └────────────────────────────────────────┘   │    │
│  │                                               │    │
│  │  forward: _ep_experts_forward                 │    │
│  │  ┌────────────────────────────────────────┐   │    │
│  │  │ 对每个 local expert i:                  │   │    │
│  │  │   x_e = tokens[offset : offset+count]  │   │    │
│  │  │   gate, up = Linear(x_e, gate_up[i])   │   │    │
│  │  │   out = Linear(SiLU(gate) * up, down[i])│   │    │
│  │  │   → cat(outputs)                        │   │    │
│  │  └────────────────────────────────────────┘   │    │
│  │                                               │    │
│  │  output_fn: _token_combine                    │    │
│  │  ┌────────────────────────────────────────┐   │    │
│  │  │ DefaultTokenPermuteBackend.unpermute()  │   │    │
│  │  │                                         │   │    │
│  │  │ a) 反向重排序：恢复 all-to-all 接收顺序  │   │    │
│  │  │                                         │   │    │
│  │  │ b) all_to_all: 把计算结果发回源 rank     │   │    │
│  │  │    splits 反转（input↔output）           │   │    │
│  │  │                                         │   │    │
│  │  │ → 每个 rank 拿回自己 token 的专家输出    │   │    │
│  │  └────────────────────────────────────────┘   │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  ⑤ 加权聚合                                          │
│     weighted_output = routed_output * routing_weights│
│     scatter_add_ 到 final_hidden_states              │
│     （top-k 路由下同一 token 的多个 expert 输出求和）  │
│                                                      │
│  → return [batch_size, seq_len, hidden_dim]          │
└─────────────────────────────────────────────────────┘
```

---

## 三、Token Dispatch/Combine 数据流示例

以 `ep_size=2, num_experts=4, top_k=2` 为例，每个 rank 持有 2 个 expert：

```
Rank 0 持有: Expert 0, Expert 1
Rank 1 持有: Expert 2, Expert 3

Rank 0 的 router 输出:
  Token A → Expert 0, Expert 3  (weights: 0.6, 0.4)
  Token B → Expert 1, Expert 2  (weights: 0.7, 0.3)

num_tokens_per_expert (全局视角，Rank 0):
  [1, 1, 1, 1]  →  E0:1个, E1:1个, E2:1个, E3:1个

排序后:
  routed_input = [A, B, B, A]  (按 expert 0,1,2,3 顺序)
```

**Dispatch (permute):**

```
Rank 0:
  要发给自己 (Expert 0,1): [A, B]     → input_split[0] = 2
  要发给 Rank1 (Expert 2,3): [B, A]   → input_split[1] = 2

all-to-all 后 Rank 0 收到:
  来自自己: [A, B]       → 给 Expert 0,1 处理
  来自 Rank1: [...]      → 给 Expert 0,1 处理
  重排后按 expert 分组: [Expert0 的所有 token | Expert1 的所有 token]

local_tokens_per_expert = [来自所有 rank 的 E0 token 数, E1 token 数]
```

**Local Compute:**

```
对 Expert 0: gate_up_proj[0], down_proj[0] → SwiGLU
对 Expert 1: gate_up_proj[1], down_proj[1] → SwiGLU
```

**Combine (unpermute):**

```
反向重排 → 反向 all-to-all → 每个 rank 拿回自己 token 的结果
Rank 0 拿回: A 过 Expert 0 的输出, B 过 Expert 1 的输出,
             B 过 Expert 2 的输出, A 过 Expert 3 的输出

scatter_add_:
  final[A] = 0.6 * Expert0(A) + 0.4 * Expert3(A)
  final[B] = 0.7 * Expert1(B) + 0.3 * Expert2(B)
```

---

## 四、关键类/函数速查

| 组件 | 文件 | 作用 |
|------|------|------|
| `FSDP2Engine.shard_model` | `fsdp2.py` | 总入口：EP → FSDP → 加载权重 |
| `FSDP2Engine.apply_ep` | `fsdp2.py` | 获取 adapter，调用 `prepare_and_apply_ep` |
| `FSDP2Engine.prepare_model` | `fsdp2.py` | 按层 `fully_shard`，expert 用 `efsdp_mesh` |
| `get_ep_adapter` | `ep_adapters/__init__.py` | 从注册表获取 adapter 实例 |
| `Qwen3MoeEPAdapter` | `ep_adapters/qwen3_moe.py` | 定位 MoE block，patch forward，apply EP |
| `_ep_moe_block_forward` | `ep_adapters/qwen3_moe.py` | 替换 MoE block forward：route → sort → experts → scatter |
| `_ep_experts_forward` | `ep_adapters/qwen3_moe.py` | 替换 experts forward：按 count 切分 → SwiGLU → cat |
| `ExpertParallel` | `expert_parallel.py` | `ParallelStyle` 子类：partition + dispatch/combine hooks |
| `DefaultTokenPermuteBackend` | `expert_parallel.py` | all-to-all token 交换 + 重排逻辑 |

---

## 五、调用顺序总结

```
shard_model
  └→ apply_ep
       └→ get_ep_adapter → Qwen3MoeEPAdapter
       └→ adapter.prepare_and_apply_ep
            ├→ patch experts.forward     → _ep_experts_forward
            ├→ patch moe_block.forward   → _ep_moe_block_forward
            └→ parallelize_module(experts, ep_mesh, ExpertParallel())
                 └→ distribute_module
                      ├→ _partition_fn: Shard(0) 分片 gate_up_proj, down_proj
                      ├→ 注册 input_fn:  _token_dispatch  → permute (all-to-all)
                      └→ 注册 output_fn: _token_combine   → unpermute (all-to-all)
  └→ prepare_model
       ├→ fully_shard(experts, efsdp_mesh)   # expert 局部 FSDP
       ├→ fully_shard(layer, fsdp_mesh)      # transformer 层 FSDP
       └→ fully_shard(model, fsdp_mesh)      # 根模型 FSDP

训练 forward:
  hidden_states → _ep_moe_block_forward
    ├→ gate(hidden_states) → routing_weights, selected_experts
    ├→ histc → num_tokens_per_expert
    ├→ argsort → sorted routed_input
    ├→ self.experts(routed_input, num_tokens_per_expert)
    │    ├→ [hook] _token_dispatch: all-to-all 发送 token 到目标 rank
    │    ├→ _ep_experts_forward: 本地 expert 计算 (SwiGLU)
    │    └→ [hook] _token_combine: all-to-all 接收结果回源 rank
    └→ scatter_add_(weighted outputs) → final_hidden_states
```
