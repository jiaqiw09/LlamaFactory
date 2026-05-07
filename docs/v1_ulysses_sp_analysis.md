# LlamaFactory v1 ULySSes Sequence Parallelism (SP) 实现逻辑拆解

## 1. 概述

ULySSes 是一种**序列并行**策略：将输入序列沿**注意力头维度**切分到多个 GPU，每个 GPU 只持有部分注意力头的 Q/K/V，通过 **All-to-All 通信**在头维度和序列维度之间转换，使每个 GPU 在少量注意力头上完成**完整序列**的注意力计算。

核心思路：**头维度与序列维度的交换** — 每个 GPU 放弃部分头，换取完整序列长度，保证注意力计算的正确性。

---

## 2. 整体流程

```
初始化: 设置进程组 → 校验头数整除性 → Monkey-patch Flash Attention
                              │
                              ▼
训练循环: 切分 → 通信 → 计算 → 汇聚 → 损失 → 反向传播
         │       │       │       │       │        │
         │       │       │       │       │        └─ 梯度范数校正
         │       │       │       │       └─ All-gather 标签/权重 + 加权交叉熵
         │       │       │       └─ All-to-All: scatter序列维, gather头维
         │       │       └─ 本地 Flash Attention (完整序列, 部分头)
         │       └─ All-to-All: scatter头维, gather序列维
         └─ padding_and_split_data 沿序列维度切分
```

---

## 3. 运行示例设定

以 Qwen3-0.6B 为例，4 张卡，FSDP2，cp_size=2：

```
模型: Qwen3-0.6B
  num_attention_heads = 16
  num_key_value_heads = 8  (GQA)
  head_size = 64
  hidden_size = 1024

硬件: 4 × GPU
分布式: FSDP2, cp_size=2, dp_size=2

DeviceMesh: (dp=2, cp=2)
  DP组0: GPU0, GPU1  (cp_rank=0, cp_rank=1)
  DP组1: GPU2, GPU3  (cp_rank=0, cp_rank=1)

输入: batch_size=2, seq_len=8
```

> 下文以 **DP组0（GPU0 + GPU1）** 为例，DP组1 完全独立执行相同逻辑。

---

## 4. 切分阶段

训练数据在进入模型前，沿**序列维度**切分到各 CP rank。

**流程**：
1. All-gather 收集所有 rank 的序列长度，计算最大长度
2. 对齐到 `cp_size` 的倍数（不足则填充）
3. 填充规则：`labels` → `-100`，`loss_weights` → `0.0`，其他 → `0`
4. 沿最后一维切分为 `cp_size` 份，每个 rank 取对应 chunk

**Shape 变化**：

```
原始输入 (每个 rank 收到相同的完整数据):
  input_ids:     (2, 8)    例: [[1,2,3,4,5,6,7,8], [9,10,11,12,13,14,15,16]]
  labels:        (2, 8)    例: [[1,2,3,4,5,6,7,8], [9,10,11,12,13,14,15,16]]
  attention_mask: (2, 8)   全1
  loss_weights:  (2, 8)    全1.0
  position_ids:  (2, 8)

  ↓ padding_and_split_data (seq_len=8 已能被 cp_size=2 整除，无需额外填充)

GPU0 (cp_rank=0): 取 chunk[0]，序列前半
  input_ids:     (2, 4)    [[1,2,3,4], [9,10,11,12]]
  labels:        (2, 4)    [[1,2,3,4], [9,10,11,12]]
  attention_mask: (2, 4)
  loss_weights:  (2, 4)
  position_ids:  (2, 4)

GPU1 (cp_rank=1): 取 chunk[1]，序列后半
  input_ids:     (2, 4)    [[5,6,7,8], [13,14,15,16]]
  labels:        (2, 4)    [[5,6,7,8], [13,14,15,16]]
  attention_mask: (2, 4)
  loss_weights:  (2, 4)
  position_ids:  (2, 4)
```

---

## 5. 通信阶段

### 5.1 All-to-All 通信原语

底层封装为自定义 `autograd.Function`（`SeqAllToAll4D`），支持自动求导：

| 方向 | scatter_dim | gather_dim | 含义 |
|------|-------------|------------|------|
| Forward | 指定维度 | 指定维度 | 沿 scatter_dim 切分 → all_to_all → 沿 gather_dim 拼接 |
| Backward | gather_dim | scatter_dim | **维度互换**，即通信的转置 |

### 5.2 Attention 内的两次 All-to-All

每层 Attention 包含两次 All-to-All，以 Qwen3-0.6B 为例：

**第一次 All-to-All（scatter头维, gather序列维）**：

```
Embedding + QKV投影后:
  GPU0: Q=(2,4,16,64)  K=(2,4,8,64)  V=(2,4,8,64)   — 部分序列, 全部头
  GPU1: Q=(2,4,16,64)  K=(2,4,8,64)  V=(2,4,8,64)   — 部分序列, 全部头

  ↓ All-to-All (scatter_dim=2, gather_dim=1)
  每个 rank 将头维度切2份，交换后沿序列维度拼接

  GPU0 发出头[0:8] 给自己, 头[8:16] 给 GPU1
  GPU1 发出头[0:8] 给 GPU0, 头[8:16] 给自己

  结果:
  GPU0: Q'=(2,8,8,64)  K'=(2,8,4,64)  V'=(2,8,4,64)   — 完整序列, 半数头
  GPU1: Q'=(2,8,8,64)  K'=(2,8,4,64)  V'=(2,8,4,64)   — 完整序列, 半数头

  注意: GQA 下 K/V 的头数从 8 变为 4
```

**第二次 All-to-All（scatter序列维, gather头维）**：

```
Flash Attention 计算后:
  GPU0: context=(2,8,8,64)   — 完整序列, 半数头
  GPU1: context=(2,8,8,64)   — 完整序列, 半数头

  ↓ All-to-All (scatter_dim=1, gather_dim=2)
  每个 rank 将序列维度切2份，交换后沿头维度拼接

  GPU0 发出 seq[0:4] 给自己, seq[4:8] 给 GPU1
  GPU1 发出 seq[0:4] 给 GPU0, seq[4:8] 给自己

  结果:
  GPU0: output=(2,4,16,64)   — 部分序列, 全部头
  GPU1: output=(2,4,16,64)   — 部分序列, 全部头
```

### 5.3 Attention Mask 的 All-Gather

```
  GPU0: mask=(2,4)    GPU1: mask=(2,4)
  ↓ All-gather 沿序列维度拼接
  两个 rank 都得到: global_mask=(2,8)
```

---

## 6. 计算阶段

在两次 All-to-All 之间，每个 rank 在**完整序列 + 部分注意力头**上执行本地 Flash Attention：

```
GPU0 (持有头 0~7):
  Q'=(2,8,8,64)   K'=(2,8,4,64)   V'=(2,8,4,64)
  ↓ Flash Attention (is_causal=True, softmax_scale=1/√64)
  context=(2,8,8,64)

GPU1 (持有头 8~15):
  Q'=(2,8,8,64)   K'=(2,8,4,64)   V'=(2,8,4,64)
  ↓ Flash Attention (is_causal=True, softmax_scale=1/√64)
  context=(2,8,8,64)
```

关键：每个注意力头看到**完整序列**，因此注意力计算结果与单卡等价。

---

## 7. 汇聚阶段

Attention 输出经过第二次 All-to-All 后恢复为**部分序列、全部头**的格式，与模型其他层格式一致：

```
  output=(2,4,16,64)
  ↓ reshape: (2,4,1024)
  ↓ 残差连接 + MLP + LayerNorm (与单卡完全相同)
  ↓ 下一层继续相同的 Attention 流程
```

---

## 8. 数据阶段（损失计算）

损失计算需要跨 rank 对齐标签，因为 next-token prediction 的标签可能跨越 rank 边界。

**Shape 变化详解**：

```
=== Step 1: 前向推理 ===

  GPU0: model(split_inputs) → logits=(2,4,151936)   — 只有 seq[0:4] 的 logits
  GPU1: model(split_inputs) → logits=(2,4,151936)   — 只有 seq[4:8] 的 logits

=== Step 2: 标签对齐 ===

  GPU0: labels=(2,4)  [[1,2,3,4], [9,10,11,12]]
  GPU1: labels=(2,4)  [[5,6,7,8], [13,14,15,16]]

  ↓ All-gather labels
  两个 rank 都得到: global_labels=(2,8)
    [[1,2,3,4,5,6,7,8], [9,10,11,12,13,14,15,16]]

  ↓ shift: labels[..., 1:]
    [[2,3,4,5,6,7,8], [10,11,12,13,14,15,16]]    shape=(2,7)

  ↓ pad 末尾: F.pad(..., (0,1), value=-100)
    [[2,3,4,5,6,7,8,-100], [10,11,12,13,14,15,16,-100]]    shape=(2,8)

  ↓ 重新切分: chunk(shift_labels, cp_size=2, dim=1)
  GPU0 (cp_rank=0): [[2,3,4,5], [10,11,12,13]]    shape=(2,4)
  GPU1 (cp_rank=1): [[6,7,8,-100], [14,15,16,-100]]  shape=(2,4)

  注意: GPU0 的 shift_labels[0] = [2,3,4,5]
        对应 GPU0 的 logits 预测位置 0,1,2,3 的下一个token
        其中位置3的预测目标=5，这个"5"原本在 GPU1 的 labels 中！

=== Step 3: 损失权重对齐 ===

  GPU0: loss_weights=(2,4)  全1.0
  GPU1: loss_weights=(2,4)  全1.0

  ↓ All-gather loss_weights
  global_loss_weights=(2,8)  全1.0

  ↓ shift: loss_weights[..., 1:]
  shift_loss_weights=(2,7)  全1.0

=== Step 4: 计算损失 ===

  ↓ cross_entropy (逐 token)
  GPU0: logits=(2,4,151936) → flatten → (8, 151936)
        shift_labels=(2,4) → flatten → (8,)
        log_probs = -cross_entropy(...) → reshape → (2,4)

  GPU1: 同理 → log_probs=(2,4)

  ↓ All-gather log_probs
  global_log_probs=(2,8)

  ↓ shift: log_probs[..., :-1]  去掉最后一个位置
  shift_log_probs=(2,7)

  ↓ 加权求和
  loss = (-shift_log_probs * shift_loss_weights).sum() / (shift_loss_weights.sum() + 1e-6)
```

**为什么需要 All-gather 标签？** 以上面 GPU0 为例，位置 3 的 token 是 `4`，它要预测的下一个 token 是 `5`，而 `5` 原本在 GPU1 的 labels 中。必须跨 rank 拼接后统一 shift 再切分，才能正确对齐。

---

## 9. Trainer 阶段

### 9.1 初始化

当 `cp_size > 1` 时自动激活 SP：
1. 校验模型注意力头数能被 `cp_size` 整除
2. 若注意力实现非 `flash_attention_2`，自动切换并警告
3. Monkey-patch `transformers` 中所有 `_flash_attention_forward`，替换为 ULySSes 版本

### 9.2 训练循环

- **损失路由**：SP 激活时，损失计算走 `sequence_parallel_loss()` 插件而非普通 `compute_loss()`
- **损失缩放**：`loss = loss * mini_step_valid_tokens * dp_size / (step_valid_tokens + 1e-6)`
  - dp_size=2 时，loss 会被乘以 2（因为 FSDP2 使用 mean reduction，需要补偿 DP 维度）

---

## 10. Backward 阶段

### 10.1 梯度通信

`SeqAllToAll4D` 的 backward 自动将 scatter/gather 维度互换，梯度沿相反方向流动：

```
Forward:  scatter=2, gather=1  (头→序列)
Backward: scatter=1, gather=2  (序列→头，自动反转)

梯度流:
  output_grad=(2,4,16,64)
    ↓ All-to-All backward (scatter=1, gather=2)
  context_grad=(2,8,8,64)      — 完整序列, 半数头的梯度
    ↓ Flash Attention backward
  Q'_grad, K'_grad, V'_grad=(2,8,8/4,64)
    ↓ All-to-All backward (scatter=2, gather=1)
  Q_grad, K_grad, V_grad=(2,4,16/8,64)  — 恢复为部分序列, 全部头
```

### 10.2 梯度范数校正

每个 rank 只计算了部分序列的梯度，局部梯度范数需要在 CP 组内校正：

```
GPU0: local_grad_norm = 3.2
GPU1: local_grad_norm = 2.8

  ↓ grad_norm²
  GPU0: 10.24    GPU1: 7.84

  ↓ All-Reduce (SUM, dim=CP)
  两个 rank 都得到: 10.24 + 7.84 = 18.08

  ↓ grad_norm^0.5
  两个 rank 都得到: √18.08 ≈ 4.25

  这就是正确的全局梯度范数: √(3.2² + 2.8²) ≈ 4.25
```

---

## 11. 通信模式总结

| 阶段 | 通信操作 | 维度 | 数据 | 示例 Shape |
|------|----------|------|------|-----------|
| Attention Forward | All-to-All | scatter=头, gather=序列 | Q, K, V | (2,4,16,64)→(2,8,8,64) |
| Attention Forward | All-Gather | 序列 | attention_mask | (2,4)→(2,8) |
| Attention Forward | All-to-All | scatter=序列, gather=头 | context_layer | (2,8,8,64)→(2,4,16,64) |
| Attention Backward | All-to-All | 自动反转维度 | 梯度 | 同 forward 反向 |
| 数据切分 | All-Gather | 序列长度 | 各 rank 的序列长度 | 标量 |
| 损失计算 | All-Gather | 序列 | labels | (2,4)→(2,8) |
| 损失计算 | All-Gather | 序列 | loss_weights | (2,4)→(2,8) |
| 损失计算 | All-Gather | 序列 | log_probs | (2,4)→(2,8) |
| 梯度范数校正 | All-Reduce (SUM) | CP | grad_norm² | 标量 |
