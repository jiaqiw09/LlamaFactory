# NPU（Ascend）

Ascend NPU 后端的差异说明。默认文档以 GPU 为准，以下仅记录 NPU 的差异部分。

## 软件依赖

| 必需项 | 至少 | 推荐 |
|:---:|---|---|
| torch-npu | 2.7.1 | 2.7.1 |

## 分布式训练

- FSDP2 和 Context Parallel 在 NPU 上支持
- DeepSpeed 在 NPU 上不支持

## 融合算子

NPU 提供以下专属融合算子，通过 `kernel_config` 配置启用：

| Kernel ID | 功能 | 说明 |
|-----------|------|------|
| `npu_fused_rmsnorm` | RMSNorm 融合算子 | 通过 `torch_npu.npu_rms_norm` 接口实现 |
| `npu_fused_swiglu` | SwiGLU 融合算子 | 通过 `torch_npu.npu_swiglu` 接口实现 |
| `npu_fused_rope` | RoPE 融合算子 | 通过 `torch_npu.npu_rotary_mul` 接口实现 |
| `npu_fused_moe` | MoE 融合算子 | 通过 `torch_npu.npu_grouped_matmul` 接口实现 |

### npu_fused_rmsnorm

RMSNorm 融合算子将 bias、residual 等操作合并执行，减少显存访问次数。通过替换模型中 RMSNorm 类的 `forward` 方法启用：

```python
def _npu_rms_forward(self, hidden_states):
    return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]
```

### npu_fused_swiglu

SwiGLU 融合算子将分割、激活、矩阵乘等多个操作融合为单一硬件指令：

```python
def _npu_swiglu_forward(self, hidden_state):
    return self.down_proj(
        torch_npu.npu_swiglu(torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1), dim=-1)
    )
```

### npu_fused_rope

RoPE 融合算子将旋转位置编码的计算流程合并为单个硬件优化算子：

```python
def _apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed
```

### npu_fused_moe

MoE 融合算子利用 GMM（Grouped Matrix Multiplication）技术，支持在单个硬件指令内并行处理多组矩阵乘法：

```python
def _npu_moe_forward(self, hidden_states, routing_weights, router_indices):
    permuted_states, row_map = torch_npu.npu_moe_token_permute(hidden_states, router_indices)
    tokens_per_expert = torch.histc(router_indices, bins=self.num_experts, min=0, max=self.num_experts)
    inter_states = torch_npu.npu_grouped_matmul(permuted_states, self.gate_up_proj_weights, split_sizes=tokens_per_expert, ...)
    inter_states = torch_npu.npu_swiglu(inter_states)
    output = torch_npu.npu_grouped_matmul(inter_states, self.down_proj_weights, split_sizes=tokens_per_expert, ...)
    return torch_npu.npu_moe_token_unpermute(output, row_map, routing_weights)
```

## 配置示例

```yaml
kernel_config:
  name: auto
  include_kernels: auto
```

开发者文档见 [Kernel 插件 API](../../developer-guide/plugins/custom-kernels/kernels_api.md)。
