# 融合算子

LLaMA-Factory 通过 Kernel 插件系统管理融合算子实现。这些算子位于 `src/llamafactory/v1/plugins/model_plugins/kernels/ops` 目录下。

系统启动时，`scan_all_kernels` 函数会自动扫描该目录，注册所有与当前硬件匹配的算子。可以通过 `apply_default_kernels(model, include_kernels="auto")` 启用默认算子，也可以使用 `apply_kernel` 单独启用某个算子。

## 算子类型

当前支持的融合算子按功能分为以下几类：

| 类别 | 目录 | 功能 |
|------|------|------|
| MLP | `ops/mlp/` | SwiGLU、MoE 等融合算子 |
| RMS Norm | `ops/rms_norm/` | RMSNorm 融合算子 |
| RoPE | `ops/rope/` | 旋转位置编码融合算子 |

每个算子通过 `@register_kernel` 装饰器注册，声明其 `_kernel_id` 和 `_device`（支持的设备类型）。系统自动跳过与当前设备不匹配的算子。

## 替换机制

融合算子通过替换模型中对应模块的 `forward` 方法接入模型，不修改模型权重，保持数值一致性。以下以 NPU RMSNorm 为例说明替换方式：

```python
# 原始 forward
def forward(self, hidden_states):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states

# 融合算子替换后（以 NPU 为例）
def forward(self, hidden_states):
    return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]
```

各硬件后端的具体差异和可用列表见 [多后端支持](../../../multi-backend/index.md)。

## 扩展新算子

详见 [Kernel 插件 API](kernels_api.md)。
