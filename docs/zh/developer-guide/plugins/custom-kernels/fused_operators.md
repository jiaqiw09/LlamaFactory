# 融合算子

`kernels/ops/` 目录下每个子目录对应一类算子。`scan_all_kernels` 在启动时扫描整个目录，把当前设备能用的实现注册进 `Registry`。

代码位置：`src/llamafactory/v1/plugins/model_plugins/kernels/ops/`。

## 算子分类

| 子目录 | 替换目标 | 说明 |
|--------|----------|------|
| `mlp/` | 各种 `*MLP` 模块 | SwiGLU、Fused MoE |
| `rms_norm/` | 各种 `*RMSNorm` 模块 | 融合 RMSNorm |
| `rope/` | 旋转位置编码 | 融合 RoPE |

## 仓库内置 kernel

| Kernel ID | 文件 | 设备 | 替换内容 |
|-----------|------|------|---------|
| `npu_fused_swiglu` | `mlp/npu_swiglu.py` | NPU | 各种 LLM `*MLP` 的 SwiGLU forward（Qwen / Llama / GLM / Phi / Gemma / DeepSeek 等） |
| `npu_fused_moe` | `mlp/npu_fused_moe.py` | NPU | MoE 融合（含 grouped GEMM） |
| `npu_fused_rmsnorm` | `rms_norm/npu_rms_norm.py` | NPU | 各种 `*RMSNorm` forward |
| `npu_fused_rope` | `rope/npu_rope.py` | NPU | 旋转位置编码 forward |

具体支持的模块名见各 kernel 文件中的 `expect_modules`。

## 替换原则

替换只发生在 forward 上，不动权重：

```python
def npu_swiglu_forward(self, hidden_state):
    return self.down_proj(
        torch_npu.npu_swiglu(
            torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1),
            dim=-1,
        )
    )

# apply 内部
module.forward = types.MethodType(npu_swiglu_forward, module)
```

因此：

- 同一个模型在 NPU 和 GPU 上保存出来的权重格式一致
- kernel 替换仅在当前进程内生效，关闭 `kernel_config` 即回退默认实现

## 添加新算子

1. 在合适的子目录下新建 `<device>_<feature>.py`
2. 写好 forward 函数
3. 用 `@register_kernel` 装饰一个继承 `BaseKernel` 的类，设置 `_kernel_id` 和 `_device`
4. 在 `apply` 里遍历 `model.named_modules()`，按类名替换 forward

完整接口与最小例子见 [kernels_api](kernels_api.md)。
