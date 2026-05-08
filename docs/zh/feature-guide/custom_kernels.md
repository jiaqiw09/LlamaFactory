# 自定义算子

`kernel_config` 控制是否启用注册到 `KernelPlugin` 的算子替换。哪些算子可用取决于当前硬件后端 —— 启动时只有与当前设备匹配的 kernel 才会被注册。

参数细节见 [KernelConfig](../parameter-reference/kernel_config.md)，扩展接口见 [Kernel API](../developer-guide/plugins/custom-kernels/kernels_api.md)。

## 启用方式

```yaml
kernel_config:
  name: auto
  include_kernels: auto       # 全部启用
```

`include_kernels` 取值：

| 值 | 行为 |
|----|------|
| `null` / `false` / 空串 | 不启用任何 kernel |
| `auto` 或 `true` | 启用所有当前可用的默认 kernel |
| `"id1,id2"` | 仅启用列表内 kernel id |

显式指定不存在或当前设备不可用的 id 会直接 `ValueError`。

## GPU

GPU 默认依赖 Flash Attention 2，由 `flash-attn` 包提供：

```bash
pip install flash-attn
```

模型加载时 `transformers` 会自动检测使用，无需 `kernel_config`。`kernel_config: auto` 此时通常不会启用任何额外算子，除非有为 CUDA 注册的 kernel。

## 其它硬件后端

NPU / XPU / 其它后端的可用 kernel 列表与依赖见 [多后端支持](../multi-backend/index.md)。其中 NPU 当前已注册的 kernel 包括：

- `npu_fused_swiglu`
- `npu_fused_moe`
- `npu_fused_rmsnorm`
- `npu_fused_rope`

支持范围与触发条件见 [fused_operators](../developer-guide/plugins/custom-kernels/fused_operators.md)。

## 注意

- Kernel 只替换 forward，不动权重 → 保存出去的模型与未启用 kernel 时一致
- 同一进程多次调用没有副作用：`apply_default_kernels` 会跳过已经注册过的 kernel
- 关闭 `kernel_config` 即可回退默认实现，方便排查算子相关问题
