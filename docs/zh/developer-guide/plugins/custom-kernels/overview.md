# 自定义算子总览

`KernelPlugin` 把"针对特定硬件的高性能 forward 实现"统一管理起来。它的设计目的是：在不改模型权重和数值语义的前提下，根据当前设备替换若干模块的 `forward`，从而让模型自动跑在最快的实现上。

代码位置：`src/llamafactory/v1/plugins/model_plugins/kernels/`。

## 基本概念

- **Kernel**：一个继承自 `BaseKernel` 的类，绑定一个 `_kernel_id`（字符串）和一个 `_device`（`DeviceType`）；`apply(model=...)` 把 forward 替换上去
- **注册表（`Registry`）**：进程级单例字典，存放当前设备能用的所有 kernel
- **`KernelPlugin("auto")`**：唯一的 `BasePlugin` 入口，按 YAML 里的 `include_kernels` 字段决定启用哪些 kernel

## 启动时发生了什么

`kernels/interface.py` 在 import 时执行：

```python
default_kernels = scan_all_kernels()
```

`scan_all_kernels` 遍历 `kernels/ops/` 下所有 `.py` 文件并 `importlib.import_module(...)`，触发每个文件里的 `@register_kernel` 装饰器。`Registry.register` 在注册时检查 kernel 的 `_device` 是否等于当前 accelerator 类型——不匹配就跳过。

结果：`default_kernels` 只包含**当前硬件可用**的 kernel id。

## 使用方式

### 1. 通过 YAML 启用

```yaml
kernel_config:
  name: auto
  include_kernels: auto         # 启用全部
```

或显式列出：

```yaml
kernel_config:
  name: auto
  include_kernels: npu_fused_swiglu,npu_fused_rmsnorm
```

字段语义见 [KernelConfig](../../../parameter-reference/kernel_config.md)。

### 2. 通过 API 启用

```python
from llamafactory.v1.plugins.model_plugins.kernels.interface import (
    apply_default_kernels,
    apply_kernel,
    get_default_kernels,
)

print(get_default_kernels())
model = apply_default_kernels(model, include_kernels="auto")
apply_kernel("npu_fused_rmsnorm", model=model)
```

`apply_default_kernels` 内部就是按 `include_kernels` 调用 `apply_kernel`，YAML 与 API 路径完全等价。

## 替换是怎么生效的

每个 kernel 的 `apply` 实现遍历 `model.named_modules()`，找到目标类（通常按 `module.__class__.__name__` 匹配），把 `module.forward` 改成融合实现：

```python
import types
module.forward = types.MethodType(npu_swiglu_forward, module)
```

权重不动，因此 kernel 替换不会改变保存出去的模型——它只在当前进程内生效。

## 加新 Kernel

接口与最小例子见 [kernels_api](kernels_api.md)；现有算子按类别在 [fused_operators](fused_operators.md) 列出。
