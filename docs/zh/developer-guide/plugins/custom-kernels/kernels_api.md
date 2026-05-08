# Kernel API

写一个新 kernel 涉及三件事：继承 `BaseKernel`、定义 `_kernel_id` 与 `_device`、用 `@register_kernel` 装饰类。本页给出完整接口和最小可工作例子。

代码位置：

- `kernels/base.py`：`BaseKernel`
- `kernels/registry.py`：`Registry` + `register_kernel`
- `kernels/interface.py`：`scan_all_kernels` / `apply_kernel` / `apply_default_kernels` / `KernelPlugin`

## BaseKernel

```python
class BaseKernel(ABC):
    _kernel_id: Any = ""
    _device: DeviceType = DeviceType.CPU

    @classmethod
    def get_kernel_id(cls) -> str: ...

    @classmethod
    def get_device(cls) -> str: ...

    @classmethod
    def check_deps(cls) -> bool:
        # 默认实现：当前 accelerator 类型必须等于 _device
        ...

    @classmethod
    @abstractmethod
    def apply(cls, **kwargs) -> HFModel: ...
```

子类至少要：

- 设置 `_kernel_id`：进程内唯一字符串
- 设置 `_device`：`DeviceType.CUDA` / `DeviceType.NPU` 等
- 实现 `apply(cls, **kwargs)`：从 `kwargs` 里取 `model`，遍历替换 forward，返回 `model`

`check_deps()` 可以重写，加更细的依赖判断（如检测 `torch_npu` 版本）；默认只看设备类型。

## Registry

```python
class Registry:
    _kernels: dict[str, type[BaseKernel]] = {}

    @classmethod
    def register(cls, kernel_cls: type[BaseKernel]) -> type[BaseKernel] | None:
        if not issubclass(kernel_cls, BaseKernel): raise TypeError
        kernel_id = kernel_cls.get_kernel_id()
        if kernel_cls.get_device() != get_current_accelerator().type:
            return                                  # 设备不匹配，跳过
        if not kernel_id: raise ValueError
        if kernel_id in cls._kernels: raise ValueError("already registered")
        cls._kernels[kernel_id] = kernel_cls
        return kernel_cls
```

注意：

- 注册同一个 `_kernel_id` 两次会直接 `ValueError`
- 设备不匹配时静默 return（不报错），所以同一个文件可以同时为 CUDA 和 NPU 写实现，跑哪个取决于当前设备

`register_kernel = Registry.register`，作为装饰器使用。

## 入口函数

```python
def scan_all_kernels(): ...                    # 遍历 ops/ 下所有 .py 触发注册
def get_default_kernels() -> list[str]: ...    # 返回当前注册的 kernel id 列表
def apply_kernel(kernel_id: str, **kwargs): ...
```

`apply_kernel` 找不到对应 id 时直接 `ValueError`。

## 最小例子

文件路径：`kernels/ops/<category>/<your_kernel>.py`。`scan_all_kernels` 会自动 import `ops/` 下所有文件，无需在任何注册表里手写注册。

```python
import types
from ......accelerator.helper import DeviceType
from ......utils.types import HFModel
from ...base import BaseKernel
from ...registry import register_kernel


def cuda_my_norm_forward(self, hidden_states):
    # 高性能实现
    ...


@register_kernel
class CudaMyNormKernel(BaseKernel):
    _kernel_id = "cuda_my_norm"
    _device = DeviceType.CUDA

    expect_modules = frozenset({"LlamaRMSNorm", "Qwen3RMSNorm"})

    @classmethod
    def apply(cls, **kwargs) -> HFModel:
        model = kwargs.get("model")
        if model is None:
            raise ValueError(f"model is required for {cls.__name__}")
        if not cls.check_deps():
            raise RuntimeError(f"{cls.__name__} dependency check failed")

        for _, module in model.named_modules():
            if module.__class__.__name__ in cls.expect_modules:
                module.forward = types.MethodType(cuda_my_norm_forward, module)
        return model
```

写完之后：

- `import` 该文件即触发注册（`scan_all_kernels` 启动时自动完成）
- YAML 里 `include_kernels: cuda_my_norm` 即可启用
- API 调用：`apply_kernel("cuda_my_norm", model=model)`

## 出错点速查

| 现象 | 原因 |
|------|------|
| `check_deps` 通过但替换没生效 | 类名匹配条件没命中；检查 `expect_modules` 与 `model.named_modules()` 实际类名 |
| 启动时 `Failed to import ...` 警告 | `kernels/ops/<file>.py` import 阶段报错；通常是依赖缺失（如 `torch_npu` 在非 NPU 环境） |
| `Kernel xxx not found` | 当前设备类型不匹配；或 `_kernel_id` 拼写不一致 |
| 注册同名报错 | `_kernel_id` 已被另一个类占用，改名或合并 |
