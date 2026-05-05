# Kernel 插件 API

## 概览

Kernel 插件系统通过注册表机制管理所有 kernel。`@register_kernel` 装饰器负责在定义后自动注册，`apply_kernel` 用于启用指定 kernel，`apply_default_kernels` 用于启用当前环境所有可用的默认 kernel。

## 架构设计

### Registry（注册表）

`Registry` 是管理所有 kernel 实现的静态类，维护字典结构 `{kernel_id: KernelClass}`。

```python
class Registry:
    @classmethod
    def register(cls, kernel_cls: type[BaseKernel]) -> type[BaseKernel] | None:
        """注册一个 kernel 类"""

    @classmethod
    def get(cls, kernel_id: str) -> type[BaseKernel] | None:
        """根据 ID 获取 kernel 类"""
```

### register_kernel（装饰器）

`@register_kernel` 是 `Registry.register` 的别名。注册机制：

1. 检查类是否继承自 `BaseKernel`
2. 检查类是否定义了 `_kernel_id` 和 `_device` 属性
3. 检查 `_device` 是否与当前运行环境的加速器类型匹配，不匹配则跳过注册
4. 符合要求则注册到全局注册表

### BaseKernel（基类）

所有 kernel 必须继承自 `BaseKernel` 抽象基类：

```python
class BaseKernel(ABC):
    _kernel_id: Any = ""
    _device: DeviceType = DeviceType.CPU

    @classmethod
    def check_deps(cls) -> bool:
        """检查依赖项"""

    @classmethod
    @abstractmethod
    def apply(cls, **kwargs) -> HFModel:
        """应用 kernel 到模型"""
```

### 标识系统

- **Kernel ID**（`_kernel_id`）：唯一字符串标识符，如 `"npu_fused_rmsnorm"`
- **Device Type**（`_device`）：支持的设备类型，如 `DeviceType.CUDA`、`DeviceType.NPU`

## API

### scan_all_kernels

自动扫描 `ops` 目录下的所有 `.py` 文件并导入，触发 `@register_kernel` 完成自动注册。

### apply_kernel

```python
def apply_kernel(kernel_id: str, **kwargs) -> HFModel:
    """应用指定的 kernel 到模型"""
```

### apply_default_kernels

```python
def apply_default_kernels(model: HFModel, include_kernels: str = None) -> HFModel:
    """应用所有默认 kernel"""
```

## 扩展 Kernel

### 创建新 Kernel

在 `src/llamafactory/v1/plugins/model_plugins/kernels/ops` 下的相应子目录中创建实现文件：

```python
from ......accelerator.helper import DeviceType
from ......utils.types import HFModel
from ...base import BaseKernel
from ...registry import register_kernel

@register_kernel
class CudaSwiGluKernel(BaseKernel):
    _kernel_id = "cuda_fused_swiglu"
    _device = DeviceType.CUDA

    @classmethod
    def apply(cls, **kwargs) -> HFModel:
        model = kwargs.get("model")
        if model is None:
            raise ValueError("model is required")
        if not cls.check_deps():
            raise RuntimeError("Dependencies not met")
        for name, module in model.named_modules():
            pass
        return model
```

### 自动发现

`scan_all_kernels` 会自动扫描 `ops` 目录，文件位于该目录下即可自动注册，无需手动修改注册表代码。

## 异常处理

- **依赖不可用**：`check_deps()` 返回 `False` 时，`apply()` 应抛出异常
- **Kernel ID 未找到**：调用 `apply_kernel` 时传入不存在的 `kernel_id` 会抛出 `ValueError`
