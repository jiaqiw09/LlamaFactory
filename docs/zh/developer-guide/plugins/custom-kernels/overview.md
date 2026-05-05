# Kernel 插件系统

## 概述

LLaMA-Factory Kernel 插件系统用于管理不同硬件设备提供的高性能计算内核（kernel）实现。该系统通过替换模型中的关键模块（如 RMSNorm、SwiGLU、RoPE、MoE 等）为硬件优化的版本，从而提升模型训练和推理的性能。

## 核心特性

- **自动注册机制**：基于 `@register_kernel` 装饰器实现自动注册。系统启动时自动扫描 `ops` 目录下的 kernel 实现，注册到全局注册表
- **设备适配感知**：自动检测当前硬件设备并应用相应的优化，跳过不支持的设备
- **模块化设计**：每个 kernel 独立实现，互不干扰，可单独或批量应用
- **后向兼容**：kernel 替换不修改模型权重，保持数值一致性
- **灵活扩展**：通过继承 `BaseKernel` 基类并使用装饰器，可轻松添加新 kernel

## 使用方式

### 通过训练 YAML 配置文件使用

```yaml
kernel_config:
  name: auto
  include_kernels: auto
```

### 调用 API 启用

```python
from llamafactory.v1.plugins.model_plugins.kernels import apply_default_kernels, apply_kernel

# 自动应用所有默认 kernels
model = apply_default_kernels(model, include_kernels="auto")

# 单独应用某个 kernel
model = apply_kernel("npu_fused_rmsnorm", model=model)
```

### 查询已注册的可用 kernels

```python
from llamafactory.v1.plugins.model_plugins.kernels import get_default_kernels

available_kernels = get_default_kernels()
```

各硬件后端提供的具体 kernel 列表见 [多后端支持](../../../multi-backend/index.md)。

## 融合算子详情

详见 [融合算子](fused_operators.md)。

## 开发新 Kernel

详见 [Kernel 插件 API](kernels_api.md)。
