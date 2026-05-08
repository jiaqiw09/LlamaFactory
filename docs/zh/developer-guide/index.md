# 开发者指南

面向需要理解 v1 内部机制或扩展框架的开发者。文档按"先骨架，再核心模块，再插件"组织：先掌握插件机制，剩下的页面都是它的具体应用。

## 骨架

| 页面 | 说明 |
|------|------|
| [架构概览](architecture_overview.md) | 目录布局、训练入口路径、并行拓扑、扩展点速查 |
| [BasePlugin](baseplugin_mechanism.md) | 命名注册表、多方法注册、`name` 路由、懒导入 |

## Core 模块

| 页面 | 对应源码 |
|------|---------|
| [DataEngine](core/data_engine.md) | `core/data_engine.py` |
| [ModelEngine](core/model_engine.md) | `core/model_engine.py` |
| [BaseTrainer](core/base_trainer.md) | `core/base_trainer.py` |
| [BatchGenerator](core/batch_generator.md) | `core/utils/batching.py` |
| [Renderer](core/renderer.md) | `core/utils/rendering.py` + `plugins/model_plugins/rendering.py` |
| [Callback](core/callback.md) | `utils/callbacks/` |
| [Accelerator](core/accelerator.md) | `accelerator/` |

## 插件

| 页面 | 内容 |
|------|------|
| [data_plugins](plugins/data_plugins.md) | `DataLoaderPlugin` / `DataConverterPlugin` |
| [model_plugins](plugins/model_plugins.md) | PEFT / Quant / Init / Rendering / Kernel / Sequence Parallel |
| [trainer_plugins](plugins/trainer_plugins.md) | Distributed (FSDP2/DeepSpeed) / Batching / Optimizer / LRScheduler |
| [custom-kernels/overview](plugins/custom-kernels/overview.md) | Kernel 系统总览与启用方式 |
| [custom-kernels/kernels_api](plugins/custom-kernels/kernels_api.md) | `BaseKernel` / `Registry` / `apply_kernel` 接口与扩展步骤 |
| [custom-kernels/fused_operators](plugins/custom-kernels/fused_operators.md) | 当前内置算子分类与替换内容 |
