# 开发者指南

设计原理与扩展模式，面向需要理解内部机制或扩展框架的开发者。

| 页面 | 说明 |
|------|------|
| [架构概览](architecture_overview.md) | 分层插件化架构总览、核心设计、关键路径、扩展点 |
| [BasePlugin 机制](baseplugin_mechanism.md) | 命名注册表、装饰器注册、参数分发、延迟导入 |
| **核心模块** | |
| [DataEngine](core/data_engine.md) | 数据集加载、索引构建、格式转换的协调入口 |
| [ModelEngine](core/model_engine.md) | 模型加载管线：Processor → Config → Model → Plugin |
| [BaseTrainer](core/base_trainer.md) | 训练循环基类：batch 生成、分布式集成、Checkpoint、Callback |
| [Accelerator 层](core/accelerator.md) | 硬件抽象、DeviceMesh 并行拓扑、分布式通信原语 |
| [Renderer 与 Template 系统](core/renderer.md) | Messages → ModelInput 的渲染管线、可插拔模板注册 |
| [BatchGenerator](core/batch_generator.md) | 批次生成、梯度累积、StatefulDataLoader、断点续训 |
| [Callback 系统](core/callback.md) | 训练钩子机制、LoggingCallback、自定义 Callback |
| **插件模块** | |
| [数据插件](plugins/data_plugins.md) | DataConverterPlugin、DataLoaderPlugin、索引调整函数 |
| [模型插件](plugins/model_plugins.md) | PeftPlugin、QuantizationPlugin、InitPlugin、RenderingPlugin、SequenceParallel |
| [训练器插件](plugins/trainer_plugins.md) | DistributedPlugin、BatchingPlugin、OptimizerPlugin、LRSchedulerPlugin |
| [Kernel 插件系统](plugins/custom-kernels/overview.md) | Kernel 注册机制、使用方式、融合算子总览 |
| [Kernel 插件 API](plugins/custom-kernels/kernels_api.md) | BaseKernel、Registry、扩展新 Kernel 的完整步骤 |
| [融合算子](plugins/custom-kernels/fused_operators.md) | 算子分类、替换机制、实现入口 |
