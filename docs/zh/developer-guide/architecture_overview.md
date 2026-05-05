# 架构概览

LlamaFactory v1 采用分层插件化架构，核心由 Config → Engine → Plugin 三层组成。

## 整体架构

```text
launcher.py                     ← 入口：解析命令、启动 torchrun
    ↓
config/                          ← 配置层：4 个参数 Dataclass
    ├── data_args.py             ← DataArguments
    ├── model_args.py            ← ModelArguments
    ├── training_args.py         ← TrainingArguments
    └── sample_args.py           ← SampleArguments
    ↓
core/                            ← 引擎层：训练核心逻辑
    ├── model_engine.py          ← 模型加载管线
    ├── data_engine.py           ← 数据集加载管线
    ├── base_trainer.py          ← 训练循环基类
    ├── base_sampler.py          ← 推理采样基类
    └── utils/
        ├── batching.py          ← 批次生成（StatefulDataLoader）
        ├── rendering.py         ← 模板渲染（ChatML/Qwen3）
        ├── checkpoint.py        ← Checkpoint 管理
        ├── callback.py          ← 训练回调系统
        └── inference_engine.py  ← 推理引擎
    ↓
plugins/                         ← 插件层：可扩展模块
    ├── data_plugins/            ← 数据加载/转换
    ├── model_plugins/           ← PEFT、量化、Kernel、初始化
    ├── trainer_plugins/         ← 分布式、优化器、调度器
    └── sampler_plugins/         ← vLLM 等采样后端
    ↓
accelerator/                     ← 硬件抽象层
    ├── interface.py             ← DistributedInterface (DeviceMesh)
    └── helper.py                ← 设备检测、通信原语
    ↓
trainers/                        ← 具体训练器
    ├── sft_trainer.py           ← SFT 训练器
    ├── dpo_trainer.py           ← DPO 训练器（预留）
    └── rm_trainer.py            ← RM 训练器（预留）
```

## 核心设计

### Plugin 机制

一切可插拔组件都是 Plugin。每个 Plugin 通过 `BasePlugin(name).register()` 注册函数，系统通过 `name` 查找和调用。

```text
BasePlugin("lora")              ← PeftPlugin 实例
    .register()                 ← 注册 __call__
    .register("save_model")     ← 注册额外方法
```

详见 [BasePlugin Mechanism](baseplugin_mechanism.md)。

### DeviceMesh 并行拓扑

```text
Model Mesh: mp_replicate × mp_shard  → 模型权重分片
Data Mesh:  dp × cp                   → 数据 & 序列并行
```

详见 `src/llamafactory/v1/accelerator/interface.py:DistributedStrategy`。

### 参数系统

两级参数结构：
- **顶层**：4 个 Dataclass（`ModelArguments`、`DataArguments`、`TrainingArguments`、`SampleArguments`）
- **嵌套**：`PluginConfig` 子配置（`peft_config`、`dist_config`、`kernel_config` 等），通过 `name` 字段路由到对应 Plugin

## 关键路径

### 训练路径

```text
get_args() → ModelEngine(is_train=True) → DataEngine(dataset_path)
  → SFTTrainer(args, model, renderer, dataset)
    → trainer.fit() → trainer.save_model()
```

### 推理路径

```text
get_args() → ModelEngine(is_train=False)
  → BaseSampler(args, model_args, model, renderer)
    → engine.generate(messages)
```

## 扩展点

1. **新模型**：添加 template 到 `plugins/model_plugins/templates/`
2. **新数据格式**：注册 `DataConverterPlugin`
3. **新分布式后端**：注册 `DistributedPlugin`
4. **新 kernel**：在 `kernels/ops/` 下添加 `BaseKernel` 子类
5. **新训练方法**：继承 `BaseTrainer`，实现 `compute_loss()`
