# 架构概览

LlamaFactory v1（`src/llamafactory/v1/`）是一套以"配置 → 引擎 → 插件"为骨干的训练框架。配置类只描述参数，引擎负责加载与训练流程，所有可替换部分以 Plugin 形式挂在引擎上。

## 目录结构

```text
src/llamafactory/v1/
├── launcher.py              入口分派；多卡时 re-exec 到 torchrun
├── config/                  4 个顶层 dataclass + arg_parser/arg_utils
├── core/
│   ├── data_engine.py       数据集加载与索引（继承 torch.utils.data.Dataset）
│   ├── model_engine.py      Processor → Renderer → Config → Model 加载管线
│   ├── base_trainer.py      训练循环基类（fit / save_model / compute_loss）
│   ├── base_sampler.py      推理基类
│   └── utils/
│       ├── batching.py      BatchGenerator + StatefulBuffer
│       ├── rendering.py     Renderer + 内置 chatml
│       ├── checkpoint.py    TrainingCheckpointCoordinator
│       └── inference_engine.py
├── plugins/
│   ├── data_plugins/        DataLoaderPlugin / DataConverterPlugin
│   ├── model_plugins/       PEFT / Quant / Init / Kernels / Rendering / Parallelization
│   ├── trainer_plugins/     Distributed (FSDP2/DeepSpeed) / Batching / Optimizer / LRScheduler
│   └── sampler_plugins/     vLLM 采样后端（预留）
├── accelerator/
│   ├── interface.py         DistributedInterface（单例）+ DistributedStrategy
│   └── helper.py            设备类型、通信原语、进程组初始化辅助
├── trainers/
│   ├── sft_trainer.py       SFTTrainer：实现 compute_loss
│   ├── dpo_trainer.py       占位，launcher 抛 NotImplementedError
│   └── rm_trainer.py        占位，launcher 抛 NotImplementedError
├── samplers/
│   └── cli_sampler.py       chat 子命令入口
└── utils/
    ├── plugin.py            BasePlugin 基类
    ├── callbacks/           TrainerCallback / CallbackHandler / LoggingCallback
    ├── types.py             共用 TypedDict / NamedTuple
    └── ...
```

## 三层骨干

```text
config/  ─────►  core/  ─────►  plugins/
  参数描述         编排流程         可替换实现
                       │
                       ▼
                accelerator/      硬件与分布式抽象
```

- **config/** 只声明字段。`arg_parser` 把命令行 / YAML 装载到四个 dataclass：`DataArguments` / `ModelArguments` / `TrainingArguments` / `SampleArguments`
- **core/** 把这些参数变成可执行步骤：`DataEngine` 拉数据，`ModelEngine` 拉模型，`BaseTrainer` 跑训练循环
- **plugins/** 提供具体实现，由 `core/` 在需要的时候按 `name` 字段实例化

## 训练入口路径

```text
llamafactory-cli sft config.yaml
   └─ launcher.launch()
        ├─ get_device_count() > 1 → 走 torchrun re-exec
        └─ run_sft()  (trainers/sft_trainer.py)
              ├─ get_args()                       → 4 个 dataclass
              ├─ DistributedInterface(dist_config) → 进程组 + DeviceMesh
              ├─ ModelEngine(model_args, is_train=True)
              ├─ DataEngine(data_args.train_dataset)
              ├─ SFTTrainer(args, model, renderer, dataset)
              │     └─ fit()  (BaseTrainer.fit)
              └─ trainer.save_model()
```

推理入口对称：`run_chat()` 调 `ModelEngine(is_train=False)` + `BaseSampler`。

## 并行拓扑

`accelerator/interface.py:DistributedStrategy` 把 `world_size` 拆成两个 2D mesh：

```text
Model Mesh:  mp_replicate × mp_shard   (模型权重切分)
Data Mesh:   dp × cp                   (数据并行 + 上下文并行)
```

约束：`mp_replicate × mp_shard = world_size`，`dp × cp = world_size`。维度名由 `Dim` 枚举（`MP_REPLICATE` / `MP_SHARD` / `DP` / `CP`）。

## Plugin 即扩展点

所有"可替换"组件统一走 `BasePlugin` 命名注册表。`core/` 通过 `name` 字符串调用对应实现，新增能力只需注册新 `name`，无需改 core。

| 想要扩展 | 注册到 | 入口 |
|---------|--------|------|
| 新对话模板 | `RenderingPlugin("xxx").register("render_messages")` 等 | `templates/xxx.py` |
| 新数据格式 | `DataConverterPlugin("xxx").register()` | `data_plugins/converter.py` |
| 新分布式后端 | `DistributedPlugin("xxx").register(...)` | `trainer_plugins/distributed/` |
| 新算子 | `@register_kernel class ...` | `model_plugins/kernels/ops/` |
| 新训练算法 | 继承 `BaseTrainer` 实现 `compute_loss` | `trainers/` |

机制细节见 [baseplugin_mechanism](baseplugin_mechanism.md)。
