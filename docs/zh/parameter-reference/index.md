# 参数参考

按 v1 配置类组织的参数表。每页只列字段、类型、默认值与说明，不包含端到端流程。

## 顶层参数

入口配置文件直接使用的四个 dataclass：

| 页面 | 说明 |
|------|------|
| [DataArguments](data_arguments.md) | 训练/验证集路径 |
| [ModelArguments](model_arguments.md) | 模型、模板与模型相关插件 |
| [TrainingArguments](training_arguments.md) | 训练超参数、Checkpoint 与分布式插件 |
| [SampleArguments](sample_arguments.md) | 推理采样配置 |

## 插件子配置

通过顶层参数中的 `*_config` 字段引用，由 `name` 字段决定具体插件分支：

| 页面 | 上层字段 | 适用 `name` |
|------|----------|-------------|
| [InitConfig](init_config.md) | `ModelArguments.init_config` | `init_on_default` / `init_on_meta` / `init_on_rank0` |
| [PeftConfig](peft_config.md) | `ModelArguments.peft_config` | `lora` / `freeze` |
| [KernelConfig](kernel_config.md) | `ModelArguments.kernel_config` | `auto` |
| [QuantConfig](quant_config.md) | `ModelArguments.quant_config` | `auto` / `bnb` |
| [DistConfig](dist_config.md) | `TrainingArguments.dist_config` | `fsdp2` / `deepspeed` |

## 数据集配置文件

| 页面 | 说明 |
|------|------|
| [DatasetInfo](dataset_info.md) | `train_dataset` / `eval_dataset` 指向 YAML 文件时的字段格式 |
