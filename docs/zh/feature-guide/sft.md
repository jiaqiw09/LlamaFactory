# 监督微调（SFT）

SFT 支持全参数微调、LoRA 和 Freeze 三种模式。

## 快速开始

```bash
export USE_V1=1
llamafactory-cli sft examples/v1/train_full/train_full_fsdp2.yaml
```

## 全参数微调

```yaml
model: Qwen/Qwen3-0.6B
template: qwen3_nothink
train_dataset: data/v1_sft_demo.yaml

micro_batch_size: 1
global_batch_size: 8
cutoff_len: 2048
learning_rate: 5e-5
num_train_epochs: 3
bf16: true
enable_activation_checkpointing: true

dist_config:
  name: fsdp2
  reshard_after_forward: true
```

## LoRA 微调

```yaml
model: Qwen/Qwen3-0.6B
template: qwen3_nothink
train_dataset: data/v1_sft_demo.yaml

peft_config:
  name: lora
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules: all

micro_batch_size: 2
global_batch_size: 16
cutoff_len: 2048
learning_rate: 1e-4
num_train_epochs: 3
```

`target_modules: all` 自动发现所有 Linear 层（排除 lm_head）。

## Freeze 微调

```yaml
peft_config:
  name: freeze
  freeze_trainable_layers: 2
  freeze_trainable_modules: all
  cast_trainable_params_to_fp32: true
```

- `freeze_trainable_layers: 2` — 只训练最后 2 层，负数则训练前 N 层
- `freeze_trainable_modules: all` — 训练指定层内所有模块

## 数据格式

数据集通过 YAML 配置文件指定，参考 [数据准备](data_preparation.md) 和 [DatasetInfo](../parameter-reference/dataset_info.md)。

支持 converter：`alpaca`（instruction+input+output）、`sharegpt`（多轮对话）、`pair`（DPO）。

## 恢复训练

```yaml
resume_from_checkpoint: auto
```

`auto` 自动找最新有效 checkpoint，恢复 optimizer、scheduler、dataloader 和 RNG 状态。不同后端的 checkpoint 格式见 [模型保存与恢复](model_saving.md)。

## 模型保存

训练完成自动保存到 `output_dir`。中间 checkpoint 策略：

```yaml
save_steps: 500
save_epochs: 1.0
save_ckpt_as_hf: false
save_total_limit: 3
```

最终模型保存、DCP checkpoint、额外 HF checkpoint 和恢复逻辑见 [模型保存与恢复](model_saving.md)。
