# 监督微调（SFT）

SFT 支持三种参数选择策略：全参（不写 `peft_config`）、LoRA（`peft_config.name=lora`）、Freeze（`peft_config.name=freeze`）。三种模式共用同一个训练循环，区别只在 `peft_config`。

## 最短运行

```bash
export USE_V1=1
llamafactory-cli sft examples/v1/train_full/train_full_fsdp2.yaml
```

`sft` 与 `train` 等价。多 GPU 自动 `torchrun`，单卡可设 `FORCE_TORCHRUN=1` 强制走分布式路径。

## 全参微调

```yaml
model: Qwen/Qwen3-0.6B
model_class: llm
template: qwen3_nothink

train_dataset: data/v1_sft_demo.yaml

output_dir: outputs/qwen3_sft_full
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

kernel_config:
  name: auto
  include_kernels: auto
```

字段细节：[ModelArguments](../parameter-reference/model_arguments.md) / [TrainingArguments](../parameter-reference/training_arguments.md) / [DistConfig](../parameter-reference/dist_config.md) / [KernelConfig](../parameter-reference/kernel_config.md)。

## LoRA 微调

```yaml
model: Qwen/Qwen3-4B
model_class: llm
template: qwen3_nothink

train_dataset: data/v1_sft_demo.yaml

peft_config:
  name: lora
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: all       # 自动识别所有 Linear，排除 lm_head / output_layer / output

dist_config:
  name: fsdp2

output_dir: outputs/qwen3_sft_lora
micro_batch_size: 2
global_batch_size: 16
cutoff_len: 2048
learning_rate: 1e-4
num_train_epochs: 3
bf16: true
```

完整字段见 [PeftConfig](../parameter-reference/peft_config.md)。

LoRA + 大模型常用 `init_on_rank0` 节省加载显存：

```yaml
init_config:
  name: init_on_rank0
```

继续训练已有 adapter（注意：LoRA 超参数会被 adapter 自身覆盖）：

```yaml
peft_config:
  name: lora
  adapter_name_or_path: outputs/run/checkpoint-1000
```

## Freeze 微调

```yaml
peft_config:
  name: freeze
  freeze_trainable_layers: 2          # 正数 = 后 N 层；负数 = 前 N 层
  freeze_trainable_modules: all       # 该层内全部子模块；也可指定如 q_proj
  freeze_extra_modules: null          # 额外解冻 embedding / lm_head 等
  cast_trainable_params_to_fp32: true
```

可训练参数转 fp32 是为了数值稳定，关闭后 grad 数值范围可能变窄。

## QLoRA（4-bit + LoRA）

```yaml
peft_config:
  name: lora
  r: 16
  lora_alpha: 32
  target_modules: all

quant_config:
  name: bnb
  quantization_bit: 4

dist_config:
  name: fsdp2

bf16: true
```

bitsandbytes 仅在 GPU 后端可用，且训练只允许 4-bit。完整说明见 [QuantConfig](../parameter-reference/quant_config.md)。

## Checkpoint 与断点续训

```yaml
save_steps: 500              # 或 save_epochs: 1.0
save_total_limit: 3
save_ckpt_as_hf: false
resume_from_checkpoint: auto
```

`auto` 在 `output_dir` 下查找带 `CHECKPOINT_COMPLETE` 标记的最新 checkpoint。各后端的 checkpoint 内容差异见 [模型保存与恢复](model_saving.md)。

## 现成示例

| 文件 | 场景 |
|------|------|
| `examples/v1/train_full/train_full_fsdp2.yaml` | 全参 + FSDP2 |
| `examples/v1/train_full/train_full_deepspeed.yaml` | 全参 + DeepSpeed ZeRO-3 |
| `examples/v1/train_full/train_full_ulysses_cp.yaml` | 全参 + FSDP2 + Ulysses CP |
| `examples/v1/train_lora/train_lora_sft.yaml` | LoRA + FSDP2 |
| `examples/v1/train_lora/train_lora_sft_rank0.yaml` | LoRA + `init_on_rank0` |
| `examples/v1/train_freeze/train_freeze_sft.yaml` | Freeze |
| `examples/v1/train_qlora/quantization.yaml` | QLoRA |
