# 模型插件

模型相关的所有可替换部件：PEFT、量化、初始化设备、自定义算子、模板、序列并行。

代码位置：`src/llamafactory/v1/plugins/model_plugins/`。

## PeftPlugin

`peft.py` 注册两个分支，分别对应 `peft_config.name=lora` 与 `freeze`。

```python
class PeftPlugin(BasePlugin):
    def __call__(self, model: HFModel, config: dict, is_train: bool) -> HFModel:
        return super().__call__(model, config, is_train)
```

### `lora`

调用 HF PEFT 的 `LoraConfig` + `get_peft_model`。重要分支：

- `target_modules == "all"`：扫描所有 `Linear` 模块，剔除 `lm_head` / `output_layer` / `output`
- `target_modules` 为字符串：作为单个名字
- `target_modules` 为列表：直接传给 `LoraConfig`

`adapter_name_or_path` 处理：

- 训练模式只允许单个 adapter；继续训练该 adapter，**所有 LoRA 超参数被 adapter 自身的配置覆盖**
- 推理模式可传多个 adapter，逐一 `merge_and_unload`

参数清单见 [PeftConfig](../../parameter-reference/peft_config.md)。

### `freeze`

按层位/模块设 `requires_grad`：

- `freeze_trainable_layers > 0`：放开"后 N 层"
- `freeze_trainable_layers < 0`：放开"前 N 层"
- `freeze_trainable_modules`：层内具体的子模块（如 `q_proj`）；`["all"]` 表示该层全部
- `freeze_extra_modules`：非分层模块（如 embedding）
- `cast_trainable_params_to_fp32`：可训练参数转 fp32 提稳定性

层数从 `model.config` 的 `num_hidden_layers` / `num_layers` / `n_layer` 中取第一个非空值；若都没有，直接 `ValueError`。

### 模型导出（merge）

`peft.py:merge_and_export_model` 是 `lmf merge` 的入口：

```text
ModelEngine(is_train=False) → merge adapters → 转 dtype → save_pretrained → push_to_hub (可选)
```

- 仅支持 `name=lora`
- 必填 `peft_config.export_dir`
- `infer_dtype="auto"` 在 fp32 + 支持 bf16 时转 bf16
- `export_legacy_format=true` 输出 `.bin` 而非 safetensors

详见 [model_export](../../feature-guide/model_export.md)。

## QuantizationPlugin

`quantization.py` 注册 `auto` 与 `bnb`：

- `auto`：根据 `quantization_bit` 选择具体后端，目前一律分派到 `bnb`
- `bnb`：bitsandbytes 4-bit / 8-bit

构造 `BitsAndBytesConfig` 时读取的字段：

| 字段 | 4-bit | 8-bit |
|------|-------|-------|
| `quantization_bit` | `4` | `8` |
| `compute_dtype` | `bnb_4bit_compute_dtype` | — |
| `double_quantization` | `bnb_4bit_use_double_quant` | — |
| `quantization_type` | `bnb_4bit_quant_type` | — |

> **注**：训练模式只允许 4-bit（用于 FSDP + QLoRA），8-bit 仅推理可用。该限制在 `quantization_with_bnb` 内部直接 `ValueError`。

参数细节见 [QuantConfig](../../parameter-reference/quant_config.md)。

## InitPlugin

`initialization.py` 注册三种初始化策略：

```python
@InitPlugin("init_on_default").register()
def init_on_default() -> torch.device:
    return DistributedInterface().current_device

@InitPlugin("init_on_meta").register()
def init_on_meta() -> torch.device:
    return torch.device(DeviceType.META.value)

@InitPlugin("init_on_rank0").register()
def init_on_rank0() -> torch.device:
    if DistributedInterface().get_rank() == 0:
        return torch.device(DeviceType.CPU.value)
    return torch.device(DeviceType.META.value)
```

`InitPlugin(name)()` 返回的 `torch.device` 被 `ModelEngine._init_model` 用作 `device_map`。`init_on_meta` 与 `init_on_rank0` 都依赖 FSDP2 后续在 `shard_model` 阶段填充权重，仅在 FSDP2 路径下有效。

## RenderingPlugin

`rendering.py` 提供懒导入的模板注册表，机制见 [renderer](../core/renderer.md#renderingplugin)。仓库内置 `qwen3` / `qwen3_nothink`，加上 `Renderer` 内置的 `chatml`。

新增模板时只要把文件放进 `templates/`，第一次按名字调用即自动 import。

## KernelPlugin

`kernels/interface.py` 注册唯一的 `auto` 分支，根据 `include_kernels` 启用已注册的 kernel：

```python
@KernelPlugin("auto").register()
def apply_default_kernels(model, include_kernels=None):
    if not include_kernels:
        return model
    if include_kernels == "auto" or include_kernels is True:
        use_kernels = default_kernels.keys()
    else:
        use_kernels = include_kernels.split(",")
    for k in use_kernels:
        apply_kernel(k, model=model)
    return model
```

`default_kernels` 在 `scan_all_kernels()` 里通过遍历 `kernels/ops/` 目录得到。kernel 注册机制见 [custom-kernels](custom-kernels/overview.md)。

## SequenceParallel*Plugin

`parallelization/sequence_parallel.py` 注册两个插件：

| Plugin | 注册 name | 用途 |
|--------|-----------|------|
| `SequenceParallelModelPlugin` | `ulysses` | 替换 transformers 的 `_flash_attention_forward` 为 Ulysses attention |
| `SequenceParallelLossPlugin` | `sequence_parallel_loss` | 在 cp 维度切分序列后计算 loss，再用 all_gather 拼回完整 logits |

`BaseTrainer.__init__` 在 `cp_size > 1` 时调用 `SequenceParallelModelPlugin(cp_mode)(model, dist_config)`，这是一个 monkey-patch：把 `transformers.modeling_flash_attention_utils._flash_attention_forward` 全局替换。

`fit()` 的 forward 路径在 `cp_size > 1` 时调 `SequenceParallelLossPlugin("sequence_parallel_loss")` 代替 `compute_loss`。

要求：

- `num_attention_heads % cp_size == 0`
- `num_key_value_heads % cp_size == 0` 或 `cp_size % num_key_value_heads == 0`
- 默认 attention 实现会被强制设为 `flash_attention_2`
- qwen3.5 因 attention 实现差异不支持，`BaseTrainer` 会直接 `RuntimeError`

底层 all-to-all 通信封装在 `parallelization/ulysses.py` 与 `seq_comm.py`。
