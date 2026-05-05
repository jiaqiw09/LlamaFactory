# 模型插件

模型相关的 Plugin 集合，包括 PEFT（LoRA/Freeze）、量化、初始化和自定义 Kernel。

## PeftPlugin（LoRA / Freeze）

位于 `src/llamafactory/v1/plugins/model_plugins/peft.py`。

### LoRA (`name: lora`)

使用 HuggingFace PEFT 库的标准 LoRA 实现：

```python
@PeftPlugin("lora").register()
def get_lora_model(model, config, is_train=False):
    peft_config = LoraConfig(
        task_type=CAUSAL_LM,
        inference_mode=not is_train,
        r=config.get("r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.05),
        use_rslora=config.get("use_rslora", False),
        use_dora=config.get("use_dora", False),
        target_modules=target_modules,
        modules_to_save=config.get("modules_to_save"),
    )
    model = get_peft_model(model, peft_config)
```

`target_modules: "all"` 时自动发现所有 linear 层（排除 `lm_head`）。

### Adapter 加载与合并

- **加载已有 adapter**：通过 `adapter_name_or_path` 参数
- **训练模式**：只加载单个 adapter 继续训练，使用 adapter 自身的超参数
- **推理模式**：支持合并多个 adapter（逐一 `merge_and_unload`）

### Freeze (`name: freeze`)

通过设置 `requires_grad` 实现部分参数冻结：

```python
@PeftPlugin("freeze").register()
def get_freeze_model(model, config, is_train=False):
    # freeze_trainable_layers: 正数=最后N层，负数=前N层
    # freeze_trainable_modules: 要解冻的模块类型
    # freeze_extra_modules: 额外解冻的非隐藏层模块
```

自动识别模型的 `num_hidden_layers`（兼容不同模型架构的命名）。

### 模型导出

`merge_and_export_model()` 函数用于将 LoRA adapter 合并到基座模型：

```text
加载基座模型 → merge_adapters → 转换 dtype → save_pretrained → [push_to_hub]
```

## QuantizationPlugin（量化）

位于 `src/llamafactory/v1/plugins/model_plugins/quantization.py`。

### Auto (`name: auto`)

自动选择量化方法，当前仅支持 BNB。

### BNB (`name: bnb`)

BitsAndBytes 4-bit / 8-bit 量化：

```python
@QuantizationPlugin("bnb").register()
def quantization_with_bnb(init_kwargs, model_args, **kwargs):
    if quantization_bit == 8:
        init_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization_bit == 4:
        init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=...,
            bnb_4bit_use_double_quant=...,
            bnb_4bit_quant_type=...,
        )
```

训练模式下仅支持 4-bit（用于 FSDP+QLoRA）。

### 量化参数

| 参数 | 说明 |
|------|------|
| `quantization_bit` | `4` 或 `8` |
| `compute_dtype` | 计算用 dtype |
| `double_quantization` | 是否双重量化 |
| `quantization_type` | `nf4` 等量化类型 |

## InitPlugin（模型初始化）

位于 `src/llamafactory/v1/plugins/model_plugins/initialization.py`。

| 模式 | 说明 |
|------|------|
| `init_on_default` | 每张卡分别加载（默认） |
| `init_on_meta` | 在 meta device 上初始化（省内存，由 FSDP2 后续实际加载权重） |
| `init_on_rank0` | Rank 0 加载到 CPU，再由 FSDP2 分发给其他 rank |

## KernelPlugin（自定义算子）

启用硬件优化的融合算子，详见 [Custom Kernels 开发者文档](custom-kernels/overview.md)。

## RenderingPlugin（模板渲染）

通过 `RenderingPlugin(template_name)` 渲染对话为 token 序列。Template 文件位于 `plugins/model_plugins/templates/`，使用延迟导入。

### 已实现的 Template

| Template | 文件 | 特性 |
|----------|------|------|
| `qwen3` | `templates/qwen3.py` | 支持 reasoning、tool_call |
| `qwen3_nothink` | `templates/qwen3_nothink.py` | Qwen3 无思考模式 |
| `chatml` | 内建于 `rendering.py` | 通用 ChatML 格式 |

### Template 开发

注册新 template 需实现两个方法：

```python
@RenderingPlugin("my_template").register("render_messages")
def render_my_template(processor, messages, tools, is_generate, enable_thinking):
    ...
    return ModelInput(input_ids=..., attention_mask=..., labels=..., loss_weights=...)

@RenderingPlugin("my_template").register("parse_message")
def parse_my_template(generated_text):
    ...
    return Message(role="assistant", content=[...])
```

## Sequence Parallel Plugins

位于 `src/llamafactory/v1/plugins/model_plugins/parallelization/`。

### SequenceParallelModelPlugin

Ulysses 序列并行：替换 `_flash_attention_forward` 为并行版本。

### SequenceParallelLossPlugin

序列并行 loss 计算：在 attention 维度拆分输入，通过 all_gather 收集完整 log_probs 后计算 loss。

### 序列并行通信

位于 `ulysses.py` 和 `seq_comm.py`，实现 Ulysses attention 的 all-to-all 通信。
