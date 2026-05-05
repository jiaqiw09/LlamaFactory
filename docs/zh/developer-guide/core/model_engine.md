# ModelEngine

ModelEngine 是模型加载的统一入口，负责从 HuggingFace 模型标识符加载 Processor → Config → Model → Adapter 的完整管线。

## 初始化流程

```text
ModelEngine(model_args, is_train=True)
  1. _init_processor()     → AutoProcessor.from_pretrained()
  2. Renderer(template)    → 对话模板渲染器
  3. _init_model_config()  → AutoConfig.from_pretrained()
  4. _init_model()         → 加载模型 + 应用 plugin
```

## 模型加载管线

`_init_model()` 是核心，按以下顺序加载：

### 初始化设备

```python
if init_config is not None:
    init_device = InitPlugin(name)()  # init_on_meta / init_on_rank0 / init_on_default
else:
    init_device = DistributedInterface().current_device  # 默认
```

支持三种初始化策略：
- `init_on_default`：每个进程各自加载（默认）
- `init_on_meta`：在 meta device 上创建空模型
- `init_on_rank0`：rank 0 加载到 CPU，其余 rank 在 meta device

### 量化

```python
if quant_config is not None:
    init_kwargs = QuantizationPlugin(name)(
        init_kwargs=init_kwargs,
        config=model_config,
        ...
    )
```

- `auto`：自动选择（当前仅支持 BNB）
- `bnb`：BitsAndBytes 4-bit / 8-bit 量化

### 模型类选择

根据 `model_class` 选择 HF AutoClass：
- `LLM` → `AutoModelForCausalLM` 或 `AutoModelForImageTextToText`
- `CLS` → `AutoModelForTokenClassification`
- `OTHER` → `AutoModel`

### 模型实例化

```python
if init_device.type == META:
    model = AutoClass.from_config(model_config)   # 空壳
else:
    model = AutoClass.from_pretrained(model, ...) # 加载权重
```

### PEFT 适配

```python
if peft_config is not None:
    model = PeftPlugin(name)(model, config, is_train)
```

- `lora`：应用 LoRA adapter
- `freeze`：冻结指定层

训练模式时，完整模型转为 fp32。

### Custom Kernel

```python
if kernel_config is not None:
    model = KernelPlugin(name)(model, include_kernels=...)
```

在模型上应用硬件优化的融合算子。

## 关键属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `processor` | `PreTrainedTokenizer \| ProcessorMixin` | Tokenizer 或多模态 processor |
| `renderer` | `Renderer` | 对话模板渲染器 |
| `model_config` | `PretrainedConfig` | 模型配置 |
| `model` | `PreTrainedModel` | 加载完成的模型（已应用所有 plugin） |
| `is_train` | `bool` | 是否为训练模式 |

## DeepSpeed ZeRO-3 特殊处理

当使用 DeepSpeed ZeRO-3 时，模型加载在 `setup_deepspeed_zero3_model_loading` 上下文中进行，确保参数在加载阶段正确分片。

## 扩展点

添加新的模型初始化模式：注册 `InitPlugin`。

```python
@InitPlugin("my_init_mode").register()
def my_init_mode() -> torch.device:
    ...
```
