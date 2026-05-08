# ModelEngine

`ModelEngine` 把"加载模型"这件事的所有副作用集中在一个对象里：分词器、模板渲染器、模型 config、模型本体，以及作用在模型上的所有插件（init、quant、peft、kernel）。

代码位置：`src/llamafactory/v1/core/model_engine.py`。

## 接口

```python
class ModelEngine:
    def __init__(self, model_args: ModelArguments, is_train: bool = False) -> None: ...
```

构造完成后暴露：

| 属性 | 类型 | 来源 |
|------|------|------|
| `processor` | tokenizer 或多模态 processor | `AutoProcessor.from_pretrained` |
| `renderer` | `Renderer` | `Renderer(template, processor)` |
| `model_config` | `PretrainedConfig` | `AutoConfig.from_pretrained` |
| `model` | `PreTrainedModel` | `AutoClass.from_pretrained` 后逐步应用插件 |

## 初始化流程

```text
__init__(model_args, is_train)
  1. _init_processor()      AutoProcessor.from_pretrained
  2. Renderer(template, processor)
  3. _init_model_config()   AutoConfig.from_pretrained
  4. [DeepSpeed ZeRO-3] setup → _init_model() → teardown
     [其它]                  _init_model()
```

`is_train` 影响后续插件分支（推理路径 / 训练路径）。

## _init_model 内部

按以下顺序构造模型，每一步可能由插件接管：

### 1. 选择初始化设备

```python
if model_args.init_config is not None:
    init_device = InitPlugin(name)()      # init_on_default / init_on_meta / init_on_rank0
else:
    init_device = DistributedInterface().current_device
```

详见 [InitConfig](../../parameter-reference/init_config.md)。

### 2. 量化 init_kwargs

```python
if model_args.quant_config is not None:
    init_kwargs = QuantizationPlugin(name)(
        init_kwargs=init_kwargs, config=..., model_args=..., is_trainable=is_train,
    )
```

ZeRO-3 下不传 `device_map`；其它情况下 `device_map=init_device`。

### 3. 选择 AutoClass

| `model_class` | AutoClass |
|---------------|-----------|
| `llm` | `AutoModelForCausalLM`，若 config 命中 image-text-to-text 映射则改 `AutoModelForImageTextToText` |
| `cls` | `AutoModelForTokenClassification` |
| `other` | `AutoModel` |

### 4. 实例化模型

```python
if init_device.type == DeviceType.META:
    assert quant_config is None, "meta device 与量化不兼容"
    with init_empty_weights():
        model = AutoClass.from_config(model_config)
else:
    model = AutoClass.from_pretrained(name, dtype="auto", **init_kwargs)
```

模型实例上挂 `model._init_mode`，FSDP2 在 `shard_model` 时根据这个值决定加载策略。

### 5. PEFT

- `peft_config is None` 且训练：`model.to(torch.float32)`，全参训练
- `peft_config is None` 且推理：原模型直接返回
- `peft_config is not None`：调用 `PeftPlugin(name)(model, config, is_train)`
- `peft_config.name == "lora"` 且 `init_mode == "init_on_meta"`：直接 `ValueError`，不支持

### 6. Kernel

```python
if model_args.kernel_config is not None:
    model = KernelPlugin(name)(model, include_kernels=...)
```

详见 [KernelConfig](../../parameter-reference/kernel_config.md)。

## DeepSpeed ZeRO-3 特殊路径

ZeRO-3 在 transformers 模型构造期间需要把 nn.Parameter 替换成 ZeRO 的代理对象。`ModelEngine` 用 `setup_deepspeed_zero3_model_loading` / `teardown_deepspeed_zero3_model_loading` 包住 `_init_model`：

```python
self._deepspeed_zero3_plugin = setup_deepspeed_zero3_model_loading(is_train, dist_config)
try:
    self.model = self._init_model()
finally:
    teardown_deepspeed_zero3_model_loading(self._deepspeed_zero3_plugin)
```

仅当 `is_train` 且 `dist_config.name == "deepspeed"` 时启用。

## 扩展

- 新增模型初始化策略：注册 `InitPlugin("xxx")`
- 新增量化方式：注册 `QuantizationPlugin("xxx")`
- 新增 PEFT 方式：注册 `PeftPlugin("xxx")`
- 新增对话模板：在 `templates/` 下创建文件，注册 `RenderingPlugin("xxx")` 的 `render_messages` / `parse_message`

各插件的具体接口见 [model_plugins](../plugins/model_plugins.md)。

## 示例

```python
from llamafactory.v1.config.arg_parser import get_args
from llamafactory.v1.core.model_engine import ModelEngine

model_args, *_ = get_args()
engine = ModelEngine(model_args, is_train=True)
print(engine.model_config)
print(engine.model)
```
