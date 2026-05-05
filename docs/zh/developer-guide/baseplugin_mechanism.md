# BasePlugin 机制

BasePlugin 是 v1 框架的核心扩展机制。所有可插拔组件（PEFT、分布式、数据加载、Template 等）都通过统一的 Plugin 系统注册和调用。

## 设计理念

Plugin 本质上是一个**命名注册表**（named registry），将一个名称（`name`）映射到一个或多个**可调用函数**（`method_name` → function）。

```python
class BasePlugin:
    _registry: dict[str, dict[str, Callable]] = defaultdict(dict)

    def __init__(self, name: str | None = None):
        self.name = name

    def register(self, method_name: str = "__call__") -> Callable:
        """装饰器：将函数注册为 Plugin 方法"""

    def __call__(self, *args, **kwargs):
        """调用 __call__ 注册的函数"""
        return self["__call__"](*args, **kwargs)

    def __getitem__(self, method_name: str):
        """按方法名获取注册的函数"""
```

## 使用模式

### 注册

```python
class PrintPlugin(BasePlugin):
    def again(self):
        self["again"]()

@PrintPlugin("hello").register()
def print_hello():
    print("Hello world!")

@PrintPlugin("hello").register("again")
def print_hello_again():
    print("Hello world! Again.")

PrintPlugin("hello")()       # → "Hello world!"
PrintPlugin("hello").again() # → "Hello world! Again."
```

### 多方法注册（实际例子）

```python
class DistributedPlugin(BasePlugin):
    def __call__(self, model, dist_config, **kwargs):
        return super().__call__(model, dist_config, **kwargs)

@DistributedPlugin("fsdp2").register()           # 注册 __call__ → shard_model
def shard_model_fsdp2(model, dist_config, **kwargs):
    ...

@DistributedPlugin("fsdp2").register("save_model")  # 注册 save_model
def save_model_fsdp2(model, output_dir, processor):
    ...

# 调用
engine = DistributedPlugin("fsdp2")(model, dist_config)  # 调用 __call__
DistributedPlugin("fsdp2").save_model(model, dir, proc)   # 调用 save_model
```

## 各 Plugin 类型

| Plugin 类 | 位置 | 用途 |
|-----------|------|------|
| `PeftPlugin` | `plugins/model_plugins/peft.py` | LoRA / Freeze 模型适配 |
| `QuantizationPlugin` | `plugins/model_plugins/quantization.py` | BNB 量化 |
| `InitPlugin` | `plugins/model_plugins/initialization.py` | 模型初始化策略 |
| `KernelPlugin` | `plugins/model_plugins/kernels/interface.py` | 自定义算子 |
| `RenderingPlugin` | `plugins/model_plugins/rendering.py` | 对话模板渲染 |
| `DistributedPlugin` | `plugins/trainer_plugins/distributed/hub.py` | FSDP2 / DeepSpeed |
| `BatchingPlugin` | `plugins/trainer_plugins/batching.py` | 批次策略 |
| `OptimizerPlugin` | `plugins/trainer_plugins/optimizer.py` | 优化器（预留） |
| `LRSchedulerPlugin` | `plugins/trainer_plugins/lr_scheduler.py` | LR 调度器（预留） |
| `DataLoaderPlugin` | `plugins/data_plugins/loader.py` | 数据加载 |
| `DataConverterPlugin` | `plugins/data_plugins/converter.py` | 数据格式转换 |
| `SequenceParallel*Plugin` | `plugins/model_plugins/parallelization/` | 序列并行 |

## 参数分发

顶层参数类的 `*_config` 字段（如 `peft_config`、`dist_config`）通过 `name` 字段选择对应 Plugin：

```python
# model_args.py
@dataclass
class ModelArguments:
    peft_config: PluginConfig | None = None

# 使用
peft_config:
  name: lora     # → PeftPlugin("lora")
  r: 8

# 内部调用
model = PeftPlugin(self.args.peft_config.name)(model, config, is_train)
```

## Template 延迟导入

`RenderingPlugin` 使用了延迟导入模式：

```python
class RenderingPlugin(BasePlugin):
    def __getitem__(self, method_name):
        self._ensure_template_imported()  # 按需导入 templates/{name}.py
        return super().__getitem__(method_name)
```

这样 template 文件只在被实际使用时才导入，减少启动开销。
