# BasePlugin

所有可替换组件——PEFT、量化、分布式、模板、自定义 kernel、数据转换器——共用同一个 `BasePlugin` 注册机制。理解它一次，所有插件类的扩展方式就一致了。

代码位置：`src/llamafactory/v1/utils/plugin.py`。

## 设计

`BasePlugin` 是一个全局命名注册表：

```text
_registry: dict[str, dict[str, Callable]]
            │       │       └── method_name → 函数
            │       └── 插件实例的 name
            └── 共享在 BasePlugin 及其所有子类
```

- 注册表是 `BasePlugin` 类属性，所有子类共享一份
- 第一层 key 是插件实例的 `name`（如 `"lora"`、`"fsdp2"`）
- 第二层 key 是方法名，默认为 `"__call__"`，也可注册其它名字（如 `"save_model"`）

```python
class BasePlugin:
    _registry: dict[str, dict[str, Callable]] = defaultdict(dict)

    def __init__(self, name: str | None = None) -> None:
        self.name = name

    def register(self, method_name: str = "__call__") -> Callable:
        def decorator(func):
            self._registry[self.name][method_name] = func
            return func
        return decorator

    def __call__(self, *args, **kwargs):
        return self["__call__"](*args, **kwargs)

    def __getitem__(self, method_name):
        return self._registry[self.name][method_name]
```

## 注册

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


PrintPlugin("hello")()        # → "Hello world!"
PrintPlugin("hello").again()  # → "Hello world! Again."
```

每个子类按需声明显式方法（如 `again`），否则通过 `__getattr__` 自动转发到注册表。`PrintPlugin("hello")()` 总是调用 `__call__`，其它方法用属性访问或 `__getitem__`。

## 多方法注册：以 DistributedPlugin 为例

`DistributedPlugin` 把分布式后端的多个生命周期方法挂在同一个 `name` 下：

```python
class DistributedPlugin(BasePlugin):
    def __call__(self, model, dist_config, **kwargs):
        return super().__call__(model, dist_config, **kwargs)


@DistributedPlugin("fsdp2").register()                  # __call__ → shard_model
def shard_model_fsdp2(model, dist_config, **kwargs): ...

@DistributedPlugin("fsdp2").register("save_model")
def save_model_fsdp2(model, output_dir, processor): ...

@DistributedPlugin("fsdp2").register("save_checkpoint")
def save_checkpoint_fsdp2(model, optimizer, ckpt_dir, **kwargs): ...

@DistributedPlugin("fsdp2").register("load_checkpoint")
def load_checkpoint_fsdp2(model, optimizer, ckpt_dir, **kwargs): ...
```

调用：

```python
DistributedPlugin("fsdp2")(model, dist_config, bf16=True)            # __call__
DistributedPlugin("fsdp2").save_model(model, output_dir, processor)   # save_model
```

## 由 `name` 字段路由

顶层参数里的 `*_config` 字段通过 `name` 选择具体插件：

```yaml
peft_config:
  name: lora        # → PeftPlugin("lora")
  r: 8

dist_config:
  name: fsdp2       # → DistributedPlugin("fsdp2")
```

`core/` 内部直接用 `name` 实例化插件并调用：

```python
model = PeftPlugin(self.args.peft_config.name)(model, self.args.peft_config, self.is_train)
```

`PluginConfig`（`config/arg_utils.py`）是一个继承 `dict` 的类型，强制要求 `name` 字段存在；`get_plugin_config` 会把 YAML 字符串和嵌套结构都规范成它。

## 内置插件类

| Plugin | 文件 | 说明 |
|--------|------|------|
| `PeftPlugin` | `model_plugins/peft.py` | LoRA / Freeze |
| `QuantizationPlugin` | `model_plugins/quantization.py` | bitsandbytes |
| `InitPlugin` | `model_plugins/initialization.py` | 模型初始化设备策略 |
| `KernelPlugin` | `model_plugins/kernels/interface.py` | 启用注册的算子替换 |
| `RenderingPlugin` | `model_plugins/rendering.py` | 模板渲染（懒导入） |
| `SequenceParallelModelPlugin` / `SequenceParallelLossPlugin` | `model_plugins/parallelization/sequence_parallel.py` | Ulysses 序列并行 |
| `DistributedPlugin` | `trainer_plugins/distributed/hub.py` | FSDP2 / DeepSpeed |
| `BatchingPlugin` | `trainer_plugins/batching.py` | 非 NORMAL 批次策略（预留） |
| `OptimizerPlugin` | `trainer_plugins/optimizer.py` | 优化器（预留） |
| `LRSchedulerPlugin` | `trainer_plugins/lr_scheduler.py` | 学习率调度器（预留） |
| `DataLoaderPlugin` | `data_plugins/loader.py` | 数据加载（`local`） |
| `DataConverterPlugin` | `data_plugins/converter.py` | 格式转换（`alpaca` / `sharegpt` / `pair`） |

## 懒导入：`RenderingPlugin`

模板文件可能很多，全部 import 会拖慢启动。`RenderingPlugin` 重写 `__getitem__`，第一次按 `name` 取方法时才导入对应的 `templates/{name}.py`，触发其中的注册装饰器：

```python
class RenderingPlugin(BasePlugin):
    _attempted_template_imports: set[str] = set()

    def __getitem__(self, method_name):
        self._ensure_template_imported()  # importlib.import_module(...)
        return super().__getitem__(method_name)
```

新增模板时，只要把文件放进 `templates/`，第一次以该名字调用就会自动注册，无需在任何地方修改注册列表。
