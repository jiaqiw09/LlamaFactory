# 插件设计与参数流

本文说明当前 v1 插件系统的设计、参数解析方式、懒加载机制，以及 `interface.py`
模块的职责边界。

## 目标

插件层主要有三个目标：

1. 让 YAML 里的用户配置保持扁平、可读。
2. 用稳定的小 key 路由插件，比如 `lora`、`fsdp2`、`ulysses`。
3. 保持重实现懒加载，避免 import 一个 interface 时就提前加载分布式后端、kernel、
   transformers patch 或其它可选依赖。

整体链路是：

```text
config dict -> PluginConfig -> plugin interface -> typed params -> implementation
```

## 配置入口

YAML 里的插件字段会先通过 `config.arg_utils.get_plugin_config` 转成
`PluginConfig`。

示例：

```yaml
dist_config:
  name: fsdp2
  cp_size: 2
  cp_mode: ulysses
  reshard_after_forward: true
```

这一层只做通用转换：

- JSON string 或 dict 转成 `PluginConfig`。
- 要求必须有 `name`。
- `"true"`、`"1"`、`"1.5"` 这类简单字符串会转成 bool、int、float。

这一层不理解 `fsdp2`、`lora`、`bnb`、`ulysses` 的具体 schema。具体参数校验
属于对应的 plugin family。

## BasePlugin 注册表

每个 plugin family 都继承自 `BasePlugin`。

示例：

```python
class PeftPlugin(BasePlugin): ...
class QuantizationPlugin(BasePlugin): ...
class DistributedPlugin(BasePlugin): ...
class SequenceParallelModelPlugin(BasePlugin): ...
class SequenceParallelLossPlugin(BasePlugin): ...
```

注册时使用插件名作为路由 key：

```python
@PeftPlugin("lora").register(params=LoraParams, parse_arg="peft_config")
def get_lora_model(model, peft_config, is_train=False):
    ...
```

调用时使用同一个 key：

```python
PeftPlugin("lora")(model, peft_config=peft_config, is_train=True)
```

`BasePlugin` 按 plugin family、plugin name、method name 存储 callable。对于单方法
插件，默认 method 是 `__call__`。

## 多方法 interface

有些 plugin family 会在同一个插件名下暴露多个操作。分布式后端是主要例子：

```python
@DistributedPlugin("fsdp2").register_methods(params=FSDP2Params)
class FSDP2Plugin:
    @staticmethod
    def shard_model(...): ...

    @staticmethod
    def save_model(...): ...

    @staticmethod
    def save_checkpoint(...): ...

    @staticmethod
    def load_checkpoint(...): ...
```

多方法插件只走显式方法，不再提供 default `__call__` 兼容写法。调用侧必须写清楚
这次要做的是 `shard_model`、`save_model` 还是 checkpoint 操作：

```python
DistributedPlugin("fsdp2").shard_model(model, dist_config)
DistributedPlugin("fsdp2").save_checkpoint(model, optimizer, ckpt_dir)
```

## interface.py 的含义

`interface.py` 是某个插件域的轻量注册和分发层。

它应该包含：

- plugin family class，比如 `SequenceParallelModelPlugin`；
- 注册 wrapper，比如 `@SequenceParallelModelPlugin("ulysses")`；
- 小而局部的 dataclass params；
- wrapper 函数内部对实现模块的懒加载 import。

它应该避免：

- 重后端逻辑；
- 可以懒加载的大型框架 import；
- 属于实现模块的 helper，比如应放在 `ulysses.py`、`loss.py`、`bnb.py`、
  `fsdp2.py` 里的内容；
- 通过 `__init__.py` 做包级 public export。

以 sequence parallel 为例，分层是：

```text
model_plugins/sequence_parallel/
  __init__.py   # 空 package marker
  interface.py  # plugin family 与懒加载注册 wrapper
  ulysses.py    # Ulysses SP mode 实现
  loss.py       # sequence-parallel loss 实现
  seq_comm.py   # 通信工具
```

## 懒加载

懒加载通过“把 import 放在已注册 wrapper 函数内部”实现，而不是在模块 import
时加载实现。

示例：

```python
@SequenceParallelModelPlugin("ulysses").register()
def apply_sequence_parallel(model, sp_config):
    from .ulysses import apply_sequence_parallel as apply_ulysses_sequence_parallel

    return apply_ulysses_sequence_parallel(model, sp_config)
```

import `sequence_parallel.interface` 时，只会注册 `ulysses` 这个 key，不会立即 import
完整的 Ulysses 实现。只有真正调用插件时，才会进入 wrapper 内部 import
`.ulysses`。

分布式后端也用同样模式：

```python
@staticmethod
def shard_model(model, dist_config, **kwargs):
    dist_cfg = DistributedPlugin.parse_dist_config("fsdp2", dist_config)
    from .fsdp2 import FSDP2Engine

    return FSDP2Engine(...).shard_model(model)
```

这样可以把可选依赖和重模块留到“被选中的插件真正需要时”再加载。

## 标准参数解析

大多数 plugin family 是“每个 plugin name 对应一个 params dataclass”。

示例：

```python
@PeftPlugin("lora").register(params=LoraParams, parse_arg="peft_config")
def get_lora_model(model, peft_config, is_train=False):
    # peft_config 已经由 BasePlugin 自动解析成 LoraParams
    ...
```

`params=` 声明该插件使用的 params dataclass；`parse_arg=` 声明哪个入参是这个插件
自己的二级配置。`BasePlugin` 会在分发前自动对这个入参执行 `parse_params`：

- alias 映射到 canonical 字段名；
- 拒绝未知 key；
- 检查必填字段；
- 构造 params dataclass。

实现层拿到的是 typed params，而不是原始用户 dict。

`parse_arg` 必须显式声明，原因是插件函数里可能同时存在多个运行时入参，比如
`model`、`init_kwargs`、`is_trainable`、`num_micro_batch`。框架不猜哪个参数叫
config，注册者必须把插件自己的配置入口标出来。

## PEFT 参数记录

PEFT 插件的二级配置统一命名为 `peft_config`，不再使用泛化的 `config` 作为入参名。
这样和模型参数字段 `ModelArguments.peft_config` 保持一致，也和量化侧的
`quant_config` 形成同一套命名规则。

调用侧：

```python
PeftPlugin(self.args.peft_config.name)(
    model,
    peft_config=self.args.peft_config,
    is_train=self.is_train,
)
```

注册侧：

```python
@PeftPlugin("lora").register(params=LoraParams, parse_arg="peft_config")
def get_lora_model(model, peft_config: LoraParams, is_train=False):
    ...

@PeftPlugin("freeze").register(params=FreezeParams, parse_arg="peft_config")
def get_freeze_model(model, peft_config: FreezeParams, is_train=False):
    ...
```

`BasePlugin` 分发前会把 `peft_config` 从 `PluginConfig` 自动解析成对应的 params：
`lora` 对应 `LoraParams`，`freeze` 对应 `FreezeParams`。实现函数只接收 typed params，
不再在函数体里手动调用 `PeftPlugin.parse_params(...)`。

同理，量化插件使用 `quant_config`：

```python
@QuantizationPlugin("bnb").register(params=BnbParams, parse_arg="quant_config")
def quantization_with_bnb(init_kwargs, quant_config: BnbParams, is_trainable=False):
    ...
```

这个规则只覆盖“某个插件自己的二级配置”。如果插件还需要运行时上下文，例如
`model`、`init_kwargs`、`is_train`、`is_trainable`，这些上下文应作为明确命名的普通
入参传入，不混进插件自己的 config。

## dist_config 参数解析

`dist_config` 比较特殊，因为一个扁平 YAML 块里混了多组逻辑：

- backend params，比如 `reshard_after_forward` 或 `config_file`；
- sequence/context parallel params，目前是 `cp_size` 和 `cp_mode`；
- topology params，比如 `timeout`、`dp_size`、`mp_replicate_size`、
  `mp_shard_size`。

分布式插件 family 用 `DistributedPlugin.parse_dist_config` 解析它。

解析结果是：

```python
DistConfig(
    backend=FSDP2Params(...),
    topology=TopologyParams(...),
    sp=SequenceParallelParams(...),
)
```

路由规则是：

- 选中的 backend params dataclass 里的字段进入 `backend`；
- `TopologyParams` 里的字段进入 `topology`；
- `SequenceParallelParams` 里的字段进入 `sp`；
- backend、topology、SP params 之间如果字段名冲突，直接报错；
- 三边都不认识的 key，直接报 unknown。

backend 实现只接收 `dist_cfg.backend`：

```python
FSDP2Engine(asdict(dist_cfg.backend), ...)
```

sequence-parallel model plugin 只接收 `dist_cfg.sp`：

```python
SequenceParallelModelPlugin(dist_cfg.sp.cp_mode)(model, dist_cfg.sp)
```

这样 backend 参数不会泄漏到 SP mode plugin，SP 参数也不会泄漏到 backend engine。

## 端到端流程

以 FSDP2 + Ulysses CP/SP 训练为例：

```yaml
dist_config:
  name: fsdp2
  cp_size: 2
  cp_mode: ulysses
  reshard_after_forward: true
```

流程是：

1. `TrainingArguments.__post_init__` 把 YAML dict 转成 `PluginConfig`。
2. `DistributedInterface(training_args.dist_config)` 从原始 config 里读取 mesh
   相关 key，初始化 process group / device mesh。
3. `BaseTrainer` 用 `DistributedPlugin.parse_dist_config` 对原始 config parse 一次。
4. `BaseTrainer` 把 typed `DistConfig` 传给 distributed backend interface。
5. `FSDP2Plugin.shard_model` 使用 `dist_cfg.backend` 构造 `FSDP2Engine`。
6. 如果 `dist_cfg.sp.cp_size > 1`，`BaseTrainer` 分发到
   `SequenceParallelModelPlugin(dist_cfg.sp.cp_mode)`。
7. `sequence_parallel.interface` 懒加载被选中的 SP mode 实现。
8. 训练过程中，SP loss 通过
   `SequenceParallelLossPlugin("sequence_parallel_loss")` 分发。

这条链路里目前唯一还消费 raw config 的是 `DistributedInterface`，因为它是更底层的
mesh 初始化器，只读取 mesh/topology 相关字段，不实例化 backend engine 或 SP mode
plugin。
