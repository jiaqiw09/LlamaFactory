# 训练器插件

训练循环用到的可替换部件：分布式后端、批次策略、优化器、学习率调度器。当前只有 `DistributedPlugin` 有实际实现，其它三个是预留扩展点。

代码位置：`src/llamafactory/v1/plugins/trainer_plugins/`。

## DistributedPlugin

`distributed/hub.py` 是统一入口，每个后端注册四个方法：

| 方法 | 行为 |
|------|------|
| `__call__` | shard 模型，返回包装后的 model 或 engine |
| `save_model` | 最终保存（HF 格式） |
| `save_checkpoint` | 中间保存（恢复用） |
| `load_checkpoint` | 中间恢复 |

```python
class DistributedPlugin(BasePlugin):
    def __call__(self, model: HFModel, dist_config: PluginConfig, **kwargs) -> HFModel:
        return super().__call__(model, dist_config, **kwargs)
```

### `fsdp2`

实现位于 `distributed/fsdp2.py`，核心类 `FSDP2Engine`：

- `prepare_model`：识别 transformer layer 类（先看 `_no_split_modules`，再 fallback 到 `model.model.layers[0]`），逐层 `fully_shard`；LoRA 模块单独识别后独立分片；最后整模型再 `fully_shard` 一次
- 混合精度：参数 `bf16` → `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`，否则全 fp32
- offload：`offload_params=true` 时使用 `CPUOffloadPolicy(pin_memory=...)`
- `shard_model` 按 `model._init_mode` 分支：
  - `init_on_rank0`：`prepare_model` → `to_empty(device)` → 从 rank 0 broadcast 完整 state dict
  - `init_on_meta`：`prepare_model` → `materialize_and_load`（优先 DCP，否则从 HF checkpoint 逐 shard 加载）
  - `init_on_default`：直接 `prepare_model`

完成后调用 `_warmup_grad_norm`：先给所有 `requires_grad` 参数挂零梯度，跑一次 `clip_grad_norm_` 来初始化 NCCL 通信组，避免训练第一步时延迟过高。

`save_checkpoint`：DCP 格式 model + optimizer，可选并存一份 HF 格式（`save_ckpt_as_hf=true` 时）。
`load_checkpoint`：DCP 恢复 model + optimizer。

字段细节见 [DistConfig](../../parameter-reference/dist_config.md) FSDP2 段。

### `deepspeed`

实现位于 `distributed/deepspeed.py`，核心类 `DeepSpeedEngine`：

```python
ds_plugin = DeepSpeedPlugin(hf_ds_config=config_file)
ds_plugin.set_mixed_precision(infer_deepspeed_mixed_precision(...))
self.accelerator = Accelerator(deepspeed_plugin=ds_plugin, gradient_accumulation_steps=num_micro_batch)
```

约束：

- `dist_config.config_file` 必填，指向 DeepSpeed 原生 JSON
- `train_micro_batch_size_per_gpu` 若为 `"auto"`，由 `micro_batch_size` 填回，避免 `accelerate.prepare` 需要 dataloader 推断

`shard_model` 在这条路径上是 no-op，真正 wrap 发生在 `prepare(model, optimizer, lr_scheduler)`：

- 内部触发 `deepspeed.initialize`
- `model._accelerator` 被设为 `Accelerator` 实例，方便后续 `save_state` / `load_state` 反查
- backward 用 `accelerator.backward(loss)`，optimizer.step 在 sync 边界自动触发

`save_checkpoint`：`accelerator.save_state(ckpt_dir)`，可选 HF 格式。
`load_checkpoint`：`accelerator.load_state(ckpt_dir)`。

### 注册新分布式后端

```python
@DistributedPlugin("my_backend").register()
def shard_model(model, dist_config, **kwargs): ...

@DistributedPlugin("my_backend").register("save_model")
def save_model(model, output_dir, processor): ...

@DistributedPlugin("my_backend").register("save_checkpoint")
def save_checkpoint(model, optimizer, ckpt_dir, **kwargs): ...

@DistributedPlugin("my_backend").register("load_checkpoint")
def load_checkpoint(model, optimizer, ckpt_dir, **kwargs): ...
```

`BaseTrainer` 还会按 `name == "deepspeed"` 走特殊分支（先创建 engine，再创建 optimizer，再 `engine.prepare`）。新增类似的"先 wrap 再创建 optimizer"后端时需要同步修改 `BaseTrainer.__init__` 里的 dist 分支。

## BatchingPlugin

`batching.py` 定义抽象接口：

```python
class BatchingPlugin(BasePlugin):
    def compute_length(self, data_provider: DataLoader) -> int: ...
    def fill_buffer(self, buffer: StatefulBuffer, batch_info: BatchInfo) -> None: ...
    def generate_batch(self, buffer, batch_info) -> list[BatchInput] | None: ...
```

`NORMAL` 策略由 `BatchGenerator` 直接实现（不走插件），其它三种 `BatchingStrategy` 值留给插件扩展，但目前还没有注册过任何 name —— `BatchGenerator._init_data_provider` 在非 NORMAL 路径上直接 `NotImplementedError`。

## OptimizerPlugin

```python
class OptimizerPlugin(BasePlugin):
    pass
```

预留位。`BaseTrainer._init_optimizer` 在 `optim_config is None` 时使用默认 `torch.optim.AdamW(lr=learning_rate)`，否则调 `OptimizerPlugin(name)(model, optim_config)` —— 但目前没有任何注册。

## LRSchedulerPlugin

```python
class LRSchedulerPlugin(BasePlugin):
    pass
```

预留位。默认使用 `LambdaLR(lambda x: 1.0)`（即恒定学习率）；自定义时调用 `LRSchedulerPlugin(name)(optimizer, num_training_steps, lr_scheduler_config)`。
