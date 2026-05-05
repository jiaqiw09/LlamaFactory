# 训练器插件

训练过程相关的 Plugin 集合，包括分布式后端、批次策略、优化器和学习率调度器。

## DistributedPlugin（分布式训练）

位于 `src/llamafactory/v1/plugins/trainer_plugins/distributed/hub.py`。

### 支持的后端

| 后端 | 注册方法 | 实现文件 |
|------|---------|--------|
| FSDP2 | `__call__`、`save_model`、`save_checkpoint`、`load_checkpoint` | `fsdp2.py` |
| DeepSpeed | `__call__`、`save_model`、`save_checkpoint`、`load_checkpoint` | `deepspeed.py` |

### FSDP2

核心类：`FSDP2Engine`（`distributed/fsdp2.py`）

**模型分片**（`shard_model`）：
1. 根据 `init_mode` 选择权重加载策略
2. `prepare_model`：逐 Transformer Layer `fully_shard`，LoRA 参数独立分片
3. 在 meta device 模式下，通过 `materialize_and_load` 实际加载权重
   - 优先使用 DCP 分布式 checkpoint
   - 否则从 HF checkpoint 逐文件加载（支持 WeightRenaming/WeightConverter）

**权重加载**（`_load_weights_from_hf_checkpoint`）：
- 支持 safetensors 和 pytorch .bin 格式
- 支持权重名映射（WeightRenaming）和权重转换（WeightConverter）
- 分布式 rank0 下载后广播、或各个 rank 按本地缓存只读

**保存/恢复**：
- `save_checkpoint`：DCP 格式 + 可选 HF 格式
- `load_checkpoint`：DCP 格式恢复
- `save_model`：收集 full_state_dict → `save_pretrained`

### DeepSpeed

核心类：`DeepSpeedEngine`（`distributed/deepspeed.py`）

通过 HuggingFace Accelerate 集成：
- `Accelerator(deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=config_file))`
- `accelerator.prepare(model, optimizer, lr_scheduler)` 自动处理 deepspeed.initialize()
- `accelerator.backward(loss)` + `sync_gradients` 管理梯度累积边界
- `accelerator.save_state(ckpt_dir)` / `accelerator.load_state(ckpt_dir)` 管理 checkpoint

### 扩展新分布式后端

```python
@DistributedPlugin("new_backend").register()
def shard_model_new(model, dist_config, **kwargs):
    ...

@DistributedPlugin("new_backend").register("save_model")
def save_model_new(model, output_dir, processor):
    ...

@DistributedPlugin("new_backend").register("save_checkpoint")
def save_checkpoint_new(model, optimizer, ckpt_dir, **kwargs):
    ...

@DistributedPlugin("new_backend").register("load_checkpoint")
def load_checkpoint_new(model, optimizer, ckpt_dir, **kwargs):
    ...
```

## BatchingPlugin（批次策略）

位于 `src/llamafactory/v1/plugins/trainer_plugins/batching.py`。

定义三种批次生命周期方法：

```python
class BatchingPlugin(BasePlugin):
    def compute_length(self, data_provider: DataLoader) -> int:
        """计算 batch generator 长度"""

    def fill_buffer(self, buffer: StatefulBuffer, batch_info: BatchInfo):
        """填充 buffer"""

    def generate_batch(self, buffer: StatefulBuffer, batch_info: BatchInfo) -> list[BatchInput] | None:
        """从 buffer 生成 batch"""
```

当前仅 `NORMAL` 策略已实现（在 `BatchGenerator` 内直接处理）。`PADDING_FREE`、`DYNAMIC_BATCHING`、`DYNAMIC_PADDING_FREE` 预留为 Plugin 扩展。

## OptimizerPlugin（优化器）

位于 `src/llamafactory/v1/plugins/trainer_plugins/optimizer.py`。

```python
class OptimizerPlugin(BasePlugin):
    pass  # 暂无注册实现
```

当前使用默认 AdamW（`torch.optim.AdamW`）。扩展时注册 `OptimizerPlugin("name")`。

## LRSchedulerPlugin（学习率调度器）

位于 `src/llamafactory/v1/plugins/trainer_plugins/lr_scheduler.py`。

```python
class LRSchedulerPlugin(BasePlugin):
    pass  # 暂无注册实现
```

当前使用默认 Constant 调度器（`LambdaLR(lr_lambda=lambda x: 1.0)`）。扩展时注册 `LRSchedulerPlugin("name")`。
