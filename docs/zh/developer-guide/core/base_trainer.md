# BaseTrainer

`BaseTrainer` 实现 v1 的训练循环骨架：批次生成、分布式集成、优化器/调度器、checkpoint、callback。具体训练算法（如 SFT 的 NLL）由子类覆写 `compute_loss` 提供。

代码位置：`src/llamafactory/v1/core/base_trainer.py`。

## 接口

```python
class BaseTrainer:
    def __init__(self, args, model, renderer, train_dataset, callbacks=None): ...
    def fit(self) -> None: ...
    def save_model(self) -> None: ...

    @abstractmethod
    def compute_loss(self, batch: BatchInput) -> Tensor: ...

    def compute_log_probs(self, model, batch) -> Tensor: ...
```

实例化时必须传入已加载的 `model`、`renderer` 和 `train_dataset`，由调用方先用 `ModelEngine` / `DataEngine` 准备好；`BaseTrainer` 不重新加载它们。

## 初始化顺序

```text
__init__
  1. _create_batch_generator()                             BatchGenerator
  2. num_training_steps                                    max_steps 优先于 num_train_epochs * len(generator)
  3. save_epochs → save_steps                              steps_per_epoch * save_epochs
  4. enable_activation_checkpointing → gradient_checkpointing_enable
  5. 分布式分支：
       deepspeed: DistributedPlugin("deepspeed")(...)
                  _init_optimizer / _init_lr_scheduler
                  engine.prepare(model, optimizer, lr_scheduler)
       其它（FSDP2 / DDP / 单卡）:
                  _shard_model → _init_optimizer → _init_lr_scheduler
  6. TrainingCheckpointCoordinator + 可选 resume
  7. CallbackHandler([LoggingCallback()] + 用户回调)
  8. TrainerState 初始化
  9. cp_size > 1：SequenceParallelModelPlugin(cp_mode) 替换 attention
```

> **注**：DeepSpeed 的 optimizer 必须在 `accelerator.prepare` 里和模型一起 wrap，所以分布式分支必须先于 optimizer 初始化；FSDP2 与之相反，`fully_shard` 会替换参数对象，optimizer 必须在分片完成之后构造。

## 分布式分支差异

| 后端 | model | optimizer | scheduler | backward |
|------|-------|-----------|-----------|----------|
| 单卡 | 原模型 | `AdamW` | `LambdaLR(lambda x: 1.0)` | `loss.backward()` |
| DDP | `DDP(model)` | 同上 | 同上 | `loss.backward()` |
| FSDP2 | `fully_shard(model, ...)` | 同上 | 同上 | `loss.backward()` + `clip_grad_norm_` + `optimizer.step()` |
| DeepSpeed | `accelerator.prepare(model, ...)` | accelerate 包装 | accelerate 包装 | `accelerator.backward(loss)`，optimizer.step 由 engine 在 sync 边界自动触发 |

## fit 循环

```text
on_train_begin
while global_step < num_training_steps:
    set_epoch(epoch)
    on_epoch_begin
    for micro_batches in train_batch_generator:           # 一个 global step 的全部 micro batch
        global_step += 1
        on_step_begin
        step_valid_tokens = all_reduce(SUM)               # 跨 DP
        for i, mb in enumerate(micro_batches):
            loss = (cp_size>1) ? sequence_parallel_loss : compute_loss(mb)
            loss = loss * mb_valid_tokens * dp_size / step_valid_tokens
            if deepspeed:
                accelerator.sync_gradients = (i == last)
                engine.backward(loss)
            else:
                loss.backward()
            step_loss += loss.item()
        if not deepspeed:
            grad_norm = clip_grad_norm_(...)
            if cp_size>1: grad_norm 跨 CP 维 all_reduce
            if isfinite: optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        all_reduce([step_loss, grad_norm])
        sync()
        on_step_end
        if global_step % logging_steps == 0: on_log
        if save_steps and global_step % save_steps == 0: checkpoint.save
        if global_step >= num_training_steps: on_epoch_end + on_train_end + return
    on_epoch_end
on_train_end
```

要点：

- 梯度累积步数 = `len(micro_batches)` = `num_micro_batch`，由 `BatchGenerator` 根据 `global_batch_size / (dp_size * micro_batch_size)` 推导
- Loss 缩放：每个 micro batch 的 loss 按"该 micro batch 有效 token 数 / 整个 step 全局有效 token 数"加权，再乘以 `dp_size`，因为 FSDP 是 mean reduction
- 非有限 grad_norm 时跳过 `optimizer.step`，但 `scheduler.step` 仍执行（保持 lr 进度）
- DeepSpeed 路径不做 `clip_grad_norm_`，由 engine 自带

## compute_loss

子类实现具体损失。`BaseTrainer` 提供基础工具 `compute_log_probs`：

```python
outputs = model(**model_inputs)
logits = outputs.logits.float()
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
return -F.cross_entropy(shift_logits, shift_labels, reduction="none").view(B, -1)
```

SFT 子类对 `log_probs` 用 `loss_weights` 做加权聚合。新增训练算法只需要实现 `compute_loss(batch) -> Tensor`，如 RM 取 chosen/rejected logits 差，DPO 用参考模型计算偏好。

## save_model

```python
if dist_name in ("fsdp2", "deepspeed"):
    DistributedPlugin(dist_name).save_model(model, output_dir, processor)
else:
    model.save_pretrained(output_dir, max_shard_size="4GB")
    processor.save_pretrained(output_dir)
```

最后触发 `on_save` 回调。Checkpoint（恢复用）由 `TrainingCheckpointCoordinator` 管理，机制见 [model_saving](../../feature-guide/model_saving.md)。

## Callback

`CallbackHandler` 在每个生命周期点 fanout：

```text
on_train_begin
  ├─ on_epoch_begin
  │   ├─ on_step_begin
  │   ├─ on_step_end
  │   ├─ on_log    (logging_steps 命中时)
  │   ├─ on_save   (save_steps 命中时)
  │   └─ ...
  └─ on_epoch_end
on_train_end
```

接口契约见 [callback](callback.md)。
