# BaseTrainer

BaseTrainer 是所有训练器的基类，实现了完整的训练循环（train loop），包括 batch 生成、分布式初始化、优化器/调度器管理、checkpoint 和回调系统。

## 初始化顺序

```text
BaseTrainer.__init__
  1. _create_batch_generator()     → BatchGenerator
  2. 计算 num_training_steps        → max_steps 优先于 num_train_epochs
  3. 转换 save_epochs → save_steps
  4. 启用 gradient checkpointing
  5. [DeepSpeed] _init_optimizer + _init_lr_scheduler → engine.prepare()
  6. [FSDP2/DDP] _shard_model → _init_optimizer → _init_lr_scheduler
  7. TrainingCheckpointCoordinator   → 检查是否 resume
  8. CallbackHandler                → LoggingCallback + 用户回调
  9. [CP] SequenceParallelModelPlugin → 替换 attention
```

## 训练循环

```python
def fit(self):
    for epoch in range(start_epoch, total_epochs):
        batch_generator.set_epoch(epoch)
        for micro_batches in batch_generator:
            step_loss = 0
            for i, micro_batch in enumerate(micro_batches):
                # [CP] → sequence_parallel_loss
                # 普通 → self.compute_loss(micro_batch)
                loss = loss * valid_tokens * dp_size / total_valid_tokens

                # [DeepSpeed] → engine.backward() + auto step
                # [FSDP2/DDP] → loss.backward() + clip_grad + optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            # all_reduce loss, grad_norm
            # callbacks: on_step_end
            # logging, checkpoint save
```

### Gradient Accumulation

通过 `global_batch_size` 和 `micro_batch_size` 自动计算 micro-batch 数：

```text
num_micro_batch = global_batch_size / (dp_size * micro_batch_size)
```

每个 micro-batch 独立 forward/backward，最后一个 micro-batch 时触发 optimizer.step()（DeepSpeed）或手动 clip_grad + step。

### Loss 计算

SFT 的 loss 计算在 `SFTTrainer.compute_loss()`：

```python
def compute_loss(self, batch):
    log_probs = self.compute_log_probs(self.model, batch)
    loss = (-log_probs * shift_loss_weights).sum() / shift_loss_weights.sum()
```

`compute_log_probs` 在基类中实现：forward → 取 logits → cross_entropy(reduction="none")。

## 分布式后端集成

| 后端 | shard_model | optimizer | scheduler | backward |
|------|------------|-----------|-----------|----------|
| FSDP2 | `fully_shard(model)` | 标准 AdamW | 标准 LambdaLR | `loss.backward()` |
| DeepSpeed | `no-op` | 在 prepare 中创建 | 在 prepare 中创建 | `engine.backward()` + auto step |
| DDP | `DDP(model)` | 标准 AdamW | 标准 LambdaLR | `loss.backward()` |

## Callback 系统

```python
class TrainerCallback:
    def on_train_begin(...)
    def on_epoch_begin(...)
    def on_step_begin(...)
    def on_step_end(...)
    def on_log(...)
    def on_epoch_end(...)
    def on_train_end(...)
    def on_save(...)
```

内置 `LoggingCallback` 记录每步的 loss、grad_norm、learning_rate，输出到 stdout 和 JSONL 文件。

## Checkpoint 系统

`TrainingCheckpointCoordinator` 管理完整的训练状态保存和恢复：

**保存内容**：
- Model weights（DCP / HF / Accelerate 格式）
- Optimizer state
- LR Scheduler state
- DataLoader state（batch_generator.state_dict()）
- RNG state（python、numpy、torch、accelerator）
- Metadata（global_step、epoch）

**恢复**：通过 `resume_from_checkpoint` 恢复完整状态，实现精确断点续训。

## 扩展新训练器

继承 `BaseTrainer`，实现 `compute_loss()`：

```python
class MyTrainer(BaseTrainer):
    def compute_loss(self, batch):
        ...
```

参考 `SFTTrainer` 的实现。
