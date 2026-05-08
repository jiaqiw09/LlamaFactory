# Callback

Callback 是观察者：在不修改训练循环代码的前提下，让外部代码在生命周期的固定时间点拿到当前状态做记录、上报、保存等动作。`BaseTrainer.fit` 在每个时间点调用 `CallbackHandler` 把事件转发给所有注册的 callback。

代码位置：`src/llamafactory/v1/utils/callbacks/`。

> **注**：`core/utils/callback.py` 是空文件，所有真实代码都在 `utils/callbacks/`。

## 接口

```python
class TrainerCallback:
    def on_train_begin(self, args, state, **kwargs): ...
    def on_train_end(self, args, state, **kwargs): ...
    def on_epoch_begin(self, args, state, **kwargs): ...
    def on_epoch_end(self, args, state, **kwargs): ...
    def on_step_begin(self, args, state, **kwargs): ...
    def on_step_end(self, args, state, **kwargs): ...
    def on_log(self, args, state, logs, **kwargs): ...
    def on_save(self, args, state, **kwargs): ...
```

每个钩子都接收：

- `args`：当前训练的 `TrainingArguments`，只读
- `state`：`TrainerState` 快照，只读
- `**kwargs`：当前不强制使用，但 `CallbackHandler` 会传入 `model` / `optimizer` / `lr_scheduler` / `train_dataloader`

callback 是**观察者**，不应修改训练流程。

## 调用顺序

```text
on_train_begin
  for each epoch:
    on_epoch_begin
      for each step:
        on_step_begin
          (forward / backward / optimizer.step)
        on_step_end
        [on_log]    # 命中 logging_steps
        [on_save]   # 命中 save_steps
    on_epoch_end
on_train_end
```

`on_log` 与 `on_save` 不是每步都触发，由 `BaseTrainer` 按 `logging_steps` / `save_steps` 决定。`on_save` 由 `BaseTrainer.save_model` 在最终保存时触发；中间 checkpoint 由 `TrainingCheckpointCoordinator` 写入，本身不直接触发 `on_save`。

## TrainerState

```python
@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    num_training_steps: int = 0
    loss: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    log_history: list[dict[str, Any]] = field(default_factory=list)
```

`BaseTrainer` 在每个 step 末尾把当前指标写入 `state`，callback 应把它当只读快照看待。`log_history` 由 `LoggingCallback` 在 `on_log` 时追加。

## CallbackHandler

```python
handler = CallbackHandler([LoggingCallback()], trainer=trainer)
handler.add_callback(MyWandbCallback())
handler.on_step_end(args, state)
```

`_call` 内部会从 `trainer` 取 `model` / `optimizer` / `lr_scheduler` / `train_dataloader`（即 `train_batch_generator`），通过 `**kwargs` 传给 callback——需要它们时直接 `kwargs.get(...)` 即可。

## 内置 LoggingCallback

```python
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, logs, **kwargs):
        # 1. 追加到 state.log_history（所有 rank）
        # 2. rank 0：stdout 打印 + 写入 <output_dir>/trainer_log.jsonl
```

只 hook 了 `on_log`：`BaseTrainer` 在命中 `logging_steps` 时构造 `logs` dict（含 `epoch` / `step` / `loss` / `grad_norm` / `learning_rate`），交给 callback 决定如何展示和持久化。每行 JSONL 都附带 `total_steps`，崩溃后日志可直接重读。

## 写一个新 callback

```python
from llamafactory.v1.utils.callbacks import TrainerCallback

class WandbCallback(TrainerCallback):
    def on_train_begin(self, args, state, **kwargs):
        import wandb
        wandb.init(project="lmf", config=vars(args))

    def on_log(self, args, state, logs, **kwargs):
        import wandb
        wandb.log(logs, step=state.global_step)

    def on_train_end(self, args, state, **kwargs):
        import wandb
        wandb.finish()
```

注册：

```python
trainer = SFTTrainer(args, model, renderer, dataset, callbacks=[WandbCallback()])
```

`BaseTrainer` 会把 `callbacks` 追加到 handler，`LoggingCallback` 默认始终在最前。
