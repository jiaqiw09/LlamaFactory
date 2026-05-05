# Callback 系统

Callback 系统提供训练过程中的钩子机制，用于日志记录、指标追踪等自定义行为。

## 设计动机

训练循环（`BaseTrainer.fit()`）是固定的，但不同场景需要在特定时机执行额外逻辑（如日志输出、指标上报、早停等）。Callback 系统通过钩子方法将扩展点暴露给用户，无需修改训练循环本身。

## 核心类

### TrainerCallback

位于 `src/llamafactory/v1/utils/callbacks/trainer_callback.py`，抽象基类。

```python
class TrainerCallback:
    def on_train_begin(self, trainer): ...
    def on_epoch_begin(self, trainer, epoch): ...
    def on_step_begin(self, trainer, step): ...
    def on_step_end(self, trainer, step, loss, grad_norm, lr): ...
    def on_log(self, trainer, logs): ...
    def on_epoch_end(self, trainer, epoch): ...
    def on_train_end(self, trainer): ...
    def on_save(self, trainer, output_dir): ...
```

### CallbackHandler

管理多个 Callback，在对应时机逐一调用：

```python
class CallbackHandler:
    def __init__(self, callbacks: list[TrainerCallback]): ...

    def on_step_end(self, trainer, step, loss, grad_norm, lr):
        for callback in self.callbacks:
            callback.on_step_end(trainer, step, loss, grad_norm, lr)
```

### TrainerState

训练状态数据类，记录当前训练进度：

```python
@dataclass
class TrainerState:
    global_step: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    log_history: list = field(default_factory=list)
```

### LoggingCallback

内置 Callback，位于 `src/llamafactory/v1/utils/callbacks/logging_callback.py`：

- `on_step_end`：在 rank 0 上输出 loss、grad_norm、learning_rate 到 stdout
- `on_log`：将日志追加到 `trainer_log.jsonl` 文件

## 扩展点

创建自定义 Callback：

```python
class MyCallback(TrainerCallback):
    def on_step_end(self, trainer, step, loss, grad_norm, lr):
        if step % 100 == 0:
            print(f"Step {step}: loss={loss:.4f}")
```

在 `BaseTrainer` 初始化时传入 Callback 列表。
