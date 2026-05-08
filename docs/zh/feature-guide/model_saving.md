# 模型保存与恢复

v1 区分两种保存：

- **最终模型**：训练结束 `BaseTrainer.save_model()` 把权重写到 `output_dir`，用于推理 / 部署
- **中间 checkpoint**：训练过程中按 `save_steps` / `save_epochs` 写出，包含 optimizer / scheduler / dataloader / RNG 状态，用于断点续训

两种保存路径不同，文件结构也不同。

## 最终模型保存

```yaml
output_dir: outputs/my_model
```

不同后端的最终保存方式：

| 后端 | 保存方式 | 说明 |
|------|----------|------|
| 单卡 / DDP | rank 0 调 `save_pretrained` | DDP 自动取 `model.module` |
| FSDP2 | 收集 full state dict 到 CPU 再 rank 0 保存 | 仍是标准 HF 格式，可直接被 `from_pretrained` 加载 |
| DeepSpeed | `accelerator.get_state_dict` 收集 ZeRO 分片 | 自动处理 ZeRO-3 |

最终目录可被 `transformers.from_pretrained` 直接加载。LoRA 想要合并进基座模型见 [模型导出](model_export.md)。

## 中间 checkpoint

启用任一项即可：

```yaml
save_steps: 500
save_epochs: 1.0
```

每次保存写入 `output_dir/checkpoint-<global_step>/`。`save_epochs` 会按 `steps_per_epoch * save_epochs` 推导成 `save_steps`，两者只取一个。

每个 checkpoint 都会写一份 `metadata.json`（含 `global_step` / `epoch` / `num_training_steps`），并在保存完成后由 rank 0 写入 `CHECKPOINT_COMPLETE` 标记。`resume_from_checkpoint: auto` 只会挑带这个标记的最新目录。

## checkpoint 内容

| 内容 | 单卡 / DDP | FSDP2 | DeepSpeed |
|------|-----------|-------|-----------|
| 模型权重 | `model/`，HF 格式 | `model/`，DCP 格式 | Accelerate state |
| 优化器 | `optimizer/state_dict.pt` | `optimizer/`，DCP 格式 | Accelerate state |
| Scheduler | `scheduler.pt` | `scheduler.pt` | Accelerate state |
| Dataloader | `dataloader/rank_<rank>.pt` | `dataloader/rank_<rank>.pt` | `dataloader/rank_<rank>.pt` |
| RNG | `rng_state/rank_<rank>.pt` | `rng_state/rank_<rank>.pt` | Accelerate state |
| 完成标记 | `CHECKPOINT_COMPLETE` | `CHECKPOINT_COMPLETE` | `CHECKPOINT_COMPLETE` |

FSDP2 始终用 DCP（torch distributed checkpoint）保存，恢复时直接读分片，不需要先聚合。DeepSpeed 走 Accelerate 内置的 `save_state` / `load_state`，把 ZeRO 状态、scheduler、RNG 一起打包。

## 同时保存一份 HF 格式

```yaml
save_ckpt_as_hf: true
```

效果：

- FSDP2 / DeepSpeed：在 `checkpoint-<step>/hf_model/` 额外写一份完整 HF 格式
- 单卡 / DDP：标准 checkpoint 已经是 HF 格式，不会再额外创建 `hf_model/`

> **注**：额外 HF checkpoint 需要把完整权重收集到 CPU，显著增加保存时的内存开销。它适合临时检查或导出，不是恢复必需。

## 断点续训

```yaml
resume_from_checkpoint: auto
```

`auto` 在 `output_dir` 下找最新带 `CHECKPOINT_COMPLETE` 的目录。也可以直接给路径：

```yaml
resume_from_checkpoint: outputs/my_model/checkpoint-500
```

恢复时按后端把模型 / optimizer / scheduler / dataloader / RNG 全部回放，做到与原训练 run 字节级一致的进度。DeepSpeed 路径下 scheduler 与 RNG 由 Accelerate 一起恢复；其它路径由 v1 自己的 `TrainingCheckpointCoordinator` 分别恢复。

## 保留策略

```yaml
save_total_limit: 3
```

仅保留最新的 3 个**带完成标记**的 checkpoint。每次保存前会清掉所有不完整目录（无 `CHECKPOINT_COMPLETE`），再按时间顺序删超出数量的最旧目录。

## DCP 初始加载（容易混淆）

`dist_config.dcp_path` 与 `resume_from_checkpoint` 不是同一件事：

| 字段 | 用途 |
|------|------|
| `dist_config.dcp_path` | FSDP2 + `init_on_meta` 时用 DCP 格式直接初始化分片权重，避开从 HF checkpoint 逐 shard 加载 |
| `resume_from_checkpoint` | 恢复一次完整训练 run 的所有状态 |

完整字段见 [TrainingArguments](../parameter-reference/training_arguments.md) 与 [DistConfig](../parameter-reference/dist_config.md)。
