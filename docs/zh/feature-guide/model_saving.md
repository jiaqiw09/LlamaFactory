# 模型保存与恢复

v1 有两种保存行为：训练结束后的最终模型保存，以及训练过程中的 checkpoint 保存。最终模型用于推理或部署；checkpoint 用于断点恢复，包含 optimizer、scheduler、dataloader 和 RNG 等训练状态。

## 最终模型保存

训练正常结束后，`BaseTrainer.save_model()` 会把模型和 processor 保存到 `output_dir`。

```yaml
output_dir: outputs/my_model
```

不同分布式后端的最终保存方式不同，但目标都是生成 HuggingFace `save_pretrained` 格式：

| 后端 | 保存方式 | 说明 |
|------|----------|------|
| 单卡 / DDP | rank 0 调用 `save_pretrained` | DDP 会先取 `model.module` |
| FSDP2 | 收集 full state dict 后 rank 0 保存 | 先把完整权重收集到 CPU 再保存 |
| DeepSpeed | 通过 Accelerate 获取已处理 ZeRO 分片的 state dict | 自动收集 ZeRO-3 参数 |

最终模型目录适合推理加载。LoRA adapter 如果需要合并到基座模型，见 [模型导出](model_export.md)。

## Checkpoint 保存

配置 `save_steps` 或 `save_epochs` 后，训练过程会在 `output_dir` 下写入 `checkpoint-<global_step>` 目录。

```yaml
save_steps: 500
save_epochs: 1.0
save_total_limit: 3
save_ckpt_as_hf: false
```

每个 checkpoint 会写入 `metadata.json`，记录 `global_step`、`epoch` 和 `num_training_steps`。保存完成后 rank 0 写入 `CHECKPOINT_COMPLETE` 标记；`resume_from_checkpoint: auto` 只会选择带有该标记的最新 checkpoint。

## Checkpoint 内容

| 内容 | 单卡 / DDP | FSDP2 | DeepSpeed |
|------|------------|-------|-----------|
| 模型状态 | `model/`，HF 格式 | `model/`，DCP 格式 | Accelerate 状态 |
| 优化器状态 | `optimizer/state_dict.pt` | `optimizer/`，DCP 格式 | Accelerate 状态 |
| Scheduler | `scheduler.pt` | `scheduler.pt` | Accelerate 状态 |
| Dataloader | `dataloader/rank_<rank>.pt` | `dataloader/rank_<rank>.pt` | `dataloader/rank_<rank>.pt` |
| RNG | `rng_state/rank_<rank>.pt` | `rng_state/rank_<rank>.pt` | Accelerate 状态 |
| 完成标记 | `CHECKPOINT_COMPLETE` | `CHECKPOINT_COMPLETE` | `CHECKPOINT_COMPLETE` |

FSDP2 checkpoint 始终保存 DCP 格式，以便恢复分片训练状态。DeepSpeed checkpoint 始终保存 Accelerate / DeepSpeed 状态，以便恢复 ZeRO 状态。

## 额外 HF Checkpoint

`save_ckpt_as_hf: true` 会在中间 checkpoint 中额外保存 HuggingFace 格式模型：

- FSDP2：额外写入 `checkpoint-<step>/hf_model/`
- DeepSpeed：额外写入 `checkpoint-<step>/hf_model/`
- 单卡 / DDP：标准 checkpoint 的 `model/` 已经是 HF 格式，不再额外创建 `hf_model/`

> **注意**：额外 HF checkpoint 会收集完整权重，显著增加保存时的内存和磁盘开销。它适合临时检查或中间模型导出，不是断点恢复所需的主要格式。

## 恢复训练

```yaml
resume_from_checkpoint: auto
```

`auto` 会在 `output_dir` 下查找最新完整 checkpoint。也可以直接指定 checkpoint 路径：

```yaml
resume_from_checkpoint: outputs/my_model/checkpoint-500
```

恢复时会加载训练进度、模型状态、优化器状态、dataloader 状态，并按后端恢复 scheduler 和 RNG。DeepSpeed 的 scheduler 和 RNG 由 Accelerate 状态统一恢复；其他后端由 v1 checkpoint coordinator 单独恢复。

## Checkpoint 保留策略

```yaml
save_total_limit: 3
```

`save_total_limit` 只保留最新的完整 checkpoint。清理时会先删除未写入 `CHECKPOINT_COMPLETE` 的不完整 checkpoint，再删除超出数量限制的最旧完整 checkpoint。

## DCP 初始加载

`dist_config.dcp_path` 是 FSDP2 初始化时用于高效加载分片权重的路径，和 `resume_from_checkpoint` 不是同一个概念：

- `dcp_path`：用于从已有 DCP 权重初始化模型
- `resume_from_checkpoint`：用于恢复一次训练 run 的完整状态

完整参数见 [TrainingArguments](../parameter-reference/training_arguments.md) 和 [DistConfig](../parameter-reference/dist_config.md)。
