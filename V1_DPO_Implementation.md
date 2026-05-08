# v1 DPO 端到端实现说明

## 一、整体链路

入口在 `LlamaFactory/src/llamafactory/v1/trainers/dpo_trainer.py`，启动方式：

```bash
python -m llamafactory.v1.trainers.dpo_trainer \
    --model Qwen/Qwen3-0.6B \
    --template qwen3_nothink \
    --train_dataset data/v1_dpo_demo.yaml \
    --dpo_ref_model Qwen/Qwen3-0.6B   # 全参时必填；LoRA 时省略
```

数据 → 模型 → 训练循环的全流程：

```
DataEngine                                     ModelEngine
  │                                              │
  ├─ 读 chosen_messages / rejected_messages       ├─ 加载 base model
  ├─ Renderer 分别渲染 → 各自加 token_type_ids     ├─ (可选) get_peft_model 包 LoRA
  └─ 同 batch 输出 [chosen…, rejected…] (2N)       │
        │                                         │
        └────────────┬────────────────────────────┘
                     ↓
                DPOTrainer
                     │
                     ├─ super().__init__()  → BaseTrainer 准备 policy（DDP/FSDP2/DS）
                     ├─ _prepare_ref_model() → ref 走对应分布式路径（或 LoRA 跳过）
                     └─ fit()
                          └─ compute_loss(batch)
                                ├─ compute_segment_logps(policy)
                                ├─ ref forward (no_grad + _ref_context)
                                ├─ DPO sigmoid loss
                                └─ 返回 (loss, metrics)
```

## 二、已实现

### 2.1 算法核心
- **标准 DPO 损失**：`loss = -logsigmoid(β · (Δlogratio_chosen − Δlogratio_rejected)).mean()`
- **token-level logp 聚合**：通过 `token_type_ids=1/2` 区分 chosen/rejected token，shift + log_softmax + gather + 求和
- **`use_cache=False`**：训练 forward 不建 KV cache，避免显存浪费

### 2.2 Reference model 三条路径

| dist | 全参 ref | LoRA ref |
|---|---|---|
| `dist_config=None`（DDP fallback / 单卡） | 每卡完整 replicate | 共享 backbone（`disable_adapter()`） |
| `fsdp2` | FSDP2 sharding | 共享 backbone |
| `deepspeed` | 单独 `deepspeed.initialize`：ZeRO-3 分片 / ZeRO≤2 降为 stage 0 | 共享 backbone |

LoRA 路径自动触发：`model_args.peft_config.name == "lora"` 时 `ref_model=None`，ref forward 走 `policy.disable_adapter()` 上下文，单卡只占一份模型权重。

### 2.3 训练观测
每个 `logging_steps` 边界，logs 含：

```
loss, grad_norm, learning_rate, epoch, step
rewards/chosen, rewards/rejected
rewards/accuracies   ← DPO 健康度首要指标
rewards/margins
logps/chosen, logps/rejected
```

实现方式：`DPOTrainer.compute_loss` 返回 `(loss, metrics_dict)`；`BaseTrainer.fit` 在 micro-batch 内做平均累加，step 末尾跨 DP all-reduce mean，再 merge 进 logs。

### 2.4 工作流
- `run_dpo` 接 `callbacks` 形参并真正传给 trainer
- LoRA / 全参分流自动判定
- `trainer.fit()` + `trainer.save_model()` + `DistributedInterface().destroy()`

## 三、可观测的"训练在学"信号

启动后看 `rewards/accuracies` 这一项：
- 起步 ≈ 0.5（policy 等于 ref，分不出好坏）
- 正常应**单调爬升到 0.7+**
- 一直在 0.5 附近 → 数据有问题或 β 不合适
- **掉到 0.4 以下** → 偏好方向反了（chosen/rejected 标签可能搞错）

只看 loss 不够——loss 在降但 accuracy 反向是 DPO 常见坑。

## 四、欠缺（按优先级）

### A. 算法变体（用户主动延后）
- **多 loss 类型**：SimPO / ORPO / IPO / BCO 都没实现，`dpo_beta` 之外的相关超参也未接入（`simpo_gamma`、`pref_ftx` 等）
- **`dpo_label_smoothing`**：args 里有字段但 loss 里没用到（cDPO）
- **SFT 辅助 loss（`pref_ftx`）**：DPO + SFT 混合训练入口缺失
- **`ld_alpha`**：长度去敏感未实现
- **平均 logp 归一化**：IPO/ORPO/SimPO 需要 `logps / valid_length`，目前只有 sum

### B. 性能/工程优化
- **Precompute reference logps**：ref 不变时可以预计算整个数据集的 ref logp 存盘，训练时直接读，省一份 ref forward 的算力。当前每 step 都重算
- **峰值显存**：`run_dpo` 里 ref_model 先加载为完整未分片状态，再到 `_prepare_ref_model` 才 wrap，中间有"两份完整模型同时在显存"的瞬间。大模型 + ZeRO-3 时需要余量
- **LoRA continue-training 路径未支持**：当前 LoRA 只支持"从 base 开始训新 adapter"，TRL 风格的"复制一份冻结 ref adapter，切 active adapter 当 ref"还没实现。"先 SFT-LoRA 再 DPO-LoRA"这种串行训练目前不能用 LoRA 共享 backbone（必须显式传 `dpo_ref_model`）
- **`enable_input_require_grads()`**：LoRA + gradient checkpointing 时需要这一行，否则 grad 不会传到 adapter。当前没加，未来打开这个组合会出问题

### C. 训练流程
- **没有 eval / validation 分支**：只跑训练，没有定期在 dev set 上算 reward accuracy
- **没有 resume from checkpoint**：中断后只能从头训
- **disable_dropout**：policy 和 ref 训练时都开 dropout 会让 ref logp 抖动，破坏 KL 约束稳定性。legacy DPO trainer 有这一步，v1 没做
- **NPU + DeepSpeed 路径未测**：deepspeed 在昇腾上的 ZeRO 行为依版本而定，代码里没有 NPU 特判

## 五、建议下一步

按收益从大到小：
1. **先验证当前链路**：单卡 LoRA 跑 demo，确认 `rewards/accuracies` 在涨
2. **加 SimPO / ORPO**：不需要 ref_model，最快验证算法变体接入是否通畅
3. **加 label_smoothing + pref_ftx**：实现简单、收益明确
4. **加 disable_dropout + LoRA gradient checkpointing 修复**：稳定性
5. **eval 分支**：让训练曲线更可信
6. **Precompute ref logps + LoRA continue-training**：等你训大模型 / 串行训练时再补
