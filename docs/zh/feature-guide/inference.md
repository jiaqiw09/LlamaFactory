# 推理与部署

v1 提供 HuggingFace 推理引擎和 CLI 对话模式。

## 对话模式

```bash
export USE_V1=1
llamafactory-cli chat
```

## 推理配置

推理相关参数见 [SampleArguments](../parameter-reference/sample_arguments.md)。

```yaml
sample_backend: hf       # hf（已实现）/ vllm（预留）
max_new_tokens: 512
```

并发控制：`MAX_CONCURRENT` 环境变量，默认 1。

## LoRA 模型使用

训练后 LoRA adapter 可直接加载使用（无需额外导出）：

```yaml
model: Qwen/Qwen3-0.6B
peft_config:
  name: lora
  adapter_name_or_path: path/to/adapter
```

如需将 LoRA adapter 合并到基座模型或推送到 Hub，请参考 [模型导出](model_export.md)。
