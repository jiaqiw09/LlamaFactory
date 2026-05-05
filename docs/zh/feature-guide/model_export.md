# 模型导出

模型导出用于把 LoRA adapter 合并到基座模型，并保存为可部署的 HuggingFace 格式目录。

> **注意**：训练过程中的 checkpoint、DCP 保存、断点恢复和最终模型保存见 [模型保存与恢复](model_saving.md)。本页只覆盖 `llamafactory-cli merge` 导出路径。

## LoRA Adapter 合并

```bash
export USE_V1=1
llamafactory-cli merge config.yaml
```

```yaml
model: Qwen/Qwen3-0.6B
peft_config:
  name: lora
  adapter_name_or_path:
    - path/to/adapter_1   # 支持合并多个 adapter
  export_dir: ./exported_model
  export_size: 5           # 分片大小 (GB)
  export_hub_model_id: null
  infer_dtype: auto        # auto / float16 / float32 / bfloat16
```

完整导出参数见 [PeftConfig](../parameter-reference/peft_config.md)。

## 适用场景

- LoRA 训练后，希望得到合并后的完整模型目录
- 推理或部署系统不直接加载 PEFT adapter
- 需要将合并模型推送到 Hub

全参数微调模型训练完成后已经保存到 `output_dir`，通常无需再走 merge 导出。
