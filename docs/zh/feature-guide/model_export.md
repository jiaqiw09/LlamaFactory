# 模型导出

把 LoRA adapter 合并进基座模型，输出一份 HuggingFace `save_pretrained` 目录。需要中间 / 最终 checkpoint 的保存与恢复见 [模型保存与恢复](model_saving.md)。

## 命令

```bash
export USE_V1=1
llamafactory-cli merge config.yaml
```

仅对 `peft_config.name == "lora"` 生效；全参数微调直接使用 `output_dir` 中的最终模型即可，无需走 merge。

## 配置

```yaml
model: Qwen/Qwen3-4B

peft_config:
  name: lora
  adapter_name_or_path:                # 单个或列表，列表会被逐一 merge_and_unload
    - outputs/run/checkpoint-1000
  export_dir: ./merged_model
  export_size: 5                       # safetensors 分片大小（GB）
  export_hub_model_id: null            # 同时推送到 HF Hub
  infer_dtype: auto                    # auto / float16 / float32 / bfloat16
  export_legacy_format: false          # true 输出 .bin 旧格式
```

字段细节见 [PeftConfig](../parameter-reference/peft_config.md) 的"导出参数"段。

`infer_dtype: auto` 的语义：模型当前是 fp32 且当前设备支持 bf16 时转 bf16，否则保持原 dtype。

## 适用场景

- LoRA 训练后想得到一份"无 adapter 包袱"的完整模型目录
- 部署框架（vLLM、TGI、SGLang 等）不接受 PEFT adapter
- 希望一次推到 Hub 共享

不需要走 merge 的情况：

- 全参数微调：`output_dir` 已经是完整模型
- 推理时直接用 LoRA adapter（参考 [推理与部署](inference.md)）
