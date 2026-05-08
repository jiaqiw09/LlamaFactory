# 推理与部署

v1 内置基于 HuggingFace `generate` 的推理引擎，提供 CLI 对话和批量推理两种入口。vLLM 后端目前为预留状态。

## CLI 对话

```bash
export USE_V1=1
llamafactory-cli chat config.yaml
```

`config.yaml` 一般只需要指定模型与采样参数：

```yaml
model: Qwen/Qwen3-0.6B
template: qwen3_nothink

sample_backend: hf
max_new_tokens: 512
```

进入 REPL 后：

- 直接输入即对话
- `clear` 清空历史
- `exit` 退出

`max_new_tokens` 等字段见 [SampleArguments](../parameter-reference/sample_arguments.md)。

## 加载 LoRA adapter

LoRA 训练后无需合并即可用于推理：

```yaml
model: Qwen/Qwen3-0.6B
template: qwen3_nothink

peft_config:
  name: lora
  adapter_name_or_path: outputs/run/checkpoint-1000   # 单个 adapter

sample_backend: hf
max_new_tokens: 512
```

推理路径下，`adapter_name_or_path` 也可以传多个 adapter，会被依次 `merge_and_unload`：

```yaml
peft_config:
  name: lora
  adapter_name_or_path:
    - outputs/run_a/checkpoint-1000
    - outputs/run_b/checkpoint-2000
```

如果希望得到一个不含 adapter 的合并模型，见 [模型导出](model_export.md)。

## 批量推理

`chat` 子命令在指定 `train_dataset` 时进入批量模式，遍历整个 `DataEngine`：

```yaml
model: Qwen/Qwen3-0.6B
template: qwen3_nothink
train_dataset: data/eval_set.yaml

sample_backend: hf
max_new_tokens: 256
```

```bash
llamafactory-cli chat config.yaml
```

数据集形态与训练完全一致，见 [数据准备](data_preparation.md)。

## 并发

`HuggingFaceEngine` 通过 `asyncio.Semaphore` 控制并发数：

```bash
export MAX_CONCURRENT=4
```

默认值 `1`。批量推理与对话模式都受这个变量约束。

## 后端

| `sample_backend` | 状态 |
|------------------|------|
| `hf` | 已实现，依赖 transformers `generate` |
| `vllm` | 预留，未启用 |
