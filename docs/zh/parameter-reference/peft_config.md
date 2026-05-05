# PeftConfig

`ModelArguments.peft_config` 的子配置。

## 参数 — LoRA

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | `lora` | 插件名 |
| `r` | `int` | `8` | Rank |
| `lora_alpha` | `int` | `16` | 缩放因子 |
| `lora_dropout` | `float` | `0.05` | Dropout 比率 |
| `target_modules` | `str \| list[str]` | `"all"` | 目标模块，`"all"` 自动发现所有 Linear 层（排除 lm_head） |
| `use_rslora` | `bool` | `False` | RS-LoRA |
| `use_dora` | `bool` | `False` | DoRA |
| `modules_to_save` | `list[str]` | `None` | 额外全参数训练模块 |
| `adapter_name_or_path` | `str \| list[str]` | `None` | 已有 adapter 路径，训练时加载单个、推理时合并多个 |

## 示例 — LoRA

```yaml
peft_config:
  name: lora
  r: 8
  lora_alpha: 16
  target_modules: all
```

## 参数 — Freeze

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | `freeze` | 插件名 |
| `freeze_trainable_layers` | `int` | `2` | 可训练层数，正数=后N层，负数=前N层 |
| `freeze_trainable_modules` | `str \| list[str]` | `["all"]` | 可训练模块 |
| `freeze_extra_modules` | `list[str]` | `[]` | 额外解冻的非隐藏层模块（如 embeddings） |
| `cast_trainable_params_to_fp32` | `bool` | `True` | 可训练参数转 fp32 |

## 示例 — Freeze

```yaml
peft_config:
  name: freeze
  freeze_trainable_layers: 2
  freeze_trainable_modules: all
```

## 参数 — 导出（merge 命令）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `export_dir` | `str` | - | 导出目录 |
| `export_size` | `int` | `5` | 分片大小（GB） |
| `export_hub_model_id` | `str` | `None` | 推送到 HF Hub 的 model ID |
| `infer_dtype` | `str` | `"auto"` | `auto` / `float16` / `float32` / `bfloat16` |
| `export_legacy_format` | `bool` | `False` | 使用 `.bin` 格式导出 |

## 示例 — 导出

```yaml
peft_config:
  name: lora
  adapter_name_or_path: path/to/lora_adapter
  export_dir: ./exported_model
  export_size: 5
```
