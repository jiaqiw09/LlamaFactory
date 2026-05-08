# PeftConfig

`ModelArguments.peft_config` 的子配置，目前实现两种 PEFT 方式：`lora` 与 `freeze`。`name` 字段决定走哪一条分支，其它字段按分支独立读取。

合并 LoRA 适配器到底座模型再导出的流程也复用本配置，见下方 [导出参数](#导出参数)。

## LoRA

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | — | 必须为 `lora` |
| `r` | `int` | `8` | LoRA rank |
| `lora_alpha` | `int` | `16` | 缩放因子 |
| `lora_dropout` | `float` | `0.05` | Dropout 比例 |
| `target_modules` | `str \| list[str]` | `"all"` | 注入位置 |
| `use_rslora` | `bool` | `false` | 启用 rsLoRA |
| `use_dora` | `bool` | `false` | 启用 DoRA |
| `modules_to_save` | `list[str]` | `None` | 额外参与全参数训练的模块 |
| `adapter_name_or_path` | `str \| list[str]` | `None` | 加载已有 adapter |

`target_modules` 取值规则：

- `"all"`：自动扫描全部 `Linear` 子模块（排除 `lm_head` / `output_layer` / `output`）
- 字符串：单个模块名前缀
- 列表：多个模块名前缀

`adapter_name_or_path` 行为：

- 训练模式（`is_train=True`）：只允许传入单个 adapter 路径，在该 adapter 上继续训练；其它 LoRA 超参数会被该 adapter 自身的配置覆盖
- 推理模式：可传入多个 adapter，逐个 merge & unload

```yaml
peft_config:
  name: lora
  r: 8
  lora_alpha: 16
  target_modules: all
```

## Freeze

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | — | 必须为 `freeze` |
| `freeze_trainable_layers` | `int` | `2` | 解冻层数：正数 = 后 N 层，负数 = 前 N 层 |
| `freeze_trainable_modules` | `str \| list[str]` | `["all"]` | 解冻的子模块，`"all"` 表示该层内全部 |
| `freeze_extra_modules` | `list[str]` | `[]` | 额外解冻的非分层模块（如 embedding） |
| `cast_trainable_params_to_fp32` | `bool` | `true` | 把可训练参数转为 fp32 提升数值稳定性 |

```yaml
peft_config:
  name: freeze
  freeze_trainable_layers: 2
  freeze_trainable_modules: all
```

## 导出参数

由 `lmf merge` 或等价的导出入口读取，仅在 `name=lora` 时生效。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `export_dir` | `str` | — | 必填，导出目录 |
| `export_size` | `int` | `5` | safetensors 分片大小（GB） |
| `export_hub_model_id` | `str` | `None` | 同时推送到 HF Hub |
| `infer_dtype` | `str` | `"auto"` | `auto` / `float16` / `float32` / `bfloat16` |
| `export_legacy_format` | `bool` | `false` | 输出 `.bin` 旧格式而非 safetensors |

```yaml
peft_config:
  name: lora
  adapter_name_or_path: outputs/run/checkpoint-1000
  export_dir: outputs/merged
  export_size: 5
  infer_dtype: bfloat16
```

完整使用流程见 [model_export](../feature-guide/model_export.md)。
