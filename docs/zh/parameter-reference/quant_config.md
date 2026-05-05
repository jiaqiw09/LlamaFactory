# QuantConfig

`ModelArguments.quant_config` 的子配置。

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | - | `auto` / `bnb` |
| `quantization_bit` | `int` | `None` | 量化位数：`4` 或 `8`。`auto` 模式下必需 |
| `compute_dtype` | `torch.dtype` | `float16` | 4-bit 量化计算用 dtype |
| `double_quantization` | `bool` | `True` | 4-bit 是否启用双重量化 |
| `quantization_type` | `str` | `"nf4"` | 4-bit 量化类型 |

## 示例

```yaml
quant_config:
  name: bnb
  quantization_bit: 4
  compute_dtype: bfloat16
  double_quantization: true
  quantization_type: nf4
```

训练模式下仅支持 4-bit 量化（用于 FSDP+QLoRA），8-bit 仅用于推理。
