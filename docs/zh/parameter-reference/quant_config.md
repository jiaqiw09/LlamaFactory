# QuantConfig

`ModelArguments.quant_config` 的子配置。当前仅支持 `bitsandbytes` 量化（`name=auto` 实际也分派到 `bnb`）。

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | — | `auto` 或 `bnb` |
| `quantization_bit` | `int` | `None` | 量化位宽，`4` 或 `8` |
| `compute_dtype` | `torch.dtype` | `torch.float16` | 4-bit 计算用 dtype |
| `double_quantization` | `bool` | `true` | 4-bit 双重量化 |
| `quantization_type` | `str` | `"nf4"` | 4-bit 量化类型 |

`name` 取值差异：

- `auto`：根据 `quantization_bit` 自动选择后端，目前等价于 `bnb`
- `bnb`：直接走 bitsandbytes

## 训练 vs 推理限制

- 训练模式只允许 4-bit（用于 FSDP + QLoRA），8-bit 仅推理可用
- 推理模式会把 `device_map` 强制设为当前设备
- bitsandbytes 仅在 GPU 后端可用

## 示例

```yaml
quant_config:
  name: bnb
  quantization_bit: 4
  compute_dtype: bfloat16
  double_quantization: true
  quantization_type: nf4
```
