# SampleArguments

推理采样参数。

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sample_backend` | `SampleBackend` | `HF` | `hf`（已实现）/ `vllm`（预留） |
| `max_new_tokens` | `int` | `128` | 最大生成 token 数 |

## 示例

```yaml
sample_backend: hf
max_new_tokens: 512
```
