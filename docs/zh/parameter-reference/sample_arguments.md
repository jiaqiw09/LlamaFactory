# SampleArguments

推理采样配置，用于 `chat` / `sample` 子命令。

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sample_backend` | `SampleBackend` | `hf` | 采样后端 |
| `max_new_tokens` | `int` | `128` | 单次生成的最大 token 数 |

## 取值说明

`sample_backend` 取值：

| 值 | 状态 | 说明 |
|----|------|------|
| `hf` | 已实现 | 直接调用 HF `generate`，无需额外依赖 |
| `vllm` | 预留 | vLLM 后端，尚未启用 |

## 示例

```yaml
sample_backend: hf
max_new_tokens: 512
```

完整推理流程见 [inference](../feature-guide/inference.md)。
