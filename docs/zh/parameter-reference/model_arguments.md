# ModelArguments

模型加载、模板及插件配置。

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `str` | `Qwen/Qwen3-4B-Instruct-2507` | 模型路径或 HF Hub ID |
| `template` | `str` | `qwen3_nothink` | 对话模板，可选 `qwen3` / `qwen3_nothink` / `chatml` |
| `trust_remote_code` | `bool` | `False` | 信任 HF 远端代码 |
| `model_class` | `ModelClass` | `LLM` | `LLM`（CausalLM）/ `CLS`（TokenClassification）/ `OTHER`（AutoModel） |
| `init_config` | `PluginConfig \| None` | `None` | 初始化策略，`name` 可选 `init_on_default` / `init_on_meta` / `init_on_rank0` |
| `peft_config` | `PluginConfig \| None` | `None` | PEFT 配置，见 [PeftConfig](peft_config.md) |
| `kernel_config` | `PluginConfig \| None` | `None` | Kernel 配置，见 [KernelConfig](kernel_config.md) |
| `quant_config` | `PluginConfig \| None` | `None` | 量化配置，`name` 可选 `auto` / `bnb` |

## 示例

```yaml
model: Qwen/Qwen3-0.6B
template: qwen3_nothink
```
