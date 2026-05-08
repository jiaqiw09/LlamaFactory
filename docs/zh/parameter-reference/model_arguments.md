# ModelArguments

模型加载、对话模板与模型相关插件配置。

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `str` | `Qwen/Qwen3-4B-Instruct-2507` | 模型路径或 HF Hub ID |
| `template` | `str` | `qwen3_nothink` | 对话模板名 |
| `trust_remote_code` | `bool` | `false` | 是否信任 HF 远端代码 |
| `model_class` | `ModelClass` | `llm` | 模型类别，决定使用哪个 `AutoModel` 类加载 |
| `init_config` | `dict \| None` | `None` | 初始化插件配置 |
| `peft_config` | `dict \| None` | `None` | PEFT 插件配置 |
| `kernel_config` | `dict \| None` | `None` | Kernel 插件配置 |
| `quant_config` | `dict \| None` | `None` | 量化插件配置 |

## 取值说明

### `template`

`Renderer` 内置 `chatml` 模板；其余模板由 `RenderingPlugin` 注册，按需懒加载。当前仓库内置：

- `chatml`
- `qwen3`
- `qwen3_nothink`

注册新模板的方式见 [RenderingPlugin](../developer-guide/core/renderer.md)。

### `model_class`

| 取值 | 对应类 | 用途 |
|------|--------|------|
| `llm` | `AutoModelForCausalLM` | 因果语言模型，SFT / DPO 主流程 |
| `cls` | `AutoModelForTokenClassification` | Token 分类任务 |
| `other` | `AutoModel` | 其他自定义模型 |

### 插件类配置（`*_config`）

四个 `*_config` 字段都接收一个字典或 JSON 字符串，必须包含 `name` 字段，`name` 决定具体使用哪个插件。值为 `None` 时表示不启用对应能力。

| 字段 | 子配置页 | 常用 `name` |
|------|----------|-------------|
| `init_config` | [InitConfig](init_config.md) | `init_on_default` / `init_on_meta` / `init_on_rank0` |
| `peft_config` | [PeftConfig](peft_config.md) | `lora` / `freeze` |
| `kernel_config` | [KernelConfig](kernel_config.md) | `auto` |
| `quant_config` | [QuantConfig](quant_config.md) | `auto` / `bnb` |

## 示例

```yaml
model: Qwen/Qwen3-0.6B
template: qwen3_nothink
peft_config:
  name: lora
  r: 16
  target_modules: all
```
