# Renderer 与 Template 系统

Renderer 负责将标准 Messages 格式转换为模型输入（`ModelInput`），Template 负责具体的对话格式渲染。

## 设计动机

不同模型使用不同的对话格式（ChatML、Qwen3 thinking、Qwen3 nothink 等）。Renderer + Template 系统将格式逻辑与训练逻辑解耦，通过 `RenderingPlugin` 实现可插拔的模板注册。

## Renderer

位于 `src/llamafactory/v1/core/utils/rendering.py`。

```python
class Renderer:
    def __init__(self, template: str, processor: Processor):
        self.template = template
        self.processor = processor

    def render_messages(self, messages, tools=None, is_generate=False, enable_thinking=False) -> ModelInput:
        """将消息列表渲染为模型输入"""
        if self.template == "chatml":
            return render_chatml_messages(...)
        else:
            return RenderingPlugin(self.template).render_messages(...)

    def parse_message(self, generated_text: str) -> Message:
        """将生成文本解析为消息"""
        ...

    def process_samples(self, samples: list[Sample]) -> list[ModelInput]:
        """批量处理样本（SFT 和 DPO）"""
        ...
```

### process_samples 处理逻辑

- **SFT 样本**（含 `messages`）：直接 `render_messages`，自动添加 `position_ids`
- **DPO 样本**（含 `chosen_messages` + `rejected_messages`）：分别渲染后拼接，用 `token_type_ids` 区分（1=chosen，2=rejected）

## RenderingPlugin

位于 `src/llamafactory/v1/plugins/model_plugins/rendering.py`。

使用延迟导入模式：Template 文件只在被实际使用时才导入，减少启动开销。

```python
class RenderingPlugin(BasePlugin):
    def __getitem__(self, method_name):
        self._ensure_template_imported()  # 按需导入 templates/{name}.py
        return super().__getitem__(method_name)
```

### 注册协议

新 Template 需注册两个方法：

```python
@RenderingPlugin("my_template").register("render_messages")
def render_my_template(processor, messages, tools, is_generate, enable_thinking):
    ...
    return ModelInput(input_ids=..., attention_mask=..., labels=..., loss_weights=...)

@RenderingPlugin("my_template").register("parse_message")
def parse_my_template(generated_text):
    ...
    return Message(role="assistant", content=[...])
```

### 已实现的 Template

| Template | 文件 | 特性 |
|----------|------|------|
| `chatml` | 内建于 `rendering.py` | 通用 ChatML 格式（`<\|im_start\|>` / `<\|im_end\|>`） |
| `qwen3` | `templates/qwen3.py` | 支持 reasoning、tool_call |
| `qwen3_nothink` | `templates/qwen3_nothink.py` | Qwen3 无思考模式 |

## 扩展点

添加新 Template：在 `plugins/model_plugins/templates/` 下创建新文件，注册 `render_messages` 和 `parse_message` 两个方法。
