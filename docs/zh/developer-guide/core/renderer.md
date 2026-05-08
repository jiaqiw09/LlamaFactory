# Renderer

`Renderer` 把统一的 `Message` 列表渲染成模型能直接消费的 `ModelInput`（`input_ids` / `attention_mask` / `labels` / `loss_weights` 等）。具体的对话格式逻辑由 Template 提供，`Renderer` 本身只负责按模板名分派并提供批量处理接口。

代码位置：

- `core/utils/rendering.py`：`Renderer` 类、内置 `chatml` 实现
- `plugins/model_plugins/rendering.py`：`RenderingPlugin`（懒导入注册表）
- `plugins/model_plugins/templates/`：具体模板文件

## 接口

```python
class Renderer:
    def __init__(self, template: str, processor: Processor) -> None: ...

    def render_messages(
        self, messages, tools=None, is_generate=False, enable_thinking=False,
    ) -> ModelInput: ...

    def parse_message(self, generated_text: str) -> Message: ...

    def process_samples(self, samples: list[Sample]) -> list[ModelInput]: ...
```

`Renderer` 由 `ModelEngine` 在初始化时构造，`template` 字段来自 `ModelArguments.template`。

## 分派规则

```python
if self.template == "chatml":
    return render_chatml_messages(...)         # 内置实现
else:
    return RenderingPlugin(self.template).render_messages(...)
```

`chatml` 是 hard-coded fast path：避免导入插件文件，启动更快。其它模板都走 `RenderingPlugin`，由懒导入机制找到 `templates/{name}.py`。

## process_samples：训练 collate

`process_samples` 是 `BatchGenerator` 给 `StatefulDataLoader` 的 `collate_fn`，把一批样本统一成 `list[ModelInput]`：

| 样本类型 | 判定字段 | 处理 |
|----------|----------|------|
| SFT | 含 `messages` | 调 `render_messages`，自动补 `position_ids` |
| DPO / 偏好对 | 含 `chosen_messages` + `rejected_messages` | 分别渲染后拼接，用 `token_type_ids` 标记（1=chosen，2=rejected） |
| 其它 | — | 直接 `ValueError` |

`extra_info` 与 `_dataset_name` 字段（如果存在）会被原样传递到 `ModelInput`，方便上游按数据集打标。

## RenderingPlugin

`RenderingPlugin` 重写了 `__getitem__`，第一次按 `name` 取方法时按需 import `templates/{name}.py`：

```python
class RenderingPlugin(BasePlugin):
    _attempted_template_imports: set[str] = set()

    def __getitem__(self, method_name):
        self._ensure_template_imported()
        return super().__getitem__(method_name)
```

`_attempted_template_imports` 记录已尝试过的名字，import 失败也不会反复尝试，避免日志刷屏。

## Template 注册协议

每个 template 文件至少注册两个方法：

```python
@RenderingPlugin("my_template").register("render_messages")
def render_my_template(processor, messages, tools, is_generate, enable_thinking) -> ModelInput:
    # 把 messages 转成 token id，按 role 设置 loss_weight
    return ModelInput(
        input_ids=...,
        attention_mask=...,
        labels=...,
        loss_weights=...,
    )


@RenderingPlugin("my_template").register("parse_message")
def parse_my_template(generated_text: str) -> Message:
    # 把生成的字符串解析回 Message（提取 reasoning / tool_call 等）
    return Message(role="assistant", content=[...])
```

`render_messages` 的实现里需要按模板规则填 `loss_weights`：assistant 部分通常 `1.0`，其它 `0.0`，被忽略的 token 在 `labels` 里写 `IGNORE_INDEX`。

## 内置模板

| name | 实现位置 | 备注 |
|------|---------|------|
| `chatml` | `core/utils/rendering.py` | 通用 ChatML 格式（`<\|im_start\|>` / `<\|im_end\|>`），不依赖懒导入 |
| `qwen3` | `templates/qwen3.py` | 支持 reasoning、tool_call |
| `qwen3_nothink` | `templates/qwen3_nothink.py` | qwen3 无思考模式 |

`chatml` 的实现里：

- 每条消息渲染成 `<|im_start|>{role}\n{text}<|im_end|>\n`
- `loss_weight` 默认遵循"assistant=1.0，其它=0.0"
- 推理路径（`is_generate=True`）末尾追加 `<|im_start|>assistant\n` 作为生成起点

## 添加新模板

1. 在 `plugins/model_plugins/templates/` 下创建 `my_template.py`
2. 注册 `render_messages` 和 `parse_message`
3. 把 `template: my_template` 写到 YAML 配置里即可

无需修改 `Renderer` 或 `RenderingPlugin`，懒导入机制会在首次使用时自动加载。
