# v1 模板渲染数据流

这份说明用于解释 v1 模板 renderer 里的数据形态。目标是让每个模型模板尽量贴近自己的原生 chat template 语义，同时把训练阶段重复的 token/label 处理固定下来。

## 训练时的调用顺序

训练链路里，数据不是一开始就全部渲染好，而是在 `BatchGenerator` 取 batch 时逐步触发。

整体顺序是：

```text
DataEngine 初始化
  -> 读取 dataset yaml / jsonl
  -> 建立 data_index

ModelEngine 初始化
  -> 加载 processor
  -> 创建 Renderer(template, processor)

BatchGenerator 初始化
  -> 创建 StatefulDataLoader
  -> collate_fn = renderer.process_samples

训练循环取 batch
  -> BatchGenerator.__next__()
  -> StatefulDataLoader 取 raw samples
  -> DataEngine.__getitem__()
  -> DataEngine._convert_data_sample()
  -> renderer.process_samples(samples)
  -> renderer.render_messages(...)
  -> RenderingPlugin(template).render_messages(...)
  -> 具体模板 renderer，例如 Qwen35TemplateRenderer.render_messages(...)
  -> 得到 ModelInput
  -> default_collate_fn(...)
  -> split_model_input(...)
  -> pad_and_truncate(...)
  -> build_multimodal_tensors(...)
  -> 返回模型可用的 BatchInput
```

所以从“谁先调用谁”的角度看：

```text
BatchGenerator 是训练取数入口
Renderer 是 BatchGenerator 的 collate_fn 里调用的
具体模板 renderer 是 Renderer.render_messages 里调用的
多模态 processor 是 batch collate 阶段调用的
```

也就是说，模板文件只负责单条样本的 `messages -> ModelInput`；真正的 batch padding 和 `pixel_values/image_grid_thw` 构造发生在 `BatchGenerator` 里面。

## qwen3.py 适合 BaseTemplateRenderer 吗？

适合。`qwen3.py` 本身已经可以拆成几个清晰的概念段：

- `system/tools` 前缀
- `user/assistant/tool` 对话轮次
- 可选的 generation prompt
- 解析模型生成的 assistant 输出

现在主要的问题是：`qwen3.py` 把模板文本渲染和训练 bookkeeping 混在了一起。例如它渲染完一个 user turn 之后，会立刻通过 `_update_model_input()` 去做 tokenizer 和 label 处理。

接入 `BaseTemplateRenderer` 后，qwen3 仍然可以把模型特有的文本规则留在本文件里，但复用通用状态工具来处理：

- `tokenizer.encode(...)`
- `labels`
- `loss_weights`
- `attention_mask`

这样 qwen3 文件仍然自包含，但每个模板不用重复写同一套训练字段处理逻辑。

## 示例 1：qwen3 纯文本 SFT 样本

原始 v1 样本：

```python
sample = {
    "messages": [
        {
            "role": "system",
            "content": [{"type": "text", "value": "You are helpful."}],
            "loss_weight": 0.0,
        },
        {
            "role": "user",
            "content": [{"type": "text", "value": "What is LLM?"}],
            "loss_weight": 0.0,
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "value": "LLM stands for Large Language Model."}],
            "loss_weight": 1.0,
        },
    ]
}
```

模板语义层会把 prompt 按片段追加：

```python
append_segment(
    text="<|im_start|>system\nYou are helpful.<|im_end|>\n",
    loss_weight=0.0,
)
append_segment(
    text="<|im_start|>user\nWhat is LLM?<|im_end|>\n",
    loss_weight=0.0,
)
append_segment(
    text="<|im_start|>assistant\nLLM stands for Large Language Model.<|im_end|>\n",
    loss_weight=1.0,
)
```

`append_segment()` 会编码每段文本，并填充 token 级别的训练字段：

```python
input_ids = system_ids + user_ids + assistant_ids
labels = [-100] * len(system_ids + user_ids) + assistant_ids
loss_weights = [0.0] * len(system_ids + user_ids) + [1.0] * len(assistant_ids)
attention_mask = [1] * len(input_ids)
```

最终 `ModelInput`：

```python
{
    "input_ids": [...],
    "attention_mask": [...],
    "labels": [...],
    "loss_weights": [...],
}
```

也就是说，qwen3 模板负责 prompt 文本长什么样；`BaseTemplateRenderer` 负责重复的 token/label bookkeeping。

## 示例 2：qwen3.5 图文/视频 SFT 样本

原始 v1 样本：

```python
sample = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "value": "data/mllm_demo_data/1.jpg"},
                {"type": "video_url", "value": "data/mllm_demo_data/demo.mp4"},
                {"type": "text", "value": "Describe this image."},
            ],
            "loss_weight": 0.0,
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "value": "Two soccer players are on the field."}],
            "loss_weight": 1.0,
        },
    ]
}
```

qwen3.5 的 content renderer 会先把 v1 多模态内容映射成 qwen3.5 原生占位符：

```python
image_url -> "<|vision_start|><|image_pad|><|vision_end|>"
video_url -> "<|vision_start|><|video_pad|><|vision_end|>"
audio_url -> 报错，因为 qwen3.5 不支持 audio 输入
```

训练时，qwen3.5 还需要把媒体 pad token 展开到和 processor 特征数匹配。假设 processor 判断这张图需要 4 个 image token，这段视频需要 6 个 video token，那么 user 片段会变成：

```python
append_segment(
    text=(
        "<|im_start|>user\n"
        "<|vision_start|>"
        "<|image_pad|><|image_pad|><|image_pad|><|image_pad|>"
        "<|vision_end|>"
        "<|vision_start|>"
        "<|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|><|video_pad|>"
        "<|vision_end|>"
        "Describe this image."
        "<|im_end|>\n"
    ),
    loss_weight=0.0,
)
model_input.setdefault("images", []).append("data/mllm_demo_data/1.jpg")
model_input.setdefault("videos", []).append("data/mllm_demo_data/demo.mp4")
```

assistant 片段：

```python
append_segment(
    text="<|im_start|>assistant\nTwo soccer players are on the field.<|im_end|>\n",
    loss_weight=1.0,
)
```

最终 `ModelInput`：

```python
{
    "input_ids": [...],
    "attention_mask": [...],
    "labels": [...],
    "loss_weights": [...],
    "images": ["data/mllm_demo_data/1.jpg"],
    "videos": ["data/mllm_demo_data/demo.mp4"],
}
```

之后 `BatchGenerator` 会把原始媒体引用和文本字段分开：

```python
text_input = {
    "input_ids": [...],
    "attention_mask": [...],
    "labels": [...],
    "loss_weights": [...],
}
multimodal_input = {
    "images": ["data/mllm_demo_data/1.jpg"],
    "videos": ["data/mllm_demo_data/demo.mp4"],
}
```

再由 `build_multimodal_tensors()` 调用 processor，补上模型需要的多模态张量：

```python
{
    "input_ids": tensor(...),
    "attention_mask": tensor(...),
    "labels": tensor(...),
    "loss_weights": tensor(...),
    "pixel_values": tensor(...),
    "image_grid_thw": tensor(...),
    "pixel_values_videos": tensor(...),
    "video_grid_thw": tensor(...),
}
```

## 示例 3：parse_message 的作用

`render_messages` 和 `parse_message` 是两个方向：

```text
render_messages:
  v1 messages -> 模型输入

parse_message:
  模型输出字符串 -> v1 assistant message
```

训练主要使用 `render_messages`。`parse_message` 主要用于聊天、采样、多轮对话和工具调用场景。

假设当前对话历史是：

```python
messages = [
    {
        "role": "user",
        "content": [{"type": "text", "value": "6 * 8 等于多少？"}],
    }
]
```

推理时先调用 `render_messages(..., is_generate=True)`，把结构化 messages 渲染成模型输入。模型生成后，拿到的是一段字符串：

```text
48
```

如果下一轮用户继续问：

```text
为什么？
```

我们需要把上一轮 assistant 的回答放回 `messages` 历史里：

```python
messages = [
    {
        "role": "user",
        "content": [{"type": "text", "value": "6 * 8 等于多少？"}],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "value": "48"}],
    },
    {
        "role": "user",
        "content": [{"type": "text", "value": "为什么？"}],
    },
]
```

这个把模型输出字符串 `"48"` 变回 assistant message 的过程，就是 `parse_message`：

```python
parsed_message = renderer.parse_message("48")

# parsed_message:
{
    "role": "assistant",
    "content": [{"type": "text", "value": "48"}],
}
```

纯文本场景里，`parse_message` 看起来只是包了一层结构。但 tool call 场景就更明显。

例如 qwen3 生成：

```text
我需要调用计算工具。
<tool_call>
{"name": "multiply", "arguments": {"a": 6, "b": 8}}
</tool_call>
```

`parse_message` 会把它解析成：

```python
{
    "role": "assistant",
    "content": [
        {"type": "text", "value": "我需要调用计算工具。"},
        {
            "type": "tool_call",
            "value": "{\"name\": \"multiply\", \"arguments\": {\"a\": 6, \"b\": 8}}",
        },
    ],
}
```

这样后续逻辑就能知道：这不是普通文本，而是 assistant 请求执行一个工具。

qwen3.5 的 tool call 文本格式和 qwen3 不一样，例如：

```text
我需要调用计算工具。
<tool_call>
<function=multiply>
<parameter=a>
6
</parameter>
<parameter=b>
8
</parameter>
</function>
</tool_call>
```

对应的 `parse_message` 会输出统一的 v1 message：

```python
{
    "role": "assistant",
    "content": [
        {"type": "text", "value": "我需要调用计算工具。"},
        {
            "type": "tool_call",
            "value": "{\"name\": \"multiply\", \"arguments\": {\"a\": \"6\", \"b\": \"8\"}}",
        },
    ],
}
```

所以 `parse_message` 也是模板语义的一部分：不同模型输出的 tool call / thinking 格式不同，但最终都应该还原成 v1 统一的 assistant `Message`。

## 推荐的模板文件结构

每个模板可以保持自包含，但内部最好保持稳定结构：

```python
class XxxTemplateRenderer(BaseTemplateRenderer):
    @staticmethod
    def render_system_and_tools(...):
        ...

    @staticmethod
    def render_conversation(...):
        ...

    @staticmethod
    def render_generation_prompt(...):
        ...

    @staticmethod
    def parse_message(...):
        ...

    @staticmethod
    def _render_user(...):
        ...

    @staticmethod
    def _render_assistant(...):
        ...

    @staticmethod
    def _render_tool(...):
        ...
```

模型特有逻辑留在具体模板里：

- 原生 prompt 字符串
- tool call 格式
- thinking 格式
- 媒体占位符语义
- 模型特定校验，例如 qwen3.5 拒绝 audio

共享 bookkeeping 留在 `BaseTemplateRenderer`：

- 编码渲染后的文本片段
- 填充 `input_ids`
- 填充 `labels`
- 填充 `loss_weights`
- 填充 `attention_mask`
- 把媒体引用带入 `ModelInput`
