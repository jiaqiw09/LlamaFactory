# LlamaFactory v1 多模态数据处理接入方案

## 1. 背景

`LlamaFactory v1` 当前已经具备多模态消息类型定义能力，例如 `image_url`、`video_url`、`audio_url`，但训练链路仍以文本处理为主，还没有形成 `v1` 原生的多模态 SFT 闭环。

本文档的目标是明确：

- 只在 `src/llamafactory/v1/` 内实现多模态能力
- 不依赖 `v0` 的数据实现，不引入 `EasyR1` 的 RL 链路
- 为 `v1` 的 `SFT` 增加端到端的多模态数据处理能力

第一阶段先聚焦：

- `image + text`
- 只支持 `v1` 原生 `messages`
- 先增强 `model_type=qwen3_5` 对应的全模态模板入口
- `train + generate`

## 2. 目标

在 `v1` 中补齐一条原生多模态 SFT 数据链路：

`DataEngine -> DataConverterPlugin -> Renderer -> BatchGenerator -> Trainer -> InferenceEngine`

最终达到以下效果：

- 数据集可以表达图文样本
- renderer 可以处理多模态 content，不再只接受纯文本
- batch 阶段可以调用 `processor` 构造视觉张量
- trainer 可以直接训练多模态模型
- inference/generate 也可以使用同一套多模态输入流程

## 3. 设计边界

### 3.1 代码范围

实现仅限：

- `src/llamafactory/v1/utils/`
- `src/llamafactory/v1/core/`
- `src/llamafactory/v1/plugins/`

### 3.2 参考范围

`v0` 和 `EasyR1` 只能参考思路，不能直接复用实现。

可借鉴的内容包括：

- `v0` 的多模态分层设计
- `EasyR1` 对 Qwen-VL 输入字段和校验逻辑的处理意识

不直接引入的内容包括：

- `v0` 的 `mm_plugin.py`、`collator.py`
- `EasyR1` 的 rollout / actor / critic 双阶段处理链路

## 4. 当前现状

`v1` 已有基础：

- `Content.type` 已支持 `image_url / video_url / audio_url`
- `DataEngine` 已支持 converter 插件
- `Renderer` 已支持模板插件
- `ModelEngine` 已统一使用 `AutoProcessor`

但当前缺少以下关键能力：

- 模板层无法真正消费多模态 content
- `Renderer.process_samples()` 不会保留或处理媒体输入
- `BatchGenerator` 不会构造 `pixel_values` 等多模态张量
- `InferenceEngine.generate()` 只传文本输入，不传视觉输入

## 5. 总体思路

`v1` 的多模态支持，不是把文本链路上补几个字段，而是补齐一条完整的数据处理闭环。

拆开来看，一共做 4 件事：

1. 定义 `v1` 内部统一的多模态样本格式
2. 让 `Renderer` 能把结构化多模态消息渲染成模型可接受的输入
3. 让 `BatchGenerator` 在 batch 阶段统一构造视觉张量
4. 让训练和生成共用同一套多模态输入逻辑

## 6. 数据格式设计

这一部分最关键，必须先讲顺。

### 6.1 推荐的第一阶段输入格式

第一阶段推荐优先支持 `v1` 原生消息格式，也就是数据集直接写成 `messages`。

示例：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "value": "data/demo.jpg"},
        {"type": "text", "value": "请描述这张图片。"}
      ],
      "loss_weight": 0.0
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "value": "这是一张..."}
      ],
      "loss_weight": 1.0
    }
  ]
}
```

这个格式的优势是：

- 完全符合 `v1` 现有 schema
- 不需要依赖旧版 `<image>` 文本占位协议
- 后续扩展外部数据格式和其他模态也自然

### 6.2 v1 内部标准格式

进入 `DataEngine` 和 `Renderer` 之后，统一以 `messages` 结构流转。

每条 `message` 的 `content` 在第一阶段允许混排：

- `text`
- `image_url`

`video_url`、`audio_url` 仍保留在 schema 中，但不在第一阶段处理。

其他内容类型保持原有语义：

- `reasoning`
- `tool_call`

也就是说，`v1` 内部不再要求“图像只能是单独列”，而是让媒体成为消息内容的一部分。

### 6.3 为什么还需要中间转换

虽然外部数据可以直接是 `messages`，但训练时模型并不能直接吃这个结构。

因此中间还需要两次转换：

1. `结构化 content -> 模板可渲染文本`
2. `媒体引用 -> processor 输出张量`

这两次转换分别发生在：

- `Renderer`
- `BatchGenerator`

### 6.4 第一阶段支持的数据约束

第一阶段建议只支持以下约束：

- 每条样本只做 `text + image`
- 图片通过 `image_url` 指定本地路径
- 一条消息里可以有多个 image
- assistant 输出仍然只训练文本
- 暂不处理 `video_url` / `audio_url`

### 6.5 后续兼容格式

第一阶段跑通后，再逐步兼容外部数据格式、文本占位协议和更多模态。

也就是说，第一阶段先把“`v1` 原生格式”打通，第二阶段再扩展“历史兼容格式”。

## 7. 数据格式到训练输入的转换目标

这一节专门解释“数据格式这边到底要做什么”。

### 7.1 原始数据层

原始数据可以是两类：

1. 已经是 `v1` 原生 `messages`
2. 其他外部格式，例如 alpaca / sharegpt / 自定义 json

### 7.2 converter 层要做的事

`DataConverterPlugin` 的职责不是做图像预处理，而是做“结构统一”：

- 把外部字段整理成 `messages`
- 把图片、视频、音频字段嵌入到 `messages[].content`
- 保证用户消息中的媒体顺序和文本顺序明确

例如把：

```json
{
  "instruction": "请描述图片内容",
  "images": ["data/demo.jpg"],
  "output": "这是一张..."
}
```

转换成：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "value": "data/demo.jpg"},
        {"type": "text", "value": "请描述图片内容"}
      ],
      "loss_weight": 0.0
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "value": "这是一张..."}
      ],
      "loss_weight": 1.0
    }
  ]
}
```

也就是说：

- converter 负责把“数据长什么样”统一掉
- renderer 负责把“消息怎么渲染”
- batching 负责把“图片怎么变成张量”

### 7.3 renderer 层要做的事

`Renderer` 接到 `messages` 后，要做两件事：

1. 把媒体引用从消息中抽出来
2. 把结构化媒体内容渲染成模板可接受的占位文本

这里的关键是：

- `image_url` 不会直接送给 tokenizer
- 它会先变成模型模板层理解的 image placeholder

### 7.4 batching 层要做的事

batch 阶段再统一使用 `processor` 构造真实模型输入，例如：

- `pixel_values`
- `image_grid_thw`

所以整个链路是：

`原始样本 -> 结构化 messages -> 模板占位文本 + 原始媒体引用 -> 多模态张量输入`

## 8. 端到端处理链路

### Step 1：数据进入 DataEngine

原始样本通过 `DataEngine` 读入。

如果数据本身已经是 `v1` 标准 `messages`，则直接进入后续流程。

如果数据是其他格式，例如 alpaca/sharegpt，则通过 `DataConverterPlugin` 转成 `v1` 标准消息格式。

目标产物：

- `Sample.messages`

### Step 2：DataConverterPlugin 做结构化转换

这一层负责把外部样本转换成统一消息格式。

第一阶段要做两类支持：

1. `v1` 原生 `messages` 数据直接透传
2. 新增图文 converter，把外部图片字段变成 `image_url` content

目标产物：

- 统一结构化多模态 `messages`

### Step 3：Renderer 提取媒体并渲染模板

这一层是 `v1` 当前最大的缺口。

`Renderer` 不应该再只做文本 token 化，而应该新增两项职责：

1. 从 `messages[].content` 中提取媒体引用
2. 把 `image_url` 等结构化内容渲染成模板侧可接受的占位文本

这里要特别注意，不应该直接修改现有纯文本模板，例如 `qwen3 / qwen3_nothink`，因为它们对应的不是当前要接入的全模态模型。

更合理的做法是：

- 为目标多模态模型增强真实的模板入口
- 例如 `config.json` 中的 `model_type=qwen3_5` 对应全模态模型
- 先把 `qwen3_5` 模板槽位和调用链打通
- 后续再单独补齐该模板的具体语义

也就是说，未来应该由 `qwen3_5` 全模态模板去支持把：

```json
{"type": "image_url", "value": "data/demo.jpg"}
```

渲染为统一 image placeholder，再进入 token 化流程。

这一步只做“消息渲染”，不做真正的图像张量化。

目标产物：

- 文本 token 字段
- 原始 `images` 引用

### Step 4：BatchGenerator 统一构造多模态张量

batch 阶段再统一调用 `processor`，生成模型 forward 所需字段。

这一层负责把媒体引用转换成：

- `pixel_values`
- `image_grid_thw`
- 未来模型需要的其他字段

这样做的原因是：

- 多模态张量构造天然是 batch 级逻辑
- 不同模型需要的字段不完全一样
- 训练和推理都应该复用同样的 processor 处理路径

目标产物：

- 文本 batch tensor
- 多模态 batch tensor

### Step 5：Trainer 直接透传给模型

`BaseTrainer` 当前已经会把 batch 中所有 tensor 字段传给模型。

因此只要 batch 阶段把多模态输入准备好，trainer 本身不需要做大的结构改动。

目标产物：

- `model(**batch)` 可以直接 forward

### Step 6：InferenceEngine 复用相同逻辑

训练能跑还不够，生成也必须复用同一套多模态输入流程。

因此 `InferenceEngine.generate()` 也要改成：

1. 先走多模态 `Renderer`
2. 再构造多模态 processor 输入
3. 最终把文本张量和视觉张量一起传给 `model.generate()`

这样训练和推理的输入语义才能保持一致。

## 9. 实施阶段

这一部分按“能力阶段”来拆，不再和“文件改动顺序”混写。

### 阶段 1：打通 `v1` 原生多模态最小闭环

这一阶段的目标是先把主链路跑通，只支持：

- `v1` 原生 `messages`
- `image_url + text`
- `train + generate`
- `model_type=qwen3_5` 对应的全模态模板入口

这一阶段暂时不做：

- 外部数据格式的图片字段接入
- `video/audio`
- `<image>` 文本占位兼容

阶段 1 要完成的事：

- 补充 `ModelInput` / `BatchInput` 的多模态字段定义
- 新增 `v1` 原生多模态处理模块
- 让 `Renderer.process_samples()` 支持媒体提取和 placeholder 渲染
- 增强 `qwen3_5` 全模态模板槽位
- 先把模板注册和调用链打通，模板细节后续单独补齐
- 让 `BatchGenerator` 能构造 `pixel_values` 等多模态张量
- 让 `InferenceEngine.generate()` 复用同一套多模态输入逻辑
- 补 image-only 最小闭环测试

阶段 1 的完成标准：

- 能读取一条 `v1` 原生图文样本
- `Renderer.process_samples()` 不报错
- `BatchGenerator` 能产出 `pixel_values`
- `SFTTrainer` 能完成至少一个训练 step
- `InferenceEngine.generate()` 能处理图文输入

后续阶段再单独补外部数据格式和更多模态，不混进第一阶段。

## 10. 开发顺序

这一部分按“真正动代码的顺序”来写。

### 阶段 1 的开发顺序

1. `src/llamafactory/v1/utils/types.py`
2. 新增 `v1` 多模态处理模块
3. `src/llamafactory/v1/core/utils/rendering.py`
4. 增强全模态模板文件，例如 `src/llamafactory/v1/plugins/model_plugins/templates/qwen3_5.py`
5. `src/llamafactory/v1/core/utils/batching.py`
6. `src/llamafactory/v1/core/utils/inference_engine.py`
7. `tests_v1`

这一顺序的原因是：

- 先定字段契约
- 再补 `process_samples -> render -> batch -> generate` 主链路
- 最后再用测试锁住 image-only 最小闭环

后续如果要补外部数据格式或 `video/audio`，再分别围绕 converter 和模型特定处理单独推进。

## 11. 为什么这样分

之所以这样重新拆，是因为：

- “实施阶段”描述的是能力目标
- “开发顺序”描述的是实际落代码顺序

如果把这两个维度混在一起，就会出现“阶段 1 说先打通主链路，但顺序里又先改 converter”的冲突。

统一后的原则是：

- 阶段 1 先解决链路问题
- 后续阶段再解决外部数据格式广度问题和更复杂模态

## 12. 总结

`v1` 的多模态接入，本质上不是“给现有文本链路加几个字段”，而是要在 `src/llamafactory/v1` 内补齐一条原生多模态 SFT 数据链路。

核心原则是：

- 只在 `v1` 内闭环
- 数据先结构化，再渲染，再 batch 化
- 模型差异放到 `v1` 自己的插件层吸收
- 训练和推理共用同一套多模态输入逻辑

第一阶段先把 `image + text + Qwen3.5 + SFT` 打通，后续再逐步扩展到更多外部数据格式和更多模态。
