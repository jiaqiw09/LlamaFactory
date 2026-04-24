# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re

from ....utils.constants import IGNORE_INDEX
from ....utils.helper import get_tokenizer
from ....utils.multimodal import get_image_token_counts, render_text_content
from ....utils.types import Message, ModelInput, Processor, ToolCall
from ..rendering import RenderingPlugin


QWEN3_5_VISION_START = "<|vision_start|>"
QWEN3_5_IMAGE_PAD = "<|image_pad|>"
QWEN3_5_VISION_END = "<|vision_end|>"
QWEN3_5_IMAGE_PLACEHOLDER = QWEN3_5_VISION_START + QWEN3_5_IMAGE_PAD + QWEN3_5_VISION_END
FUNCTION_CALL_PATTERN = re.compile(r"<function=([^>]+)>\s*(.*?)\s*</function>", re.DOTALL)
PARAMETER_PATTERN = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _update_model_input(
    processor: Processor,
    input_ids: list[int],
    labels: list[int],
    loss_weights: list[float],
    temp_str: str,
    temp_weight: float,
) -> str:
    """Update model input with temporary string."""
    if not temp_str:
        return ""

    tokenizer = get_tokenizer(processor)
    temp_ids = tokenizer.encode(temp_str, add_special_tokens=False)
    input_ids.extend(temp_ids)
    loss_weights.extend([temp_weight] * len(temp_ids))
    if temp_weight > 1e-6:
        labels.extend(temp_ids)
    else:
        labels.extend([IGNORE_INDEX] * len(temp_ids))

    return ""


def _render_message_content(
    message: Message,
    *,
    collect_images: bool = True,
    allow_media: bool = True,
) -> tuple[str, list[str]]:
    content_text, images = render_text_content(
        message["content"],
        image_placeholder=QWEN3_5_IMAGE_PLACEHOLDER,
    )
    if len(images) != 0 and not allow_media:
        raise ValueError(f"{message['role']} message does not support media content in qwen3_5.")

    return content_text, (images if collect_images else [])


def _expand_image_placeholders(processor: Processor, content_text: str, images: list[str]) -> str:
    for image_token_count in get_image_token_counts(processor, images):
        image_tokens = QWEN3_5_IMAGE_PAD * image_token_count
        content_text = content_text.replace(
            QWEN3_5_IMAGE_PLACEHOLDER,
            QWEN3_5_VISION_START + image_tokens + QWEN3_5_VISION_END,
            1,
        )

    return content_text


def _get_last_query_index(messages: list[Message]) -> int:
    """Find the last user query index, excluding wrapped tool responses."""
    last_query_index = len(messages) - 1
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if message["role"] != "user":
            continue

        user_text, _ = _render_message_content(message, collect_images=False)

        if not (user_text.startswith("<tool_response>") and user_text.endswith("</tool_response>")):
            last_query_index = idx
            break

    return last_query_index


def _split_assistant_content(message: Message) -> tuple[str, str, list[ToolCall]]:
    """Split assistant message into text, reasoning and tool calls."""
    text_content = ""
    reasoning_content = ""
    tool_calls: list[ToolCall] = []

    for content in message["content"]:
        if content["type"] == "text":
            text_content += content["value"]
        elif content["type"] == "reasoning":
            reasoning_content += content["value"]
        elif content["type"] == "tool_call":
            try:
                tool_call: ToolCall = json.loads(content["value"])
            except json.JSONDecodeError:
                raise ValueError(f"Invalid tool call format: {content['value']}.")

            tool_calls.append(tool_call)
        elif content["type"] in {"image_url", "video_url", "audio_url"}:
            raise ValueError("The qwen3_5 multimodal bridge does not support assistant media outputs.")
        else:
            raise ValueError(f"Unsupported content type: {content['type']}")

    return text_content, reasoning_content, tool_calls


@RenderingPlugin("qwen3_5").register("render_messages")
def render_qwen3_5_messages(
    processor: Processor,
    messages: list[Message],
    tools: str | None = None,
    is_generate: bool = False,
    enable_thinking: bool = False,
) -> ModelInput:
    """Render messages in the Qwen3.5 format using the local tokenizer chat-template semantics."""
    input_ids, labels, loss_weights = [], [], []
    temp_str, temp_weight = "", 0.0
    images: list[str] = []

    if tools:
        temp_str += "<|im_start|>system\n"
        temp_str += "# Tools\n\nYou have access to the following functions:\n\n<tools>"
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid tools format: {str(tools)}.")

        if not isinstance(tools, list):
            tools = [tools]

        for tool in tools:
            temp_str += "\n" + json.dumps(tool, ensure_ascii=False)

        temp_str += (
            "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
            "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
            "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
            "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within "
            "<tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning "
            "for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call "
            "available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            "</IMPORTANT>"
        )
        if messages[0]["role"] == "system":
            system_text, system_images = _render_message_content(messages[0], allow_media=False)
            if len(system_images) != 0:
                raise ValueError("System message cannot contain media.")

            if system_text.strip():
                temp_str += "\n\n" + system_text.strip()

        temp_str += "<|im_end|>\n"
    elif messages[0]["role"] == "system":
        system_text, system_images = _render_message_content(messages[0], allow_media=False)
        if len(system_images) != 0:
            raise ValueError("System message cannot contain media.")

        temp_str += "<|im_start|>system\n" + system_text + "<|im_end|>\n"
        temp_weight = messages[0].get("loss_weight", 0.0)

    temp_str = _update_model_input(processor, input_ids, labels, loss_weights, temp_str, temp_weight)
    last_query_index = _get_last_query_index(messages)

    for turn_idx, message in enumerate(messages):
        if message["role"] == "system":
            if turn_idx != 0:
                raise ValueError("System message must be at the beginning.")
            continue

        if message["role"] == "user":
            message_text, message_images = _render_message_content(message)
            message_text = _expand_image_placeholders(processor, message_text, message_images)
            images.extend(message_images)
            temp_str += "<|im_start|>user\n" + message_text + "<|im_end|>\n"
            temp_weight = message.get("loss_weight", 0.0)
        elif message["role"] == "assistant":
            temp_str += "<|im_start|>assistant\n"
            text_content, reasoning_content, tool_calls = _split_assistant_content(message)
            if turn_idx > last_query_index:
                temp_str += "<think>\n" + reasoning_content.strip("\n") + "\n</think>\n\n" + text_content.lstrip("\n")
            else:
                temp_str += text_content

            for tool_call_idx, tool_call in enumerate(tool_calls):
                if tool_call_idx == 0:
                    if temp_str and not temp_str.endswith("\n"):
                        temp_str += "\n\n"
                else:
                    temp_str += "\n"

                temp_str += "<tool_call>\n<function=" + tool_call["name"] + ">\n"
                arguments = tool_call.get("arguments", {})
                if isinstance(arguments, str):
                    temp_str += arguments.rstrip("\n")
                    if not temp_str.endswith("\n"):
                        temp_str += "\n"
                else:
                    for arg_name, arg_value in arguments.items():
                        if isinstance(arg_value, (dict, list)):
                            arg_str = json.dumps(arg_value, ensure_ascii=False)
                        else:
                            arg_str = str(arg_value)

                        temp_str += "<parameter=" + arg_name + ">\n" + arg_str + "\n</parameter>\n"

                temp_str += "</function>\n</tool_call>"

            temp_str += "<|im_end|>\n"
            temp_weight = message.get("loss_weight", 1.0)
        elif message["role"] == "tool":
            tool_text, tool_images = _render_message_content(message, allow_media=False)
            if len(tool_images) != 0:
                raise ValueError("Tool message cannot contain media.")

            if turn_idx == 0 or messages[turn_idx - 1]["role"] != "tool":
                temp_str += "<|im_start|>user"

            temp_str += "\n<tool_response>\n" + tool_text + "\n</tool_response>"
            if turn_idx == len(messages) - 1 or messages[turn_idx + 1]["role"] != "tool":
                temp_str += "<|im_end|>\n"

            temp_weight = message.get("loss_weight", 0.0)
        else:
            raise ValueError(f"Unsupported role: {message['role']}")

        temp_str = _update_model_input(processor, input_ids, labels, loss_weights, temp_str, temp_weight)

    if is_generate:
        temp_str += "<|im_start|>assistant\n"
        temp_weight = 0.0
        if enable_thinking is False:
            temp_str += "<think>\n\n</think>\n\n"
        else:
            temp_str += "<think>\n"

        temp_str = _update_model_input(processor, input_ids, labels, loss_weights, temp_str, temp_weight)

    model_input = ModelInput(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
        loss_weights=loss_weights,
    )
    if len(images) != 0:
        model_input["images"] = images

    return model_input


@RenderingPlugin("qwen3_5").register("parse_message")
def parse_qwen3_5_message(generated_text: str) -> Message:
    """Parse assistant message in qwen3.5 format."""
    pattern = re.compile(r"<(think|tool_call)>\s*(.*?)\s*</\1>\s*", re.DOTALL)
    content = []
    last_end = 0

    for match in pattern.finditer(generated_text):
        start, end = match.span()
        if start > last_end:
            text = generated_text[last_end:start].strip()
            if text:
                content.append({"type": "text", "value": text})

        tag_type = match.group(1)
        tag_value = match.group(2).strip()
        if tag_type == "think":
            content.append({"type": "reasoning", "value": tag_value})
        elif tag_type == "tool_call":
            function_match = FUNCTION_CALL_PATTERN.search(tag_value)
            if function_match is None:
                raise ValueError(f"Invalid qwen3_5 tool call format: {tag_value}")

            arguments: dict[str, str] = {}
            for parameter_match in PARAMETER_PATTERN.finditer(function_match.group(2)):
                arguments[parameter_match.group(1)] = parameter_match.group(2).strip()

            content.append(
                {
                    "type": "tool_call",
                    "value": json.dumps(
                        {"name": function_match.group(1).strip(), "arguments": arguments},
                        ensure_ascii=False,
                    ),
                }
            )

        last_end = end

    if last_end < len(generated_text):
        text = generated_text[last_end:].strip()
        if text:
            content.append({"type": "text", "value": text})

    return Message(role="assistant", content=content)
