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
from typing import Any, Literal, NotRequired, TypedDict

from ...utils import logging
from ...utils.plugin import BasePlugin
from ...utils.types import DPOSample, Sample, SFTSample, ToolCall


logger = logging.get_logger(__name__)


class AlpacaSample(TypedDict, total=False):
    system: NotRequired[str]
    instruction: str
    input: NotRequired[str]
    output: str
    images: NotRequired[list[str] | str]
    videos: NotRequired[list[str] | str]
    audios: NotRequired[list[str] | str]


SharegptMessage = TypedDict(
    "SharegptMessage",
    {"from": Literal["human", "gpt", "system", "function_call", "observation"], "value": str},
)


class SharegptSample(TypedDict, total=False):
    conversations: list[SharegptMessage]
    messages: NotRequired[list[dict[str, str]]]
    tools: NotRequired[str]
    images: NotRequired[list[str] | str]
    videos: NotRequired[list[str] | str]
    audios: NotRequired[list[str] | str]


class OpenaiMessage(TypedDict, total=False):
    role: Literal["user", "assistant", "tool"]
    content: str


class OpenaiSample(TypedDict, total=False):
    messages: list[OpenaiMessage]


class PairSample(TypedDict, total=False):
    chosen: list[OpenaiMessage]
    rejected: list[OpenaiMessage]


class DataConverterPlugin(BasePlugin):
    """Plugin for data converters."""

    def __call__(self, raw_sample: dict[str, Any]) -> Sample:
        return super().__call__(raw_sample)


def _normalize_image_refs(raw_images: list[str] | str | None) -> list[str]:
    if raw_images is None:
        return []

    if isinstance(raw_images, str):
        return [raw_images] if raw_images else []

    return [image for image in raw_images if image]


def _consume_media_ref(media_refs: dict[str, list[str]], content_type: str) -> dict[str, str] | None:
    refs = media_refs[content_type]
    if len(refs) == 0:
        return None

    return {"type": content_type, "value": refs.pop(0)}


def _remaining_media_contents(media_refs: dict[str, list[str]]) -> list[dict[str, str]]:
    contents: list[dict[str, str]] = []
    for content_type in ("image_url", "video_url", "audio_url"):
        contents.extend({"type": content_type, "value": media_ref} for media_ref in media_refs[content_type])
        media_refs[content_type].clear()

    return contents


def _content_from_text_and_media(
    text: str, media_refs: dict[str, list[str]], inject_remaining: bool = False
) -> list[dict[str, str]]:
    content: list[dict[str, str]] = []
    placeholder_types = {"<image>": "image_url", "<video>": "video_url", "<audio>": "audio_url"}
    cursor = 0
    while cursor < len(text):
        matches = [(placeholder, text.find(placeholder, cursor)) for placeholder in placeholder_types]
        matches = [(placeholder, index) for placeholder, index in matches if index != -1]
        if len(matches) == 0:
            break

        placeholder, index = min(matches, key=lambda item: item[1])
        if index > cursor:
            content.append({"type": "text", "value": text[cursor:index]})

        content_type = placeholder_types[placeholder]
        media_content = _consume_media_ref(media_refs, content_type)
        if media_content is not None:
            content.append(media_content)
        else:
            logger.warning_rank0(f"Found {placeholder} placeholder but no remaining media path was provided.")

        cursor = index + len(placeholder)

    if cursor < len(text):
        content.append({"type": "text", "value": text[cursor:]})

    if inject_remaining:
        content = _remaining_media_contents(media_refs) + content

    if len(content) == 0:
        content.append({"type": "text", "value": ""})

    return content


@DataConverterPlugin("alpaca").register()
def alpaca_converter(raw_sample: AlpacaSample) -> SFTSample:
    """Convert Alpaca sample to SFT sample.

    See raw example at: https://huggingface.co/datasets/llamafactory/alpaca_gpt4_en

    Args:
        raw_sample (AlpacaSample): Alpaca sample.

    Returns:
        SFTSample: SFT sample.
    """
    messages = []
    if "system" in raw_sample:
        messages.append(
            {"role": "system", "content": [{"type": "text", "value": raw_sample["system"]}], "loss_weight": 0.0}
        )

    media_refs = {
        "image_url": _normalize_image_refs(raw_sample.get("images")),
        "video_url": _normalize_image_refs(raw_sample.get("videos")),
        "audio_url": _normalize_image_refs(raw_sample.get("audios")),
    }
    if "instruction" in raw_sample or "input" in raw_sample or any(len(refs) != 0 for refs in media_refs.values()):
        user_text = raw_sample.get("instruction", "") + raw_sample.get("input", "")
        messages.append(
            {
                "role": "user",
                "content": _content_from_text_and_media(user_text, media_refs, inject_remaining=True),
                "loss_weight": 0.0,
            }
        )

    if "output" in raw_sample:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "value": raw_sample["output"]}], "loss_weight": 1.0}
        )

    return {"messages": messages}


@DataConverterPlugin("sharegpt").register()
def sharegpt_converter(raw_sample: SharegptSample) -> SFTSample:
    """Convert ShareGPT sample to SFT sample.

    See raw example at: https://huggingface.co/datasets/llamafactory/glaive_toolcall_en

    Args:
        raw_sample (SharegptSample): ShareGPT sample.

    Returns:
        SFTSample: SFT sample.
    """
    tag_mapping = {
        "system": "system",
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "observation": "tool",
        "tool": "tool",
        "function_call": "assistant",
    }
    sample = {}
    messages = []
    media_refs = {
        "image_url": _normalize_image_refs(raw_sample.get("images")),
        "video_url": _normalize_image_refs(raw_sample.get("videos")),
        "audio_url": _normalize_image_refs(raw_sample.get("audios")),
    }
    raw_messages = raw_sample.get("conversations") or raw_sample.get("messages", [])
    for message in raw_messages:
        tag = message.get("from", message.get("role"))
        message_text = message.get("value", message.get("content", ""))
        if tag not in tag_mapping:
            logger.warning_rank0(f"Unsupported role tag {tag} in message: {message}")
        elif tag == "function_call":
            try:
                tool_calls: ToolCall | list[ToolCall] = json.loads(message_text)
            except json.JSONDecodeError:
                logger.warning_rank0(f"Invalid tool call format: {str(message_text)}")
                continue

            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "tool_call", "value": json.dumps(tool_call)} for tool_call in tool_calls],
                    "loss_weight": 1.0,
                }
            )
        else:
            role = tag_mapping[tag]
            messages.append(
                {
                    "role": role,
                    "content": _content_from_text_and_media(message_text, media_refs),
                    "loss_weight": 1.0 if role == "assistant" else 0.0,
                }
            )

    if any(len(refs) != 0 for refs in media_refs.values()):
        for message in messages:
            if message["role"] == "user":
                message["content"] = _remaining_media_contents(media_refs) + message["content"]
                break

    sample["messages"] = messages

    tools = raw_sample.get("tools")
    if tools:
        try:
            tools: list[dict[str, Any]] = json.loads(tools)
            sample["tools"] = json.dumps(tools)
        except json.JSONDecodeError:
            logger.warning_rank0(f"Invalid tools format: {str(tools)}")

    return sample


@DataConverterPlugin("pair").register()
def pair_converter(raw_sample: PairSample) -> DPOSample:
    """Convert Pair sample to DPO sample.

    See raw example at: https://huggingface.co/datasets/HuggingFaceH4/orca_dpo_pairs

    Args:
        raw_sample (PairSample): pair sample with chosen, rejected fields.

    Returns:
        DPOSample: DPO sample with chosen_messages and rejected_messages.
    """

    def process_message(raw_messages: list[OpenaiMessage]):
        messages = []
        for message in raw_messages:
            if message["role"] == "tool":
                try:
                    tool_calls: ToolCall | list[ToolCall] = json.loads(message["content"])
                except json.JSONDecodeError:
                    logger.warning_rank0(f"Invalid tool call format: {str(message['content'])}")
                    continue

                if not isinstance(tool_calls, list):
                    tool_calls = [tool_calls]

                messages.append(
                    {
                        "role": message["role"],
                        "content": [{"type": "tool_call", "value": json.dumps(tool_call)} for tool_call in tool_calls],
                        "loss_weight": 1.0 if message["role"] == "assistant" else 0.0,
                    }
                )
            else:
                messages.append(
                    {
                        "role": message["role"],
                        "content": [{"type": "text", "value": message["content"]}],
                        "loss_weight": 1.0 if message["role"] == "assistant" else 0.0,
                    }
                )

        return messages

    sample = {}
    sample["chosen_messages"] = process_message(raw_sample.get("chosen", []))
    sample["rejected_messages"] = process_message(raw_sample.get("rejected", []))

    tools = raw_sample.get("tools")
    if tools:
        try:
            tools: list[dict[str, Any]] = json.loads(tools)
            sample["tools"] = json.dumps(tools)
        except json.JSONDecodeError:
            logger.warning_rank0(f"Invalid tools format: {str(tools)}")

    return sample
