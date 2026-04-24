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

"""Utilities for the v1 multimodal bridge."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image

from .types import Content, ModelInput, Processor


MULTIMODAL_REF_KEYS = ("images",)


def render_text_content(
    content: list[Content],
    image_placeholder: str | None = None,
) -> tuple[str, list[str]]:
    """Serialize multimodal content into text and collect media refs in order."""
    text_chunks: list[str] = []
    image_refs: list[str] = []
    for item in content:
        if item["type"] == "text":
            text_chunks.append(item["value"])
        elif item["type"] == "image_url":
            if image_placeholder is None:
                raise ValueError("image_placeholder must be provided for image_url content.")
            text_chunks.append(image_placeholder)
            image_refs.append(item["value"])
        elif item["type"] == "video_url":
            raise NotImplementedError("The v1 multimodal bridge only supports image_url in stage 1.")
        elif item["type"] == "audio_url":
            raise NotImplementedError("The v1 multimodal bridge only supports image_url in stage 1.")
        else:
            raise ValueError(f"Unsupported content type in multimodal template: {item['type']}")

    return "".join(text_chunks), image_refs


def split_model_input(model_input: ModelInput) -> tuple[ModelInput, ModelInput]:
    """Split raw multimodal refs from tensorizable model input fields."""
    text_input: ModelInput = {}
    multimodal_input: ModelInput = {}
    for key, value in model_input.items():
        if key in MULTIMODAL_REF_KEYS:
            multimodal_input[key] = value
        else:
            text_input[key] = value

    return text_input, multimodal_input


def _load_image(path: str) -> Image.Image:
    resolved_path = os.path.abspath(path)
    with Image.open(resolved_path) as image:
        return image.convert("RGB").copy()


def _get_image_processor(processor: Processor) -> Any:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError(
            "The current processor does not provide an image_processor for multimodal inputs. "
            f"Got processor={processor.__class__.__name__}. Please use a multimodal model/processor such as "
            "Qwen3.5 with template=qwen3_5; pure text Qwen3 processors cannot train image_url samples."
        )

    return image_processor


def _call_image_processor(image_processor: Any, images: list[Image.Image]) -> dict[str, Any]:
    try:
        return image_processor(images=images, return_tensors="pt")
    except TypeError:
        return image_processor(images, return_tensors="pt")


def get_image_token_counts(processor: Processor, image_refs: list[str]) -> list[int]:
    """Compute expanded image token counts from the processor grid metadata."""
    if len(image_refs) == 0:
        return []

    image_processor = _get_image_processor(processor)
    images = [_load_image(path) for path in image_refs]
    mm_inputs = _call_image_processor(image_processor, images)
    image_grid_thw = mm_inputs.get("image_grid_thw")
    if image_grid_thw is None:
        return [1] * len(image_refs)

    merge_size = getattr(image_processor, "merge_size", getattr(processor, "merge_size", 1))
    merge_length = merge_size**2
    return [int(grid.prod().item() // merge_length) for grid in image_grid_thw]


def _build_image_tensors(processor: Processor, image_refs: list[str]) -> dict[str, Any]:
    image_processor = _get_image_processor(processor)
    images = [_load_image(path) for path in image_refs]
    mm_inputs = _call_image_processor(image_processor, images)

    return {key: _to_tensor(value) for key, value in mm_inputs.items()}


def _to_tensor(value: Any) -> Any:
    if torch.is_tensor(value):
        return value

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return torch.empty((0,), dtype=torch.float32)

        if isinstance(value[0], str):
            return value

        return torch.tensor(value)

    return value


def build_multimodal_tensors(processor: Processor, model_inputs: Sequence[ModelInput]) -> dict[str, Any]:
    """Build multimodal tensors for a micro batch from extracted media refs."""
    image_refs = [path for sample in model_inputs for path in sample.get("images", [])]
    if len(image_refs) == 0:
        return {}

    return _build_image_tensors(processor, image_refs)
