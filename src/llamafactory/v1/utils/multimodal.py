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
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from .types import Content, ModelInput, Processor


MULTIMODAL_REF_KEYS = ("images", "videos", "audios")


@dataclass
class RenderedMultimodalContent:
    text: str
    images: list[str]
    videos: list[str]
    audios: list[str]


@dataclass
class VideoData:
    frames: list[Image.Image]
    fps: float
    duration: float
    frames_indices: list[int]


def render_text_content(
    content: list[Content],
    image_placeholder: str | None = None,
) -> tuple[str, list[str]]:
    """Serialize multimodal content into text and collect media refs in order."""
    rendered = render_multimodal_content(content, image_placeholder=image_placeholder)
    return rendered.text, rendered.images


def render_multimodal_content(
    content: list[Content],
    image_placeholder: str | None = None,
    video_placeholder: str | None = None,
    audio_placeholder: str | None = None,
) -> RenderedMultimodalContent:
    """Serialize multimodal content into text and collect media refs in order."""
    text_chunks: list[str] = []
    image_refs: list[str] = []
    video_refs: list[str] = []
    audio_refs: list[str] = []
    for item in content:
        if item["type"] == "text":
            text_chunks.append(item["value"])
        elif item["type"] == "image_url":
            if image_placeholder is None:
                raise ValueError("image_placeholder must be provided for image_url content.")
            text_chunks.append(image_placeholder)
            image_refs.append(item["value"])
        elif item["type"] == "video_url":
            if video_placeholder is None:
                raise ValueError("video_placeholder must be provided for video_url content.")
            text_chunks.append(video_placeholder)
            video_refs.append(item["value"])
        elif item["type"] == "audio_url":
            if audio_placeholder is None:
                raise ValueError("audio_placeholder must be provided for audio_url content.")
            text_chunks.append(audio_placeholder)
            audio_refs.append(item["value"])
        else:
            raise ValueError(f"Unsupported content type in multimodal template: {item['type']}")

    return RenderedMultimodalContent(
        text="".join(text_chunks),
        images=image_refs,
        videos=video_refs,
        audios=audio_refs,
    )


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


def _get_video_processor(processor: Processor) -> Any:
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        raise ValueError(
            "The current processor does not provide a video_processor for video inputs. "
            f"Got processor={processor.__class__.__name__}. Please use a video-capable multimodal processor."
        )

    return video_processor


def _load_video(path: str, *, video_fps: float = 2.0, video_maxlen: int = 128) -> VideoData:
    try:
        import av
    except ImportError as error:
        raise ImportError("Video inputs require the `av` package to decode local video files.") from error

    resolved_path = os.path.abspath(path)
    with av.open(resolved_path) as container:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate) if stream.average_rate is not None else video_fps
        sample_stride = max(int(round(source_fps / video_fps)), 1)
        frames: list[Image.Image] = []
        frames_indices: list[int] = []

        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx % sample_stride != 0:
                continue

            frames.append(frame.to_image().convert("RGB"))
            frames_indices.append(frame_idx)
            if len(frames) >= video_maxlen:
                break

    if len(frames) == 0:
        raise ValueError(f"No decodable video frames found in {path}.")

    if len(frames) % 2 != 0:
        frames.append(frames[-1].copy())
        frames_indices.append(frames_indices[-1])

    effective_fps = min(source_fps, video_fps) if source_fps > 0 else video_fps
    duration = (frames_indices[-1] + 1) / source_fps if source_fps > 0 else len(frames) / effective_fps
    return VideoData(frames=frames, fps=effective_fps, duration=duration, frames_indices=frames_indices)


def _call_video_processor(video_processor: Any, video_data: list[VideoData], fps: float) -> dict[str, Any]:
    videos = [data.frames for data in video_data]
    video_metadata = [
        {
            "fps": data.fps,
            "duration": data.duration,
            "total_num_frames": len(data.frames),
            "frames_indices": data.frames_indices,
        }
        for data in video_data
    ]
    try:
        return video_processor(
            videos=videos,
            video_metadata=video_metadata,
            fps=fps,
            return_metadata=True,
            do_sample_frames=False,
        )
    except TypeError:
        try:
            return video_processor(videos=videos, return_tensors="pt")
        except TypeError:
            return video_processor(videos, return_tensors="pt")


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


def get_video_token_counts(processor: Processor, video_refs: list[str]) -> list[int]:
    """Compute expanded video token counts from the processor grid metadata."""
    if len(video_refs) == 0:
        return []

    video_processor = _get_video_processor(processor)
    video_fps = getattr(processor, "video_fps", 2.0)
    video_maxlen = getattr(processor, "video_maxlen", 128)
    videos = [_load_video(path, video_fps=video_fps, video_maxlen=video_maxlen) for path in video_refs]
    mm_inputs = _call_video_processor(video_processor, videos, video_fps)
    video_grid_thw = mm_inputs.get("video_grid_thw")
    if video_grid_thw is None:
        return [1] * len(video_refs)

    merge_size = getattr(video_processor, "merge_size", getattr(processor, "merge_size", 1))
    merge_length = merge_size**2
    return [int(grid.prod().item() // merge_length) for grid in video_grid_thw]


def _build_image_tensors(processor: Processor, image_refs: list[str]) -> dict[str, Any]:
    image_processor = _get_image_processor(processor)
    images = [_load_image(path) for path in image_refs]
    mm_inputs = _call_image_processor(image_processor, images)

    return {key: _to_tensor(value) for key, value in mm_inputs.items()}


def _build_video_tensors(processor: Processor, video_refs: list[str]) -> dict[str, Any]:
    video_processor = _get_video_processor(processor)
    video_fps = getattr(processor, "video_fps", 2.0)
    video_maxlen = getattr(processor, "video_maxlen", 128)
    videos = [_load_video(path, video_fps=video_fps, video_maxlen=video_maxlen) for path in video_refs]
    mm_inputs = _call_video_processor(video_processor, videos, video_fps)

    image_processor = getattr(processor, "image_processor", None)
    temporal_patch_size = getattr(image_processor, "temporal_patch_size", 2)
    if "second_per_grid_ts" in getattr(processor, "model_input_names", []):
        mm_inputs["second_per_grid_ts"] = [temporal_patch_size / video.fps for video in videos]

    return {key: _to_tensor(value) for key, value in mm_inputs.items() if key != "video_metadata"}


def _to_tensor(value: Any) -> Any:
    if torch.is_tensor(value):
        return value

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return torch.empty((0,), dtype=torch.float32)

        if isinstance(value[0], str):
            return value

        try:
            return torch.tensor(value)
        except (TypeError, ValueError):
            return value

    return value


def build_multimodal_tensors(processor: Processor, model_inputs: Sequence[ModelInput]) -> dict[str, Any]:
    """Build multimodal tensors for a micro batch from extracted media refs."""
    image_refs = [path for sample in model_inputs for path in sample.get("images", [])]
    video_refs = [path for sample in model_inputs for path in sample.get("videos", [])]

    mm_inputs: dict[str, Any] = {}
    if len(image_refs) != 0:
        mm_inputs.update(_build_image_tensors(processor, image_refs))

    if len(video_refs) != 0:
        mm_inputs.update(_build_video_tensors(processor, video_refs))

    return mm_inputs
