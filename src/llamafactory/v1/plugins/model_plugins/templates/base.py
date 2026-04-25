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

from abc import ABC, abstractmethod

from ....utils.constants import IGNORE_INDEX
from ....utils.helper import get_tokenizer
from ....utils.types import Message, ModelInput, Processor


class BaseTemplateRenderer(ABC):
    """Base contract for model-specific chat template renderers.

    A concrete renderer should keep model chat-template semantics in the render_* methods,
    and use append_segment/build_model_input for the shared training input bookkeeping.
    """

    @classmethod
    def render_messages(
        cls,
        processor: Processor,
        messages: list[Message],
        tools: str | None = None,
        is_generate: bool = False,
        enable_thinking: bool = False,
    ) -> ModelInput:
        model_input = ModelInput(input_ids=[], labels=[], loss_weights=[])
        cls.render_system_and_tools(processor, model_input, messages, tools)
        cls.render_conversation(processor, model_input, messages)
        cls.render_generation_prompt(processor, model_input, is_generate, enable_thinking)
        model_input["attention_mask"] = [1] * len(model_input["input_ids"])
        return model_input

    @staticmethod
    @abstractmethod
    def render_system_and_tools(
        processor: Processor, model_input: ModelInput, messages: list[Message], tools: str | None
    ) -> None:
        """Render leading system/tool definitions according to the model template."""

    @staticmethod
    @abstractmethod
    def render_conversation(processor: Processor, model_input: ModelInput, messages: list[Message]) -> None:
        """Render user/assistant/tool turns according to the model template."""

    @staticmethod
    @abstractmethod
    def render_generation_prompt(
        processor: Processor, model_input: ModelInput, is_generate: bool, enable_thinking: bool
    ) -> None:
        """Render generation prompt suffix when needed."""

    @staticmethod
    @abstractmethod
    def parse_message(generated_text: str) -> Message:
        """Parse generated text back into a v1 assistant message."""

    @staticmethod
    def append_segment(processor: Processor, model_input: ModelInput, text: str, loss_weight: float) -> None:
        if not text:
            return

        tokenizer = get_tokenizer(processor)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        model_input["input_ids"].extend(token_ids)
        model_input["loss_weights"].extend([loss_weight] * len(token_ids))
        if loss_weight > 1e-6:
            model_input["labels"].extend(token_ids)
        else:
            model_input["labels"].extend([IGNORE_INDEX] * len(token_ids))
