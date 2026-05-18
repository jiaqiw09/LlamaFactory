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

"""Interface for rendering plugins."""

from ....utils.plugin import BasePlugin
from ....utils.types import Message, ModelInput, Processor


class RenderingPlugin(BasePlugin):
    def render_messages(
        self,
        processor: Processor,
        messages: list[Message],
        tools: str | None = None,
        is_generate: bool = False,
        enable_thinking: bool = False,
    ) -> ModelInput:
        """Render messages in the template format."""
        return self["render_messages"](processor, messages, tools, is_generate, enable_thinking)

    def parse_message(self, generated_text: str) -> Message:
        """Parse messages in the template format."""
        return self["parse_message"](generated_text)

    def parse_messages(self, generated_text: str) -> Message:
        """Backward-compatible alias for parse_message."""
        return self.parse_message(generated_text)


@RenderingPlugin("qwen3").register_methods()
class Qwen3RenderingPlugin:
    @staticmethod
    def render_messages(
        processor: Processor,
        messages: list[Message],
        tools: str | None = None,
        is_generate: bool = False,
        enable_thinking: bool = False,
    ) -> ModelInput:
        from .qwen3 import render_qwen3_messages

        return render_qwen3_messages(processor, messages, tools, is_generate, enable_thinking)

    @staticmethod
    def parse_message(generated_text: str) -> Message:
        from .qwen3 import parse_qwen3_message

        return parse_qwen3_message(generated_text)


@RenderingPlugin("qwen3_nothink").register_methods()
class Qwen3NoThinkRenderingPlugin:
    @staticmethod
    def render_messages(
        processor: Processor,
        messages: list[Message],
        tools: str | None = None,
        is_generate: bool = False,
        enable_thinking: bool = False,
    ) -> ModelInput:
        from .qwen3_nothink import render_qwen3_nothink_messages

        return render_qwen3_nothink_messages(processor, messages, tools, is_generate, enable_thinking)

    @staticmethod
    def parse_message(generated_text: str) -> Message:
        from .qwen3_nothink import parse_qwen3_nothink_message

        return parse_qwen3_nothink_message(generated_text)
