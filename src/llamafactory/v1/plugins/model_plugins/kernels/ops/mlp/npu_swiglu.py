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

"""The definition of NPU fused SwiGLU kernels.

Init Phase:
1. Define SwiGLU forward functions.
2. Register NPU fused SwiGLU kernel.

"""

import types

import torch

from ......accelerator.helper import DeviceType
from ......utils.types import HFModel
from ...base import BaseKernel
from ...registry import register_kernel


try:
    import torch_npu
except ImportError:
    pass


def npu_swiglu_forward(self, hidden_state):
    """SwiGLU forward pass for NPU.

    Args:
        self: The MLP layer instance.
        hidden_state (Tensor): Input hidden state.

    Returns:
        Tensor: Output of SwiGLU.
    """
    return self.down_proj(
        torch_npu.npu_swiglu(torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1), dim=-1)
    )


def _npu_swiglu_glm4_forward(self, hidden_states):
    """SwiGLU forward pass for GLM4 on NPU.

    Args:
        self: The GLM4 MLP layer instance.
        hidden_states (Tensor): Input hidden states.

    Returns:
        Tensor: Output of SwiGLU.
    """
    up_states = self.gate_up_proj(hidden_states)
    gate, up_states = up_states.chunk(2, dim=-1)
    return self.down_proj(torch_npu.npu_swiglu(torch.cat((gate, up_states), dim=-1), dim=-1))


def _npu_swiglu_gemma3ntext_forward(self, hidden_states):
    """SwiGLU forward pass for Gemma3nText on NPU.

    Args:
        self: The Gemma3nText MLP layer instance.
        hidden_states (Tensor): Input hidden states.

    Returns:
        Tensor: Output of SwiGLU.
    """
    gate_proj = self.gate_proj(hidden_states)
    if self.activation_sparsity > 0.0:
        gate_proj = self._gaussian_topk(gate_proj)
    down_proj = self.down_proj(
        torch_npu.npu_swiglu(torch.cat((gate_proj, self.up_proj(hidden_states)), dim=-1), dim=-1)
    )
    return down_proj


kernel_swiglu_mapping = {
    "Qwen3ForCausalLM": {
        "Qwen3MLP": npu_swiglu_forward,
    },
    "Qwen3MoeForCausalLM": {
        "Qwen3MoeMLP": npu_swiglu_forward,
    },
    "Qwen3NextForCausalLM": {
        "Qwen3NextMLP": npu_swiglu_forward,
    },
    "Qwen3VLForConditionalGeneration": {
        "Qwen3VLTextMLP": npu_swiglu_forward,
    },
    "Qwen3VLMoeForConditionalGeneration": {
        "Qwen3VLMoeTextMLP": npu_swiglu_forward,
    },
    "Qwen3_5ForCausalLM": {
        "Qwen3_5MLP": npu_swiglu_forward,
    },
    "Qwen3_5ForConditionalGeneration": {
        "Qwen3_5MLP": npu_swiglu_forward,
    },
    "Qwen3_5MoeForCausalLM": {
        "Qwen3_5MoeMLP": npu_swiglu_forward,
    },
    "Qwen3_5MoeForConditionalGeneration": {
        "Qwen3_5MoeMLP": npu_swiglu_forward,
    },
    "Qwen3OmniMoeForConditionalGeneration": {
        "Qwen3OmniMoeCode2WavMlp": npu_swiglu_forward,
        "Qwen3OmniMoeMLP": npu_swiglu_forward,
        "Qwen3OmniMoeTalkerTextMLP": npu_swiglu_forward,
        "Qwen3OmniMoeThinkerTextMLP": npu_swiglu_forward,
    },
}


@register_kernel
class NpuSwiGluKernel(BaseKernel):
    """NPU Kernel for fused SwiGLU activation."""

    _kernel_id = "npu_fused_swiglu"
    _device = DeviceType.NPU

    @classmethod
    def apply(cls, **kwargs) -> "HFModel":
        """Applies the NPU fused SwiGLU kernel to whitelisted MLP modules.

        Args:
            **kwargs: Keyword arguments containing the model.

        Returns:
            HFModel: The model with patched SwiGLU forward functions.

        Raises:
            ValueError: If the model is not provided.
            RuntimeError: If dependencies are not met.
        """
        model = kwargs.get("model", None)
        if model is None:
            raise ValueError(f"HFModel instance is required for {cls.__name__}.")

        if not cls.check_deps():
            raise RuntimeError("torch_npu is not available but NpuSwiGluKernel was called.")

        archs = getattr(model.config, "architectures", None) or []
        target_swiglu_mapping = None
        for arch in archs:
            if arch in kernel_swiglu_mapping:
                target_swiglu_mapping = kernel_swiglu_mapping[arch]
                break

        if target_swiglu_mapping is None:
            return model

        for _, module in model.named_modules():
            class_name = module.__class__.__name__
            if class_name in target_swiglu_mapping:
                # Bind function as an instance method to preserve `self` semantics
                # and replace the original forward
                new_forward_func = target_swiglu_mapping[class_name]
                module.forward = types.MethodType(new_forward_func, module)

        return model
