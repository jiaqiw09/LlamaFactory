import torch.nn as nn

from .base import BaseEPAdapter


class DeepSeekV2EPAdapter(BaseEPAdapter):
    """Adapter for DeepSeek-V2 model family."""

    @staticmethod
    def get_expert_module(module: nn.Module) -> nn.Module | None:
        try:
            return module.get_submodule("moe.experts")
        except AttributeError:
            return None

    @staticmethod
    def get_num_experts(experts: nn.Module) -> int | None:
        return getattr(experts, "n_experts", None) or len(experts)

    @property
    def permute_backend(self) -> str:
        return "deepseek_v2"
