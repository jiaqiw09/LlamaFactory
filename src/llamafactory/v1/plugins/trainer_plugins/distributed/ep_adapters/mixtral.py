import torch.nn as nn

from .base import BaseEPAdapter


class MixtralEPAdapter(BaseEPAdapter):
    """Adapter for Mixtral model family."""

    @staticmethod
    def get_expert_module(module: nn.Module) -> nn.Module | None:
        try:
            return module.get_submodule("block_sparse_moe.experts")
        except AttributeError:
            return None

    @staticmethod
    def get_num_experts(experts: nn.Module) -> int | None:
        return getattr(experts, "num_experts", None) or len(experts)

    @property
    def permute_backend(self) -> str:
        return "mixtral"
