import torch.nn as nn

from .base import BaseEPAdapter


class DefaultEPAdapter(BaseEPAdapter):
    """Default adapter for unknown MoE models, trying common paths."""

    @staticmethod
    def get_expert_module(module: nn.Module) -> nn.Module | None:
        for attr in ("mlp.experts", "moe.experts", "block_sparse_moe.experts", "experts"):
            try:
                expert_module = module.get_submodule(attr)
                if expert_module is not None:
                    return expert_module
            except AttributeError:
                continue
        return None

    @staticmethod
    def get_num_experts(experts: nn.Module) -> int | None:
        for attr in ("num_experts", "n_experts"):
            value = getattr(experts, attr, None)
            if isinstance(value, int):
                return value
        if hasattr(experts, "__len__"):
            try:
                return len(experts)  # type: ignore[arg-type]
            except TypeError:
                return None
        return None
