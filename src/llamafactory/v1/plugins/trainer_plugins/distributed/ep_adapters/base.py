from abc import ABC, abstractmethod

import torch.nn as nn

from .....utils.types import HFModel


class BaseEPAdapter(ABC):
    """Abstract base class for expert parallelism model adapters."""

    @staticmethod
    @abstractmethod
    def get_expert_module(module: nn.Module) -> nn.Module | None:
        """Find the expert module (e.g., MoE block) within a given module."""
        pass

    @staticmethod
    @abstractmethod
    def get_num_experts(experts: nn.Module) -> int | None:
        """Get the number of experts in the expert module."""
        pass

    @property
    def permute_backend(self) -> str:
        """Preferred token permute backend for this model."""
        return "default"
