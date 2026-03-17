from abc import ABC, abstractmethod

import torch.nn as nn
from torch.distributed.tensor import DeviceMesh


class BaseEPAdapter(ABC):
    """Abstract base class for expert parallelism model adapters.

    Each adapter handles model-specific logic:
    - Locating MoE blocks and their stacked expert modules
    - Patching forward methods for EP-compatible token dispatch
    - Applying ExpertParallel to shard expert parameters
    """

    @staticmethod
    @abstractmethod
    def get_expert_module(module: nn.Module) -> nn.Module | None:
        """Find the stacked expert module within a given module.

        Used by FSDP2 to apply expert-specific sharding.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_num_experts(experts: nn.Module) -> int | None:
        """Get the number of experts in the expert module."""
        pass

    @property
    def permute_backend(self) -> str:
        """Token permute backend name for this model type."""
        return "default"

    @abstractmethod
    def prepare_and_apply_ep(self, model: nn.Module, ep_mesh: DeviceMesh) -> int:
        """Prepare model for EP and apply ExpertParallel.

        This method should:
        1. Find all MoE blocks in the model
        2. Patch expert and MoE block forward methods for EP
        3. Apply ExpertParallel to shard stacked expert parameters

        Returns the number of MoE blocks patched.
        """
        pass
