from .base import BaseEPAdapter
from .qwen3_moe import Qwen3MoeEPAdapter

from .....utils.types import HFModel


_EP_ADAPTER_REGISTRY: dict[str, type[BaseEPAdapter]] = {
    "qwen3_moe": Qwen3MoeEPAdapter,
}


def _infer_adapter_name(model: HFModel) -> str:
    """Infer adapter name from model configuration.

    New HF transformers unifies all MoE models to the same stacked-tensor
    expert structure (gate_up_proj + down_proj), so a single adapter works.
    """
    return "qwen3_moe"


def get_ep_adapter(model: HFModel, adapter_name: str = "auto") -> BaseEPAdapter:
    """Get expert parallelism adapter for the given model."""
    if adapter_name == "auto":
        adapter_name = _infer_adapter_name(model)

    adapter_cls = _EP_ADAPTER_REGISTRY.get(adapter_name)
    if adapter_cls is None:
        raise ValueError(
            f"Unknown EP adapter: {adapter_name}. Available: {sorted(_EP_ADAPTER_REGISTRY.keys())}"
        )

    return adapter_cls()
