from typing import Optional

from .....utils.types import HFModel
from .base import BaseEPAdapter
from .deepseek_v2 import DeepSeekV2EPAdapter
from .default import DefaultEPAdapter
from .mixtral import MixtralEPAdapter
from .qwen2_moe import Qwen2MoeEPAdapter


_EP_ADAPTER_REGISTRY = {
    "default": DefaultEPAdapter,
    "qwen2_moe": Qwen2MoeEPAdapter,
    "mixtral": MixtralEPAdapter,
    "deepseek_v2": DeepSeekV2EPAdapter,
}


def _infer_adapter_name(model: HFModel) -> str:
    """Infer adapter name from model configuration."""
    model_type = str(getattr(model.config, "model_type", "")).lower()
    architectures = getattr(model.config, "architectures", None) or []
    arch_text = " ".join(str(item).lower() for item in architectures)
    signature = f"{model_type} {arch_text}".strip()

    if "mixtral" in signature:
        return "mixtral"
    if "deepseek" in signature:
        return "deepseek_v2"
    if "qwen" in signature and "moe" in signature:
        return "qwen2_moe"
    # Qwen3 MoE uses the same structure as Qwen2 MoE
    if "qwen3_moe" in signature:
        return "qwen2_moe"
    return "default"


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
