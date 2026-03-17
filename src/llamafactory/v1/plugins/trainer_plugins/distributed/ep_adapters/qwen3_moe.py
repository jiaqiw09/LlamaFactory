import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module

from ..expert_parallel import ExpertParallel
from .base import BaseEPAdapter


def _ep_experts_forward(self, x: Tensor, num_tokens_per_expert: Tensor) -> Tensor:
    """EP-compatible forward for Qwen3MoeExperts.

    Accepts pre-sorted tokens grouped by expert and a count tensor, matching
    the ExpertParallel hook interface. Uses local expert weights after DTensor
    Shard(0) partitioning.

    Parameters:
        gate_up_proj: [num_local_experts, 2 * intermediate_dim, hidden_dim]
        down_proj:    [num_local_experts, hidden_dim, intermediate_dim]
    """
    try:
        from torch.distributed._tensor import DTensor

        gate_up = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
        down = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj
    except ImportError:
        gate_up, down = self.gate_up_proj, self.down_proj

    num_local_experts = gate_up.shape[0]
    hidden_size = down.shape[1]
    token_counts = num_tokens_per_expert.tolist()

    outputs = []
    offset = 0
    for i in range(num_local_experts):
        count = int(token_counts[i])
        if count > 0:
            x_e = x[offset : offset + count]
            gate, up = F.linear(x_e, gate_up[i]).chunk(2, dim=-1)
            out = F.linear(self.act_fn(gate) * up, down[i])
            outputs.append(out)
        offset += count

    if not outputs:
        return x.new_empty(0, hidden_size)
    return torch.cat(outputs, dim=0)


def _ep_moe_block_forward(self, hidden_states: Tensor) -> Tensor:
    """EP-aware forward for Qwen3MoeSparseMoeBlock.

    Replaces the per-expert loop with:
        route → sort tokens by expert → call self.experts(sorted, counts) → scatter-combine

    ExpertParallel hooks on self.experts transparently handle all-to-all
    dispatch and combine around the expert computation.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    num_tokens = hidden_states.shape[0]

    _router_logits, routing_weights, selected_experts = self.gate(hidden_states)

    num_experts = self.experts.num_experts
    top_k = self.gate.top_k

    flat_expert_indices = selected_experts.view(-1)
    num_tokens_per_expert = torch.histc(
        flat_expert_indices.float(),
        bins=num_experts,
        min=0,
        max=num_experts - 1,
    ).to(torch.int64)

    sorted_order = torch.argsort(flat_expert_indices, stable=True)
    token_indices = (
        torch.arange(num_tokens, device=hidden_states.device)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
    )
    sorted_token_indices = token_indices[sorted_order]
    sorted_weights = routing_weights.view(-1)[sorted_order]

    routed_input = hidden_states[sorted_token_indices]

    routed_output = self.experts(routed_input, num_tokens_per_expert)

    weighted_output = routed_output * sorted_weights.unsqueeze(-1)
    final_hidden_states = torch.zeros(
        num_tokens, hidden_dim,
        dtype=hidden_states.dtype, device=hidden_states.device,
    )
    final_hidden_states.scatter_add_(
        0,
        sorted_token_indices.unsqueeze(-1).expand_as(weighted_output),
        weighted_output.to(hidden_states.dtype),
    )

    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class Qwen3MoeEPAdapter(BaseEPAdapter):
    """Adapter for Qwen3-MoE (new HF transformers with stacked expert weights).

    Model structure (new transformers):
        DecoderLayer.mlp = Qwen3MoeSparseMoeBlock
            .gate: Qwen3MoeTopKRouter  (weight: [num_experts, hidden_dim])
            .experts: Qwen3MoeExperts   (gate_up_proj: [E, 2*I, H], down_proj: [E, H, I])

    The experts module already has stacked 3D parameters, so ExpertParallel
    can shard them along dim-0 directly.
    """

    @staticmethod
    def get_expert_module(module: nn.Module) -> nn.Module | None:
        try:
            return module.get_submodule("mlp.experts")
        except AttributeError:
            return None

    @staticmethod
    def get_num_experts(experts: nn.Module) -> int | None:
        return getattr(experts, "num_experts", None)

    def prepare_and_apply_ep(self, model: nn.Module, ep_mesh: DeviceMesh) -> int:
        patched = 0
        seen = set()
        for _, module in model.named_modules():
            moe_block = self._find_moe_block(module)
            if moe_block is None or id(moe_block) in seen:
                continue
            seen.add(id(moe_block))

            experts = moe_block.experts
            if not self._has_stacked_params(experts):
                continue

            experts.forward = types.MethodType(_ep_experts_forward, experts)
            moe_block.forward = types.MethodType(_ep_moe_block_forward, moe_block)
            parallelize_module(experts, ep_mesh, ExpertParallel())
            patched += 1

        return patched

    @staticmethod
    def _find_moe_block(module: nn.Module) -> nn.Module | None:
        mlp = getattr(module, "mlp", None)
        if mlp is None:
            return None
        if hasattr(mlp, "gate") and hasattr(mlp, "experts") and hasattr(mlp.experts, "gate_up_proj"):
            return mlp
        return None

    @staticmethod
    def _has_stacked_params(experts: nn.Module) -> bool:
        gate_up = getattr(experts, "gate_up_proj", None)
        down = getattr(experts, "down_proj", None)
        return isinstance(gate_up, nn.Parameter) and isinstance(down, nn.Parameter) and gate_up.ndim == 3
