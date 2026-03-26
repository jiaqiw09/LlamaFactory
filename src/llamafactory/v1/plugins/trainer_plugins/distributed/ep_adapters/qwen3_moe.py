import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module

from ....utils.logging import get_logger
from ..expert_parallel import ExpertParallel, _dist_prefix, _tensor_preview, _tensor_summary, log_module_parameters
from .base import BaseEPAdapter


logger = get_logger(__name__)


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
    device_mesh = getattr(self, "_ep_device_mesh", None)
    logger.info(
        "%s experts_forward enter x=%s num_tokens_per_expert=%s num_local_experts=%s hidden_size=%s",
        _dist_prefix(device_mesh),
        _tensor_summary(x),
        token_counts,
        num_local_experts,
        hidden_size,
    )

    outputs = []
    offset = 0
    for i in range(num_local_experts):
        count = int(token_counts[i])
        if count > 0:
            x_e = x[offset : offset + count]
            logger.info(
                "%s experts_forward local_expert=%s slice=[%s:%s] x_e=%s",
                _dist_prefix(device_mesh),
                i,
                offset,
                offset + count,
                _tensor_summary(x_e),
            )
            gate, up = F.linear(x_e, gate_up[i]).chunk(2, dim=-1)
            out = F.linear(self.act_fn(gate) * up, down[i])
            outputs.append(out)
        offset += count

    if not outputs:
        dummy = (gate_up.sum() * 0.0) + (down.sum() * 0.0)
        logger.info("%s experts_forward no tokens routed to this rank", _dist_prefix(device_mesh))
        return x + dummy.to(x.dtype)
    output = torch.cat(outputs, dim=0)
    logger.info(
        "%s experts_forward exit output=%s",
        _dist_prefix(device_mesh),
        _tensor_summary(output),
    )
    return output


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
    device_mesh = getattr(self.experts, "_ep_device_mesh", None)
    logger.info(
        "%s moe_forward enter hidden_states=%s batch_size=%s sequence_length=%s hidden_dim=%s num_experts=%s top_k=%s",
        _dist_prefix(device_mesh),
        _tensor_summary(hidden_states),
        batch_size,
        sequence_length,
        hidden_dim,
        self.experts.num_experts,
        self.gate.top_k,
    )

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
    logger.info(
        "%s moe_forward router selected_experts=%s routing_weights=%s num_tokens_per_expert=%s",
        _dist_prefix(device_mesh),
        _tensor_summary(selected_experts),
        _tensor_summary(routing_weights),
        _tensor_summary(num_tokens_per_expert),
    )

    sorted_order = torch.argsort(flat_expert_indices, stable=True)
    token_indices = (
        torch.arange(num_tokens, device=hidden_states.device)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
    )
    sorted_token_indices = token_indices[sorted_order]
    sorted_weights = routing_weights.view(-1)[sorted_order]
    logger.info(
        "%s moe_forward sorted_order=%s sorted_token_indices=%s sorted_weights=%s",
        _dist_prefix(device_mesh),
        _tensor_preview(sorted_order),
        _tensor_preview(sorted_token_indices),
        _tensor_summary(sorted_weights),
    )

    routed_input = hidden_states[sorted_token_indices]
    logger.info("%s moe_forward routed_input=%s", _dist_prefix(device_mesh), _tensor_summary(routed_input))

    routed_output = self.experts(routed_input, num_tokens_per_expert)
    logger.info("%s moe_forward routed_output=%s", _dist_prefix(device_mesh), _tensor_summary(routed_output))

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
    logger.info(
        "%s moe_forward final_hidden_states=%s",
        _dist_prefix(device_mesh),
        _tensor_summary(final_hidden_states),
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

            logger.info(
                "%s prepare_and_apply_ep patching moe_block=%s experts=%s",
                _dist_prefix(ep_mesh),
                type(moe_block).__name__,
                type(experts).__name__,
            )
            experts._ep_device_mesh = ep_mesh
            log_module_parameters(experts, f"pre-parallelize experts for {type(moe_block).__name__}", ep_mesh)
            experts.forward = types.MethodType(_ep_experts_forward, experts)
            moe_block.forward = types.MethodType(_ep_moe_block_forward, moe_block)
            parallelize_module(experts, ep_mesh, ExpertParallel())
            log_module_parameters(experts, f"post-parallelize experts for {type(moe_block).__name__}", ep_mesh)
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
