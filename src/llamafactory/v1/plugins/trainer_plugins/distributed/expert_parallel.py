from collections.abc import Callable
import os

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed._functional_collectives import all_to_all_single, all_to_all_single_autograd
from torch.distributed.tensor import DeviceMesh, Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle

from ....utils.logging import get_logger


logger = get_logger(__name__)


def _dist_prefix(device_mesh: DeviceMesh | None = None) -> str:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        global_rank = 0
        world_size = 1

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    prefix = f"[rank={global_rank} local_rank={local_rank} world={world_size}]"
    if device_mesh is not None:
        try:
            prefix += f"[ep_size={device_mesh.size()} ep_rank={device_mesh.get_local_rank()}]"
        except Exception:
            prefix += f"[ep_size={device_mesh.size()}]"
    return prefix


def _preview_list(values: list[int] | list[float], max_items: int = 16) -> str:
    if len(values) <= max_items:
        return str(values)
    head = values[:max_items]
    return f"{head} ... (len={len(values)})"


def _tensor_summary(tensor: Tensor, max_items: int = 16) -> str:
    flat = tensor.detach().cpu().reshape(-1)
    preview_values = flat[:max_items].tolist()
    preview = _preview_list(preview_values, max_items=max_items)
    if flat.numel() > max_items:
        preview = f"{preview} ... (len={flat.numel()})"
    return f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} values={preview}"


def _tensor_preview(tensor: Tensor, max_items: int = 16) -> str:
    flat = tensor.detach().cpu().reshape(-1)
    preview_values = flat[:max_items].tolist()
    preview = str(preview_values)
    if flat.numel() > max_items:
        preview = f"{preview} ... (len={flat.numel()})"
    return preview


def _module_param_summary(param: Tensor) -> str:
    try:
        from torch.distributed._tensor import DTensor
    except ImportError:
        DTensor = None

    if DTensor is not None and isinstance(param, DTensor):
        local = param.to_local()
        placements = ",".join(type(p).__name__ if not hasattr(p, "dim") else f"{type(p).__name__}(dim={p.dim})" for p in param.placements)
        return (
            f"DTensor global_shape={tuple(param.shape)} local_shape={tuple(local.shape)} "
            f"dtype={param.dtype} device={local.device} placements=[{placements}]"
        )

    return f"Tensor shape={tuple(param.shape)} dtype={param.dtype} device={param.device}"


def log_module_parameters(module: nn.Module, label: str, device_mesh: DeviceMesh | None = None) -> None:
    params = list(module.named_parameters(recurse=False))
    if not params:
        logger.info("%s %s has no direct parameters", _dist_prefix(device_mesh), label)
        return

    for param_name, param in params:
        logger.info(
            "%s %s param=%s %s",
            _dist_prefix(device_mesh),
            label,
            param_name,
            _module_param_summary(param),
        )


class TokenPermuteBackend:
    def permute(self, routed_input: Tensor, num_tokens_per_expert: Tensor, device_mesh: DeviceMesh) -> tuple[Tensor, Tensor, dict]:
        raise NotImplementedError

    def unpermute(self, routed_output: Tensor, state: dict, device_mesh: DeviceMesh) -> Tensor:
        raise NotImplementedError


class DefaultTokenPermuteBackend(TokenPermuteBackend):
    def permute(self, routed_input: Tensor, num_tokens_per_expert: Tensor, device_mesh: DeviceMesh) -> tuple[Tensor, Tensor, dict]:
        ep_degree = device_mesh.size()
        logger.info(
            "%s permute enter routed_input=%s num_tokens_per_expert=%s",
            _dist_prefix(device_mesh),
            _tensor_summary(routed_input),
            _tensor_summary(num_tokens_per_expert),
        )
        if ep_degree <= 1:
            state = {
                "input_splits": None,
                "output_splits": None,
                "combine_permutation": None,
            }
            logger.info("%s permute bypassed because ep_degree<=1", _dist_prefix(device_mesh))
            return routed_input, num_tokens_per_expert, state

        if num_tokens_per_expert.numel() % ep_degree != 0:
            raise ValueError(
                f"num_tokens_per_expert length must be divisible by ep degree, "
                f"got {num_tokens_per_expert.numel()} and {ep_degree}."
            )

        num_local_experts = num_tokens_per_expert.numel() // ep_degree
        with torch.no_grad():
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=device_mesh.get_group(),
            )
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(num_tokens_per_expert_group)

            input_splits = num_tokens_per_expert.view(ep_degree, num_local_experts).sum(dim=1).to("cpu")
            output_splits = num_tokens_per_expert_group.view(ep_degree, num_local_experts).sum(dim=1).to("cpu")
            input_splits_list = input_splits.tolist()
            output_splits_list = output_splits.tolist()

        logger.info(
            "%s permute token-count exchange num_local_experts=%s input_splits=%s output_splits=%s exchanged_counts=%s",
            _dist_prefix(device_mesh),
            num_local_experts,
            input_splits_list,
            output_splits_list,
            _tensor_summary(num_tokens_per_expert_group),
        )

        routed_input = all_to_all_single_autograd(
            routed_input,
            output_splits_list,
            input_splits_list,
            device_mesh.get_group(),
        )
        logger.info(
            "%s permute after all_to_all routed_input=%s",
            _dist_prefix(device_mesh),
            _tensor_summary(routed_input),
        )

        tokens_matrix = num_tokens_per_expert_group.view(ep_degree, num_local_experts)
        logger.info(
            "%s permute token matrix shape=%s matrix=%s",
            _dist_prefix(device_mesh),
            tuple(tokens_matrix.shape),
            _tensor_summary(tokens_matrix),
        )
        token_offsets = tokens_matrix.reshape(-1).cumsum(0)
        starts = torch.zeros_like(token_offsets)
        starts[1:] = token_offsets[:-1]

        permutation_chunks = []
        for local_expert in range(num_local_experts):
            for src_rank in range(ep_degree):
                flat_idx = src_rank * num_local_experts + local_expert
                start = starts[flat_idx].item()
                end = token_offsets[flat_idx].item()
                if end > start:
                    permutation_chunks.append(torch.arange(start, end, device=routed_input.device))

        combine_permutation = None
        if permutation_chunks:
            dispatch_permutation = torch.cat(permutation_chunks, dim=0)
            combine_permutation = torch.empty_like(dispatch_permutation)
            combine_permutation[dispatch_permutation] = torch.arange(
                dispatch_permutation.numel(),
                device=dispatch_permutation.device,
            )
            routed_input = routed_input.index_select(0, dispatch_permutation)
            logger.info(
                "%s permute dispatch_permutation=%s combine_permutation=%s rerouted_input=%s",
                _dist_prefix(device_mesh),
                _tensor_preview(dispatch_permutation),
                _tensor_preview(combine_permutation),
                _tensor_summary(routed_input),
            )
        else:
            logger.info("%s permute no token reordering was needed", _dist_prefix(device_mesh))

        state = {
            "input_splits": input_splits_list,
            "output_splits": output_splits_list,
            "combine_permutation": combine_permutation,
        }
        local_tokens_per_expert = tokens_matrix.sum(dim=0).to(num_tokens_per_expert.dtype)
        logger.info(
            "%s permute state=%s local_tokens_per_expert=%s",
            _dist_prefix(device_mesh),
            {
                "input_splits": input_splits_list,
                "output_splits": output_splits_list,
                "combine_permutation": None
                if combine_permutation is None
                else _tensor_preview(combine_permutation),
            },
            _tensor_summary(local_tokens_per_expert),
        )
        return routed_input, local_tokens_per_expert, state

    def unpermute(self, routed_output: Tensor, state: dict, device_mesh: DeviceMesh) -> Tensor:
        combine_permutation = state.get("combine_permutation")
        logger.info(
            "%s unpermute enter routed_output=%s state_keys=%s",
            _dist_prefix(device_mesh),
            _tensor_summary(routed_output),
            sorted(state.keys()),
        )
        if combine_permutation is not None:
            routed_output = routed_output.index_select(0, combine_permutation)
            logger.info(
                "%s unpermute after inverse permutation routed_output=%s",
                _dist_prefix(device_mesh),
                _tensor_summary(routed_output),
            )

        input_splits = state.get("input_splits")
        output_splits = state.get("output_splits")
        if input_splits is None or output_splits is None:
            logger.info("%s unpermute bypassed because splits are missing", _dist_prefix(device_mesh))
            return routed_output

        final_output = all_to_all_single_autograd(
            routed_output,
            input_splits,
            output_splits,
            device_mesh.get_group(),
        )
        logger.info(
            "%s unpermute after all_to_all final_output=%s",
            _dist_prefix(device_mesh),
            _tensor_summary(final_output),
        )
        return final_output


_TOKEN_PERMUTE_BACKENDS: dict[str, Callable[[], TokenPermuteBackend]] = {
    "default": DefaultTokenPermuteBackend,
}


def register_token_permute_backend(name: str, factory: Callable[[], TokenPermuteBackend]) -> None:
    _TOKEN_PERMUTE_BACKENDS[name] = factory


def create_token_permute_backend(name: str) -> TokenPermuteBackend:
    factory = _TOKEN_PERMUTE_BACKENDS.get(name, _TOKEN_PERMUTE_BACKENDS["default"])
    return factory()


class ExpertParallel(ParallelStyle):
    """Distributes expert parameters along dim-0 and adds all-to-all token dispatch/combine.

    Applied to any module with stacked expert weights (e.g. gate_up_proj, down_proj)
    and a forward(x, num_tokens_per_expert) interface.
    """

    def __init__(self, token_permute_backend: str = "default") -> None:
        super().__init__()
        self.permute_backend = create_token_permute_backend(token_permute_backend)
        self._dispatch_state: dict = {}

    @staticmethod
    def _partition_fn(name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        for param_name, param in mod.named_parameters(recurse=False):
            mod.register_parameter(param_name, nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)])))
        log_module_parameters(mod, f"partitioned module={name}", device_mesh)

    def _token_dispatch(self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh) -> tuple[Tensor, Tensor]:
        routed_input, num_tokens_per_expert = inputs
        logger.info(
            "%s token_dispatch module=%s pre_permute routed_input=%s num_tokens_per_expert=%s",
            _dist_prefix(device_mesh),
            type(mod).__name__,
            _tensor_summary(routed_input),
            _tensor_summary(num_tokens_per_expert),
        )
        routed_input, local_tokens_per_expert, state = self.permute_backend.permute(
            routed_input,
            num_tokens_per_expert,
            device_mesh,
        )
        self._dispatch_state = state
        logger.info(
            "%s token_dispatch module=%s post_permute routed_input=%s local_tokens_per_expert=%s state_keys=%s",
            _dist_prefix(device_mesh),
            type(mod).__name__,
            _tensor_summary(routed_input),
            _tensor_summary(local_tokens_per_expert),
            sorted(state.keys()),
        )
        return routed_input, local_tokens_per_expert

    def _token_combine(self, mod: nn.Module, routed_output: Tensor, device_mesh: DeviceMesh) -> Tensor:
        logger.info(
            "%s token_combine module=%s pre_unpermute routed_output=%s",
            _dist_prefix(device_mesh),
            type(mod).__name__,
            _tensor_summary(routed_output),
        )
        combined = self.permute_backend.unpermute(
            routed_output,
            self._dispatch_state,
            device_mesh,
        )
        logger.info(
            "%s token_combine module=%s post_unpermute routed_output=%s",
            _dist_prefix(device_mesh),
            type(mod).__name__,
            _tensor_summary(combined),
        )
        return combined

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )
