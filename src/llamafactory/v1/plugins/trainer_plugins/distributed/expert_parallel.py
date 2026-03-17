from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed._functional_collectives import all_to_all_single, all_to_all_single_autograd
from torch.distributed.tensor import DeviceMesh, Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle


class TokenPermuteBackend:
    def permute(self, routed_input: Tensor, num_tokens_per_expert: Tensor, device_mesh: DeviceMesh) -> tuple[Tensor, Tensor, dict]:
        raise NotImplementedError

    def unpermute(self, routed_output: Tensor, state: dict, device_mesh: DeviceMesh) -> Tensor:
        raise NotImplementedError


class DefaultTokenPermuteBackend(TokenPermuteBackend):
    def permute(self, routed_input: Tensor, num_tokens_per_expert: Tensor, device_mesh: DeviceMesh) -> tuple[Tensor, Tensor, dict]:
        ep_degree = device_mesh.size()
        if ep_degree <= 1:
            state = {
                "input_splits": None,
                "output_splits": None,
                "combine_permutation": None,
            }
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

        routed_input = all_to_all_single_autograd(
            routed_input,
            output_splits_list,
            input_splits_list,
            device_mesh.get_group(),
        )

        tokens_matrix = num_tokens_per_expert_group.view(ep_degree, num_local_experts)
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

        state = {
            "input_splits": input_splits_list,
            "output_splits": output_splits_list,
            "combine_permutation": combine_permutation,
        }
        local_tokens_per_expert = tokens_matrix.sum(dim=0).to(num_tokens_per_expert.dtype)
        return routed_input, local_tokens_per_expert, state

    def unpermute(self, routed_output: Tensor, state: dict, device_mesh: DeviceMesh) -> Tensor:
        combine_permutation = state.get("combine_permutation")
        if combine_permutation is not None:
            routed_output = routed_output.index_select(0, combine_permutation)

        input_splits = state.get("input_splits")
        output_splits = state.get("output_splits")
        if input_splits is None or output_splits is None:
            return routed_output

        return all_to_all_single_autograd(
            routed_output,
            input_splits,
            output_splits,
            device_mesh.get_group(),
        )


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

    def _token_dispatch(self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh) -> tuple[Tensor, Tensor]:
        routed_input, num_tokens_per_expert = inputs
        routed_input, local_tokens_per_expert, state = self.permute_backend.permute(
            routed_input,
            num_tokens_per_expert,
            device_mesh,
        )
        self._dispatch_state = state
        return routed_input, local_tokens_per_expert

    def _token_combine(self, mod: nn.Module, routed_output: Tensor, device_mesh: DeviceMesh) -> Tensor:
        return self.permute_backend.unpermute(
            routed_output,
            self._dispatch_state,
            device_mesh,
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )
