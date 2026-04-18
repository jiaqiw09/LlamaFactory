# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path
from typing import Any, Callable

import torch
from torch.distributed.device_mesh import DeviceMesh

from ....accelerator.helper import get_current_accelerator
from ....accelerator.interface import Dim
from ....utils.logging import get_logger
from ....utils.types import HFModel
from .fsdp2 import FSDP2Engine


logger = get_logger(__name__)


def _grad_to_local_fp32(grad: torch.Tensor) -> torch.Tensor:
    from torch.distributed._tensor import DTensor

    local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
    return local_grad.detach().to(torch.float32)


def _local_pth_sum(parameters: list[torch.nn.Parameter], norm_type: float, device: torch.device) -> torch.Tensor:
    total = torch.zeros((), device=device, dtype=torch.float32)
    for param in parameters:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        total = total + torch.norm(_grad_to_local_fp32(grad), p=norm_type).pow(norm_type)
    return total


def _allreduce_sum_(value: torch.Tensor, groups: list[object]) -> torch.Tensor:
    import torch.distributed as dist

    for group in groups:
        if group is not None:
            dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
    return value


def clip_grad_norm_(model: HFModel, max_norm: float, **kwargs) -> float:
    """
    CP-aware grad norm clipping for MindSpeed EP + EFSDP + outer FSDP2.

    Avoids torch.nn.utils.get_total_norm() since mixed DTensor meshes
    (`dp` vs `efsdp`/`ep`) may hit DTensor stack propagation failures.
    """
    from torch.distributed._tensor import DTensor

    from ....accelerator.interface import Dim, DistributedInterface

    norm_type = float(kwargs.get("norm_type", 2.0))
    dist_interface = DistributedInterface()
    device = dist_interface.current_device
    dp_group = dist_interface.get_group(Dim.DP)
    cp_group = dist_interface.get_group(Dim.CP) if dist_interface.strategy.cp_size > 1 else None
    ep_group = dist_interface.get_group(Dim.EP) if dist_interface.strategy.ep_size > 1 else None
    efsdp_group = dist_interface.get_group(Dim.EFSDP) if dist_interface.strategy.ep_size > 1 else None

    ep_params: list[torch.nn.Parameter] = []
    non_ep_params: list[torch.nn.Parameter] = []
    for param in model.parameters():
        grad = getattr(param, "grad", None)
        if grad is None:
            continue

        mesh_names = set(getattr(getattr(grad, "device_mesh", None), "mesh_dim_names", ()) or ())
        is_ep_side = isinstance(grad, DTensor) and bool(mesh_names & {Dim.EP.value, Dim.EFSDP.value})
        if is_ep_side:
            ep_params.append(param)
        else:
            non_ep_params.append(param)

    if not ep_params and not non_ep_params:
        return 0.0

    total_pth = torch.zeros((), device=device, dtype=torch.float32)
    if non_ep_params:
        non_ep_pth = _local_pth_sum(non_ep_params, norm_type, device)
        total_pth = total_pth + _allreduce_sum_(non_ep_pth, [dp_group, cp_group])
    if ep_params:
        ep_pth = _local_pth_sum(ep_params, norm_type, device)
        total_pth = total_pth + _allreduce_sum_(ep_pth, [efsdp_group, ep_group, cp_group])

    total_norm = total_pth.pow(1.0 / norm_type)
    clip_coef = min(max_norm / (float(total_norm.item()) + 1e-6), 1.0)
    if clip_coef < 1.0:
        for param in ep_params + non_ep_params:
            grad = getattr(param, "grad", None)
            if grad is not None:
                grad.detach().mul_(clip_coef)

    return float(total_norm.item())


def _import_mindspeed_master() -> tuple[Any, Any, Any]:
    """Import the MindSpeed EP backend only.

    CP/FSDP topology is owned by LlamaFactory's DistributedInterface.
    This adapter only borrows MindSpeed's EP/EFSDP implementation.
    """

    try:
        from mindspeed.fsdp.distributed.expert_parallel.expert_fully_shard_parallel import expert_fully_shard_modules
        from mindspeed.fsdp.distributed.expert_parallel.expert_parallel import expert_parallelize_modules
        from mindspeed.fsdp.parallel_engine_config import EPPlanConfig
    except ImportError:
        workspace_root = next((p for p in Path(__file__).resolve().parents if (p / "MindSpeed" / "mindspeed").exists()), None)
        if workspace_root is None:
            raise

        mindspeed_root = workspace_root / "MindSpeed"
        if str(mindspeed_root) not in sys.path:
            sys.path.insert(0, str(mindspeed_root))

        from mindspeed.fsdp.distributed.expert_parallel.expert_fully_shard_parallel import expert_fully_shard_modules
        from mindspeed.fsdp.distributed.expert_parallel.expert_parallel import expert_parallelize_modules
        from mindspeed.fsdp.parallel_engine_config import EPPlanConfig

    return expert_parallelize_modules, expert_fully_shard_modules, EPPlanConfig


def _get_model_type(model: HFModel) -> str | None:
    return getattr(getattr(model, "config", None), "model_type", None)


class MindSpeedEPModelSpec:
    _registry: dict[str, "MindSpeedEPModelSpec"] = {}

    def __init__(
        self,
        ep_modules: list[str],
        ep_fsdp_modules: list[str] | None = None,
        prepare_fn: Callable[[HFModel], HFModel] | None = None,
    ) -> None:
        self.ep_modules = ep_modules
        self.ep_fsdp_modules = ep_fsdp_modules
        self.prepare_fn = prepare_fn

    @classmethod
    def register(
        cls,
        model_type: str,
        ep_modules: list[str],
        ep_fsdp_modules: list[str] | None = None,
    ):
        def decorator(fn):
            cls._registry[model_type] = cls(
                ep_modules=ep_modules,
                ep_fsdp_modules=ep_fsdp_modules,
                prepare_fn=fn,
            )
            return fn

        return decorator

    @classmethod
    def get(cls, model: HFModel) -> "MindSpeedEPModelSpec | None":
        model_type = _get_model_type(model)
        if model_type is None:
            return None
        return cls._registry.get(model_type)

    def prepare(self, model: HFModel) -> HFModel:
        if self.prepare_fn is None:
            return model
        return self.prepare_fn(model)


@MindSpeedEPModelSpec.register(
    "qwen3_moe",
    ep_modules=["model.layers.{*}.mlp.experts"],
    ep_fsdp_modules=["model.layers.{*}.mlp"],
)
def _prepare_qwen3_moe_for_ep(model: HFModel) -> HFModel:
    prepared = 0
    for module in model.modules():
        if not all(hasattr(module, attr) for attr in ("gate_up_proj", "down_proj", "hidden_dim", "num_experts")):
            continue

        # MindSpeed eager EP dispatcher expects sparse expert blocks to expose `hidden_size`.
        if not hasattr(module, "hidden_size"):
            module.hidden_size = module.hidden_dim
        prepared += 1

    if prepared:
        logger.info_rank0(
            f"MindSpeed EP adapter: prepared {prepared} qwen3_moe experts modules for transformers 5.x structure."
        )
    else:
        logger.info_rank0("MindSpeed EP adapter: qwen3_moe uses transformers 5.x native sparse MoE structure.")
    return model


_REGISTERED_EP_PREPARE_FNS = (_prepare_qwen3_moe_for_ep,)


class MindSpeedFSDP2Engine(FSDP2Engine):
    """MindSpeed EP adapter that reuses LlamaFactory's init/load flow.

    Design:
    - MindSpeed owns EP / EFSDP only.
    - LlamaFactory owns FSDP / CP / init-load lifecycle.
    """

    def __init__(self, dist_config: dict):
        self.dist_config = dist_config
        super().__init__(dist_config)
        self.ep_size = self.dist_config.get("ep_size", 1)
        self.ep_fsdp_size = self._get_ep_fsdp_size(self.ep_size)

    def _get_ep_fsdp_size(self, ep_size: int) -> int:
        if ep_size <= 1:
            return 1
        return self.dist_interface.get_world_size(Dim.DP) // ep_size

    def _get_ep_fsdp_modules(self, model: HFModel, ep_modules: list[str]) -> list[str]:
        modules = self.dist_config.get("ep_fsdp_modules")
        if modules is not None:
            return modules

        spec = MindSpeedEPModelSpec.get(model)
        if spec is not None and spec.ep_fsdp_modules is not None:
            return spec.ep_fsdp_modules

        ep_fsdp_modules = []
        for module in ep_modules:
            if module.endswith(".experts"):
                ep_fsdp_modules.append(module.removesuffix(".experts"))
            else:
                ep_fsdp_modules.append(module)
        return ep_fsdp_modules

    def _get_external_ep_meshes(self) -> tuple[DeviceMesh, DeviceMesh | None]:
        """Reuse the cached expert-side meshes owned by DistributedInterface."""
        ep_mesh, efsdp_mesh = self.dist_interface.get_expert_meshes()
        if ep_mesh is None:
            raise RuntimeError("Expert EP mesh is not initialized in DistributedInterface.")
        if self.ep_fsdp_size > 1 and efsdp_mesh is None:
            raise RuntimeError("Expert EFSDP mesh is not initialized in DistributedInterface.")
        return ep_mesh, efsdp_mesh

    def _copy_weights(self, param, loaded_tensor):
        from torch.distributed._tensor import DTensor, Shard

        if loaded_tensor.dtype != param.dtype:
            loaded_tensor = loaded_tensor.to(param.dtype)

        if isinstance(param, DTensor):
            local_tensor = param.to_local()
            shard_placements = [(i, placement) for i, placement in enumerate(param.placements) if isinstance(placement, Shard)]

            if not shard_placements:
                local_tensor.copy_(loaded_tensor)
                return

            mesh = param.device_mesh
            my_coordinate = mesh.get_coordinate()
            if my_coordinate is None:
                return

            sliced_tensor = loaded_tensor
            for mesh_dim, shard_placement in shard_placements:
                dim = shard_placement.dim
                rank_in_dim = my_coordinate[mesh_dim]
                world_size_in_dim = mesh.size(mesh_dim)
                full_size = sliced_tensor.shape[dim]
                chunk_size = (full_size + world_size_in_dim - 1) // world_size_in_dim
                start = rank_in_dim * chunk_size
                end = min(start + chunk_size, full_size)

                if start >= full_size:
                    return

                sliced_tensor = sliced_tensor.narrow(dim, start, end - start)

            slices = [slice(None)] * local_tensor.ndim
            for _, shard_placement in shard_placements:
                dim = shard_placement.dim
                slices[dim] = slice(0, sliced_tensor.shape[dim])
            local_tensor[tuple(slices)].copy_(sliced_tensor)
            return

        param.data.copy_(loaded_tensor)

    def prepare_model_ep(self, model: HFModel) -> tuple[HFModel, set]:
        """Apply MindSpeed EP/EFSDP and return the model along with parameters to ignore in FSDP."""
        (
            expert_parallelize_modules,
            expert_fully_shard_modules,
            EPPlanConfig,
        ) = _import_mindspeed_master()

        ep_modules = self.dist_config.get("ep_modules")
        if ep_modules is None:
            spec = MindSpeedEPModelSpec.get(model)
            ep_modules = spec.ep_modules if spec is not None else None
        if ep_modules is None:
            raise ValueError(
                f"`ep_modules` is not specified and no built-in MindSpeed EP spec is registered for model_type={_get_model_type(model)}."
            )

        spec = MindSpeedEPModelSpec.get(model)
        if spec is not None:
            model = spec.prepare(model)

        if self.ep_size > 1:
            ep_plan = EPPlanConfig(
                apply_modules=ep_modules,
                dispatcher=self.dist_config.get("ep_dispatcher", "eager"),
                apply_efsdp_modules=self._get_ep_fsdp_modules(model, ep_modules),
            )
            ep_plan._gradient_divide_factor = float(
                self.ep_size * self.dist_interface.get_world_size(Dim.EFSDP)
            )
            ep_mesh, efsdp_mesh = self._get_external_ep_meshes()
            if self.rank == 0:
                logger.info("Applying MindSpeed master EP backend.")
                logger.info(f"MindSpeed EP apply patterns: {ep_modules}")
                logger.info(f"MindSpeed EP device mesh: {ep_mesh}")
                logger.info(f"MindSpeed EP gradient divide factor: {ep_plan._gradient_divide_factor}")

            model = expert_parallelize_modules(model, ep_mesh, ep_plan)

            if self.ep_fsdp_size > 1:
                if self.rank == 0:
                    logger.info(f"MindSpeed EFSDP apply patterns: {ep_plan.apply_efsdp_modules}")
                    logger.info(f"MindSpeed EFSDP device mesh: {efsdp_mesh}")
                model = expert_fully_shard_modules(model, efsdp_mesh, ep_plan)

        # Collect ignored params for the outer FSDP wrap
        fsdp_ignored_modules = list(self.dist_config.get("fsdp_ignored_modules", []))
        if self.ep_size > 1:
            fsdp_ignored_modules.extend(ep_modules)

        ignored_params = set()
        if fsdp_ignored_modules:
            try:
                from mindspeed.fsdp.utils.str_match import module_name_match
            except ImportError:
                _import_mindspeed_master()
                from mindspeed.fsdp.utils.str_match import module_name_match

            for name, module in model.named_modules():
                for pattern in fsdp_ignored_modules:
                    if module_name_match(pattern, name):
                        ignored_params.update(list(module.parameters(recurse=True)))

            if ignored_params and self.rank == 0:
                logger.info(f"MindSpeed FSDP2: Ignoring {len(ignored_params)} parameters from outer FSDP wrapping.")

        return model, ignored_params

    def prepare_model(self, model: HFModel) -> HFModel:
        # 1. Apply MindSpeed EP and collect params that need to be ignored by outer FSDP
        model, ignored_params = self.prepare_model_ep(model)

        # 2. Call the base FSDP2Engine prepare_model, passing the ignored_params
        model = super().prepare_model(model, ignored_params=ignored_params)
        
        return model
