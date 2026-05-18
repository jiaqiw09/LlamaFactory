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

"""Interface for distributed backend plugins.

Each backend (fsdp2 / deepspeed) is registered as a single ``register_methods``
interface class.  Method bodies import the heavy implementation modules lazily,
so importing this module does NOT drag in ``fsdp2.py`` / ``deepspeed.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from dataclasses import fields as dc_fields
from typing import TYPE_CHECKING, Any, Literal

from ....utils.plugin import BasePlugin


if TYPE_CHECKING:
    import torch

    from ....config.arg_utils import PluginConfig
    from ....utils.types import HFModel, Processor


# ---------------------------------------------------------------------------
# Typed params
# ---------------------------------------------------------------------------


@dataclass
class FSDP2Params:
    """Typed configuration for the FSDP2 distributed backend."""

    name: Literal["fsdp2"] = "fsdp2"
    reshard_after_forward: bool = True
    offload_params: bool = False
    pin_memory: bool = True
    dcp_path: str | None = None


@dataclass
class DeepSpeedParams:
    """Typed configuration for the DeepSpeed distributed backend."""

    name: Literal["deepspeed"] = "deepspeed"
    config_file: str = ""

    def __post_init__(self) -> None:
        if not self.config_file:
            raise ValueError("DeepSpeedParams.config_file is required.")


@dataclass
class TopologyParams:
    """Cross-cutting distributed topology parameters.

    These fields are consumed by ``DistributedInterface`` when it initializes
    process groups and device meshes. They are not backend-engine parameters.
    """

    timeout: int = 18000
    dp_size: int | None = None
    mp_replicate_size: int = 1
    mp_shard_size: int | None = None


@dataclass
class SequenceParallelParams:
    """Cross-cutting sequence / context parallel parameters.

    Lives alongside the backend params inside one ``dist_config`` block.  Fields
    here are independent of which backend is in use.
    """

    cp_size: int = 1
    cp_mode: Literal["ulysses"] = "ulysses"

    def __post_init__(self) -> None:
        if self.cp_size < 1:
            raise ValueError(f"cp_size must be >= 1, got {self.cp_size}.")


@dataclass
class DistConfig:
    """Parsed distributed configuration.

    Result of :py:meth:`DistributedPlugin.parse_dist_config` — typed
    backend params plus the topology and SP slices.
    """

    backend: Any  # FSDP2Params | DeepSpeedParams | ...
    topology: TopologyParams
    sp: SequenceParallelParams

    @property
    def name(self) -> str:
        return self.backend.name


# ---------------------------------------------------------------------------
# DistributedPlugin family
# ---------------------------------------------------------------------------


class DistributedPlugin(BasePlugin):
    """Plugin family for distributed training backends.

    Use ``DistributedPlugin("<name>").<method>(...)`` to dispatch explicitly.
    """

    # ------------------------------------------------------------------
    # dist_config helper — routes a flat user dict into typed buckets
    # ------------------------------------------------------------------

    @classmethod
    def parse_dist_config(cls, name: str, dist_config: Any) -> DistConfig:
        """Split a flat ``dist_config`` dict into typed buckets.

        Algorithm:
        1. Look up the backend's ParamsClass via ``_params[name]``.
        2. Collect field names of backend, topology, and sequence-parallel params.
        3. Defensive: error if any buckets share field names (collision).
        4. Walk *dist_config*: route each key to backend / topology / sp / unknown.
        5. Reject unknown keys with all buckets listed.
        6. Run ``parse_params`` for the backend; construct topology and SP directly.
        """
        if isinstance(dist_config, DistConfig):
            if dist_config.name != name:
                raise ValueError(f"dist_config name mismatch: expected {name!r}, got {dist_config.name!r}.")
            return dist_config

        if dist_config is None:
            raise ValueError(f"dist_config required for backend {name!r}.")
        if isinstance(dist_config, str):
            dist_config = {"name": dist_config}
        elif not isinstance(dist_config, dict):
            dist_config = dict(dist_config)

        backend_cls = cls._params.get(name)
        if backend_cls is None:
            raise ValueError(
                f"Backend {name!r} has no ParamsClass registered. "
                f"Did the interface call ``register_methods(params=...)``?"
            )

        backend_field_names = {f.name for f in dc_fields(backend_cls) if f.init}
        topology_field_names = {f.name for f in dc_fields(TopologyParams)}
        sp_field_names = {f.name for f in dc_fields(SequenceParallelParams)}

        backend_non_name_fields = backend_field_names - {"name"}
        overlap = (
            (topology_field_names & backend_non_name_fields)
            | (sp_field_names & backend_non_name_fields)
            | (topology_field_names & sp_field_names)
        )
        if overlap:
            raise RuntimeError(
                f"Field-name collision in dist_config params for {backend_cls.__name__}: "
                f"{sorted(overlap)}. Rename one side."
            )

        backend_payload: dict[str, Any] = {}
        topology_payload: dict[str, Any] = {}
        sp_payload: dict[str, Any] = {}
        unknown: list[str] = []
        for key, value in dist_config.items():
            if key in backend_field_names:
                backend_payload[key] = value
            elif key in topology_field_names:
                topology_payload[key] = value
            elif key in sp_field_names:
                sp_payload[key] = value
            else:
                unknown.append(key)

        if unknown:
            raise ValueError(
                f"Unknown keys in dist_config (name={name!r}): {sorted(unknown)}.\n"
                f"  Backend keys for {name!r}: {sorted(backend_field_names - {'name'})}\n"
                f"  Topology keys: {sorted(topology_field_names)}\n"
                f"  Sequence-parallel keys: {sorted(sp_field_names)}"
            )

        backend = cls.parse_params(name, backend_payload)
        topology = TopologyParams(**topology_payload) if topology_payload else TopologyParams()
        sp = SequenceParallelParams(**sp_payload) if sp_payload else SequenceParallelParams()
        return DistConfig(backend=backend, topology=topology, sp=sp)


# ---------------------------------------------------------------------------
# fsdp2 plugin
# ---------------------------------------------------------------------------


@DistributedPlugin("fsdp2").register_methods(params=FSDP2Params)
class FSDP2Plugin:
    """Lightweight wrappers; the heavy fsdp2 module is imported on first call."""

    @staticmethod
    def shard_model(model: HFModel, dist_config: PluginConfig | DistConfig, **kwargs) -> HFModel:
        dist_cfg = DistributedPlugin.parse_dist_config("fsdp2", dist_config)
        from .fsdp2 import FSDP2Engine

        return FSDP2Engine(asdict(dist_cfg.backend), bf16=bool(kwargs.get("bf16"))).shard_model(model)

    @staticmethod
    def save_model(model: HFModel, output_dir: str, processor: Processor) -> None:
        from .fsdp2 import save_model

        return save_model(model, output_dir, processor)

    @staticmethod
    def save_checkpoint(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str, **kwargs) -> None:
        from .fsdp2 import save_checkpoint

        return save_checkpoint(model, optimizer, ckpt_dir, **kwargs)

    @staticmethod
    def load_checkpoint(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str, **kwargs) -> None:
        from .fsdp2 import load_checkpoint

        return load_checkpoint(model, optimizer, ckpt_dir, **kwargs)


# ---------------------------------------------------------------------------
# deepspeed plugin
# ---------------------------------------------------------------------------


@DistributedPlugin("deepspeed").register_methods(params=DeepSpeedParams)
class DeepSpeedPlugin:
    """Lightweight wrappers; the heavy deepspeed module is imported on first call."""

    @staticmethod
    def shard_model(model: HFModel, dist_config: PluginConfig | DistConfig, **kwargs) -> HFModel:
        dist_cfg = DistributedPlugin.parse_dist_config("deepspeed", dist_config)
        from .deepspeed import DeepSpeedEngine

        return DeepSpeedEngine(
            asdict(dist_cfg.backend),
            num_micro_batch=kwargs.get("num_micro_batch"),
            micro_batch_size=kwargs.get("micro_batch_size"),
        ).shard_model(model)

    @staticmethod
    def save_model(model: HFModel, output_dir: str, processor: Processor) -> None:
        from .deepspeed import save_model

        return save_model(model, output_dir, processor)

    @staticmethod
    def save_checkpoint(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str) -> None:
        from .deepspeed import save_checkpoint

        return save_checkpoint(model, optimizer, ckpt_dir)

    @staticmethod
    def load_checkpoint(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str) -> None:
        from .deepspeed import load_checkpoint

        return load_checkpoint(model, optimizer, ckpt_dir)
