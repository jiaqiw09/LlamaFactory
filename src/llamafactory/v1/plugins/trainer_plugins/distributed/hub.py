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

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Literal

import torch

from ....config.arg_utils import StrictConfigMixin, normalize_plugin_argument, strict_dataclass_from_dict
from ....utils.plugin import BasePlugin


if TYPE_CHECKING:
    from ....utils.types import HFModel, Processor


@dataclass
class FSDP2Config(StrictConfigMixin):
    """Config for FSDP2 sharding backend.

    Self-contained — the fields here are exactly what FSDP2 reads. ``timeout`` is
    repeated across backend variants rather than inherited from a shared base,
    because the set of "universal" fields is small and a shared base would lie
    about applicability for backends that don't honor every field.
    """

    name: Literal["fsdp2"] = "fsdp2"
    timeout: int = 18000
    dp_size: int | None = None
    mp_replicate_size: int = 1
    mp_shard_size: int | None = None
    reshard_after_forward: bool = True
    offload_params: bool = False
    pin_memory: bool = True
    dcp_path: str | None = None


@dataclass
class DeepSpeedConfig(StrictConfigMixin):
    """Config for DeepSpeed backend.

    Note the absence of ``dp_size`` / ``mp_*`` / ``offload_params`` — DeepSpeed
    derives those from ``world_size`` and the contents of ``config_file``.
    Surfacing them here would mislead users into thinking they take effect.
    """

    name: Literal["deepspeed"] = "deepspeed"
    config_file: str = field(default="", metadata={"required": True})
    timeout: int = 18000

    def __post_init__(self) -> None:
        if not self.config_file:
            raise ValueError("DeepSpeedConfig.config_file is required.")


@dataclass
class DistConfig:
    """Façade for the user-facing single ``dist_config`` slot.

    Internally the flat YAML input is routed into two sub-objects:

    * ``backend`` — one of the registered backend variants (``FSDP2Config``,
      ``DeepSpeedConfig``, ...). Holds backend-specific settings.
    * ``sp`` — ``SequenceParallelConfig``. Holds cross-cutting parallelism
      settings (``cp_size``, ``cp_mode``) that are independent of the chosen
      backend.

    Users keep writing one flat ``dist_config:`` block in YAML; the routing is
    invisible. Consumers access ``args.dist_config.backend.X`` and
    ``args.dist_config.sp.X`` explicitly.
    """

    backend: Any  # FSDP2Config | DeepSpeedConfig | ... (variant-typed)
    sp: Any  # SequenceParallelConfig (always populated, even when cp_size==1)

    @property
    def name(self) -> str:
        """Backwards-compat: ``args.dist_config.name`` returns the backend name."""
        return self.backend.name


class DistributedPlugin(BasePlugin):
    def __call__(self, model: HFModel, dist_config: StrictConfigMixin, **kwargs) -> HFModel:
        return super().__call__(model, dist_config, **kwargs)

    @classmethod
    def parse_config(cls, raw: Any) -> DistConfig | None:
        """Parse a flat user-facing ``dist_config`` dict into a :class:`DistConfig`.

        Routes each input key into either the backend variant dataclass or the
        sequence-parallel dataclass based on which schema declares the field.
        Unknown keys are rejected with an actionable error message that lists
        both buckets.
        """
        # Local import to avoid a circular import at module load time:
        # SequenceParallelConfig lives under model_plugins, which transitively
        # imports back into config/utils.
        from ...model_plugins.parallelization.sequence_parallel import SequenceParallelConfig

        normalized = normalize_plugin_argument(raw)
        if normalized is None:
            return None

        name = normalized.get("name")
        if name is None:
            raise ValueError("dist_config must have a 'name' field.")

        if name not in cls._configs:
            raise ValueError(
                f"Unknown DistributedPlugin variant: {name!r}. Available variants: {cls.list_variants()}."
            )

        backend_cls, aliases = cls._configs[name]

        sp_field_names = {f.name for f in fields(SequenceParallelConfig)}
        backend_field_names = {f.name for f in fields(backend_cls) if f.init}

        # Defensive: a future field-name collision would silently mis-route inputs.
        overlap = sp_field_names & (backend_field_names - {"name"})
        if overlap:
            raise RuntimeError(
                f"Field-name collision between SequenceParallelConfig and "
                f"{backend_cls.__name__}: {sorted(overlap)}. Rename one side."
            )

        backend_payload: dict[str, Any] = {}
        sp_payload: dict[str, Any] = {}
        unknown: list[str] = []
        for key, value in normalized.items():
            if key in backend_field_names:
                backend_payload[key] = value
            elif key in sp_field_names:
                sp_payload[key] = value
            else:
                unknown.append(key)

        if unknown:
            raise ValueError(
                f"Unknown keys in dist_config (name={name!r}): {sorted(unknown)}.\n"
                f"  Backend keys for {name!r}: {sorted(backend_field_names - {'name'})}\n"
                f"  Sequence-parallel keys: {sorted(sp_field_names)}"
            )

        backend = strict_dataclass_from_dict(
            backend_cls,
            backend_payload,
            f"DistributedPlugin.{name}",
            aliases=aliases,
        )
        sp = strict_dataclass_from_dict(SequenceParallelConfig, sp_payload, "dist_config.sp")
        return DistConfig(backend=backend, sp=sp)


@DistributedPlugin("fsdp2", config=FSDP2Config).register()
def shard_model_fsdp2(model: HFModel, backend_config: FSDP2Config, **kwargs) -> HFModel:
    from .fsdp2 import FSDP2Engine

    return FSDP2Engine(backend_config, bf16=bool(kwargs.get("bf16"))).shard_model(model)


@DistributedPlugin("fsdp2").register("save_model")
def save_model_fsdp2(model: HFModel, output_dir: str, processor: Processor) -> None:
    from .fsdp2 import save_model

    return save_model(model, output_dir, processor)


@DistributedPlugin("fsdp2").register("save_checkpoint")
def save_checkpoint_fsdp2(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str, **kwargs) -> None:
    from .fsdp2 import save_checkpoint

    return save_checkpoint(model, optimizer, ckpt_dir, **kwargs)


@DistributedPlugin("fsdp2").register("load_checkpoint")
def load_checkpoint_fsdp2(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str, **kwargs) -> None:
    from .fsdp2 import load_checkpoint

    return load_checkpoint(model, optimizer, ckpt_dir, **kwargs)


@DistributedPlugin("deepspeed", config=DeepSpeedConfig).register()
def shard_model_deepspeed(model: HFModel, backend_config: DeepSpeedConfig, **kwargs) -> HFModel:
    from .deepspeed import DeepSpeedEngine

    return DeepSpeedEngine(
        backend_config,
        num_micro_batch=kwargs.get("num_micro_batch"),
        micro_batch_size=kwargs.get("micro_batch_size"),
    ).shard_model(model)


@DistributedPlugin("deepspeed").register("save_model")
def save_model_deepspeed(model: HFModel, output_dir: str, processor: Processor) -> None:
    from .deepspeed import save_model

    return save_model(model, output_dir, processor)


@DistributedPlugin("deepspeed").register("save_checkpoint")
def save_checkpoint_deepspeed(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str) -> None:
    from .deepspeed import save_checkpoint

    return save_checkpoint(model, optimizer, ckpt_dir)


@DistributedPlugin("deepspeed").register("load_checkpoint")
def load_checkpoint_deepspeed(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str) -> None:
    from .deepspeed import load_checkpoint

    return load_checkpoint(model, optimizer, ckpt_dir)
