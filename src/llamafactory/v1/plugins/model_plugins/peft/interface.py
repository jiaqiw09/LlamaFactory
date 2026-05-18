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

"""Interface for PEFT plugins."""

from dataclasses import dataclass, field
from typing import Literal

from ....utils.plugin import BasePlugin
from ....utils.types import HFModel


class PeftPlugin(BasePlugin):
    def __call__(self, model: HFModel, peft_config: dict, is_train: bool) -> HFModel:
        return super().__call__(model, peft_config=peft_config, is_train=is_train)


@dataclass
class LoraParams:
    """Typed configuration for the ``lora`` PEFT plugin."""

    name: Literal["lora"] = "lora"
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] | str = "all"
    use_rslora: bool = False
    use_dora: bool = False
    modules_to_save: list[str] | None = None
    adapter_name_or_path: list[str] | str | None = None
    export_dir: str | None = None
    export_size: int = 5
    export_hub_model_id: str | None = None
    infer_dtype: Literal["auto", "float16", "float32", "bfloat16"] = "auto"
    export_legacy_format: bool = False


@PeftPlugin("lora").register(params=LoraParams, parse_arg="peft_config")
def get_lora_model(model: HFModel, peft_config: LoraParams, is_train: bool = False) -> HFModel:
    from .lora import apply_lora_model

    return apply_lora_model(model, peft_config, is_train)


@dataclass
class FreezeParams:
    """Typed configuration for the ``freeze`` PEFT plugin."""

    name: Literal["freeze"] = "freeze"
    freeze_trainable_layers: int = 2
    freeze_trainable_modules: list[str] | str = "all"
    freeze_extra_modules: list[str] | str = field(default_factory=list)
    cast_trainable_params_to_fp32: bool = True


@PeftPlugin("freeze").register(params=FreezeParams, parse_arg="peft_config")
def get_freeze_model(model: HFModel, peft_config: FreezeParams, is_train: bool = False) -> HFModel:
    from .freeze import apply_freeze_model

    return apply_freeze_model(model, peft_config, is_train)
