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

import re
from typing import TYPE_CHECKING

from ....utils import logging
from ....utils.types import HFModel


if TYPE_CHECKING:
    from .interface import FreezeParams


logger = logging.get_logger(__name__)


def apply_freeze_model(model: HFModel, peft_config: "FreezeParams", is_train: bool = False) -> HFModel:
    import torch

    logger.info_rank0("Fine-tuning method: Freeze")

    if not is_train:
        return model

    freeze_trainable_layers = peft_config.freeze_trainable_layers
    freeze_trainable_modules: list[str] | str = peft_config.freeze_trainable_modules
    freeze_extra_modules: list[str] | str = peft_config.freeze_extra_modules
    cast_trainable_params_to_fp32 = peft_config.cast_trainable_params_to_fp32

    if isinstance(freeze_trainable_modules, str):
        freeze_trainable_modules = [module.strip() for module in freeze_trainable_modules.split(",")]

    if isinstance(freeze_extra_modules, str):
        freeze_extra_modules = [module.strip() for module in freeze_extra_modules.split(",")]

    num_layers = (
        getattr(model.config, "num_hidden_layers", None)
        or getattr(model.config, "num_layers", None)
        or getattr(model.config, "n_layer", None)
    )

    if not num_layers:
        raise ValueError("Current model does not support freeze tuning.")

    if freeze_trainable_layers > 0:
        trainable_layer_ids = range(max(0, num_layers - freeze_trainable_layers), num_layers)
    else:
        trainable_layer_ids = range(min(-freeze_trainable_layers, num_layers))

    hidden_modules = set()
    non_hidden_modules = set()
    for name, _ in model.named_parameters():
        if ".0." in name:
            hidden_modules.add(name.split(".0.")[-1].split(".")[0])
        elif ".1." in name:
            hidden_modules.add(name.split(".1.")[-1].split(".")[0])

        if re.search(r"\.\d+\.", name) is None:
            non_hidden_modules.add(name.split(".")[-2])

    trainable_layers = []
    for module_name in freeze_trainable_modules:
        if module_name == "all":
            for idx in trainable_layer_ids:
                trainable_layers.append(f".{idx:d}.")
        elif module_name in hidden_modules:
            for idx in trainable_layer_ids:
                trainable_layers.append(f".{idx:d}.{module_name}")
        else:
            raise ValueError(f"Module {module_name} not found in hidden modules: {hidden_modules}")

    if freeze_extra_modules:
        for module_name in freeze_extra_modules:
            if module_name in non_hidden_modules:
                trainable_layers.append(module_name)
            else:
                raise ValueError(f"Module {module_name} not found in non-hidden modules: {non_hidden_modules}")

    forbidden_modules = {"quant_state", "quantization_weight", "qweight", "qzeros", "scales"}
    for name, param in model.named_parameters():
        if any(trainable_layer in name for trainable_layer in trainable_layers) and not any(
            forbidden_module in name for forbidden_module in forbidden_modules
        ):
            param.requires_grad_(True)
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)

    logger.info_rank0(f"Set trainable layers: {trainable_layers}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info_rank0(
        f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.4f}"
    )

    return model
