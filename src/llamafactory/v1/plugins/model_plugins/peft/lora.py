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

from typing import TYPE_CHECKING

from ....utils import logging
from ....utils.types import HFModel


if TYPE_CHECKING:
    from .interface import LoraParams


logger = logging.get_logger(__name__)


def _find_all_linear_modules(model: HFModel) -> list[str]:
    r"""Find all available modules to apply LoRA."""
    forbidden_modules = {"lm_head", "output_layer", "output"}
    module_names = set()
    for name, module in model.named_modules():
        if any(forbidden_module in name for forbidden_module in forbidden_modules):
            continue

        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            module_names.add(name.split(".")[-1])

    return list(module_names)


def merge_adapters(model: HFModel, adapter_name_or_path: list[str] | str) -> HFModel:
    from peft import PeftModel

    if not isinstance(adapter_name_or_path, list):
        adapter_name_or_path = [adapter_name_or_path]

    for adapter_path in adapter_name_or_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        logger.info_rank0(f"Merged adapter from {adapter_path}")

    return model


def load_adapter(model: HFModel, adapter_name_or_path: list[str] | str, is_train: bool) -> HFModel:
    r"""Loads adapter(s) into the model."""
    from peft import PeftModel

    if not isinstance(adapter_name_or_path, list):
        adapter_name_or_path = [adapter_name_or_path]

    if is_train and len(adapter_name_or_path) > 1:
        raise ValueError(
            "When `adapter_name_or_path` is provided for training, only a single LoRA adapter is supported. "
            "Training will continue on the specified adapter. "
            "Please merge multiple adapters before starting a new LoRA adapter."
        )

    if is_train:
        adapter_to_merge = []
        adapter_to_resume = adapter_name_or_path[0]
    else:
        adapter_to_merge = adapter_name_or_path
        adapter_to_resume = None

    if adapter_to_merge:
        model = merge_adapters(model, adapter_to_merge)

    if adapter_to_resume is not None:
        model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_train)
        if is_train:
            logger.info_rank0(
                f"Resuming training from existing LoRA adapter at {adapter_to_resume}. "
                "LoRA hyperparameters will be loaded from the adapter itself; "
                "the current LoRA configuration will be ignored. "
                "Merge the adapter into the base model before training if you want to start a new adapter."
            )

    return model


def apply_lora_model(model: HFModel, peft_config: "LoraParams", is_train: bool = False) -> HFModel:
    from peft import LoraConfig, TaskType, get_peft_model

    if peft_config.adapter_name_or_path:
        return load_adapter(model, peft_config.adapter_name_or_path, is_train)

    logger.info_rank0("Fine-tuning method: LoRA")

    target_modules = peft_config.target_modules
    if target_modules == "all":
        target_modules = _find_all_linear_modules(model)
    elif isinstance(target_modules, str):
        target_modules = [target_modules]

    logger.info_rank0(f"LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=not is_train,
        r=peft_config.r,
        lora_alpha=peft_config.lora_alpha,
        lora_dropout=peft_config.lora_dropout,
        use_rslora=peft_config.use_rslora,
        use_dora=peft_config.use_dora,
        target_modules=target_modules,
        modules_to_save=peft_config.modules_to_save,
    )

    model = get_peft_model(model, lora_config)

    if is_train:
        model.print_trainable_parameters()

    return model
