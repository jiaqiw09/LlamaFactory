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

import torch

from ....config import InputArgument, get_args
from ....core.model_engine import ModelEngine
from ....utils import logging
from .interface import LoraParams, PeftPlugin


logger = logging.get_logger(__name__)


def merge_and_export_model(args: InputArgument = None):
    model_args, _, _, _ = get_args(args)

    raw_config = model_args.peft_config
    if raw_config is None:
        raise ValueError("Please specify peft_config to merge and export model.")
    if raw_config.get("name") != "lora":
        raise ValueError("Currently merge and export model function is only supported for lora.")

    peft_config: LoraParams = PeftPlugin.parse_params("lora", raw_config)

    if peft_config.export_dir is None:
        raise ValueError("Please specify export_dir.")
    if peft_config.adapter_name_or_path is None:
        raise ValueError("Please set adapter_name_or_path to merge adapters into base model.")

    logger.info_rank0("Loading model for export...")
    model_engine = ModelEngine(model_args, is_train=False)
    model = model_engine.model
    tokenizer = model_engine.processor

    if peft_config.infer_dtype == "auto":
        if model.config.torch_dtype == torch.float32 and torch.cuda.is_bf16_supported():
            model = model.to(torch.bfloat16)
            logger.info_rank0("Converted model to bfloat16.")
    else:
        target_dtype = getattr(torch, peft_config.infer_dtype)
        model = model.to(target_dtype)
        logger.info_rank0(f"Converted model to {peft_config.infer_dtype}.")

    logger.info_rank0(f"Exporting model to {peft_config.export_dir}...")
    model.save_pretrained(
        peft_config.export_dir,
        max_shard_size=f"{peft_config.export_size}GB",
        safe_serialization=not peft_config.export_legacy_format,
    )
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, "padding_side"):
                tokenizer.padding_side = "left"
            tokenizer.save_pretrained(peft_config.export_dir)
        except Exception as e:
            logger.warning(f"Failed to save tokenizer: {e}")

    if peft_config.export_hub_model_id:
        logger.info_rank0(f"Pushing to hub: {peft_config.export_hub_model_id}...")
        model.push_to_hub(peft_config.export_hub_model_id)
        if tokenizer is not None:
            tokenizer.push_to_hub(peft_config.export_hub_model_id)

    logger.info_rank0("Model exported successfully.")
