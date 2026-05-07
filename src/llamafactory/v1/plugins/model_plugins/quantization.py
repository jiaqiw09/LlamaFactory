# Copyright 2025 HuggingFace Inc., the KVCache.AI team, Approaching AI, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from transformers import BitsAndBytesConfig

from ...accelerator.helper import get_current_device
from ...config.arg_utils import StrictConfigMixin
from ...config.model_args import ModelArguments
from ...utils import logging
from ...utils.packages import check_version
from ...utils.plugin import BasePlugin


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer

logger = logging.get_logger(__name__)


@dataclass
class QuantConfig(StrictConfigMixin):
    """Quantization config. ``auto`` and ``bnb`` share the same fields today;
    they remain a single dataclass until a variant grows divergent fields."""

    name: Literal["auto", "bnb"] = "auto"
    quantization_bit: int | None = None
    compute_dtype: str | torch.dtype | None = None
    double_quantization: bool = True
    quantization_type: Literal["nf4", "fp4"] = "nf4"


def _resolve_compute_dtype(compute_dtype: str | torch.dtype | None) -> torch.dtype:
    if compute_dtype is None:
        return torch.float16
    elif isinstance(compute_dtype, str):
        if not hasattr(torch, compute_dtype):
            raise ValueError(f"Unknown torch dtype: {compute_dtype}.")

        return getattr(torch, compute_dtype)
    else:
        return compute_dtype


class QuantizationPlugin(BasePlugin):
    r"""Plugin for model quantization."""

    def __call__(
        self,
        init_kwargs: dict[str, Any] = None,
        config: "PretrainedConfig" = None,
        tokenizer: "PreTrainedTokenizer" = None,
        model_args: "ModelArguments" = None,
        is_trainable: bool = False,
    ) -> dict[str, Any]:
        return super().__call__(
            init_kwargs, config=config, tokenizer=tokenizer, model_args=model_args, is_trainable=is_trainable
        )


@QuantizationPlugin("auto", config=QuantConfig).register()
def quantization_auto(
    init_kwargs: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    """Automatic quantization selection. Currently dispatches to bnb."""
    model_args: ModelArguments = kwargs.get("model_args", None)
    quant_config: QuantConfig = model_args.quant_config

    if quant_config.quantization_bit is not None:
        logger.info_rank0(f"Loading {quant_config.quantization_bit}-bit quantized model.")
        if quant_config.quantization_bit in (8, 4):
            return quantization_with_bnb(init_kwargs, **kwargs)
        else:
            raise ValueError(
                f"Unsupported quantization bit: {quant_config.quantization_bit} for auto quantization."
            )
    logger.warning_rank0("No quantization method applied.")
    return init_kwargs


@QuantizationPlugin("bnb", config=QuantConfig).register()
def quantization_with_bnb(
    init_kwargs: dict[str, Any],
    model_args: "ModelArguments" = None,
    **kwargs,
) -> dict[str, Any]:
    r"""Quantization with BNB."""
    logger.info_rank0("Using Bitsandbytes quantization.")
    quant_config: QuantConfig = model_args.quant_config
    quantization_bit = quant_config.quantization_bit
    if quantization_bit is None:
        logger.warning_rank0("quantization_bit is not specified, default to 4-bit quantization.")
        quantization_bit = 4
    if quantization_bit not in (8, 4):
        raise ValueError("Bitsandbytes only accepts 4-bit or 8-bit quantization.")

    if quantization_bit == 8:
        check_version("bitsandbytes>=0.37.0", mandatory=True)
        init_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:  # 4-bit
        check_version("bitsandbytes>=0.39.0", mandatory=True)
        compute_dtype = _resolve_compute_dtype(quant_config.compute_dtype)
        init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_config.double_quantization,
            bnb_4bit_quant_type=quant_config.quantization_type,
            bnb_4bit_quant_storage=compute_dtype,  # crucial for fsdp+qlora
        )

    # TODO: improve deepspeed zero3 and fsdp detection.
    if kwargs.get("is_trainable", False):
        logger.info_rank0("Detected inference mode, setting device_map for bitsandbytes quantization.")
        init_kwargs["device_map"] = {"": get_current_device()}  # change auto device map for inference
    else:
        logger.info_rank0("Detected training mode, skip setting device_map for bitsandbytes quantization.")
        if quantization_bit != 4:
            raise ValueError("Only 4-bit quantized model can use fsdp+qlora or auto device map.")

        check_version("bitsandbytes>=0.43.0", mandatory=True)

    logger.info_rank0(f"Quantizing model to {quantization_bit} bit with bitsandbytes.")
    return init_kwargs
