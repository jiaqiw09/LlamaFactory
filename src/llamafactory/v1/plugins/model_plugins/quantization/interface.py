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

"""Interface for quantization plugins."""

from dataclasses import dataclass
from typing import Any, Literal

from ....utils import logging
from ....utils.plugin import BasePlugin


logger = logging.get_logger(__name__)


class QuantizationPlugin(BasePlugin):
    r"""Plugin for model quantization."""

    def __call__(
        self,
        init_kwargs: dict[str, Any] = None,
        quant_config=None,
        is_trainable: bool = False,
    ) -> dict[str, Any]:
        return super().__call__(
            init_kwargs,
            quant_config=quant_config,
            is_trainable=is_trainable,
        )


@dataclass
class BnbParams:
    """Typed configuration for the BitsAndBytes quantization plugin."""

    name: Literal["bnb", "auto"] = "bnb"
    quantization_bit: int | None = None
    compute_dtype: str | Any = "float16"
    double_quantization: bool = True
    quantization_type: str = "nf4"

    def __post_init__(self) -> None:
        import torch

        if isinstance(self.compute_dtype, str):
            resolved = getattr(torch, self.compute_dtype, None)
            if not isinstance(resolved, torch.dtype):
                raise ValueError(f"compute_dtype={self.compute_dtype!r} is not a torch dtype name.")
            self.compute_dtype = resolved
        elif not isinstance(self.compute_dtype, torch.dtype):
            raise TypeError(f"compute_dtype must be str or torch.dtype; got {type(self.compute_dtype).__name__}.")


@QuantizationPlugin("auto").register(params=BnbParams, parse_arg="quant_config")
def quantization_auto(
    init_kwargs: dict[str, Any],
    quant_config: BnbParams,
    is_trainable: bool = False,
) -> dict[str, Any]:
    """Dispatcher: pick a concrete quantization backend from ``quantization_bit``."""
    quantization_bit = quant_config.quantization_bit
    if quantization_bit is None:
        logger.warning_rank0("No quantization method applied.")
        return init_kwargs
    if quantization_bit not in (8, 4):
        raise ValueError(f"Unsupported quantization bit: {quantization_bit} for auto quantization.")
    logger.info_rank0(f"Loading {quantization_bit}-bit quantized model.")
    return QuantizationPlugin("bnb")(init_kwargs, quant_config=quant_config, is_trainable=is_trainable)


@QuantizationPlugin("bnb").register(params=BnbParams, parse_arg="quant_config")
def quantization_with_bnb(
    init_kwargs: dict[str, Any],
    quant_config: BnbParams,
    is_trainable: bool = False,
) -> dict[str, Any]:
    r"""Quantization with BNB."""
    from .bnb import apply_bnb_quantization

    return apply_bnb_quantization(init_kwargs, quant_config, is_trainable=is_trainable)
