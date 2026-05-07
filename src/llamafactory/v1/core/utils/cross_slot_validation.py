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

"""Cross-slot constraint validation for v1 plugin configurations.

Some constraints span multiple slots (e.g. ``peft_config.name == 'lora'`` is
incompatible with ``init_config.name == 'init_on_meta'``) and cannot be
expressed inside any single per-slot dataclass schema. They are collected here
as named rules and run after all per-slot parsing is done, before any plugin is
invoked.

Each rule receives the parsed ``ModelArguments`` and ``TrainingArguments`` and
raises ``ValueError`` (with an actionable message) if violated.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ...config.model_args import ModelArguments
    from ...config.training_args import TrainingArguments


CrossSlotRule = Callable[["ModelArguments", "TrainingArguments"], None]


def _rule_lora_disallows_init_on_meta(model_args: "ModelArguments", training_args: "TrainingArguments") -> None:
    peft_name = getattr(model_args.peft_config, "name", None)
    init_name = getattr(model_args.init_config, "name", None)
    if peft_name == "lora" and init_name == "init_on_meta":
        raise ValueError(
            "peft_config.name='lora' is incompatible with init_config.name='init_on_meta': "
            "LoRA initialization needs concrete weights. Use init_on_default or init_on_rank0."
        )


def _rule_meta_init_disallows_quantization(model_args: "ModelArguments", training_args: "TrainingArguments") -> None:
    init_name = getattr(model_args.init_config, "name", None)
    if init_name == "init_on_meta" and model_args.quant_config is not None:
        raise ValueError(
            "init_config.name='init_on_meta' is incompatible with quant_config: "
            "quantization needs real weights to operate on."
        )


def _rule_qwen35_disallows_sequence_parallel(
    model_args: "ModelArguments", training_args: "TrainingArguments"
) -> None:
    # Note: model_type can only be checked after the model is loaded.
    # This rule lives here for centralization but the actual check still happens
    # in base_trainer where model.config is available. Documented for visibility.
    return


_RULES: list[CrossSlotRule] = [
    _rule_lora_disallows_init_on_meta,
    _rule_meta_init_disallows_quantization,
]


def validate_cross_slot_constraints(
    model_args: "ModelArguments", training_args: "TrainingArguments"
) -> None:
    """Run every registered cross-slot rule.

    Call this AFTER per-slot dataclass parsing (i.e. ``parse_training_plugin_configs``
    and ``ModelEngine._parse_plugin_configs``) so that every config is its
    final dataclass form, and BEFORE any plugin function is invoked so that
    failures surface early with full context.
    """
    for rule in _RULES:
        rule(model_args, training_args)


def register_cross_slot_rule(rule: CrossSlotRule) -> CrossSlotRule:
    """Register a custom cross-slot rule. Useful for downstream extensions."""
    _RULES.append(rule)
    return rule
