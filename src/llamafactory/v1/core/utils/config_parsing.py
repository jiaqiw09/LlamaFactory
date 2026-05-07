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

"""Strict parsing of TrainingArguments plugin slots.

This is invoked from the trainer entry points (e.g. ``run_sft``) before the
``TrainingArguments`` is consumed by ``DistributedInterface`` or ``BaseTrainer``.
Keeping this in ``core/utils`` rather than in ``config/`` preserves the v1
layering rule: ``config/`` does not depend on ``plugins/``.
"""

from __future__ import annotations


def parse_training_plugin_configs(training_args) -> None:
    """In-place parse ``training_args`` plugin slots into strict dataclasses.

    Idempotent: dicts are parsed; already-parsed dataclasses are left alone.

    ``dist_config`` is the only user-facing slot for distributed concerns. Its
    ``DistributedPlugin.parse_config`` routes flat user input into a backend
    dataclass plus a sequence-parallel dataclass under a single ``DistConfig``
    façade — see ``llamafactory.v1.plugins.trainer_plugins.distributed.hub``.
    """
    # Importing the plugin modules here triggers their @PluginCls(..., config=...).register()
    # decorators, which populate the BasePlugin._configs registry that parse_config reads.
    from ...plugins.trainer_plugins.distributed import hub as _dist_hub  # noqa: F401
    from ...plugins.trainer_plugins.distributed.hub import DistributedPlugin

    if isinstance(training_args.dist_config, dict):
        training_args.dist_config = DistributedPlugin.parse_config(training_args.dist_config)


def parse_model_plugin_configs(model_args) -> None:
    """In-place parse ``model_args`` plugin slots into strict dataclasses.

    Idempotent: dicts are parsed; already-parsed dataclasses are left alone.
    Provided so trainers can run cross-slot validation before constructing
    ``ModelEngine`` (which would otherwise materialize the model first).
    """
    from ...plugins.model_plugins import initialization as _init_module  # noqa: F401
    from ...plugins.model_plugins import peft as _peft_module  # noqa: F401
    from ...plugins.model_plugins import quantization as _quant_module  # noqa: F401
    from ...plugins.model_plugins.initialization import InitPlugin
    from ...plugins.model_plugins.kernels.interface import KernelPlugin
    from ...plugins.model_plugins.peft import PeftPlugin
    from ...plugins.model_plugins.quantization import QuantizationPlugin

    if isinstance(model_args.init_config, dict):
        model_args.init_config = InitPlugin.parse_config(model_args.init_config)

    if isinstance(model_args.peft_config, dict):
        model_args.peft_config = PeftPlugin.parse_config(model_args.peft_config)

    if isinstance(model_args.kernel_config, dict):
        model_args.kernel_config = KernelPlugin.parse_config(model_args.kernel_config)

    if isinstance(model_args.quant_config, dict):
        model_args.quant_config = QuantizationPlugin.parse_config(model_args.quant_config)
