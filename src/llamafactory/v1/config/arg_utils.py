# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v5.0.0rc0/src/transformers/training_args.py
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


import json
from dataclasses import MISSING, fields, is_dataclass
from enum import StrEnum, unique
from typing import Any


class PluginConfig(dict):
    """Dictionary that allows attribute access. Kept for transitional callsites."""

    @property
    def name(self) -> str:
        """Plugin name."""
        if "name" not in self:
            raise ValueError("Plugin configuration must have a 'name' field.")

        return self["name"]


PluginArgument = PluginConfig | dict | str | None


class StrictConfigMixin:
    """Marker base for strict plugin config dataclasses.

    Provides a small dict-like surface (``get`` / ``[]`` / ``in``) so that
    callsites still using ``config.get("key", default)`` continue to work during
    the migration. New code should use attribute access (``config.r``) directly.

    Semantic note: ``__contains__`` returns True for any declared field even if
    its value is ``None``. ``get(key, default)`` returns the declared field's
    value, including ``None``, rather than ``default`` — this differs from
    ``dict.get`` semantics. Migrate consumers off this bridge when convenient.
    """

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)

        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


@unique
class ModelClass(StrEnum):
    """Auto class for model config."""

    LLM = "llm"
    CLS = "cls"
    OTHER = "other"


@unique
class SampleBackend(StrEnum):
    HF = "hf"
    VLLM = "vllm"


@unique
class BatchingStrategy(StrEnum):
    NORMAL = "normal"
    PADDING_FREE = "padding_free"
    DYNAMIC_BATCHING = "dynamic_batching"
    DYNAMIC_PADDING_FREE = "dynamic_padding_free"


def _convert_str_dict(data: dict) -> dict:
    """Parse string representation inside the dictionary.

    Args:
        data: The string or dictionary to convert.

    Returns:
        The converted dictionary.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = _convert_str_dict(value)
        elif isinstance(value, str):
            lower_value = value.lower()
            if lower_value in ("true", "false"):
                data[key] = lower_value == "true"
            elif lower_value in ("none", "null"):
                data[key] = None
            elif value.isdigit():
                data[key] = int(value)
            elif value.replace(".", "", 1).isdigit():
                data[key] = float(value)

    return data


def normalize_plugin_argument(config: PluginArgument) -> PluginConfig | None:
    """Normalize a raw plugin argument into a ``PluginConfig`` dict (or None).

    Accepts: ``None`` | ``str`` (variant name shortcut or JSON) | ``dict``-like.
    Returns a ``PluginConfig`` (dict subclass with ``.name`` property) so that
    pre-strict-parse callsites can still dispatch by name.
    """
    if config is None:
        return None

    if isinstance(config, str):
        if config.startswith("{"):
            config = json.loads(config)
        else:
            config = {"name": config}

    return PluginConfig(_convert_str_dict(dict(config)))


def get_plugin_config(config: PluginArgument) -> PluginConfig | None:
    """Legacy dict-style normalizer kept for slots that have not yet migrated to strict schemas."""
    normalized = normalize_plugin_argument(config)
    if normalized is None:
        return None

    if "name" not in normalized:
        raise ValueError("Plugin configuration must have a 'name' field.")

    return PluginConfig(normalized)


def strict_dataclass_from_dict(
    cls: type,
    raw_config: dict[str, Any],
    config_name: str,
    aliases: dict[str, str] | None = None,
) -> Any:
    """Build a dataclass config and reject unknown keys.

    Plugin-aware parsing should go through ``BasePlugin.parse_config``; this
    helper is the lower-level building block it delegates to.

    Args:
        cls: Target dataclass type.
        raw_config: Already-normalized dict of fields (must contain ``name`` if discriminated).
        config_name: Human-readable slot name used in error messages.
        aliases: ``{old_key: canonical_key}`` map applied before validation.

    Raises:
        TypeError: ``cls`` is not a dataclass.
        ValueError: Both alias and canonical key present, unknown keys, or missing required fields.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} must be a dataclass.")

    config = dict(raw_config)
    for old_key, new_key in (aliases or {}).items():
        if old_key in config:
            if new_key in config:
                raise ValueError(
                    f"`{old_key}` and `{new_key}` are both specified in {config_name}. "
                    f"Please use `{new_key}` only."
                )

            config[new_key] = config.pop(old_key)

    field_names = {field.name for field in fields(cls) if field.init}
    unknown_keys = sorted(set(config) - field_names)
    if unknown_keys:
        raise ValueError(
            f"Unknown parameters for {config_name}: {unknown_keys}. "
            f"Allowed parameters: {sorted(field_names)}."
        )

    required_keys = [
        field.name
        for field in fields(cls)
        if field.init and field.default is MISSING and field.default_factory is MISSING  # type: ignore[attr-defined]
    ]
    missing_keys = sorted(set(required_keys) - set(config))
    if missing_keys:
        raise ValueError(f"Missing required parameters for {config_name}: {missing_keys}.")

    return cls(**config)
