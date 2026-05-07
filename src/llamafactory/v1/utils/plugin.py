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


from collections import defaultdict
from collections.abc import Callable
from typing import Any, ClassVar

from . import logging


logger = logging.get_logger(__name__)


class BasePlugin:
    """Base class for plugins.

    A plugin is a callable object registered by name. Each variant can also
    declare its config schema and field aliases at registration time, so the
    config dataclass and the consumer function live together in source.

    Example:
    ```python
    @dataclass
    class LoraConfig(StrictConfigMixin):
        name: Literal["lora"] = "lora"
        r: int = 8

    @PeftPlugin("lora", config=LoraConfig, aliases={"lora_rank": "r"}).register()
    def get_lora_model(model, config: LoraConfig, is_train): ...

    # later, parsing user input:
    parsed = PeftPlugin.parse_config({"name": "lora", "r": 16})
    # -> LoraConfig(r=16)
    ```
    """

    # Per-subclass registries. Populated by __init_subclass__ so subclasses get
    # isolated namespaces — otherwise e.g. KernelPlugin("auto") and
    # QuantizationPlugin("auto") would clobber each other in a shared registry.
    _registry: ClassVar[dict[str, dict[str, Callable]]]
    _configs: ClassVar[dict[str, tuple[type, dict[str, str] | None]]]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._registry = defaultdict(dict)
        cls._configs = {}

    def __init__(
        self,
        name: str | None = None,
        config: type | None = None,
        aliases: dict[str, str] | None = None,
    ) -> None:
        """Initialize the plugin handle.

        Args:
            name: Variant name (e.g. "lora", "fsdp2").
            config: Optional dataclass type that validates this variant's input.
                Only consumed by ``register()`` for the primary ``__call__`` registration.
            aliases: Optional ``{old_key: canonical_key}`` map for backwards-compatible
                renames. Applied during ``parse_config``.
        """
        self.name = name
        self._config_cls = config
        self._aliases = aliases

    def register(self, method_name: str = "__call__") -> Callable:
        """Register a function as this variant's implementation.

        When ``method_name == "__call__"`` and a config schema was passed to
        ``__init__``, the schema is also recorded in the per-class config registry.
        """
        if self.name is None:
            raise ValueError("Plugin name should be specified.")

        if method_name in self._registry[self.name]:
            logger.warning_rank0_once(f"Method {method_name} of plugin {self.name} is already registered.")

        if method_name == "__call__" and self._config_cls is not None:
            existing = type(self)._configs.get(self.name)
            if existing is not None and existing[0] is not self._config_cls:
                logger.warning_rank0_once(
                    f"Config for plugin {type(self).__name__}({self.name!r}) is being reassigned "
                    f"from {existing[0].__name__} to {self._config_cls.__name__}."
                )
            type(self)._configs[self.name] = (self._config_cls, self._aliases)

        def decorator(func: Callable) -> Callable:
            self._registry[self.name][method_name] = func
            return func

        return decorator

    def __call__(self, *args, **kwargs) -> Any:
        """Call the registered function with the given arguments."""
        return self["__call__"](*args, **kwargs)

    def __getattr__(self, method_name: str) -> Callable:
        """Get the registered function with the given name."""
        return self[method_name]

    def __getitem__(self, method_name: str) -> Callable:
        """Get the registered function with the given name."""
        if method_name not in self._registry[self.name]:
            raise ValueError(f"Method {method_name} of plugin {self.name} is not registered.")

        return self._registry[self.name][method_name]

    @classmethod
    def list_variants(cls) -> list[str]:
        """List all variant names registered with a config schema for this plugin class."""
        return sorted(cls._configs.keys())

    @classmethod
    def parse_config(cls, raw: Any) -> Any:
        """Parse a raw plugin argument into the matching variant's config dataclass.

        ``raw`` may be ``None``, a string short-form (``"lora"`` → ``{"name": "lora"}``),
        a JSON string, or a dict. Unknown keys raise ``ValueError``.

        Returns ``None`` when ``raw`` is ``None``.
        """
        # Local import keeps utils/plugin.py free of config-layer dependencies at import time.
        from ..config.arg_utils import normalize_plugin_argument, strict_dataclass_from_dict

        normalized = normalize_plugin_argument(raw)
        if normalized is None:
            return None

        name = normalized.get("name")
        if name is None:
            raise ValueError(f"{cls.__name__} config must have a 'name' field.")

        if name not in cls._configs:
            raise ValueError(
                f"Unknown {cls.__name__} variant: {name!r}. Available variants: {cls.list_variants()}."
            )

        config_cls, aliases = cls._configs[name]
        return strict_dataclass_from_dict(
            config_cls,
            normalized,
            f"{cls.__name__}.{name}",
            aliases=aliases,
        )


if __name__ == "__main__":
    """
    python -m llamafactory.v1.utils.plugin
    """

    class PrintPlugin(BasePlugin):
        def again(self):  # optional
            self["again"]()

    @PrintPlugin("hello").register()
    def print_hello():
        print("Hello world!")

    @PrintPlugin("hello").register("again")
    def print_hello_again():
        print("Hello world! Again.")

    PrintPlugin("hello")()
    PrintPlugin("hello").again()
