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

"""Lightweight wrapper-style plugin registry.

The shape mirrors the original BasePlugin (``_registry[name][method]``):

* Plugin **name** is the routing key — callers write
  ``Plugin("fsdp2").shard_model(...)`` or ``Plugin("lora")(...)`` and the base
  class looks up the matching callable.
* Registered callables stay **lightweight wrappers** — the heavy implementation
  modules are imported lazily inside the function body, not at registration
  time.  So importing an ``interface.py`` does *not* drag in ``fsdp2.py`` /
  ``deepspeed.py``.

Beyond the original API this module adds:

* ``register_methods()`` — a class decorator that registers every ``@staticmethod``
  on an interface class as one method of the same plugin name, replacing the "four
  decorators per backend" boilerplate.
* ``params=`` / ``parse_arg=`` / ``aliases=`` kwargs on ``register()`` and
  ``register_methods()`` — declares the ParamsClass, the argument to auto-parse,
  and alias map for that plugin.
* ``parse_params(name, config)`` — runs the same validation pipeline used by
  the previous design (alias remap → unknown-key reject → required-field check
  → dataclass construct), keyed by plugin name.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from collections import defaultdict
from collections.abc import Callable
from dataclasses import fields
from typing import Any

from . import logging


logger = logging.get_logger(__name__)


class BasePlugin:
    """Base class for plugins.

    A plugin is a callable object that can be registered and called by name.

    Single-method usage::

        class PrintPlugin(BasePlugin):
            pass


        @PrintPlugin("hello").register()
        def print_hello():
            print("Hello world!")


        PrintPlugin("hello")()

    Multi-method usage (one class, all staticmethods become methods of the same
    plugin name)::

        @MyPlugin("foo").register_methods()
        class FooPlugin:
            @staticmethod
            def run(...): ...

            @staticmethod
            def save(...): ...


        MyPlugin("foo").run(...)
        MyPlugin("foo").save(...)

    Declaring params and aliases::

        @MyPlugin("foo").register(params=FooParams, parse_arg="config", aliases={"old": "new"})
        def run(...): ...

    Auto-parsing the user config before dispatch::

        @MyPlugin("foo").register(params=FooParams, parse_arg="config")
        def run(model, config, **kwargs):
            # config is already a FooParams instance.
            ...
    """

    # Each subclass shares its own per-class registry via __init_subclass__.
    _registry: dict[str, dict[str, Callable]]
    _params: dict[str, type | None]
    _aliases: dict[str, dict[str, str] | None]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Give each subclass its own buckets so different families don't share
        # a registry by accident.
        cls._registry = defaultdict(dict)
        cls._params = {}
        cls._aliases = {}

    # Base class itself also has empty buckets so direct ``BasePlugin("x")``
    # never resolves to nothing unexpectedly.
    _registry = defaultdict(dict)
    _params = {}
    _aliases = {}

    def __init__(self, name: str | None = None) -> None:
        self.name = name

    # ------------------------------------------------------------------
    # Single-method registration
    # ------------------------------------------------------------------

    def register(
        self,
        method_name: str = "__call__",
        *,
        params: type | None = None,
        aliases: dict[str, str] | None = None,
        parse_arg: str | None = None,
    ) -> Callable:
        """Decorator to register a function under ``self.name`` / *method_name*.

        Args:
            method_name: Which "method slot" to fill.  Defaults to ``__call__``
                so ``Plugin("foo")(...)`` dispatches to this function.
            params: Optional dataclass for typed parameter parsing.  Stored on
                the plugin's name (not the function) so multi-method plugins
                share one schema.
            aliases: Optional ``{old_key: canonical_key}`` map for backward
                compatibility on YAML field names.
            parse_arg: Optional function argument name to auto-parse with
                ``parse_params`` before dispatch. Explicit opt-in avoids
                guessing which argument is the plugin config.
        """
        if self.name is None:
            raise ValueError("Plugin name should be specified.")

        cls = type(self)
        if method_name in cls._registry[self.name]:
            logger.warning_rank0_once(f"Method {method_name} of plugin {self.name} is already registered.")

        # Store params / aliases once per (cls, name). Multi-decorator usage on
        # the same name should not contradict — last writer wins, but warn on
        # conflict.
        if params is not None:
            existing = cls._params.get(self.name)
            if existing is not None and existing is not params:
                logger.warning_rank0_once(
                    f"Params for {cls.__name__}({self.name!r}) re-declared ({existing.__name__} → {params.__name__})."
                )
            cls._params[self.name] = params

        if aliases is not None:
            existing_a = cls._aliases.get(self.name)
            if existing_a is not None and existing_a != aliases:
                logger.warning_rank0_once(f"Aliases for {cls.__name__}({self.name!r}) re-declared.")
            cls._aliases[self.name] = aliases

        def decorator(func: Callable) -> Callable:
            target = cls._wrap_with_auto_parse(func, self.name, params=params, parse_arg=parse_arg)
            cls._registry[self.name][method_name] = target
            return target

        return decorator

    # ------------------------------------------------------------------
    # Multi-method class-based registration
    # ------------------------------------------------------------------

    def register_methods(
        self,
        *,
        params: type | None = None,
        aliases: dict[str, str] | None = None,
        parse_arg: str | None = None,
    ) -> Callable:
        """Register static methods from a decorated class.

        Class decorator: register every ``@staticmethod`` on the decorated
        class as a method of plugin ``self.name``.

        Args:
            params: Same as :py:meth:`register`.
            aliases: Same as :py:meth:`register`.
            parse_arg: Same as :py:meth:`register`.

        Example::

            @MyPlugin("fsdp2").register_methods(params=FSDP2Params)
            class FSDP2Plugin:
                @staticmethod
                def shard_model(...): ...

                @staticmethod
                def save_model(...): ...
        """
        if self.name is None:
            raise ValueError("Plugin name should be specified.")

        outer = self  # capture for the closure below

        def decorator(hub_cls: type) -> type:
            methods: list[tuple[str, Callable]] = []
            for attr_name, attr_value in vars(hub_cls).items():
                if attr_name.startswith("__") and attr_name.endswith("__"):
                    continue
                if not isinstance(attr_value, staticmethod):
                    raise TypeError(
                        f"{hub_cls.__name__}.{attr_name} must be a staticmethod. "
                        "`register_methods()` only supports stateless method groups."
                    )

                methods.append((attr_name, attr_value.__func__))

            if not methods:
                raise ValueError(
                    f"register_methods({outer.name!r}) found no methods on "
                    f"{hub_cls.__name__}; declare @staticmethod methods."
                )

            for attr_name, func in methods:
                outer.register(attr_name, params=params, aliases=aliases, parse_arg=parse_arg)(func)
            return hub_cls

        return decorator

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    @classmethod
    def _wrap_with_auto_parse(
        cls,
        func: Callable,
        name: str,
        *,
        params: type | None,
        parse_arg: str | None,
    ) -> Callable:
        if params is None or parse_arg is None:
            return func

        signature = inspect.signature(func)
        if parse_arg not in signature.parameters:
            raise ValueError(
                f"{cls.__name__}({name!r}) requested parse_arg={parse_arg!r}, "
                f"but {func.__name__} has no such parameter."
            )

        @functools.wraps(func)
        def auto_parse_wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            value = bound.arguments[parse_arg]
            if not isinstance(value, params):
                bound.arguments[parse_arg] = cls.parse_params(name, value)
            return func(*bound.args, **bound.kwargs)

        return auto_parse_wrapper

    def __call__(self, *args, **kwargs) -> Any:
        return self["__call__"](*args, **kwargs)

    def __getattr__(self, method_name: str) -> Callable:
        # ``__getattr__`` is only consulted when normal attribute lookup fails,
        # so this never shadows ``name``/``_registry``/etc.
        return self[method_name]

    def __getitem__(self, method_name: str) -> Callable:
        cls = type(self)
        if self.name is None:
            raise ValueError(f"{cls.__name__} must be constructed with a name.")
        bucket = cls._registry.get(self.name)
        if bucket is None or method_name not in bucket:
            raise ValueError(
                f"Method {method_name!r} of plugin {self.name!r} is not registered. "
                f"Available: {sorted(bucket.keys()) if bucket else []}"
            )
        return bucket[method_name]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @classmethod
    def names(cls) -> list[str]:
        """List plugin names registered under this family."""
        return sorted(cls._registry.keys())

    # ------------------------------------------------------------------
    # Typed config parsing
    # ------------------------------------------------------------------

    @classmethod
    def parse_params(cls, name: str, config: Any) -> Any:
        """Validate and convert *config* to the ParamsClass of plugin *name*.

        Steps:
        1. If no ParamsClass was declared, return *config* unchanged.
        2. If *config* is already an instance of the ParamsClass, return as-is.
        3. Apply aliases (``{old: canonical}``) — a key clash between an alias
           and its canonical name raises ``ValueError``.
        4. Reject unknown keys (``"name"`` is always permitted as the routing
           key).
        5. Reject when a required field (no default / default_factory) is
           absent.
        6. Construct and return the ParamsClass.
        """
        params_cls = cls._params.get(name)
        if params_cls is None:
            return config

        if isinstance(config, params_cls):
            return config

        aliases = cls._aliases.get(name) or {}
        resolved: dict[str, Any] = {}
        raw_items = config.items() if isinstance(config, dict) else dict(config).items()
        for k, v in raw_items:
            canonical = aliases.get(k, k)
            if canonical in resolved:
                raise ValueError(
                    f"Conflicting keys for {cls.__name__}({name!r}): "
                    f"'{k}' and its canonical name '{canonical}' both present."
                )
            resolved[canonical] = v

        all_fields = fields(params_cls)
        known = {f.name for f in all_fields}

        unknown = set(resolved.keys()) - known - {"name"}
        if unknown:
            raise ValueError(
                f"Unknown params for {cls.__name__}({name!r}): {sorted(unknown)}. Expected keys: {sorted(known)}"
            )

        missing = [
            f.name
            for f in all_fields
            if f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
            and f.name not in resolved
        ]
        if missing:
            raise ValueError(f"Missing required params for {cls.__name__}({name!r}): {missing}.")

        valid = {k: v for k, v in resolved.items() if k in known}
        return params_cls(**valid)


if __name__ == "__main__":
    """python -m llamafactory.v1.utils.plugin"""

    class PrintPlugin(BasePlugin):
        pass

    @PrintPlugin("hello").register()
    def print_hello():
        print("Hello world!")

    @PrintPlugin("hello").register("again")
    def print_hello_again():
        print("Hello world! Again.")

    PrintPlugin("hello")()
    PrintPlugin("hello").again()
