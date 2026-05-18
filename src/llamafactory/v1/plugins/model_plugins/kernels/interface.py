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

"""Concrete kernel registrations and public helpers.

Each wrapper lazily imports its ``*Kernel`` class and delegates to
``KernelClass.apply()``, which calls ``check_deps()`` then ``_apply()``.

Routing by ``kernel_config.name``:

* ``name: auto``              → the registered auto selector
* ``name: npu_fused_rmsnorm`` → that single kernel
* ``name: [a, b]``            → routed by :func:`apply_kernels`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import KernelNotAvailableError, KernelPlugin


if TYPE_CHECKING:
    from ....utils.types import HFModel


# ---------------------------------------------------------------------------
# Concrete kernel wrappers
# ---------------------------------------------------------------------------


@KernelPlugin("npu_fused_rmsnorm").register()
def apply_npu_fused_rmsnorm(model: HFModel, **kwargs) -> HFModel:
    from .ops.rms_norm.npu_rms_norm import NpuRMSNormKernel

    return NpuRMSNormKernel.apply(model=model, **kwargs)


@KernelPlugin("npu_fused_rope").register()
def apply_npu_fused_rope(model: HFModel, **kwargs) -> HFModel:
    from .ops.rope.npu_rope import NpuRoPEKernel

    return NpuRoPEKernel.apply(model=model, **kwargs)


@KernelPlugin("npu_fused_swiglu").register()
def apply_npu_fused_swiglu(model: HFModel, **kwargs) -> HFModel:
    from .ops.mlp.npu_swiglu import NpuSwiGLUKernel

    return NpuSwiGLUKernel.apply(model=model, **kwargs)


@KernelPlugin("npu_fused_moe").register()
def apply_npu_fused_moe(model: HFModel, **kwargs) -> HFModel:
    from .ops.mlp.npu_fused_moe import NpuFusedMoEKernel

    return NpuFusedMoEKernel.apply(model=model, **kwargs)


# ---------------------------------------------------------------------------
# auto selector
# ---------------------------------------------------------------------------


_AUTO_KERNELS = (
    "npu_fused_moe",
    "npu_fused_rmsnorm",
    "npu_fused_rope",
    "npu_fused_swiglu",
)


@KernelPlugin("auto").register()
def apply_auto_kernels(model: HFModel, **kwargs) -> HFModel:
    """Apply the kernels selected by the auto selector."""
    for kernel_name in _AUTO_KERNELS:
        try:
            model = KernelPlugin(kernel_name)(model=model, **kwargs)
        except KernelNotAvailableError:
            continue
    return model


def get_auto_kernels() -> list[str]:
    """List kernel names selected by ``auto``."""
    return list(_AUTO_KERNELS)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def apply_kernels(model: HFModel, config: dict[str, Any]) -> HFModel:
    """Apply kernels described by ``kernel_config.name``.

    ``name`` accepts ``"auto"``, a single kernel name, or a list of
    kernel names. Multi-name routing is scoped to kernels; other plugin
    families still use a single route name.
    """
    kernel_names = config["name"]
    if not isinstance(kernel_names, list):
        kernel_names = [kernel_names]

    for kernel_name in kernel_names:
        if not isinstance(kernel_name, str):
            raise TypeError(f"kernel_config.name must be a string or a list of strings; got item {kernel_name!r}.")
        model = KernelPlugin(kernel_name)(model=model, config=config)

    return model


def apply_kernel(kernel_id: str, **kwargs) -> HFModel:
    """Apply a single named kernel."""
    return KernelPlugin(kernel_id)(**kwargs)
