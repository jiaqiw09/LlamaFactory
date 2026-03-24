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

"""Gradient clipping compatible with mixed DeviceMesh DTensor gradients (e.g. FSDP2 + expert parallel).

PyTorch's ``clip_grad_norm_`` stacks per-parameter norms; when parameters use different
``DeviceMesh`` instances, that ``aten.stack`` fails during sharding propagation.
This module aggregates the global L2 norm without cross-mesh stacking.
"""

from __future__ import annotations

from typing import Iterable

import torch


def _try_dtensor_type():
    try:
        from torch.distributed._tensor import DTensor

        return DTensor
    except ImportError:
        return None


def _any_dtensor_grad(parameters: Iterable[torch.Tensor]) -> bool:
    DTensor = _try_dtensor_type()
    if DTensor is None:
        return False
    for p in parameters:
        g = getattr(p, "grad", None)
        if g is not None and isinstance(g, DTensor):
            return True
    return False


@torch.no_grad()
def clip_grad_norm_(
    parameters: Iterable[torch.Tensor] | torch.Tensor,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    """Clip gradient norm; safe when gradients are ``DTensor`` on different meshes.

    For ``norm_type == 2``, mixed ``dp`` / ``efsdp``+``ep`` meshes are handled by summing
    each parameter's global squared norm via ``(g * g).sum().full_tensor()`` (scalar reductions
    only), then applying the same clip coefficient to every ``.grad``.

    Other ``norm_type`` values delegate to :func:`torch.nn.utils.clip_grad_norm_` (may fail
    if meshes differ).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    grads_params = [p for p in parameters if p.grad is not None]
    if not grads_params:
        return torch.zeros((), dtype=torch.float32)

    if norm_type == 2.0 or norm_type == 2:
        if not _any_dtensor_grad(grads_params):
            return torch.nn.utils.clip_grad_norm_(
                parameters, max_norm, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite, foreach=foreach
            )

        DTensor = _try_dtensor_type()
        assert DTensor is not None

        first_grad = grads_params[0].grad
        assert first_grad is not None
        norm_device = first_grad.device

        total_norm_sq = 0.0
        for p in grads_params:
            g = p.grad
            if g is None:
                continue
            sq = (g.detach().float() * g.detach().float()).sum()
            if isinstance(sq, DTensor):
                total_norm_sq += float(sq.full_tensor().item())
            else:
                total_norm_sq += float(sq.item())

        total_norm = total_norm_sq**0.5
        out = torch.tensor(total_norm, dtype=torch.float32, device=norm_device)

        if error_if_nonfinite and not torch.isfinite(out):
            raise RuntimeError(f"Non-finite gradient norm after clipping aggregation: {total_norm}")

        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for p in parameters:
                if p.grad is not None:
                    p.grad.mul_(scale)

        return out

    return torch.nn.utils.clip_grad_norm_(
        parameters, max_norm, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite, foreach=foreach
    )
