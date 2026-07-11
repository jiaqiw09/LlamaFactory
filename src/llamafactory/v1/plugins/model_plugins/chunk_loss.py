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

import types
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.types import HFModel, Tensor


@dataclass
class _ChunkLossState:
    labels: Tensor
    loss_weights: Tensor
    chunk_size: int


_CHUNK_LOSS_STATE: ContextVar[_ChunkLossState | None] = ContextVar("chunk_loss_state", default=None)


class _ChunkedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: Tensor,
        head_weight: Tensor,
        head_bias: Tensor | None,
        labels: Tensor,
        loss_weights: Tensor,
        denominator: Tensor,
        chunk_size: int,
    ) -> Tensor:
        needs_hidden_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        accumulated_loss = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
        grad_hidden = torch.empty_like(hidden_states) if needs_hidden_grad else None
        grad_weight = torch.zeros_like(head_weight) if needs_weight_grad else None
        grad_bias = torch.zeros_like(head_bias) if head_bias is not None and needs_bias_grad else None

        hidden_chunks = torch.split(hidden_states, chunk_size, dim=1)
        label_chunks = torch.split(labels, chunk_size, dim=1)
        loss_weight_chunks = torch.split(loss_weights, chunk_size, dim=1)
        grad_hidden_chunks = torch.split(grad_hidden, chunk_size, dim=1) if grad_hidden is not None else None

        for index, (hidden_chunk, label_chunk, loss_weight_chunk) in enumerate(
            zip(hidden_chunks, label_chunks, loss_weight_chunks, strict=True)
        ):
            with torch.enable_grad():
                hidden_arg = hidden_chunk.detach().requires_grad_(needs_hidden_grad)
                weight_arg = head_weight.detach().requires_grad_(needs_weight_grad)
                bias_arg = None
                if head_bias is not None:
                    bias_arg = head_bias.detach().requires_grad_(needs_bias_grad)

                logits = F.linear(hidden_arg, weight_arg, bias_arg).float()
                token_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    label_chunk.reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                )
                chunk_loss = (token_loss * loss_weight_chunk.reshape(-1)).sum() / denominator

                grad_targets = []
                if needs_hidden_grad:
                    grad_targets.append(hidden_arg)
                if needs_weight_grad:
                    grad_targets.append(weight_arg)
                if bias_arg is not None and needs_bias_grad:
                    grad_targets.append(bias_arg)
                chunk_grads = torch.autograd.grad(chunk_loss, grad_targets) if grad_targets else []

            accumulated_loss.add_(chunk_loss.detach())
            grad_index = 0
            if grad_hidden_chunks is not None:
                grad_hidden_chunks[index].copy_(chunk_grads[grad_index])
                grad_index += 1
            if grad_weight is not None:
                grad_weight.add_(chunk_grads[grad_index])
                grad_index += 1
            if grad_bias is not None:
                grad_bias.add_(chunk_grads[grad_index])

        empty = torch.empty(0, device=hidden_states.device)
        ctx.save_for_backward(
            grad_hidden if grad_hidden is not None else empty,
            grad_weight if grad_weight is not None else empty,
            grad_bias if grad_bias is not None else empty,
        )
        ctx.has_hidden_grad = grad_hidden is not None
        ctx.has_weight_grad = grad_weight is not None
        ctx.has_bias_grad = grad_bias is not None
        return accumulated_loss

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_hidden, grad_weight, grad_bias = ctx.saved_tensors
        return (
            grad_hidden * grad_output if ctx.has_hidden_grad else None,
            grad_weight * grad_output if ctx.has_weight_grad else None,
            grad_bias * grad_output if ctx.has_bias_grad else None,
            None,
            None,
            None,
            None,
        )


def chunked_linear_cross_entropy(
    hidden_states: Tensor,
    head_weight: Tensor,
    head_bias: Tensor | None,
    labels: Tensor,
    loss_weights: Tensor,
    chunk_size: int,
) -> Tensor:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if hidden_states.shape[:2] != labels.shape or labels.shape != loss_weights.shape:
        raise ValueError(
            "Chunk loss expects hidden_states, labels and loss_weights to have matching batch/sequence dimensions."
        )

    denominator = loss_weights.float().sum() + 1e-6
    return _ChunkedLinearCrossEntropy.apply(
        hidden_states,
        head_weight,
        head_bias,
        labels,
        loss_weights,
        denominator,
        chunk_size,
    )


def _get_local_tensor(tensor: Tensor | None) -> Tensor | None:
    if tensor is None:
        return None

    try:
        from torch.distributed.tensor import DTensor

        if isinstance(tensor, DTensor):
            return tensor.to_local()
    except ImportError:
        pass

    return tensor


def _get_lm_head(model: HFModel) -> nn.Linear:
    candidates = [model]
    if hasattr(model, "get_base_model"):
        candidates.insert(0, model.get_base_model())

    for candidate in candidates:
        get_output_embeddings = getattr(candidate, "get_output_embeddings", None)
        if callable(get_output_embeddings):
            lm_head = get_output_embeddings()
            if isinstance(lm_head, nn.Linear):
                return lm_head

    raise TypeError("SFT chunk loss currently requires get_output_embeddings() to return torch.nn.Linear.")


def enable_sft_chunk_loss(model: HFModel) -> None:
    lm_head = _get_lm_head(model)
    if getattr(lm_head, "_llamafactory_chunk_loss_enabled", False):
        return

    original_forward = lm_head.forward

    def chunk_loss_forward(self, hidden_states: Tensor, *args, **kwargs):
        state = _CHUNK_LOSS_STATE.get()
        if state is None:
            return original_forward(hidden_states, *args, **kwargs)
        if args or kwargs:
            raise TypeError("SFT chunk loss does not support extra lm_head forward arguments.")
        if hidden_states.shape[:2] != state.labels.shape:
            raise ValueError(
                "The lm_head hidden-state shape does not match labels. "
                "This model may slice logits or require a model-specific chunk-loss adapter."
            )

        weight = _get_local_tensor(self.weight)
        bias = _get_local_tensor(self.bias)
        return chunked_linear_cross_entropy(
            hidden_states=hidden_states[..., :-1, :].contiguous(),
            head_weight=weight,
            head_bias=bias,
            labels=state.labels[..., 1:].contiguous(),
            loss_weights=state.loss_weights[..., 1:].contiguous(),
            chunk_size=state.chunk_size,
        )

    lm_head.forward = types.MethodType(chunk_loss_forward, lm_head)
    lm_head._llamafactory_chunk_loss_enabled = True


@contextmanager
def sft_chunk_loss_context(labels: Tensor, loss_weights: Tensor, chunk_size: int):
    token = _CHUNK_LOSS_STATE.set(_ChunkLossState(labels, loss_weights, chunk_size))
    try:
        yield
    finally:
        _CHUNK_LOSS_STATE.reset(token)
