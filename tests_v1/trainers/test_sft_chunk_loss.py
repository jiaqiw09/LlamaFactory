from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from llamafactory.v1.plugins.model_plugins.chunk_loss import (
    chunked_linear_cross_entropy,
    enable_sft_chunk_loss,
    sft_chunk_loss_context,
)


@pytest.mark.parametrize("train_head", [False, True])
def test_chunked_linear_cross_entropy_matches_eager(train_head: bool):
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 7, requires_grad=True)
    weight = torch.randn(11, 7, requires_grad=train_head)
    labels = torch.tensor([[1, 2, -100, 4, 5], [3, 7, 8, -100, 1]])
    loss_weights = torch.tensor([[1.0, 1.0, 0.0, 0.5, 1.0], [1.0, 0.25, 1.0, 0.0, 1.0]])

    eager_hidden = hidden.detach().clone().requires_grad_(True)
    eager_weight = weight.detach().clone().requires_grad_(train_head)
    eager_logits = F.linear(eager_hidden, eager_weight).float()
    eager_token_loss = F.cross_entropy(
        eager_logits.reshape(-1, eager_logits.size(-1)), labels.reshape(-1), reduction="none", ignore_index=-100
    )
    eager_loss = (eager_token_loss * loss_weights.reshape(-1)).sum() / (loss_weights.sum() + 1e-6)
    eager_loss.backward()

    chunk_loss = chunked_linear_cross_entropy(hidden, weight, None, labels, loss_weights, chunk_size=2)
    chunk_loss.backward()

    torch.testing.assert_close(chunk_loss, eager_loss)
    torch.testing.assert_close(hidden.grad, eager_hidden.grad)
    if train_head:
        torch.testing.assert_close(weight.grad, eager_weight.grad)
    else:
        assert weight.grad is None


class _TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(5, 7)
        self.lm_head = nn.Linear(7, 11, bias=False)
        self.full_forward_calls = 0

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_features, **kwargs):
        self.full_forward_calls += 1
        hidden_states = self.proj(input_features)
        return SimpleNamespace(logits=self.lm_head(hidden_states))


def test_chunk_loss_keeps_full_model_forward():
    torch.manual_seed(0)
    model = _TinyCausalLM()
    enable_sft_chunk_loss(model)
    input_features = torch.randn(2, 5, 5)
    labels = torch.randint(0, 11, (2, 5))
    loss_weights = torch.ones(2, 5)

    with sft_chunk_loss_context(labels, loss_weights, chunk_size=2):
        loss = model(input_features).logits
    loss.backward()

    assert loss.ndim == 0
    assert model.full_forward_calls == 1
    assert model.proj.weight.grad is not None
