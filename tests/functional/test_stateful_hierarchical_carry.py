"""Long diagonal sequences retain zero factors and cross-group State carries."""

from __future__ import annotations

import pytest
import torch

from attn_engine.stateful.lowering.diagonal.diagonal_scan import hierarchical_carry

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


@pytest.mark.parametrize("identity", [False, True])
def test_hierarchical_carry_matches_ordered_recurrence(gpu_device, identity):
    # 65 summaries crosses the first hierarchy threshold and leaves a tail group.
    generator = torch.Generator(device=gpu_device).manual_seed(941)
    shape = (2, 65, 3, 5)
    bias = torch.randn(shape, device=gpu_device, generator=generator) * 0.01
    scale = torch.rand(shape, device=gpu_device, generator=generator) * 0.2 + 0.8
    scale[:, 31] = 0
    scale[:, 32] = -0.5
    initial = torch.randn(2, 3, 5, device=gpu_device, generator=generator)
    carry = torch.empty_like(bias)
    final = torch.empty_like(initial)
    hierarchical_carry(
        None if identity else scale,
        bias,
        initial,
        carry,
        final,
        identity=identity,
        write_final=True,
    )
    state = initial.clone()
    expected = []
    for chunk in range(shape[1]):
        expected.append(state.clone())
        state = (
            state + bias[:, chunk]
            if identity
            else scale[:, chunk] * state + bias[:, chunk]
        )
    torch.testing.assert_close(carry, torch.stack(expected, 1), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(final, state, atol=2e-6, rtol=2e-5)


def test_hierarchical_carry_does_not_write_unrequested_final(gpu_device):
    shape = (1, 65, 1, 1)
    bias = torch.ones(shape, device=gpu_device)
    initial = torch.full((1, 1, 1), 7.0, device=gpu_device)
    carry = torch.empty_like(bias)
    hierarchical_carry(
        None, bias, initial, carry, initial, identity=True, write_final=False
    )
    torch.testing.assert_close(initial, torch.full_like(initial, 7.0))
    torch.testing.assert_close(
        carry.flatten(), torch.arange(7, 72, device=gpu_device).float()
    )
