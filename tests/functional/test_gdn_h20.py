from __future__ import annotations

import pytest
import torch

from attn_engine import GDNEngine
from attn_engine.gdn_reference import gated_delta_rule_reference


pytestmark = [pytest.mark.functional, pytest.mark.gpu, pytest.mark.h20]


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (actual.float() - expected.float()).norm()
    return (difference / expected.float().norm().clamp_min(1e-12)).item()


def _inputs(device, *, hq=1, hv=1, length=64, initial=False, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(0)
    q = (torch.randn(1, hq, length, 128, generator=generator, device=device) * 0.05).bfloat16()
    k = (torch.randn(1, hq, length, 128, generator=generator, device=device) * 0.05).bfloat16()
    v = (torch.randn(1, hv, length, 128, generator=generator, device=device) * 0.05).bfloat16()
    g = -torch.rand(1, hv, length, generator=generator, device=device, dtype=torch.float32) * 0.1
    beta = torch.rand(1, hv, length, generator=generator, device=device, dtype=torch.float32)
    state = None
    if initial:
        state = torch.randn(1, hv, 128, 128, generator=generator, device=device) * 0.02
    tensors = [q, k, v, g, beta]
    if requires_grad:
        tensors = [tensor.requires_grad_() for tensor in tensors]
        if state is not None:
            state.requires_grad_()
    return (*tensors, state)


@pytest.fixture
def require_h20(gpu_device):
    if torch.version.hip is not None:
        pytest.skip("H20 GDN requires CUDA")
    name = torch.cuda.get_device_name(gpu_device)
    if "H20" not in name.upper():
        pytest.skip(f"H20 GDN requires an NVIDIA H20; found {name}")
    return gpu_device


@pytest.mark.parametrize("hq,hv,length,initial,scale", [(1, 1, 64, False, None), (1, 2, 128, True, 0.1)])
def test_h20_forward_state_and_gva(require_h20, hq, hv, length, initial, scale):
    q, k, v, g, beta, state = _inputs(require_h20, hq=hq, hv=hv, length=length, initial=initial)
    engine = GDNEngine(require_h20)
    actual, actual_state = engine(
        q, k, v, g, beta, scale=scale, initial_state=state, output_final_state=True
    )
    expected, expected_state = gated_delta_rule_reference(
        q, k, v, g, beta, scale=scale, initial_state=state, output_final_state=True
    )
    assert _relative_l2(actual, expected) <= 0.02
    assert _relative_l2(actual_state, expected_state) <= 0.02


@pytest.mark.parametrize("loss_path", ["output", "state", "joint"])
@pytest.mark.parametrize("hq,hv,length,initial", [(1, 1, 128, False), (1, 2, 64, True)])
def test_h20_all_gradients(require_h20, loss_path, hq, hv, length, initial):
    inputs = _inputs(require_h20, hq=hq, hv=hv, length=length, initial=initial, requires_grad=True)
    q, k, v, g, beta, state = inputs
    engine = GDNEngine(require_h20)
    output, final_state = engine(q, k, v, g, beta, initial_state=state, output_final_state=True)
    weights = torch.randn_like(output)
    state_weights = torch.randn_like(final_state)
    gradient_inputs = inputs if state is not None else inputs[:5]
    loss = output.float().mul(weights.float()).sum() if loss_path in ("output", "joint") else output.sum() * 0
    if loss_path in ("state", "joint"):
        loss = loss + (final_state * state_weights).sum()
    actual = torch.autograd.grad(loss, gradient_inputs)

    reference_inputs = tuple(tensor.detach().requires_grad_() for tensor in gradient_inputs)
    reference_output, reference_state = gated_delta_rule_reference(
        *reference_inputs[:5], initial_state=reference_inputs[5] if state is not None else None, output_final_state=True
    )
    expected_loss = reference_output.float().mul(weights.float()).sum() if loss_path in ("output", "joint") else reference_output.sum() * 0
    if loss_path in ("state", "joint"):
        expected_loss = expected_loss + (reference_state * state_weights).sum()
    expected = torch.autograd.grad(expected_loss, reference_inputs)
    for actual_gradient, expected_gradient in zip(actual, expected):
        assert _relative_l2(actual_gradient, expected_gradient) <= 0.02


@pytest.mark.parametrize(
    ("gate_value", "beta_value"),
    [(-1e-4, 1e-4), (-1e-4, 1.0 - 1e-4), (-1.0, 0.5)],
)
def test_h20_gate_and_beta_regimes(require_h20, gate_value, beta_value):
    q, k, v, g, beta, _ = _inputs(require_h20, length=64)
    g.fill_(gate_value)
    beta.fill_(beta_value)
    actual = GDNEngine(require_h20)(q, k, v, g, beta)
    expected, _ = gated_delta_rule_reference(q, k, v, g, beta)
    assert _relative_l2(actual, expected) <= 0.02


def test_h20_output_only_loss_with_requested_final_state(require_h20):
    q, k, v, g, beta, _ = _inputs(require_h20, requires_grad=True)
    output, final_state = GDNEngine(require_h20)(
        q, k, v, g, beta, output_final_state=True
    )
    assert final_state is not None
    torch.autograd.grad(output.float().square().mean(), (q, k, v, g, beta))


def test_h20_output_only_does_not_return_state(require_h20):
    q, k, v, g, beta, _ = _inputs(require_h20, requires_grad=True)
    output = GDNEngine(require_h20)(q, k, v, g, beta)
    assert isinstance(output, torch.Tensor)
    torch.autograd.grad(output.float().square().mean(), (q, k, v, g, beta))
