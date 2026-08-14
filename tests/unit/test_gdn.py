from __future__ import annotations

import pytest
import torch

from attn_engine.gdn_engine import DEFAULT_SCALE, validate_gdn_inputs
from attn_engine.gdn_reference import gated_delta_rule_reference


pytestmark = pytest.mark.unit


def _inputs(*, batch=1, hq=1, hv=1, length=64, dim=128):
    q = torch.randn(batch, hq, length, dim, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, hv, length, dim, dtype=torch.bfloat16)
    g = -torch.rand(batch, hv, length, dtype=torch.float32)
    beta = torch.rand(batch, hv, length, dtype=torch.float32)
    return q, k, v, g, beta


def test_reference_supports_state_gva_and_gradients():
    q, k, v, g, beta = _inputs(hq=1, hv=2)
    tensors = [x.requires_grad_() for x in (q, k, v, g, beta)]
    initial_state = torch.randn(1, 2, 128, 128, dtype=torch.float32, requires_grad=True)

    output, final_state = gated_delta_rule_reference(
        *tensors,
        initial_state=initial_state,
        output_final_state=True,
    )

    assert output.shape == (1, 2, 64, 128)
    assert output.dtype == torch.bfloat16
    assert final_state is not None
    assert final_state.shape == (1, 2, 128, 128)
    assert final_state.dtype == torch.float32
    torch.autograd.grad(output.float().square().mean() + final_state.square().mean(), tensors + [initial_state])


def test_reference_default_scale_matches_explicit_scale():
    inputs = _inputs()
    implicit, _ = gated_delta_rule_reference(*inputs)
    explicit, _ = gated_delta_rule_reference(*inputs, scale=DEFAULT_SCALE)
    torch.testing.assert_close(implicit, explicit)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda xs: xs.__setitem__(0, xs[0].float()), "query must have dtype"),
        (lambda xs: xs.__setitem__(3, xs[3].to(torch.bfloat16)), "gate must have dtype"),
        (lambda xs: xs.__setitem__(0, xs[0][..., :64].contiguous()), "head dimensions must be 128"),
        (lambda xs: xs.__setitem__(2, xs[2][:, :, :-1].contiguous()), "sequence lengths must match"),
        (lambda xs: xs.__setitem__(0, xs[0][:, :0]), "head counts must be positive"),
        (lambda xs: (xs.__setitem__(0, xs[0].repeat(1, 2, 1, 1)), xs.__setitem__(1, xs[1].repeat(1, 2, 1, 1))), "Hv must be divisible by Hk"),
    ],
)
def test_input_contract_rejects_invalid_tensors(mutate, message):
    inputs = list(_inputs())
    mutate(inputs)
    with pytest.raises(ValueError, match=message):
        validate_gdn_inputs(*inputs, check_device=False)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda xs: xs.__setitem__(1, torch.randn(2, 1, 64, 128, dtype=torch.bfloat16)), "batch dimensions must match"),
        (lambda xs: xs.__setitem__(3, torch.zeros(1, 1, 63, dtype=torch.float32)), "gate must have shape"),
        (lambda xs: xs.__setitem__(4, torch.zeros(1, 2, 64, dtype=torch.float32)), "beta must have shape"),
        (lambda xs: xs.__setitem__(2, xs[2].transpose(-1, -2)), "value must be contiguous"),
    ],
)
def test_input_contract_rejects_mismatched_shapes_and_strides(mutate, message):
    inputs = list(_inputs())
    mutate(inputs)
    with pytest.raises(ValueError, match=message):
        validate_gdn_inputs(*inputs, check_device=False)


def test_input_contract_rejects_bad_state_dtype():
    inputs = _inputs()
    state = torch.empty(1, 1, 128, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="initial_state must have dtype"):
        validate_gdn_inputs(*inputs, initial_state=state, check_device=False)



@pytest.mark.parametrize("length", [0, 63, 65])
def test_input_contract_rejects_unaligned_lengths(length):
    inputs = _inputs(length=length)
    with pytest.raises(ValueError, match="positive multiple of 64"):
        validate_gdn_inputs(*inputs, check_device=False)


def test_input_contract_rejects_bad_state_and_scale():
    inputs = _inputs()
    with pytest.raises(ValueError, match="initial_state must have shape"):
        validate_gdn_inputs(*inputs, initial_state=torch.empty(1, 1, 64, 128), check_device=False)
    with pytest.raises(TypeError, match="scale must be a Python float"):
        validate_gdn_inputs(*inputs, scale=torch.tensor(DEFAULT_SCALE), check_device=False)
