from __future__ import annotations

import pytest
import torch

from attn_engine import StateTuple, Target
from examples.linear_attention import (
    gla_attention,
    linear_attention,
    simple_gla_attention,
)


pytestmark = [pytest.mark.functional, pytest.mark.gpu, pytest.mark.slow]


def _compile(operator, heads: int, dim: int, dim_value: int, *, backward: bool = False):
    roles = {
        role.name: role
        for spec in (*operator.algorithm.inputs, *operator.algorithm.states)
        for role in spec.roles
        if hasattr(role, "name")
    }
    return operator.compile(
        role_sizes={
            roles["query_heads"]: heads,
            roles["key_heads"]: heads,
            roles["value_heads"]: heads,
            roles["key_dim"]: dim,
            roles["value_dim"]: dim_value,
        },
        target=Target(),
        mode="dense",
        optimized=True,
        backward=backward,
    )


def _reference(query, key, value, gate=None, initial_state=None):
    state = (
        torch.zeros(
            query.shape[0],
            query.shape[1],
            query.shape[-1],
            value.shape[-1],
            device=query.device,
            dtype=torch.float32,
        )
        if initial_state is None
        else initial_state.float().clone()
    )
    # Pointwise DSL nodes round in the input dtype before State algebra
    # promotes their values to FP32. Keep this independent oracle faithful to
    # that order, including BF16 addition and query scaling.
    write_key = (key + 0.1).float()
    read_query = (query * query.shape[-1] ** -0.5).float()
    outputs = []
    for index in range(query.shape[2]):
        if gate is not None:
            if gate.ndim == 3:
                state = state * gate[:, :, index, None, None].float().exp()
            else:
                state = state * gate[:, :, index, :, None].float().exp()
        state = state + torch.einsum(
            "bhk,bhv->bhkv",
            write_key[:, :, index],
            value[:, :, index].float(),
        )
        outputs.append(
            torch.einsum(
                "bhk,bhkv->bhv",
                read_query[:, :, index],
                state,
            )
        )
    return torch.stack(outputs, dim=2).to(query.dtype), state


def _log_gate(gate_kind, batch, heads, seqlen, dim, device):
    if gate_kind == "scalar":
        gate = torch.randn(batch, heads, seqlen, device=device, dtype=torch.float32)
        gate = gate.clamp_max(0)
        # Exact zero plateaus exercise the exp(0) boundary at chunk edges and tails.
        gate[:, :, :2] = 0
        gate[:, :, seqlen // 2 - 1 : seqlen // 2 + 2] = 0
        gate[:, :, -2:] = 0
        return gate
    gate = torch.randn(batch, heads, seqlen, dim, device=device, dtype=torch.float32)
    return gate.clamp_max(0)


@pytest.mark.parametrize(
    ("factory", "gate_kind", "seqlen"),
    [
        (linear_attention, None, 65),
        (simple_gla_attention, "scalar", 129),
        (gla_attention, "feature", 65),
    ],
    ids=["linear-65", "simple-gla-129-zero-plateaus", "gla-65"],
)
def test_linear_attention_dsl_matches_fp32_recurrence(
    factory, gate_kind, seqlen, gpu_device, seed
):
    batch, heads, dim, dim_value = 2, 3, 64, 64
    dtype = torch.bfloat16
    compiled = _compile(
        factory(batch, heads, seqlen, dim, dim_value, dtype=dtype),
        heads,
        dim,
        dim_value,
    )
    query = torch.randn(batch, heads, seqlen, dim, device=gpu_device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn(batch, heads, seqlen, dim_value, device=gpu_device, dtype=dtype)
    initial = torch.randn(
        batch, heads, dim, dim_value, device=gpu_device, dtype=torch.float32
    )
    inputs = {"query": query, "key": key, "value": value}
    gate = None
    if gate_kind is not None:
        gate = _log_gate(gate_kind, batch, heads, seqlen, dim, gpu_device)
        inputs["gate"] = gate

    result = compiled(
        **inputs,
        initial_state=StateTuple(("memory",), (initial,)),
        return_final_state=True,
    )
    expected, expected_final = _reference(query, key, value, gate, initial)
    assert result.final_state is not None
    actual = result.outputs["output"]
    assert actual.shape == (batch, heads, seqlen, dim_value)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.05)
    torch.testing.assert_close(
        result.final_state["memory"], expected_final, rtol=0.01, atol=0.02
    )

    # A returned state must be a valid input for continuation, not merely a
    # diagnostic. This also checks the tail after a non-chunk-aligned prefix.
    if gate_kind == "scalar":
        split = 64
        prefix_inputs = {
            name: tensor[:, :, :split].contiguous() for name, tensor in inputs.items()
        }
        suffix_inputs = {
            name: tensor[:, :, split:].contiguous() for name, tensor in inputs.items()
        }
        prefix = compiled(
            **prefix_inputs,
            initial_state=StateTuple(("memory",), (initial,)),
            return_final_state=True,
        )
        assert prefix.final_state is not None
        suffix = compiled(
            **suffix_inputs,
            initial_state=prefix.final_state,
            return_final_state=True,
        )
        assert suffix.final_state is not None
        torch.testing.assert_close(
            torch.cat((prefix.outputs["output"], suffix.outputs["output"]), dim=2),
            actual,
            rtol=0.02,
            atol=0.05,
        )
        torch.testing.assert_close(
            suffix.final_state["memory"],
            result.final_state["memory"],
            rtol=0.01,
            atol=0.02,
        )


@pytest.mark.parametrize(
    "factory,gate_kind,seqlen,dim,dim_value",
    [
        (gla_attention, "feature", 128, 64, 64),
        (gla_attention, "feature", 192, 128, 128),
        (simple_gla_attention, "scalar", 128, 64, 64),
        (simple_gla_attention, "scalar", 192, 128, 64),
        (simple_gla_attention, "scalar", 2048, 64, 64),
    ],
)
def test_gated_chunked_state_continuation(
    factory, gate_kind, seqlen, dim, dim_value, gpu_device, seed
):
    batch, heads = 1, 2
    dtype = torch.bfloat16
    compiled = _compile(
        factory(batch, heads, seqlen, dim, dim_value, dtype=dtype),
        heads,
        dim,
        dim_value,
    )
    query = torch.randn(batch, heads, seqlen, dim, device=gpu_device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn(batch, heads, seqlen, dim_value, device=gpu_device, dtype=dtype)
    gate_shape = (
        (batch, heads, seqlen) if gate_kind == "scalar" else (batch, heads, seqlen, dim)
    )
    gate = torch.randn(*gate_shape, device=gpu_device).clamp(-0.05, 0)
    gate[:, :, 63:66] = 0
    gate[:, :, 127:130] = 0
    initial = torch.randn(batch, heads, dim, dim_value, device=gpu_device)
    inputs = {"query": query, "key": key, "value": value, "gate": gate}
    state = StateTuple(("memory",), (initial,))
    result = compiled(**inputs, initial_state=state, return_final_state=True)
    expected, expected_final = _reference(query, key, value, gate, initial)
    torch.testing.assert_close(result.outputs["output"], expected, rtol=0.02, atol=0.05)
    assert result.final_state is not None
    torch.testing.assert_close(
        result.final_state["memory"], expected_final, rtol=0.01, atol=0.02
    )
    split = 64
    prefix = compiled(
        **{name: tensor[:, :, :split].contiguous() for name, tensor in inputs.items()},
        initial_state=state,
        return_final_state=True,
    )
    assert prefix.final_state is not None
    suffix = compiled(
        **{name: tensor[:, :, split:].contiguous() for name, tensor in inputs.items()},
        initial_state=prefix.final_state,
        return_final_state=True,
    )
    assert suffix.final_state is not None
    torch.testing.assert_close(
        torch.cat((prefix.outputs["output"], suffix.outputs["output"]), dim=2),
        expected,
        rtol=0.02,
        atol=0.05,
    )
    torch.testing.assert_close(
        suffix.final_state["memory"], expected_final, rtol=0.01, atol=0.02
    )


@pytest.mark.parametrize(
    ("factory", "gate_kind"),
    [
        (linear_attention, None),
        (simple_gla_attention, "scalar"),
        (gla_attention, "feature"),
    ],
    ids=["linear", "simple-gla", "gla"],
)
def test_linear_attention_dsl_gradients_match_fp32_recurrence(
    factory, gate_kind, gpu_device, seed
):
    """The compiled elementwise VJP must match the recurrence's own gradients.

    A per-axis (feature-wise) gate lowers to a Broadcast Axis Scale, so this case
    checks the VJP that reduces a ``[T,H,K,1]`` scale gradient, not just the
    per-token scalar one.
    """
    batch, heads, seqlen, dim, dim_value = 2, 3, 130, 64, 64
    dtype = torch.bfloat16
    compiled = _compile(
        factory(batch, heads, seqlen, dim, dim_value, dtype=dtype),
        heads,
        dim,
        dim_value,
        backward=True,
    )
    actual = {
        "query": torch.randn(batch, heads, seqlen, dim, device=gpu_device, dtype=dtype),
        "key": torch.randn(batch, heads, seqlen, dim, device=gpu_device, dtype=dtype),
        "value": torch.randn(
            batch, heads, seqlen, dim_value, device=gpu_device, dtype=dtype
        ),
    }
    if gate_kind is not None:
        actual["gate"] = _log_gate(gate_kind, batch, heads, seqlen, dim, gpu_device)
    actual = {
        name: tensor.detach().requires_grad_(True) for name, tensor in actual.items()
    }
    expected = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in actual.items()
    }

    result = compiled(**actual)
    reference = _reference(
        expected["query"], expected["key"], expected["value"], expected.get("gate")
    )[0]
    torch.testing.assert_close(
        result.outputs["output"], reference, rtol=0.02, atol=0.05
    )

    actual_gradients = torch.autograd.grad(
        result.outputs["output"].float().square().mean(), tuple(actual.values())
    )
    expected_gradients = torch.autograd.grad(
        reference.square().mean(), tuple(expected.values())
    )
    for name, actual_gradient, expected_gradient in zip(
        actual, actual_gradients, expected_gradients, strict=True
    ):
        scale = expected_gradient.abs().max().item()
        assert scale > 0.0, name
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=0.08, atol=0.05 * scale + 1e-5
        )
