"""Two-Axis Scale Programs compose one elementwise factor per axis.

A Transition that decays the key axis and the value axis of one matrix State has
no single per-element factor, but it does not need the dense ``[T,H,K,K]`` and
``[T,H,V,V]`` summaries either: the prefix kernel takes one factor per axis and
composes them per element. These tests prove the pair matches an FP32 token
recurrence in dense and packed modes, and that a compilation which asks for a
backward keeps the dense summary whose VJP owns those factors.
"""

from __future__ import annotations

import pytest
import torch

from attn_engine import StateTuple, Target
from examples.linear_attention import dual_gate_gla_attention

pytestmark = [pytest.mark.functional, pytest.mark.gpu, pytest.mark.slow]


def _compile(
    operator, heads: int, dim: int, *, mode: str = "dense", backward: bool = False
):
    state = operator.algorithm.states[0]
    return operator.compile(
        role_sizes={
            operator.algorithm.inputs[0].roles[1]: heads,
            operator.algorithm.inputs[1].roles[1]: heads,
            operator.algorithm.inputs[2].roles[1]: heads,
            state.head_role: heads,
            state.feature_roles[0]: dim,
            state.feature_roles[1]: dim,
        },
        target=Target(),
        mode=mode,
        optimized=True,
        backward=backward,
    )


def _bindings(device, lengths: tuple[int, ...], heads: int, dim: int):
    """Token-major packed inputs for one or more logical sequences."""
    generator = torch.Generator(device=device).manual_seed(11)
    total = sum(lengths)
    query = torch.randn(
        total, heads, dim, device=device, dtype=torch.bfloat16, generator=generator
    )
    key = torch.randn(
        total, heads, dim, device=device, dtype=torch.bfloat16, generator=generator
    )
    value = torch.randn(
        total, heads, dim, device=device, dtype=torch.bfloat16, generator=generator
    )
    key_gate = -torch.rand(total, heads, dim, device=device, generator=generator)
    value_gate = -torch.rand(total, heads, dim, device=device, generator=generator)
    return {
        "query": query,
        "key": key,
        "value": value,
        "key_gate": key_gate,
        "value_gate": value_gate,
    }


def _reference(bindings, lengths, initial, scale):
    """FP32 State recurrence with the factory's input-dtype expressions."""
    write_key = (bindings["key"] + 0.1).float()
    read_query = (bindings["query"] * scale).float()
    outputs = []
    states = []
    offset = 0
    for sequence, length in enumerate(lengths):
        state = initial[sequence].float().clone()
        for token in range(offset, offset + length):
            decay = bindings["key_gate"][token].float().exp().unsqueeze(-1) * bindings[
                "value_gate"
            ][token].float().exp().unsqueeze(-2)
            state = state * decay + torch.einsum(
                "hk,hv->hkv",
                write_key[token],
                bindings["value"][token].float(),
            )
            outputs.append(
                torch.einsum("hk,hkv->hv", read_query[token], state).to(torch.bfloat16)
            )
        states.append(state)
        offset += length
    stacked = torch.stack(states)
    output = (
        torch.stack(outputs)
        if outputs
        else torch.empty(
            0, states[0].shape[0], states[0].shape[1], device=initial.device
        )
    )
    return output, stacked


def test_two_axis_gla_dense_matches_the_token_recurrence(gpu_device):
    heads, dim, length = 2, 64, 70
    bindings = _bindings(gpu_device, (length,), heads, dim)
    dense = {
        name: value.transpose(0, 1).unsqueeze(0).contiguous()
        for name, value in bindings.items()
    }
    initial = 0.1 * torch.randn(
        1, heads, dim, dim, device=gpu_device, dtype=torch.float32
    )
    operator = dual_gate_gla_attention(1, heads, length, dim, dim)
    compiled = _compile(operator, heads, dim)

    with torch.no_grad():
        result = compiled(
            **dense,
            initial_state=StateTuple(("memory",), (initial,)),
            return_final_state=True,
        )
    expected_output, expected_state = _reference(
        bindings, (length,), initial, dim**-0.5
    )
    assert result.final_state is not None
    torch.testing.assert_close(
        result.outputs["output"],
        expected_output.permute(1, 0, 2).unsqueeze(0),
        rtol=0.04,
        atol=0.05,
    )
    torch.testing.assert_close(
        result.final_state["memory"], expected_state, rtol=0.04, atol=0.05
    )


def test_two_axis_gla_packed_isolates_sequences_and_empty_runs(gpu_device):
    heads, dim = 2, 64
    lengths = (0, 33, 31, 0)
    offsets = torch.tensor((0, 0, 33, 64, 64), device=gpu_device, dtype=torch.int32)
    bindings = _bindings(gpu_device, lengths, heads, dim)
    initial = 0.1 * torch.randn(
        len(lengths), heads, dim, dim, device=gpu_device, dtype=torch.float32
    )
    operator = dual_gate_gla_attention(1, heads, sum(lengths), dim, dim)
    compiled = _compile(operator, heads, dim, mode="packed")
    assert compiled.operator._plans[compiled.specialization].dense_summary is False

    with torch.no_grad():
        result = compiled(
            **bindings,
            sequence_offsets=offsets,
            initial_state=StateTuple(("memory",), (initial,)),
            return_final_state=True,
        )
    # the leading empty sequence resets the carry before the first real token
    main_lengths = (0, 33, 31, 0)
    expected_output, expected_state = _reference(
        bindings, main_lengths, initial, dim**-0.5
    )
    assert result.final_state is not None
    torch.testing.assert_close(
        result.outputs["output"], expected_output, rtol=0.04, atol=0.05
    )
    torch.testing.assert_close(
        result.final_state["memory"], expected_state, rtol=0.04, atol=0.05
    )
    assert torch.equal(result.final_state["memory"][0], initial[0])
    assert torch.equal(result.final_state["memory"][3], initial[3])


def test_two_axis_gradients_match_the_token_recurrence(gpu_device):
    """The paired VJP contracts each factor along the axis the other spans."""
    heads, dim, length = 2, 64, 33
    bindings = _bindings(gpu_device, (length,), heads, dim)
    dense = {
        name: value.transpose(0, 1).unsqueeze(0).contiguous().requires_grad_()
        for name, value in bindings.items()
    }
    initial = (
        0.1 * torch.randn(1, heads, dim, dim, device=gpu_device, dtype=torch.float32)
    ).requires_grad_()
    operator = dual_gate_gla_attention(1, heads, length, dim, dim)
    compiled = _compile(operator, heads, dim, backward=True)
    assert compiled.operator._plans[compiled.specialization].dense_summary is False

    result = compiled(
        **dense,
        initial_state=StateTuple(("memory",), (initial,)),
        return_final_state=True,
    )
    assert result.final_state is not None
    loss = result.outputs["output"].float().square().mean()
    loss = loss + result.final_state["memory"].float().square().mean()
    gradients = torch.autograd.grad(loss, (*dense.values(), initial))

    reference_initial = initial.detach().clone().requires_grad_()
    token_major = {
        name: value.detach().squeeze(0).transpose(0, 1).float().requires_grad_()
        for name, value in dense.items()
    }
    expected_output, expected_state = _reference(
        token_major, (length,), reference_initial, dim**-0.5
    )
    expected_loss = expected_output.float().square().mean()
    expected_loss = expected_loss + expected_state.square().mean()
    expected = torch.autograd.grad(
        expected_loss,
        (*token_major.values(), reference_initial),
    )
    for index, (actual, want) in enumerate(zip(gradients, expected, strict=True)):
        assert actual is not None
        if index < len(dense):
            # the oracle works token-major; the compiled call is head-first
            want = want.transpose(0, 1).unsqueeze(0)
        # the compiled call differentiates bf16 inputs and the oracle fp32 ones
        torch.testing.assert_close(actual.float(), want.float(), rtol=0.08, atol=0.05)
