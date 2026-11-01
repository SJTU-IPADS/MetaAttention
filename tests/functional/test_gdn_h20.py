from __future__ import annotations

import pytest
import torch

from attention_engine.attn_engine.gdn_reference import gated_delta_rule_reference
from attn_engine import StateTuple, Target
from examples.gated_delta_rule import (
    compositional_pressure_profile,
    gated_delta_rule,
)


pytestmark = [
    pytest.mark.functional,
    pytest.mark.gpu,
    pytest.mark.h20,
    pytest.mark.usefixtures("gpu_device", "seed"),
]


def _compile_gdn(operator, *, packed=False, backward=True):
    return operator.compile(
        role_sizes={
            operator.algorithm.inputs[0].roles[1]: 1,
            operator.algorithm.inputs[1].roles[1]: 1,
            operator.algorithm.inputs[2].roles[1]: 2,
            operator.algorithm.states[0].head_role: 2,
            operator.algorithm.states[0].feature_roles[0]: 128,
            operator.algorithm.states[0].feature_roles[1]: 128,
        },
        target=Target(),
        mode="packed" if packed else "dense",
        backward=backward,
    )


def _compile_pressure(operator, *, packed):
    roles = {
        role
        for spec in operator.algorithm.inputs
        for role in spec.roles
        if hasattr(role, "name") and str(role) not in ("batch", "sequence", "one")
    }
    sizes = {}
    for role in roles:
        if role == operator.algorithm.states[0].head_role:
            sizes[role] = 4
        elif role in operator.algorithm.states[0].feature_roles:
            sizes[role] = (
                64 if role == operator.algorithm.states[0].feature_roles[0] else 128
            )
        else:
            sizes[role] = 2
    return operator.compile(
        role_sizes=sizes,
        target=Target(),
        mode="packed" if packed else "dense",
        backward=True,
    )


def _bindings(device: torch.device, *, length: int = 64):
    batch, query_heads, state_heads, dim = 1, 1, 2, 128
    query = torch.randn(
        batch,
        query_heads,
        length,
        dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    key = (
        torch.nn.functional.normalize(torch.randn_like(query), p=2, dim=-1)
        .detach()
        .requires_grad_()
    )
    value = torch.randn(
        batch,
        state_heads,
        length,
        dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    gate = (-torch.rand(batch, state_heads, length, device=device)).requires_grad_()
    beta = torch.rand(batch, state_heads, length, device=device, requires_grad=True)
    return query, key, value, gate, beta


@pytest.mark.parametrize("length", [64, 257])
def test_gdn_generic_profile_matches_independent_reference(gpu_device, length):
    query, key, value, gate, beta = _bindings(gpu_device, length=length)
    initial = torch.randn(
        1, 2, 128, 128, device=gpu_device, dtype=torch.float32, requires_grad=True
    )
    operator = gated_delta_rule(1, 2, length, 128, 128, HQ=1)
    compiled = _compile_gdn(operator)
    with torch.no_grad():
        result = compiled(
            query=query,
            key=key,
            value=value,
            gate=gate,
            beta=beta,
            initial_state=StateTuple(("memory",), (initial,)),
            return_final_state=True,
        )
    expected_output, expected_state = gated_delta_rule_reference(
        query,
        key,
        value,
        gate,
        beta,
        initial_state=initial,
        output_final_state=True,
    )
    assert result.final_state is not None
    assert expected_state is not None
    torch.testing.assert_close(
        result.outputs["output"], expected_output, rtol=0.1, atol=0.1
    )
    torch.testing.assert_close(
        result.final_state["memory"], expected_state, rtol=0.1, atol=0.1
    )


@pytest.mark.parametrize("length", [128, 257])
def test_gdn_continuation_matches_full_invocation(gpu_device, length):
    query, key, value, gate, beta = (
        tensor.detach() for tensor in _bindings(gpu_device, length=length)
    )
    initial = torch.randn(1, 2, 128, 128, device=gpu_device) * 0.1
    initial_before = initial.clone()
    operator = gated_delta_rule(1, 2, length, 128, 128, HQ=1)
    compiled = _compile_gdn(operator)
    prefix_query = query[:, :, :64].contiguous()
    prefix_key = key[:, :, :64].contiguous()
    prefix_value = value[:, :, :64].contiguous()
    prefix_gate = gate[:, :, :64].contiguous()
    prefix_beta = beta[:, :, :64].contiguous()
    prefix_compiled = compiled
    full = compiled(
        query=query,
        key=key,
        value=value,
        gate=gate,
        beta=beta,
        initial_state=StateTuple(("memory",), (initial,)),
        return_final_state=True,
    )
    prefix = prefix_compiled(
        query=prefix_query,
        key=prefix_key,
        value=prefix_value,
        gate=prefix_gate,
        beta=prefix_beta,
        return_final_state=True,
        initial_state=StateTuple(("memory",), (initial,)),
    )
    assert prefix.final_state is not None
    suffix_query = query[:, :, 64:].contiguous()
    suffix_key = key[:, :, 64:].contiguous()
    suffix_value = value[:, :, 64:].contiguous()
    suffix_gate = gate[:, :, 64:].contiguous()
    suffix_beta = beta[:, :, 64:].contiguous()
    suffix_compiled = compiled
    suffix = suffix_compiled(
        query=suffix_query,
        key=suffix_key,
        value=suffix_value,
        gate=suffix_gate,
        beta=suffix_beta,
        initial_state=prefix.final_state,
        return_final_state=True,
    )
    assert full.final_state is not None
    assert suffix.final_state is not None
    torch.testing.assert_close(
        torch.cat((prefix.outputs[0], suffix.outputs[0]), dim=2),
        full.outputs[0],
        rtol=0.1,
        atol=0.1,
    )
    torch.testing.assert_close(
        suffix.final_state[0], full.final_state[0], rtol=0.1, atol=0.1
    )
    torch.testing.assert_close(initial, initial_before, rtol=0, atol=0)


def _pressure_bindings(device: torch.device, lengths: tuple[int, ...]):
    generator = torch.Generator(device=device).manual_seed(29)
    token_count = sum(lengths)

    def sample(heads: int, features: int, dtype=torch.bfloat16):
        return (
            0.05
            * torch.randn(
                token_count,
                heads,
                features,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        ).requires_grad_()

    return {
        "query": sample(2, 64),
        "auxiliary_query": sample(2, 64),
        "left": sample(2, 64),
        "right": sample(2, 64),
        "left_gate": (
            -0.01 * torch.rand(token_count, 4, 64, device=device, generator=generator)
        ).requires_grad_(),
        "right_gate": (
            -0.01 * torch.rand(token_count, 4, 128, device=device, generator=generator)
        ).requires_grad_(),
        "coefficient": (
            0.001 * torch.randn(token_count, 4, device=device, generator=generator)
        ).requires_grad_(),
        "value": sample(2, 128),
    }


def _pressure_reference(bindings, lengths, initial):
    repeated = {
        name: value.repeat_interleave(2, dim=1) if value.shape[1] == 2 else value
        for name, value in bindings.items()
    }
    outputs = []
    auxiliaries = []
    finals = []
    offset = 0
    for sequence, length in enumerate(lengths):
        state = initial[sequence]
        for token in range(offset, offset + length):
            left_gate = repeated["left_gate"][token].float().exp()
            right_gate = repeated["right_gate"][token].float().exp()
            left = repeated["left"][token].float()
            right = repeated["right"][token].float()
            value = repeated["value"][token].float()
            coefficient = repeated["coefficient"][token].float()
            state = left_gate[:, :, None] * state
            contraction = torch.einsum("hd,hdf->hf", right, state)
            state = (
                state
                + coefficient[:, None, None]
                * left[:, :, None]
                * contraction[:, None, :]
            )
            state = state * right_gate[:, None, :]
            state = state + left[:, :, None] * value[:, None, :]
            state = (
                state
                + right[:, :, None] * value[:, None, :] * coefficient[:, None, None]
            )
            outputs.append(
                torch.einsum("hd,hdf->hf", repeated["query"][token].float(), state).to(
                    torch.bfloat16
                )
            )
            auxiliaries.append(
                torch.einsum(
                    "hd,hdf->hf",
                    repeated["auxiliary_query"][token].float(),
                    state,
                ).to(torch.bfloat16)
            )
        finals.append(state)
        offset += length
    output_shape = (0, 4, 128)
    return (
        torch.stack(outputs)
        if outputs
        else initial.new_empty(output_shape).to(torch.bfloat16),
        torch.stack(auxiliaries)
        if auxiliaries
        else initial.new_empty(output_shape).to(torch.bfloat16),
        torch.stack(finals),
    )


def test_compositional_pressure_profile_packed_forward_backward(gpu_device):
    lengths = (0, 65, 0, 64)
    offsets = torch.tensor((0, 0, 65, 65, 129), device=gpu_device, dtype=torch.int32)
    actual_bindings = _pressure_bindings(gpu_device, lengths)
    reference_bindings = {
        name: value.detach().clone().requires_grad_()
        for name, value in actual_bindings.items()
    }
    actual_initial = (
        0.05 * torch.randn(4, 4, 64, 128, device=gpu_device, dtype=torch.float32)
    ).requires_grad_()
    reference_initial = actual_initial.detach().clone().requires_grad_()
    operator = compositional_pressure_profile()
    compiled = _compile_pressure(operator, packed=True)
    result = compiled(
        **actual_bindings,
        sequence_offsets=offsets,
        initial_state=StateTuple(("memory",), (actual_initial,)),
        return_final_state=True,
    )
    expected_output, expected_auxiliary, expected_state = _pressure_reference(
        reference_bindings, lengths, reference_initial
    )
    assert result.final_state is not None
    torch.testing.assert_close(
        result.outputs["output"], expected_output, rtol=0.04, atol=0.1
    )
    torch.testing.assert_close(
        result.outputs["auxiliary"], expected_auxiliary, rtol=0.04, atol=0.1
    )
    torch.testing.assert_close(
        result.final_state[0], expected_state, rtol=0.04, atol=0.1
    )
    assert torch.equal(result.final_state[0][0], actual_initial[0])
    assert torch.equal(result.final_state[0][2], actual_initial[2])

    actual_loss = sum(output.float().square().mean() for output in result.outputs)
    actual_loss = actual_loss + result.final_state[0].square().mean()
    expected_loss = expected_output.float().square().mean()
    expected_loss = expected_loss + expected_auxiliary.float().square().mean()
    expected_loss = expected_loss + expected_state.square().mean()
    actual_gradients = torch.autograd.grad(
        actual_loss, (*actual_bindings.values(), actual_initial)
    )
    expected_gradients = torch.autograd.grad(
        expected_loss, (*reference_bindings.values(), reference_initial)
    )
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.08, atol=0.2)


def test_compositional_pressure_profile_dense_unaligned_tail(gpu_device):
    lengths = (129,)
    packed = _pressure_bindings(gpu_device, lengths)
    dense = {
        name: value.transpose(0, 1).unsqueeze(0).contiguous()
        for name, value in packed.items()
    }
    initial = 0.05 * torch.randn(1, 4, 64, 128, device=gpu_device)
    operator = compositional_pressure_profile()
    compiled = _compile_pressure(operator, packed=False)
    result = compiled(
        **dense,
        initial_state=StateTuple(("memory",), (initial,)),
        return_final_state=True,
    )
    expected_output, expected_auxiliary, expected_state = _pressure_reference(
        packed, lengths, initial
    )
    assert result.final_state is not None
    torch.testing.assert_close(
        result.outputs["output"],
        expected_output.permute(1, 0, 2).unsqueeze(0),
        rtol=0.04,
        atol=0.1,
    )
    torch.testing.assert_close(
        result.outputs["auxiliary"],
        expected_auxiliary.permute(1, 0, 2).unsqueeze(0),
        rtol=0.04,
        atol=0.1,
    )
    torch.testing.assert_close(
        result.final_state[0], expected_state, rtol=0.04, atol=0.1
    )


def test_compositional_pressure_profile_all_empty(gpu_device):
    operator = compositional_pressure_profile()
    bindings = _pressure_bindings(gpu_device, (0, 0))
    initial = torch.randn(2, 4, 64, 128, device=gpu_device)
    offsets = torch.tensor((0, 0, 0), device=gpu_device, dtype=torch.int32)
    compiled = _compile_pressure(operator, packed=True)
    result = compiled(
        **bindings,
        sequence_offsets=offsets,
        initial_state=StateTuple(("memory",), (initial,)),
        return_final_state=True,
    )
    assert result.outputs["output"].shape == (0, 4, 128)
    assert result.final_state is not None
    assert torch.equal(result.final_state[0], initial)


def test_gdn_factored_recurrence_matches_independent_reference(gpu_device):
    """FP32 affine carries and chunk replay reproduce the GDN recurrence."""
    query, key, value, gate, beta = (
        tensor.detach() for tensor in _bindings(gpu_device, length=130)
    )
    initial = torch.randn(1, 2, 128, 128, device=gpu_device, dtype=torch.float32)
    operator = gated_delta_rule(1, 2, 130, 128, 128, HQ=1)
    compiled = _compile_gdn(operator, backward=False)
    plan = compiled.operator._plans[compiled.specialization]
    assert plan.plan.candidate.lowerer == "tilelang-factored-recurrence"
    assert plan.rank_one_factored is not None

    with torch.no_grad():
        result = compiled(
            query=query,
            key=key,
            value=value,
            gate=gate,
            beta=beta,
            initial_state=StateTuple(("memory",), (initial,)),
            return_final_state=True,
        )
    expected_output, expected_state = gated_delta_rule_reference(
        query,
        key,
        value,
        gate,
        beta,
        initial_state=initial,
        output_final_state=True,
    )
    assert result.final_state is not None
    assert expected_state is not None
    torch.testing.assert_close(
        result.outputs["output"], expected_output, rtol=0.1, atol=0.1
    )
    torch.testing.assert_close(
        result.final_state["memory"], expected_state, rtol=0.1, atol=0.1
    )


def test_gdn_factored_recurrence_continues_from_a_returned_state(gpu_device):
    """A factored result must be usable as the next invocation's initial State."""
    query, key, value, gate, beta = _bindings(gpu_device, length=128)
    operator = gated_delta_rule(1, 2, 128, 128, 128, HQ=1)
    compiled = _compile_gdn(operator, backward=False)
    with torch.no_grad():
        full = compiled(
            query=query,
            key=key,
            value=value,
            gate=gate,
            beta=beta,
            return_final_state=True,
        )
        prefix = compiled(
            query=query[:, :, :64].contiguous(),
            key=key[:, :, :64].contiguous(),
            value=value[:, :, :64].contiguous(),
            gate=gate[:, :, :64].contiguous(),
            beta=beta[:, :, :64].contiguous(),
            return_final_state=True,
        )
        assert prefix.final_state is not None
        assert full.final_state is not None
        suffix = compiled(
            query=query[:, :, 64:].contiguous(),
            key=key[:, :, 64:].contiguous(),
            value=value[:, :, 64:].contiguous(),
            gate=gate[:, :, 64:].contiguous(),
            beta=beta[:, :, 64:].contiguous(),
            initial_state=prefix.final_state,
            return_final_state=True,
        )
    assert suffix.final_state is not None
    torch.testing.assert_close(
        torch.cat((prefix.outputs[0], suffix.outputs[0]), dim=2),
        full.outputs[0],
        rtol=0.1,
        atol=0.1,
    )
    torch.testing.assert_close(
        suffix.final_state[0], full.final_state[0], rtol=0.1, atol=0.1
    )


def test_gdn_factored_recurrence_supports_non_square_state(gpu_device):
    """A State whose Feature sizes differ must lower and match the reference.

    The kernel stages one State column per thread, so the column axis and the
    contracting axis are different sizes here.
    """
    rows, columns, heads, length = 64, 128, 2, 70
    generator = torch.Generator(device=gpu_device).manual_seed(7)
    query = torch.randn(
        1, 1, length, rows, device=gpu_device, dtype=torch.bfloat16, generator=generator
    )
    key = torch.nn.functional.normalize(
        torch.randn(
            1,
            1,
            length,
            rows,
            device=gpu_device,
            dtype=torch.bfloat16,
            generator=generator,
        ),
        p=2,
        dim=-1,
    )
    value = torch.randn(
        1,
        heads,
        length,
        columns,
        device=gpu_device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    gate = -torch.rand(1, heads, length, device=gpu_device, generator=generator)
    beta = torch.rand(1, heads, length, device=gpu_device, generator=generator)
    initial = 0.1 * torch.randn(1, heads, rows, columns, device=gpu_device)
    operator = gated_delta_rule(1, heads, length, rows, columns, HQ=1)
    compiled = operator.compile(
        role_sizes={
            operator.algorithm.inputs[0].roles[1]: 1,
            operator.algorithm.inputs[1].roles[1]: 1,
            operator.algorithm.inputs[2].roles[1]: heads,
            operator.algorithm.states[0].head_role: heads,
            operator.algorithm.states[0].feature_roles[0]: rows,
            operator.algorithm.states[0].feature_roles[1]: columns,
        },
        target=Target(),
        mode="dense",
        backward=False,
    )
    assert (
        compiled.operator._plans[compiled.specialization].plan.candidate.lowerer
        == "tilelang-factored-recurrence"
    )
    with torch.no_grad():
        result = compiled(
            query=query,
            key=key,
            value=value,
            gate=gate,
            beta=beta,
            initial_state=StateTuple(("memory",), (initial,)),
            return_final_state=True,
        )
    # the oracle defaults to the 128-dim acceptance-profile scale; this Program
    # scales the readout by its own key dimension
    expected_output, expected_state = gated_delta_rule_reference(
        query,
        key,
        value,
        gate,
        beta,
        scale=rows**-0.5,
        initial_state=initial,
        output_final_state=True,
    )
    assert result.final_state is not None
    torch.testing.assert_close(
        result.outputs["output"], expected_output, rtol=0.1, atol=0.1
    )
    torch.testing.assert_close(
        result.final_state["memory"], expected_state, rtol=0.1, atol=0.1
    )


def test_gdn_factored_recurrence_gradients_match_independent_reference(gpu_device):
    """The factored reverse walk must match the FP64 oracle's gradients.

    Covers the rank-one adjoint, the per-token factor gradients, the injection
    gradients, the readout operand, and the initial State.
    """
    query, key, value, gate, beta = _bindings(gpu_device, length=130)
    initial = torch.randn(
        1, 2, 128, 128, device=gpu_device, dtype=torch.float32, requires_grad=True
    )
    operator = gated_delta_rule(1, 2, 130, 128, 128, HQ=1)
    compiled = _compile_gdn(operator)
    assert (
        compiled.operator._plans[compiled.specialization].plan.candidate.lowerer
        == "tilelang-factored-recurrence"
    )

    result = compiled(
        query=query,
        key=key,
        value=value,
        gate=gate,
        beta=beta,
        initial_state=StateTuple(("memory",), (initial,)),
        return_final_state=True,
    )
    assert result.final_state is not None

    actual_inputs = (query, key, value, gate, beta, initial)
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in actual_inputs
    )
    expected_output, expected_state = gated_delta_rule_reference(
        reference_inputs[0],
        reference_inputs[1],
        reference_inputs[2],
        reference_inputs[3],
        reference_inputs[4],
        initial_state=reference_inputs[5],
        output_final_state=True,
    )
    assert expected_state is not None

    actual_loss = result.outputs["output"].float().square().mean()
    actual_loss = actual_loss + result.final_state["memory"].square().mean()
    expected_loss = expected_output.float().square().mean()
    expected_loss = expected_loss + expected_state.square().mean()
    actual_gradients = torch.autograd.grad(actual_loss, actual_inputs)
    expected_gradients = torch.autograd.grad(expected_loss, reference_inputs)

    for name, actual_gradient, expected_gradient in zip(
        ("query", "key", "value", "gate", "beta", "initial_state"),
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        scale = expected_gradient.abs().max().item()
        assert scale > 0.0, name
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=0.08, atol=0.08 * scale + 1e-6
        )


@pytest.mark.parametrize("rank_first", [False, True])
def test_ordered_lowrank_chunks_preserve_order_and_continuation(gpu_device, rank_first):
    """Noncommuting row decay and rank-one updates retain authored order."""
    from dataclasses import replace
    from attn_engine import AlgorithmIR, StatefulOperator
    from testing.benchmark_stateful_plans import CASES, inputs_for, role_sizes_for

    case = replace(
        next(case for case in CASES if case.name == "pressure"), batch=2, heads=2
    )
    length = 67
    operator = case.factory(**case.kwargs(length))
    transition = operator.algorithm.transition
    if rank_first:
        row, rank, column = transition.propagations
        transition = replace(transition, propagations=(rank, row, column))
        operator = StatefulOperator(
            AlgorithmIR(
                inputs=operator.algorithm.inputs,
                states=operator.algorithm.states,
                transition=transition,
                readouts=operator.algorithm.readouts,
                head_mapping=operator.algorithm.head_mapping,
            )
        )
    sizes = role_sizes_for(operator, case, length)
    compiled = operator.compile(role_sizes=sizes, target=Target(), mode="dense")
    bindings = inputs_for(operator, case, length, gpu_device)
    initial = torch.randn(2, 2, 64, 64, device=gpu_device) * 0.1
    state = initial.clone()
    expected = {"output": [], "auxiliary": []}
    for token in range(length):
        left = bindings["left"][:, :, token].float()
        right = bindings["right"][:, :, token].float()
        coefficient = bindings["coefficient"][:, :, token, None, None]
        row_decay = bindings["left_gate"][:, :, token].exp().unsqueeze(-1)
        if not rank_first:
            state = state * row_decay
        state = state + coefficient * left.unsqueeze(-1) * (
            state * right.unsqueeze(-1)
        ).sum(-2, keepdim=True)
        if rank_first:
            state = state * row_decay
        state = state * bindings["right_gate"][:, :, token].exp().unsqueeze(-2)
        value = bindings["value"][:, :, token].float()
        state = state + left.unsqueeze(-1) * value.unsqueeze(-2)
        state = state + right.unsqueeze(-1) * (value * coefficient[..., 0]).unsqueeze(
            -2
        )
        for name, query in (("output", "query"), ("auxiliary", "auxiliary_query")):
            expected[name].append(
                (state * bindings[query][:, :, token].float().unsqueeze(-1)).sum(-2)
            )
    result = compiled(
        **bindings,
        initial_state=StateTuple(("memory",), (initial,)),
        return_final_state=True,
    )
    prefix = compiled(
        **{name: tensor[:, :, :65].contiguous() for name, tensor in bindings.items()},
        initial_state=StateTuple(("memory",), (initial,)),
        return_final_state=True,
    )
    suffix = compiled(
        **{name: tensor[:, :, 65:].contiguous() for name, tensor in bindings.items()},
        initial_state=prefix.final_state,
        return_final_state=True,
    )
    for name, reference in expected.items():
        reference = torch.stack(reference, 2)
        torch.testing.assert_close(
            result.outputs[name].float(), reference, rtol=0.02, atol=0.02
        )
        torch.testing.assert_close(
            torch.cat((prefix.outputs[name], suffix.outputs[name]), 2).float(),
            reference,
            rtol=0.02,
            atol=0.02,
        )
    torch.testing.assert_close(
        result.final_state["memory"], state, rtol=0.002, atol=0.002
    )
    torch.testing.assert_close(
        suffix.final_state["memory"], state, rtol=0.002, atol=0.002
    )
