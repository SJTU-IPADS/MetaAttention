"""Public weighted low-rank execution against authored-order token arithmetic."""

import pytest
import torch

from attn_engine import StateTuple, Target
from tests.unit.test_weighted_lowrank import HEAD, SOURCE, KEY, VALUE, weighted_operator

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


def _case(device, order, dtype, *, length=35, backward=False):
    operator = weighted_operator(order, dtype=dtype)
    sizes = {HEAD: 2, SOURCE: 1, KEY: 64, VALUE: 128}
    generator = torch.Generator(device=device).manual_seed(374)
    inputs = {}
    for spec in operator.algorithm.inputs:
        shape = tuple(
            sizes.get(role, 1 if role.name == "batch" else length)
            for role in spec.roles
        )
        value = torch.randn(shape, device=device, generator=generator) * 0.1
        if spec.name == "decay":
            value = torch.full(shape, 0.95, device=device)
            value[:, :, 3, 0] = 0
        elif spec.name == "weight":
            value[:, :, 7] = 0
        inputs[spec.name] = value.to(spec.dtype).requires_grad_(backward)
    initial = (
        torch.randn(1, 2, 64, 128, device=device, generator=generator) * 0.1
    ).requires_grad_(backward)
    compiled = operator.compile(
        role_sizes=sizes, target=Target(), mode="dense", backward=backward
    )
    executable = operator._plans[compiled.specialization]
    assert executable.plan.candidate.lowerer == "tilelang-factored-recurrence"
    return operator, compiled, inputs, initial


def _oracle(inputs, initial, order):
    state = initial
    heads = state.shape[1]
    x = {
        name: value.repeat_interleave(heads // value.shape[1], dim=1)
        for name, value in inputs.items()
    }
    # Bind each authored expression once for the whole sequence. Moving casts
    # inside the token loop rounds each token's BF16 VJP separately; sharing
    # casts across different nodes also changes gradient accumulation rounding.
    rank_key = x["key"].float()
    prediction = x["prediction"].float()
    write_key = x["key"].float()
    plain_key = x["key"].float()
    # Contract is FP32, while the authored BF16 Multiply rounds before use.
    write_value = (x["value"] * x["write"]).float()
    plain_value = (x["value"] * x["write"]).float()
    overlap = (x["key"].float() * x["prediction"].float()).sum(-1)
    weight = x["weight"].float()
    query = x["query"].float()
    output = []
    for token in range(query.shape[2]):
        factors = {
            -1: write_key[:, :, token, :, None],
            -2: write_value[:, :, token, None, :],
            0: weight[:, :, token, None, None],
            1: overlap[:, :, token, None, None],
        }
        state = state * x["decay"][:, :, token, :, None]
        projection = (state * prediction[:, :, token, :, None]).sum(-2, keepdim=True)
        state = state + (-0.2 * rank_key[:, :, token, :, None]) * projection
        product = factors[order[0]]
        for factor in order[1:]:
            product = product * factors[factor]
        state = state + product
        state = (
            state + plain_key[:, :, token, :, None] * plain_value[:, :, token, None, :]
        )
        output.append((state * query[:, :, token, :, None]).sum(-2))
    return torch.stack(output, dim=2), state


def _invoke(compiled, inputs, initial):
    return compiled(
        **inputs,
        initial_state=StateTuple(("memory",), (initial,)),
        return_final_state=True,
    )


@pytest.mark.parametrize("order", [(-1, -2, 0, 1), (0, -2, 1, -1), (-2, 0, -1, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_weighted_public_output_final_and_continuation(gpu_device, order, dtype):
    _, compiled, inputs, initial = _case(gpu_device, order, dtype)
    actual = _invoke(compiled, inputs, initial)
    expected, final = _oracle(inputs, initial, order)
    torch.testing.assert_close(actual.outputs["output"], expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        actual.final_state["memory"], final, atol=2e-6, rtol=2e-5
    )
    prefix_inputs = {
        name: value[:, :, :13].contiguous() for name, value in inputs.items()
    }
    tail_inputs = {
        name: value[:, :, 13:].contiguous() for name, value in inputs.items()
    }
    prefix = _invoke(compiled, prefix_inputs, initial)
    tail = _invoke(compiled, tail_inputs, prefix.final_state["memory"])
    torch.testing.assert_close(
        tail.outputs["output"], expected[:, :, 13:], atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(tail.final_state["memory"], final, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("loss", ["both", "output", "final"])
def test_weighted_zero_scalar_gradients_and_expression_vjp(gpu_device, loss, dtype):
    order = (0, -2, 1, -1)
    _, compiled, inputs, initial = _case(
        gpu_device, order, dtype, length=19, backward=True
    )
    actual = _invoke(compiled, inputs, initial)
    expected, final = _oracle(inputs, initial, order)
    generator = torch.Generator(device=gpu_device).manual_seed(117)
    dy = torch.randn(expected.shape, device=gpu_device, generator=generator)
    df = torch.randn(final.shape, device=gpu_device, generator=generator)
    actual_loss = (actual.outputs["output"] * dy).sum() if loss != "final" else 0
    expected_loss = (expected * dy).sum() if loss != "final" else 0
    if loss != "output":
        actual_loss = actual_loss + (actual.final_state["memory"] * df).sum()
        expected_loss = expected_loss + (final * df).sum()
    leaves = (*inputs.values(), initial)
    gradients = torch.autograd.grad(actual_loss, leaves)
    reference = torch.autograd.grad(expected_loss, leaves, allow_unused=True)
    for actual_gradient, expected_gradient, leaf in zip(
        gradients, reference, leaves, strict=True
    ):
        if expected_gradient is None:
            expected_gradient = torch.zeros_like(leaf)
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=2e-5 if dtype == torch.float32 else 2e-4,
            rtol=2e-4 if dtype == torch.float32 else 0.02,
        )


@pytest.mark.parametrize("packed", [False, True])
def test_multiple_weighted_terms_long_and_empty_packed_vjp(gpu_device, packed):
    from attn_engine.stateful.lowering.lowrank.checkpoint import forward, backward

    torch.manual_seed(93)
    lengths = (0, 17, 35) if packed else (130, 130)
    tokens, heads, rows, columns = sum(lengths), 2, 7, 19
    offsets = (
        torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device=gpu_device,
            dtype=torch.int64,
        )
        if packed
        else None
    )
    left = torch.full((1, tokens, heads, rows, 1), 0.97, device=gpu_device)
    right = torch.zeros_like(left)
    coefficient = torch.ones(1, tokens, heads, 1, 1, device=gpu_device)
    key = torch.randn(2, tokens, heads, rows, 1, device=gpu_device) * 0.1
    value = torch.randn(2, tokens, heads, 1, columns, device=gpu_device) * 0.1
    query = torch.randn(tokens, heads, rows, 1, device=gpu_device) * 0.1
    initial = torch.randn(len(lengths), heads, rows, columns, device=gpu_device) * 0.1
    scalars = torch.randn(3, tokens, heads, 1, 1, device=gpu_device) * 0.1
    scalars[1, 7] = 0
    orders = ((0, -2, -1, 1), (-1, 2, -2))
    arguments = dict(
        operation_codes=("row-scale",),
        left_factors=left,
        right_factors=right,
        coefficients=coefficient,
        injection_left=key,
        injection_right=value,
        readout=query,
        initial_state=initial,
        injection_scalars=scalars,
        injection_orders=orders,
        sequence_offsets=offsets,
    )
    output, final, checkpoints = forward(
        **arguments, output_dtype=torch.float32, record=True
    )
    # Exercise the 64-token inference schedule as well as recorded replay.
    if not packed:
        inference, inference_final, _ = forward(
            **arguments, output_dtype=torch.float32, record=False
        )
        torch.testing.assert_close(inference, output, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(inference_final, final, atol=2e-6, rtol=2e-5)
    leaves = tuple(
        t.double().detach().requires_grad_()
        for t in (left, right, coefficient, key, value, query, initial, scalars)
    )
    gates, _, _, k, v, q, start, weights = leaves
    outputs, finals = [], []
    begin = 0
    for sequence, length in enumerate(lengths):
        state = start[sequence]
        local = []
        for token in range(begin, begin + length):
            state = state * gates[0, token]
            for index, order in enumerate(orders):
                factors = {-1: k[index, token], -2: v[index, token]}
                factors.update({j: weights[j, token] for j in range(3)})
                product = factors[order[0]]
                for factor in order[1:]:
                    product = product * factors[factor]
                state = state + product
            local.append((state * q[token]).sum(-2))
        if local:
            outputs.append(torch.stack(local, dim=0 if packed else 1))
        finals.append(state)
        begin += length
    expected = torch.cat(outputs, dim=0) if packed else torch.stack(outputs)
    expected_final = torch.stack(finals)
    torch.testing.assert_close(output.double(), expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(final.double(), expected_final, atol=2e-6, rtol=2e-5)
    dy, df = torch.randn_like(output), torch.randn_like(final)
    gradients = backward(
        **arguments, checkpoints=checkpoints, grad_output=dy, grad_final=df
    )
    reference = torch.autograd.grad(
        (expected * dy).sum() + (expected_final * df).sum(), leaves, allow_unused=True
    )
    for actual, expected_gradient, leaf in zip(
        gradients, reference, leaves, strict=True
    ):
        if expected_gradient is None:
            expected_gradient = torch.zeros_like(leaf)
        torch.testing.assert_close(
            actual.double(), expected_gradient, atol=2e-6, rtol=2e-4
        )
    if packed:
        from attn_engine.stateful.lowering.packed import execute_lowrank_packed

        transport = tuple(
            tensor.detach().requires_grad_()
            for tensor in (
                left,
                right,
                coefficient,
                key,
                value,
                query,
                initial,
                scalars,
            )
        )
        gates, right, coefficient, key, value, query, initial, scalars = transport
        (packed_output,), packed_final = execute_lowrank_packed(
            operation_codes=("row-scale",),
            left_factors=list(gates),
            right_factors=list(right),
            coefficients=list(coefficient),
            injection_left=list(key),
            injection_right=list(value),
            readouts=[query],
            initial_state=initial,
            sequence_offsets=offsets,
            output_dtypes=(torch.float32,),
            return_final_state=True,
            injection_scalars=list(scalars),
            injection_orders=orders,
        )
        torch.testing.assert_close(packed_output, output, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(packed_final, final, atol=2e-6, rtol=2e-5)
        transported_gradients = torch.autograd.grad(
            (packed_output * dy).sum() + (packed_final * df).sum(),
            transport,
        )
        for actual, expected_gradient in zip(
            transported_gradients, gradients, strict=True
        ):
            torch.testing.assert_close(actual, expected_gradient, atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("backward", [False, True])
def test_scalar_first_product_avoids_reassociation_overflow(gpu_device, backward):
    from dataclasses import replace
    from attn_engine import (
        AlgorithmIR,
        AxisScale,
        Constant,
        Input,
        ProductFactor,
        ProductInjection,
        RankOnePropagation,
        StatefulOperator,
    )

    algorithm = weighted_operator().algorithm
    injection = ProductInjection(
        (
            ProductFactor(Constant(1e-20), ()),
            ProductFactor(Input("key"), (KEY,)),
            ProductFactor(Input("value"), (VALUE,)),
        )
    )
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=algorithm.inputs,
            states=algorithm.states,
            transition=replace(
                algorithm.transition,
                propagations=(
                    AxisScale(Constant(1.0), ()),
                    RankOnePropagation(KEY, Input("query"), Input("query"), 0.0),
                ),
                injections=(injection,),
            ),
            readouts=algorithm.readouts,
            head_mapping=algorithm.head_mapping,
        )
    )
    sizes = {HEAD: 2, SOURCE: 1, KEY: 64, VALUE: 128}
    inputs = {}
    for spec in algorithm.inputs:
        shape = tuple(sizes.get(role, 1) for role in spec.roles)
        fill = 1e20 if spec.name in ("key", "value") else 1e-20
        inputs[spec.name] = torch.full(
            shape, fill, device=gpu_device, dtype=spec.dtype, requires_grad=backward
        )
    initial = torch.zeros(1, 2, 64, 128, device=gpu_device, requires_grad=backward)
    compiled = operator.compile(
        role_sizes=sizes, target=Target(), mode="dense", backward=backward
    )
    assert (
        operator._plans[compiled.specialization].plan.candidate.lowerer
        == "tilelang-factored-recurrence"
    )
    actual = _invoke(compiled, inputs, initial)
    # Forming the outer product first overflows FP32. The authored scalar-first
    # multiplication and its readout stay finite, making reassociation observable.
    assert torch.isinf(inputs["key"][0, 0, 0, 0] * inputs["value"][0, 0, 0, 0])
    assert torch.isfinite(actual.final_state["memory"]).all()
    torch.testing.assert_close(
        actual.final_state["memory"], torch.full_like(initial, 1e20), atol=0, rtol=1e-6
    )
    torch.testing.assert_close(
        actual.outputs["output"],
        torch.full_like(actual.outputs["output"], 64),
        atol=1e-5,
        rtol=1e-6,
    )
