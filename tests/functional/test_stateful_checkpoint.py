"""Checkpoint recurrence behavior against an independent FP64 autograd oracle."""

from __future__ import annotations

import pytest
import torch

from attn_engine.stateful.lowering.lowrank.checkpoint import (
    CHUNK_SIZE,
    backward,
    forward,
)
from attn_engine.stateful.lowering.lowrank.factored import (
    factored_recurrence_backward,
    factored_recurrence_dense,
)

pytestmark = [
    pytest.mark.functional,
    pytest.mark.gpu,
    pytest.mark.usefixtures("gpu_device", "seed"),
]


def _inputs(device, codes, *, length=35, decay=0.8, rows=7, columns=19):
    batch, heads = 2, 2
    tokens, width = batch * length, max(rows, columns)
    left = torch.randn(len(codes), tokens, heads, width, 1, device=device) * 0.1
    right = torch.randn(len(codes), tokens, heads, rows, 1, device=device) * 0.1
    coefficient = torch.randn(len(codes), tokens, heads, 1, 1, device=device) * 0.1
    for op, code in enumerate(codes):
        if code == "scale":
            coefficient[op].fill_(decay)
            coefficient[op, 9, 0] = 0
        elif code in ("row-scale", "column-scale"):
            left[op].fill_(decay)
            left[op, 21, 1, 2] = 0
    ik = torch.randn(2, tokens, heads, rows, 1, device=device) * 0.1
    iv = torch.randn(2, tokens, heads, 1, columns, device=device) * 0.1
    query = torch.randn(tokens, heads, rows, 1, device=device) * 0.2
    initial = torch.randn(batch, heads, rows, columns, device=device) * 0.2
    return left, right, coefficient, ik, iv, query, initial


def _oracle(codes, inputs):
    left, right, coefficient, ik, iv, query, initial = inputs
    batch, heads, rows, columns = initial.shape
    length = query.shape[0] // batch
    state = initial
    outputs, checkpoints = [], []
    for token in range(length):
        if token % CHUNK_SIZE == 0:
            checkpoints.append(state)
        index = torch.arange(batch, device=query.device) * length + token
        for op, code in enumerate(codes):
            if code == "scale":
                state = state * coefficient[op, index]
            elif code == "row-scale":
                state = state * left[op, index, :, :rows]
            elif code == "column-scale":
                state = state * left[op, index, :, :columns, 0].unsqueeze(-2)
            else:
                projection = (state * right[op, index]).sum(-2, keepdim=True)
                state = (
                    state
                    + coefficient[op, index] * left[op, index, :, :rows] * projection
                )
        for injection in range(ik.shape[0]):
            state = state + ik[injection, index] * iv[injection, index]
        outputs.append((state * query[index]).sum(-2))
    return torch.stack(outputs, dim=2), state, torch.stack(checkpoints, dim=2)


def _arguments(codes, inputs):
    return dict(
        zip(
            (
                "operation_codes",
                "left_factors",
                "right_factors",
                "coefficients",
                "injection_left",
                "injection_right",
                "readout",
                "initial_state",
            ),
            (codes, *inputs),
            strict=True,
        )
    )


@pytest.mark.parametrize("rank_first", [False, True])
@pytest.mark.parametrize("decay", [0.8, 0.001])
@pytest.mark.parametrize("cotangent", ["both", "final", "output"])
def test_checkpoint_vjp_all_operands_match_fp64(
    gpu_device, rank_first, decay, cotangent
):
    codes = (
        ("rank-one", "row-scale", "scale", "column-scale", "rank-one")
        if rank_first
        else ("row-scale", "rank-one", "scale", "column-scale", "rank-one")
    )
    inputs = _inputs(gpu_device, codes, decay=decay)
    arguments = _arguments(codes, inputs)
    output, final, checkpoints = forward(**arguments, output_dtype=torch.float32)
    reference_inputs = tuple(x.double().detach().requires_grad_() for x in inputs)
    expected, expected_final, expected_checkpoints = _oracle(codes, reference_inputs)
    torch.testing.assert_close(output.double(), expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(final.double(), expected_final, atol=2e-6, rtol=2e-5)
    assert checkpoints.dtype == torch.float32
    assert checkpoints.shape == (2, 2, 3, 7, 19)
    torch.testing.assert_close(
        checkpoints.double(), expected_checkpoints, atol=2e-6, rtol=2e-5
    )
    dy, df = torch.randn_like(output), torch.randn_like(final)
    dy[1, 1] *= 2.7
    df[0, 1] *= -1.9
    gradients = dict(
        grad_output=None if cotangent == "final" else dy,
        grad_final=None if cotangent == "output" else df,
    )
    actual_gradients = backward(**arguments, checkpoints=checkpoints, **gradients)
    reference_loss = (
        expected_final * (0 if cotangent == "output" else df.double())
    ).sum()
    if cotangent != "final":
        reference_loss = reference_loss + (expected * dy.double()).sum()
    expected_gradients = torch.autograd.grad(
        reference_loss, reference_inputs, allow_unused=True
    )
    for actual, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        if expected_gradient is None:
            expected_gradient = torch.zeros_like(actual, dtype=torch.float64)
        torch.testing.assert_close(
            actual.double(), expected_gradient, atol=2e-5, rtol=2e-4
        )
    repeated = backward(**arguments, checkpoints=checkpoints, **gradients)
    for actual, repeat in zip(actual_gradients, repeated, strict=True):
        torch.testing.assert_close(actual, repeat, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [7, 32, 64])
def test_inference_chunk_tail_and_continuation_match_fp64(gpu_device, rows):
    codes = ("row-scale", "rank-one", "column-scale", "scale")
    inputs = _inputs(gpu_device, codes, length=130, rows=rows)
    arguments = _arguments(codes, inputs)
    output, final, _ = forward(**arguments, output_dtype=torch.float32, record=False)
    expected, expected_final, _ = _oracle(codes, tuple(x.double() for x in inputs))
    torch.testing.assert_close(output.double(), expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(final.double(), expected_final, atol=2e-6, rtol=2e-5)
    first_inputs = _slice(inputs, 0, 63)
    first, carried, _ = forward(
        **_arguments(codes, first_inputs), output_dtype=torch.float32, record=False
    )
    tail_inputs = _slice(inputs, 63, 130)
    tail, split_final, _ = forward(
        **_arguments(codes, (*tail_inputs[:-1], carried)),
        output_dtype=torch.float32,
        record=False,
    )
    torch.testing.assert_close(
        torch.cat((first, tail), dim=2), output, atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(split_final, final, atol=2e-6, rtol=2e-5)


def _binding(context, gradients):
    return factored_recurrence_backward(
        operation_codes=context["factored.operation_codes"],
        left_factors=context["factored.left"],
        right_factors=context["factored.right"],
        coefficients=context["factored.coefficient"],
        injection_left=context["factored.injection_left"],
        injection_right=context["factored.injection_right"],
        readout=context["factored.readout"],
        recorded=context["factored.recorded"],
        initial_state=context["factored.initial_state"],
        grad_output=gradients[0],
        grad_final=gradients[1],
    )


def _dense(codes, inputs, readouts=None):
    left, right, coefficient, ik, iv, query, initial = inputs
    rows, columns = initial.shape[2:]
    readouts = [query] if readouts is None else readouts
    readout_args = dict(
        readouts=readouts, output_dtypes=(torch.float32,) * len(readouts)
    )
    return factored_recurrence_dense(
        operation_codes=codes,
        left_factors=[
            x[:, :, : columns if code == "column-scale" else rows]
            for x, code in zip(left, codes, strict=True)
        ],
        right_factors=list(right),
        coefficients=list(coefficient),
        injection_left=list(ik),
        injection_right=list(iv),
        initial_state=initial,
        backward=_binding,
        **readout_args,
    )


def _slice(inputs, start, stop):
    *factors, query, initial = inputs
    batch, heads = initial.shape[:2]
    length = query.shape[0] // batch
    sliced = []
    for value in factors:
        operations = value.shape[0]
        shape = (operations, batch, length, *value.shape[2:])
        sliced.append(
            value.reshape(shape)[:, :, start:stop].reshape(
                operations, batch * (stop - start), *value.shape[2:]
            )
        )
    sliced_query = query.reshape(batch, length, *query.shape[1:])[
        :, start:stop
    ].reshape(batch * (stop - start), heads, *query.shape[2:])
    return (*sliced, sliced_query, initial)


def test_checkpoint_non_aligned_continuation_preserves_vjp(gpu_device):
    codes = ("row-scale", "rank-one", "column-scale", "scale")
    inputs = tuple(x.requires_grad_() for x in _inputs(gpu_device, codes, length=35))
    (full,), full_final = _dense(codes, inputs)
    (first,), carried = _dense(codes, _slice(inputs, 0, 13))
    tail_inputs = _slice(inputs, 13, 35)
    (tail,), split_final = _dense(codes, (*tail_inputs[:-1], carried))
    continued = torch.cat((first, tail), dim=2)
    torch.testing.assert_close(continued, full, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(split_final, full_final, atol=2e-6, rtol=2e-5)
    dy, df = torch.randn_like(full), torch.randn_like(full_final)
    full_grad = torch.autograd.grad((full * dy).sum() + (full_final * df).sum(), inputs)
    split_grad = torch.autograd.grad(
        (continued * dy).sum() + (split_final * df).sum(), inputs
    )
    for actual, expected in zip(split_grad, full_grad, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("used", [(0, 1), (1,), ()])
def test_checkpoint_multi_readout_vjp_matches_fp64(gpu_device, used):
    codes = ("rank-one", "row-scale", "column-scale", "rank-one")
    inputs = tuple(x.requires_grad_() for x in _inputs(gpu_device, codes))
    auxiliary = torch.randn_like(inputs[-2], requires_grad=True)
    outputs, final = _dense(codes, inputs, readouts=[inputs[-2], auxiliary])
    reference = tuple(
        x.double().detach().requires_grad_() for x in (*inputs, auxiliary)
    )
    expected, expected_final, _ = _oracle(codes, reference[:7])
    auxiliary_expected, _, _ = _oracle(
        codes, (*reference[:5], reference[7], reference[6])
    )
    for actual, oracle in zip(outputs, (expected, auxiliary_expected), strict=True):
        torch.testing.assert_close(actual.double(), oracle, atol=2e-6, rtol=2e-5)
    dy = [torch.randn_like(output) for output in outputs]
    df = torch.randn_like(final)
    loss = (final * df).sum()
    oracle_loss = (expected_final * df.double()).sum()
    for readout in used:
        loss = loss + (outputs[readout] * dy[readout]).sum()
        oracle_loss = (
            oracle_loss
            + ((expected, auxiliary_expected)[readout] * dy[readout].double()).sum()
        )
    actual_grad = torch.autograd.grad(loss, (*inputs, auxiliary), allow_unused=True)
    expected_grad = torch.autograd.grad(oracle_loss, reference, allow_unused=True)
    for actual, oracle, source in zip(
        actual_grad, expected_grad, (*inputs, auxiliary), strict=True
    ):
        if actual is None:
            actual = torch.zeros_like(source)
        if oracle is None:
            oracle = torch.zeros_like(source, dtype=torch.float64)
        torch.testing.assert_close(actual.double(), oracle, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("rows", [32, 64, 65])
def test_checkpoint_carry_exact_and_partial_reduction_tiles_match_fp64(
    gpu_device, rows
):
    codes = ("row-scale", "rank-one")
    inputs = _inputs(gpu_device, codes, rows=rows)
    arguments = _arguments(codes, inputs)
    output, final, checkpoints = forward(**arguments, output_dtype=torch.float32)
    reference = tuple(x.double().detach().requires_grad_() for x in inputs)
    expected, expected_final, expected_checkpoints = _oracle(codes, reference)
    torch.testing.assert_close(output.double(), expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(final.double(), expected_final, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        checkpoints.double(), expected_checkpoints, atol=2e-6, rtol=2e-5
    )
    dy, df = torch.randn_like(output), torch.randn_like(final)
    gradients = backward(
        **arguments, checkpoints=checkpoints, grad_output=dy, grad_final=df
    )
    expected_gradients = torch.autograd.grad(
        (expected * dy.double()).sum() + (expected_final * df.double()).sum(), reference
    )
    for actual, oracle in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual.double(), oracle, atol=2e-5, rtol=2e-4)
