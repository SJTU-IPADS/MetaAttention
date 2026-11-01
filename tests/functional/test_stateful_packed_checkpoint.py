"""Packed checkpoint outputs and all VJPs against independent FP64 recurrence."""

from __future__ import annotations

import pytest
import torch

from attn_engine.stateful.lowering.packed import execute_lowrank_packed

pytestmark = [
    pytest.mark.functional,
    pytest.mark.gpu,
    pytest.mark.usefixtures("gpu_device", "seed"),
]

CODES = ("rank-one", "row-scale", "scale", "column-scale", "rank-one")


def _inputs(device, lengths):
    tokens = sum(lengths)
    heads, groups, rows, columns = 4, 2, 7, 19
    left, right, coefficient = [], [], []
    for code in CODES:
        width = columns if code == "column-scale" else rows
        factor = torch.randn(tokens, groups, width, 1, device=device) * 0.05
        if code in ("row-scale", "column-scale"):
            factor = factor + 0.7
            if tokens:
                factor[0, 0, 2] = 0
        scalar = torch.randn(tokens, groups, 1, 1, device=device) * 0.05
        if code == "scale":
            scalar.fill_(0.8)
            if tokens > 2:
                scalar[2, 1] = 0
        left.append(factor.requires_grad_())
        right.append(
            (
                torch.randn(tokens, groups, rows, 1, device=device) * 0.05
            ).requires_grad_()
        )
        coefficient.append(scalar.requires_grad_())
    ik = [
        (torch.randn(tokens, groups, rows, 1, device=device) * 0.1).requires_grad_()
        for _ in range(2)
    ]
    iv = [
        (torch.randn(tokens, groups, 1, columns, device=device) * 0.1).requires_grad_()
        for _ in range(2)
    ]
    readouts = [
        (torch.randn(tokens, heads, rows, 1, device=device) * 0.2).requires_grad_()
        for _ in range(2)
    ]
    initial = (
        torch.randn(len(lengths), heads, rows, columns, device=device) * 0.2
    ).requires_grad_()
    return left, right, coefficient, ik, iv, readouts, initial


def _flat(inputs):
    *lists, initial = inputs
    return (*tuple(value for values in lists for value in values), initial)


def _oracle(lengths, inputs):
    left, right, coefficient, ik, iv, readouts, initial = inputs
    heads = initial.shape[1]

    def expand(value):
        return value.repeat_interleave(heads // value.shape[1], dim=1)

    left, right, coefficient = [
        list(map(expand, values)) for values in (left, right, coefficient)
    ]
    ik, iv = [list(map(expand, values)) for values in (ik, iv)]
    outputs = [[] for _ in readouts]
    final = []
    start = 0
    for sequence, length in enumerate(lengths):
        state = initial[sequence]
        for token in range(start, start + length):
            for operation, code in enumerate(CODES):
                if code == "scale":
                    state = state * coefficient[operation][token]
                elif code == "row-scale":
                    state = state * left[operation][token]
                elif code == "column-scale":
                    state = state * left[operation][token].transpose(-1, -2)
                else:
                    projection = (state * right[operation][token]).sum(-2, keepdim=True)
                    state = (
                        state
                        + coefficient[operation][token]
                        * left[operation][token]
                        * projection
                    )
            for key, value in zip(ik, iv, strict=True):
                state = state + key[token] * value[token]
            for index, query in enumerate(readouts):
                outputs[index].append((state * query[token]).sum(-2))
        final.append(state)
        start += length
    empty = initial.new_empty((0, heads, initial.shape[-1]))
    return tuple(
        torch.stack(values) if values else empty for values in outputs
    ), torch.stack(final)


@pytest.mark.parametrize("lengths", [(0, 1, 17, 0, 35, 2), (0, 0, 0), (33, 0, 3), (1,)])
@pytest.mark.parametrize("used", [(0, 1), (1,), ()])
def test_packed_checkpoint_all_gradients_match_fp64(gpu_device, lengths, used):
    inputs = _inputs(gpu_device, lengths)
    left, right, coefficient, ik, iv, readouts, initial = inputs
    offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device=gpu_device,
        dtype=torch.int64,
    )
    outputs, final = execute_lowrank_packed(
        operation_codes=CODES,
        left_factors=left,
        right_factors=right,
        coefficients=coefficient,
        injection_left=ik,
        injection_right=iv,
        readouts=readouts,
        initial_state=initial,
        sequence_offsets=offsets,
        output_dtypes=(torch.float32, torch.float32),
        return_final_state=True,
    )
    reference = (
        *tuple(
            [value.detach().double().requires_grad_() for value in values]
            for values in inputs[:-1]
        ),
        initial.detach().double().requires_grad_(),
    )
    expected, expected_final = _oracle(lengths, reference)
    for actual, oracle in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual.double(), oracle, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(final.double(), expected_final, atol=2e-6, rtol=2e-5)
    dy = [torch.randn_like(value) for value in outputs]
    df = torch.randn_like(final)
    loss, oracle_loss = (final * df).sum(), (expected_final * df.double()).sum()
    for index in used:
        loss = loss + (outputs[index] * dy[index]).sum()
        oracle_loss = oracle_loss + (expected[index] * dy[index].double()).sum()
    gradients = torch.autograd.grad(loss, _flat(inputs), allow_unused=True)
    expected_gradients = torch.autograd.grad(
        oracle_loss, _flat(reference), allow_unused=True
    )
    for actual, oracle, source in zip(
        gradients, expected_gradients, _flat(inputs), strict=True
    ):
        actual = torch.zeros_like(source) if actual is None else actual
        oracle = (
            torch.zeros_like(source, dtype=torch.float64) if oracle is None else oracle
        )
        torch.testing.assert_close(actual.double(), oracle, atol=2e-5, rtol=2e-4)


def test_packed_checkpoint_dynamic_segmentation(gpu_device):
    for lengths in ((17, 0, 2), (0, 3, 16, 0), (1, 35)):
        inputs = _inputs(gpu_device, lengths)
        left, right, coefficient, ik, iv, readouts, initial = inputs
        offsets = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device=gpu_device,
            dtype=torch.int32,
        )
        output, final = execute_lowrank_packed(
            operation_codes=CODES,
            left_factors=left,
            right_factors=right,
            coefficients=coefficient,
            injection_left=ik,
            injection_right=iv,
            readouts=readouts,
            initial_state=initial,
            sequence_offsets=offsets,
            output_dtypes=(torch.float32, torch.float32),
            return_final_state=True,
        )
        expected, expected_final = _oracle(lengths, inputs)
        for actual, oracle in zip(output, expected, strict=True):
            torch.testing.assert_close(actual, oracle, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(final, expected_final, atol=2e-6, rtol=2e-5)
