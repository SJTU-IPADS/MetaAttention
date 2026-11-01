"""Authored-order ABC oracle for native softmax regions and vector States."""

from dataclasses import replace

import pytest
import torch

from attn_engine import (
    AlgorithmIR,
    Readout,
    Softmax,
    StateTerm,
    StatefulOperator,
    StateTuple,
    Target,
)
from attn_engine.stateful.lowering.chunked import select_chunked_schedule
from examples import abc_attention as abc

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


def _fixture(length, dtype, *, altered=False, mixed=False):
    op = abc.abc_attention(2, 4, length, 13, 19, 11, dtype=dtype)
    if altered or mixed:
        ir = op.algorithm
        outer = ir.readouts[0].expression
        first = outer.operand.left.operand.left
        before = StateTerm("normalizer", 1, None, prior=mixed)
        after = StateTerm("normalizer", 1, None)
        scores = (first + 0.07) / (before + 0.03) if altered else first / before
        weights = Softmax(scores, abc.SLOT) / after
        if altered:
            weights = (weights + 0.001) * 0.9
        out = StateTerm(outer.state, weights, outer.role, prior=mixed)
        op = StatefulOperator(
            AlgorithmIR(
                inputs=ir.inputs,
                states=ir.states,
                transition=ir.transitions,
                readouts=(Readout("output", out, dtype),),
                head_mapping=ir.head_mapping,
            )
        )
    sizes = {
        abc.KEY: 13,
        abc.SLOT: 19,
        abc.VALUE: 11,
        abc.QUERY_HEADS: 2,
        abc.KEY_HEADS: 1,
        abc.VALUE_HEADS: 2,
        abc.STATE_HEADS: 4,
    }
    compiled = op.compile(role_sizes=sizes, target=Target(), mode="dense")
    return op, compiled


def _data(length, dtype, device):
    g = torch.Generator().manual_seed(739)
    host = {
        "query": torch.randn(2, 2, length, 13, generator=g).to(dtype) * 0.3,
        "key": torch.randn(2, 1, length, 13, generator=g).to(dtype) * 0.3,
        "value": torch.randn(2, 2, length, 11, generator=g).to(dtype) * 0.3,
        "slot_logits": torch.randn(2, 1, length, 19, generator=g) * 0.2,
    }
    initial = {
        "key_slot": torch.randn(2, 4, 13, 19, generator=g) * 0.1,
        "slot_value": torch.randn(2, 4, 19, 11, generator=g) * 0.1,
        "normalizer": torch.rand(2, 4, 19, generator=g) + 0.5,
    }
    # Contiguous nonzero-offset views satisfy the public runtime contract and
    # exercise raw address/cache rebinding during continuation.
    values = {}
    for name, x in host.items():
        storage = torch.empty(x.numel() + 7, device=device, dtype=x.dtype)
        values[name] = storage[7:].reshape(x.shape)
        values[name].copy_(x)
    return host, initial, values


def _oracle(host, initial, *, altered=False, mixed=False):
    x = {n: v.repeat_interleave(4 // v.shape[1], 1) for n, v in host.items()}
    # The scalar multiply happens in authored query dtype before contraction.
    q = (x["query"] * (13**-0.5)).float()
    states = {n: v.clone() for n, v in initial.items()}
    out = []
    for t in range(q.shape[2]):
        zprev, vprev = states["normalizer"], states["slot_value"]
        w = x["slot_logits"][:, :, t].exp()
        states["key_slot"] = (
            states["key_slot"] + x["key"][:, :, t].float()[..., None] * w[..., None, :]
        )
        states["slot_value"] = (
            states["slot_value"]
            + w[..., None] * x["value"][:, :, t].float()[..., None, :]
        )
        states["normalizer"] = states["normalizer"] + w
        z = zprev if mixed else states["normalizer"]
        scores = torch.einsum("bhr,bhrc->bhc", q[:, :, t], states["key_slot"])
        logits = (scores + 0.07) / (z + 0.03) if altered else scores / z
        weights = logits.softmax(-1) / states["normalizer"]
        if altered:
            weights = (weights + 0.001) * 0.9
        out.append(
            torch.einsum(
                "bhr,bhrc->bhc", weights, vprev if mixed else states["slot_value"]
            )
        )
    return torch.stack(out, 2).to(host["query"].dtype), states


@pytest.mark.parametrize("schedule", ["fused", "cut"])
@pytest.mark.parametrize("length", [1, 63, 64, 65, 130])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_abc_native_outputs_finals_continuation(gpu_device, schedule, length, dtype):
    op, compiled = _fixture(length, dtype)
    executable = op._plans[compiled.specialization]
    assert executable.chunked_recipe.native_diagonal is not None
    if schedule in ("cut", "fused"):
        candidate = next(
            c
            for c in executable.plan.lowered_candidates
            if (
                c.lowerer.endswith("readout-cut")
                if schedule == "cut"
                else c.lowerer == "tilelang-native-diagonal"
            )
        )
        op._plans[compiled.specialization] = replace(
            executable,
            plan=replace(executable.plan, candidate=candidate),
            chunked_recipe=select_chunked_schedule(
                executable.chunked_recipe, candidate.lowerer
            ),
        )
    host, initial, values = _data(length, dtype, gpu_device)
    start = StateTuple(
        tuple(initial), tuple(v.to(gpu_device) for v in initial.values())
    )
    expected, finals = _oracle(host, initial)
    result = compiled(**values, initial_state=start, return_final_state=True)
    tolerance = (
        dict(rtol=2e-4, atol=2e-5)
        if dtype == torch.float32
        else dict(rtol=2e-2, atol=2e-3)
    )
    torch.testing.assert_close(result.outputs["output"].cpu(), expected, **tolerance)
    only = compiled(**values, initial_state=start)
    torch.testing.assert_close(only.outputs["output"].cpu(), expected, **tolerance)
    assert only.final_state is None
    for name in finals:
        torch.testing.assert_close(
            result.final_state[name].cpu(), finals[name], rtol=2e-4, atol=2e-5
        )
        torch.testing.assert_close(start[name].cpu(), initial[name], rtol=0, atol=0)
    if length > 1:
        split = min(17, length - 1)
        prefix = compiled(
            **{n: v[:, :, :split].contiguous() for n, v in values.items()},
            initial_state=start,
            return_final_state=True,
        )
        suffix = compiled(
            **{n: v[:, :, split:].contiguous() for n, v in values.items()},
            initial_state=prefix.final_state,
            return_final_state=True,
        )
        torch.testing.assert_close(
            torch.cat((prefix.outputs["output"], suffix.outputs["output"]), 2).cpu(),
            expected,
            **tolerance,
        )
        for name in finals:
            torch.testing.assert_close(
                suffix.final_state[name].cpu(), finals[name], rtol=2e-4, atol=2e-5
            )


@pytest.mark.parametrize("schedule", ["fused", "cut"])
@pytest.mark.parametrize("mixed", [False, True])
def test_softmax_pointwise_and_temporal_vector_views(gpu_device, schedule, mixed):
    op, compiled = _fixture(65, torch.float32, altered=True, mixed=mixed)
    executable = op._plans[compiled.specialization]
    if schedule in ("cut", "fused"):
        candidate = next(
            c
            for c in executable.plan.lowered_candidates
            if (
                c.lowerer.endswith("readout-cut")
                if schedule == "cut"
                else c.lowerer == "tilelang-native-diagonal"
            )
        )
        op._plans[compiled.specialization] = replace(
            executable,
            plan=replace(executable.plan, candidate=candidate),
            chunked_recipe=select_chunked_schedule(
                executable.chunked_recipe, candidate.lowerer
            ),
        )
    host, initial, values = _data(65, torch.float32, gpu_device)
    expected, finals = _oracle(host, initial, altered=True, mixed=mixed)
    start = StateTuple(
        tuple(initial), tuple(v.to(gpu_device) for v in initial.values())
    )
    result = compiled(**values, initial_state=start, return_final_state=True)
    torch.testing.assert_close(
        result.outputs["output"].cpu(), expected, rtol=2e-4, atol=2e-5
    )
    for name in finals:
        torch.testing.assert_close(
            result.final_state[name].cpu(), finals[name], rtol=2e-4, atol=2e-5
        )


def test_oversized_abc_cut_outputs_and_public_vector_finals(gpu_device):
    from tests.unit.test_stateful_abc_recipe import _sizes

    torch.manual_seed(393)
    op = abc.abc_attention(1, 1, 7, 96, 96, 33, dtype=torch.float32)
    compiled = op.compile(role_sizes=_sizes(96, 96, 33), target=Target(), mode="dense")
    recipe = op._plans[compiled.specialization].chunked_recipe.native_diagonal
    assert recipe is not None and recipe.fused_softmax is False
    values = {
        name: torch.randn(1, 1, 7, width, device=gpu_device) * 0.1
        for name, width in (
            ("query", 96),
            ("key", 96),
            ("value", 33),
            ("slot_logits", 96),
        )
    }
    initial = {
        "key_slot": torch.zeros(1, 1, 96, 96),
        "slot_value": torch.zeros(1, 1, 96, 33),
        "normalizer": torch.zeros(1, 1, 96),
    }
    # Independent authored-order oracle at this rejected fused shape.
    host = {name: value.cpu() for name, value in values.items()}
    states = {name: value.clone() for name, value in initial.items()}
    expected = []
    for token in range(7):
        weight = host["slot_logits"][:, :, token].exp()
        states["key_slot"] += host["key"][:, :, token, :, None] * weight[:, :, None, :]
        states["slot_value"] += (
            weight[:, :, :, None] * host["value"][:, :, token, None, :]
        )
        states["normalizer"] += weight
        scores = torch.einsum(
            "bhr,bhrc->bhc", host["query"][:, :, token] * (96**-0.5), states["key_slot"]
        )
        weights = (scores / states["normalizer"]).softmax(-1) / states["normalizer"]
        expected.append(torch.einsum("bhr,bhrc->bhc", weights, states["slot_value"]))
    result = compiled(**values, return_final_state=True)
    torch.testing.assert_close(
        result.outputs["output"].cpu(), torch.stack(expected, 2), rtol=2e-4, atol=2e-5
    )
    for name in states:
        torch.testing.assert_close(
            result.final_state[name].cpu(), states[name], rtol=2e-4, atol=2e-5
        )
