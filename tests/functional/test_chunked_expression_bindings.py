"""Compiled dependency/cache metadata preserves dependent readout execution."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from attn_engine import AlgorithmIR, StateTuple, StatefulOperator, Target
from attn_engine.stateful.lowering import chunked
from tests.unit.test_delta_family_examples import _case

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


def _fixture(device, dims, dtype, length, *, mode="dense", backward=False):
    operator, sizes, inputs, initial = _case(
        "abc", device=device, dim=dims[0], dim_value=dims[1], dtype=dtype
    )
    ir = operator.algorithm
    # Exercise the supported matrix path with shared query/key heads.
    slot = next(role for role in sizes if role.name == "slot_dim")
    sizes[slot] = 64
    inputs["slot_logits"] = inputs["slot_logits"].repeat(1, 1, 1, 16)
    initial = {
        "key_slot": initial["key_slot"].repeat(1, 1, 1, 16),
        "slot_value": initial["slot_value"].repeat(1, 1, 16, 1),
        "normalizer": initial["normalizer"].repeat(1, 1, 16),
    }
    # Distinct but structurally equal expressions must still share cache slots.
    second = replace(
        ir.readouts[0],
        name="second",
        expression=deepcopy(ir.readouts[0].expression),
        output_dtype=torch.float32,
    )
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=ir.inputs,
            states=ir.states,
            transition=ir.transitions,
            readouts=(ir.readouts[0], second),
            head_mapping=ir.head_mapping,
        )
    )
    inputs = {
        name: tensor.repeat(1, 1, (length + 4) // 5, *([1] * (tensor.ndim - 3)))[
            :, :, :length
        ].contiguous()
        for name, tensor in inputs.items()
    }
    if mode == "packed":
        inputs = {
            name: tensor[0].transpose(0, 1).contiguous()
            for name, tensor in inputs.items()
        }
        initial = {
            name: tensor.repeat(4, *([1] * (tensor.ndim - 1)))
            for name, tensor in initial.items()
        }
    inputs = {name: tensor.requires_grad_(backward) for name, tensor in inputs.items()}
    initial = {
        name: tensor.requires_grad_(backward) for name, tensor in initial.items()
    }
    compiled = operator.compile(
        role_sizes=sizes, target=Target(), mode=mode, backward=backward
    )
    if mode == "dense" and not backward:
        executable = operator._plans[compiled.specialization]
        assert executable.runtime_family == "chunked-stateful"
        recipe = executable.chunked_recipe
        first, second = (expression for _, expression in recipe.readouts)
        assert first is not second and first == second
        assert recipe.expression_slots[id(first)] == recipe.expression_slots[id(second)]
        assert id(first) not in recipe.input_only
    return compiled, inputs, initial


def _oracle(inputs, initial):
    heads = initial["normalizer"].shape[1]
    values = {
        name: tensor.repeat_interleave(heads // tensor.shape[1], dim=1)
        for name, tensor in inputs.items()
    }
    scale = torch.tensor(
        values["query"].shape[-1] ** -0.5,
        device=values["query"].device,
        dtype=torch.float32,
    )
    query = (values["query"] * scale).float()
    key, value = values["key"].float(), values["value"].float()
    weight = values["slot_logits"].exp()
    states = dict(initial)
    outputs = []
    for token in range(query.shape[2]):
        states["key_slot"] = (
            states["key_slot"]
            + key[:, :, token, :, None] * weight[:, :, token, None, :]
        )
        states["slot_value"] = (
            states["slot_value"]
            + weight[:, :, token, :, None] * value[:, :, token, None, :]
        )
        states["normalizer"] = states["normalizer"] + weight[:, :, token]
        scores = (states["key_slot"] * query[:, :, token, :, None]).sum(-2)
        normalizer = states["normalizer"]
        probabilities = (scores / normalizer).softmax(-1)
        outputs.append(
            (states["slot_value"] * (probabilities / normalizer)[..., None]).sum(-2)
        )
    return torch.stack(outputs, dim=2), states


def _invoke(compiled, inputs, initial, offsets=None):
    return compiled(
        **inputs,
        initial_state=StateTuple(tuple(initial), tuple(initial.values())),
        sequence_offsets=offsets,
        return_final_state=True,
    )


@pytest.mark.parametrize("dims", [(64, 128), (128, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dependent_readouts_rebind_inputs_and_continue_without_runtime_ir_walk(
    gpu_device, dims, dtype
):
    compiled, inputs, initial = _fixture(gpu_device, dims, dtype, 67)
    # Runtime must consume the compiled proof; reference-oracle construction
    # happens outside this patch and never participates in production execution.
    with patch.object(
        chunked, "_contracted_terms", side_effect=AssertionError("runtime IR walk")
    ):
        actual = _invoke(compiled, inputs, initial)
        prefix = _invoke(
            compiled, {n: x[:, :, :31].contiguous() for n, x in inputs.items()}, initial
        )
        tail = _invoke(
            compiled,
            {n: x[:, :, 31:].contiguous() for n, x in inputs.items()},
            prefix.final_state.as_dict(),
        )
        changed = {n: (x * 0.7).contiguous() for n, x in inputs.items()}
        rebound = _invoke(compiled, changed, initial)
    expected, finals = _oracle(inputs, initial)
    changed_expected, changed_finals = _oracle(changed, initial)
    tolerance = (
        dict(atol=2e-3, rtol=0.02)
        if dtype == torch.bfloat16
        else dict(atol=3e-6, rtol=3e-5)
    )
    for name, output_dtype in (("output", dtype), ("second", torch.float32)):
        torch.testing.assert_close(
            actual.outputs[name], expected.to(output_dtype), **tolerance
        )
        torch.testing.assert_close(
            tail.outputs[name], expected[:, :, 31:].to(output_dtype), **tolerance
        )
        torch.testing.assert_close(
            rebound.outputs[name], changed_expected.to(output_dtype), **tolerance
        )
    for name in finals:
        torch.testing.assert_close(
            actual.final_state[name], finals[name], atol=3e-5, rtol=3e-5
        )
        torch.testing.assert_close(
            tail.final_state[name], finals[name], atol=3e-5, rtol=3e-5
        )
        torch.testing.assert_close(
            rebound.final_state[name], changed_finals[name], atol=3e-5, rtol=3e-5
        )


def _segmented_oracle(inputs, initial, offsets):
    outputs, finals = [], {name: [] for name in initial}
    boundaries = offsets.tolist()  # Independent test oracle only.
    for segment, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        states = {
            name: tensor[segment : segment + 1] for name, tensor in initial.items()
        }
        if begin != end:
            output, states = _oracle(
                {
                    name: tensor[begin:end].transpose(0, 1)[None]
                    for name, tensor in inputs.items()
                },
                states,
            )
            outputs.append(output[0].transpose(0, 1))
        for name in finals:
            finals[name].append(states[name][0])
    return torch.cat(outputs), {
        name: torch.stack(tensors) for name, tensors in finals.items()
    }


@pytest.mark.parametrize("dims", [(64, 128), (128, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["dense", "packed"])
def test_dependent_readout_input_and_initial_gradients(gpu_device, dims, dtype, mode):
    compiled, inputs, initial = _fixture(
        gpu_device, dims, dtype, 5, mode=mode, backward=True
    )
    offsets = None
    if mode == "packed":
        offsets = torch.tensor([0, 0, 2, 5, 5], device=gpu_device, dtype=torch.int32)
    actual = _invoke(compiled, inputs, initial, offsets)
    expected, finals = (
        _oracle(inputs, initial)
        if offsets is None
        else _segmented_oracle(inputs, initial, offsets)
    )
    dy = torch.linspace(-1, 1, expected.numel(), device=gpu_device).reshape_as(expected)
    actual_loss = (actual.outputs["second"] * dy).sum()
    expected_loss = (expected * dy).sum()
    for name, final in finals.items():
        torch.testing.assert_close(
            actual.final_state[name], final, atol=3e-5, rtol=3e-5
        )
        actual_loss = actual_loss + actual.final_state[name].square().sum() * 0.1
        expected_loss = expected_loss + final.square().sum() * 0.1
        if mode == "packed":
            torch.testing.assert_close(
                actual.final_state[name][[0, 3]], initial[name][[0, 3]], rtol=0, atol=0
            )
    torch.testing.assert_close(actual.outputs["second"], expected, atol=2e-3, rtol=0.02)
    leaves = (*inputs.values(), *initial.values())
    gradients = torch.autograd.grad(actual_loss, leaves)
    reference = torch.autograd.grad(expected_loss, leaves)
    for actual_gradient, expected_gradient in zip(gradients, reference, strict=True):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=2e-4 if dtype == torch.bfloat16 else 2e-5,
            rtol=0.02 if dtype == torch.bfloat16 else 2e-4,
        )
