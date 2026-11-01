"""Independent update/readout axes through the public Oja execution boundary."""

from dataclasses import replace

import pytest
import torch

from attn_engine import AlgorithmIR, StateTuple, StatefulOperator, Target
from tests.unit.test_delta_family_examples import _case

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


def _fixture(device, dtype, dims, mode, *, length=35, backward=False):
    operator, roles, inputs, initial = _case(
        "oja", device=device, dtype=dtype, dim=dims[0], dim_value=dims[1]
    )
    algorithm = operator.algorithm
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=algorithm.inputs,
            states=algorithm.states,
            transition=algorithm.transitions,
            readouts=(
                algorithm.readouts[0],
                replace(
                    algorithm.readouts[0], name="second", output_dtype=torch.float32
                ),
            ),
            head_mapping=algorithm.head_mapping,
        )
    )
    inputs = {
        name: value.repeat(1, 1, (length + 4) // 5, *([1] * (value.ndim - 3)))[
            :, :, :length
        ].contiguous()
        for name, value in inputs.items()
    }
    # Bound the recurrence; a zero gate exercises an uninvertible update.
    inputs["value"] = (inputs["value"] * 0.25).contiguous()
    if length > 3:
        inputs["gate"][:, :, 3, 0] = -float("inf")
    inputs = {name: value.requires_grad_(backward) for name, value in inputs.items()}
    state = initial["memory"].requires_grad_(backward)
    lengths = (0, 17, length - 17, 0) if length else (0, 0)
    if mode == "packed":
        inputs = {
            name: value[0].transpose(0, 1).contiguous()
            for name, value in inputs.items()
        }
        state = state.repeat(len(lengths), 1, 1, 1).contiguous()
        offsets = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device=device,
        )
    else:
        offsets = None
    compiled = operator.compile(
        role_sizes=roles, target=Target(), mode=mode, backward=backward
    )
    executable = operator._plans[compiled.specialization]
    assert executable.plan.candidate.lowerer == "tilelang-factored-recurrence"
    assert executable.rank_one_factored.transpose_state
    assert executable.rank_one_factored.readout_axis == 1
    return compiled, inputs, state, offsets


def _dense_oracle(inputs, state):
    heads = state.shape[1]
    values = {
        name: value.repeat_interleave(heads // value.shape[1], dim=1)
        for name, value in inputs.items()
    }
    # Preserve the authored expression dtype and separate use-site casts before
    # differentiating the independent residual recurrence. A BF16 query multiply
    # rounds before contraction; casting it first changes even FP32 State VJPs.
    scale = torch.tensor(
        state.shape[-2] ** -0.5, device=state.device, dtype=torch.float32
    )
    query = (scale * values["query"]).float()
    prediction = values["value"].float()
    correction = values["value"].float()
    key = values["key"].float()
    write = (values["value"] * values["beta"][..., None]).float()
    gate = values["gate"].exp()
    coefficient = -values["beta"]
    output = []
    for token in range(query.shape[2]):
        state = state * gate[:, :, token, None, :]
        residual = (state * prediction[:, :, token, None, :]).sum(-1)
        state = (
            state
            + (coefficient[:, :, token, None] * residual)[..., None]
            * correction[:, :, token, None, :]
        )
        state = state + key[:, :, token, :, None] * write[:, :, token, None, :]
        output.append((state * query[:, :, token, :, None]).sum(-2))
    return torch.stack(output, dim=2), {"memory": state}


def _oracle(inputs, state, offsets):
    if offsets is None:
        return _dense_oracle(inputs, state)
    # Test-only host offsets; the production pipeline keeps offsets on device.
    values = offsets.tolist()
    outputs, finals = [], []
    for index, (begin, end) in enumerate(zip(values, values[1:])):
        if begin == end:
            finals.append(state[index])
            continue
        output, final = _dense_oracle(
            {
                name: value[begin:end].transpose(0, 1)[None]
                for name, value in inputs.items()
            },
            state[index : index + 1],
        )
        outputs.append(output[0].transpose(0, 1))
        finals.append(final["memory"][0])
    width, heads = state.shape[-1], state.shape[1]
    output = torch.cat(outputs) if outputs else state.new_empty((0, heads, width))
    return output, {"memory": torch.stack(finals)}


def _invoke(compiled, inputs, state, offsets):
    return compiled(
        **inputs,
        initial_state=StateTuple(("memory",), (state,)),
        sequence_offsets=offsets,
        return_final_state=True,
    )


@pytest.mark.parametrize("dims", [(64, 128), (128, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["dense", "packed"])
def test_axis_outputs_final_and_continuation(gpu_device, dims, dtype, mode):
    compiled, inputs, state, offsets = _fixture(gpu_device, dtype, dims, mode)
    actual = _invoke(compiled, inputs, state, offsets)
    output, final = _oracle(inputs, state, offsets)
    tolerance = (
        dict(atol=2e-3, rtol=0.02)
        if dtype == torch.bfloat16
        else dict(atol=3e-6, rtol=3e-5)
    )
    for name in ("output", "second"):
        expected = output.to(dtype if name == "output" else torch.float32)
        torch.testing.assert_close(actual.outputs[name], expected, **tolerance)
    torch.testing.assert_close(
        actual.final_state["memory"], final["memory"], **tolerance
    )
    if mode == "dense":
        prefix = _invoke(
            compiled,
            {n: x[:, :, :13].contiguous() for n, x in inputs.items()},
            state,
            None,
        )
        tail = _invoke(
            compiled,
            {n: x[:, :, 13:].contiguous() for n, x in inputs.items()},
            prefix.final_state["memory"],
            None,
        )
        torch.testing.assert_close(
            tail.outputs["second"], actual.outputs["second"][:, :, 13:], **tolerance
        )
        torch.testing.assert_close(
            tail.final_state["memory"], actual.final_state["memory"], **tolerance
        )
    else:
        torch.testing.assert_close(
            actual.final_state["memory"][[0, 3]], state[[0, 3]], rtol=0, atol=0
        )


@pytest.mark.parametrize("dims", [(64, 128), (128, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["dense", "packed"])
@pytest.mark.parametrize("loss", ["both", "output", "final"])
def test_axis_input_and_initial_gradients(gpu_device, dims, dtype, mode, loss):
    compiled, inputs, state, offsets = _fixture(
        gpu_device, dtype, dims, mode, backward=True
    )
    actual = _invoke(compiled, inputs, state, offsets)
    output, final = _oracle(inputs, state, offsets)
    dy = torch.linspace(-1, 1, output.numel(), device=gpu_device).reshape_as(output)
    actual_loss = (actual.outputs["second"] * dy).sum() if loss != "final" else 0
    expected_loss = (output * dy).sum() if loss != "final" else 0
    if loss != "output":
        actual_loss = actual_loss + actual.final_state["memory"].square().sum() * 0.1
        expected_loss = expected_loss + final["memory"].square().sum() * 0.1
    leaves = (*inputs.values(), state)
    gradients = torch.autograd.grad(actual_loss, leaves)
    reference = torch.autograd.grad(expected_loss, leaves, allow_unused=True)
    for gradient, expected, leaf in zip(gradients, reference, leaves, strict=True):
        if expected is None:
            expected = torch.zeros_like(leaf)
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(
            gradient,
            expected,
            atol=2e-4 if dtype == torch.bfloat16 else 2e-5,
            rtol=0.03 if dtype == torch.bfloat16 else 3e-4,
        )


def test_axis_all_empty_packed(gpu_device):
    compiled, inputs, state, offsets = _fixture(
        gpu_device, torch.float32, (128, 64), "packed", length=0, backward=True
    )
    actual = _invoke(compiled, inputs, state, offsets)
    assert actual.outputs["second"].shape == (0, 2, 64)
    torch.testing.assert_close(actual.final_state["memory"], state, rtol=0, atol=0)
    gradient = torch.autograd.grad(actual.final_state["memory"].sum(), state)[0]
    torch.testing.assert_close(gradient, torch.ones_like(state), rtol=0, atol=0)
