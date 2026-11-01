"""Ordered weighted updates through dense/packed public execution."""

from dataclasses import replace

import pytest
import torch

from attn_engine import AlgorithmIR, StateTuple, StatefulOperator, Target
from tests.unit.test_delta_family_examples import _case

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


def _fixture(device, dims, dtype, mode, *, length=130, backward=False, steps=2):
    operator, sizes, inputs, initial = _case(
        "delta_product",
        device=device,
        dim=dims[0],
        dim_value=dims[1],
        dtype=dtype,
        steps=steps,
    )
    ir = operator.algorithm
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=ir.inputs,
            states=ir.states,
            transition=ir.transitions,
            readouts=(
                ir.readouts[0],
                replace(ir.readouts[0], name="second", output_dtype=torch.float32),
            ),
            head_mapping=ir.head_mapping,
        )
    )
    inputs = {
        name: tensor.repeat(1, 1, (length + 4) // 5, *([1] * (tensor.ndim - 3)))[
            :, :, :length
        ].contiguous()
        for name, tensor in inputs.items()
    }
    for name in inputs:
        if name.startswith("key_"):
            inputs[name] = torch.nn.functional.normalize(
                inputs[name].float(), dim=-1
            ).to(dtype)
    if length > 7:
        inputs["beta_0"][:, :, 7] = 0
        inputs["gate"][:, :, 3] = -float("inf")
    inputs = {name: tensor.requires_grad_(backward) for name, tensor in inputs.items()}
    state = initial["memory"].requires_grad_(backward)
    offsets = None
    if mode == "packed":
        lengths = (0, 17, length - 17, 0)
        offsets = torch.tensor(
            [0, 0, 17, length, length], device=device, dtype=torch.int32
        )
        inputs = {
            name: tensor[0].transpose(0, 1).contiguous()
            for name, tensor in inputs.items()
        }
        state = state.repeat(len(lengths), 1, 1, 1).contiguous()
    compiled = operator.compile(
        role_sizes=sizes, target=Target(), mode=mode, backward=backward
    )
    assert (
        operator._plans[compiled.specialization].plan.candidate.lowerer
        == "tilelang-factored-recurrence"
    )
    return compiled, inputs, state, offsets


def _dense_oracle(inputs, state, steps, *, compact_writes=False):
    heads = state.shape[1]
    x = {
        name: tensor.repeat_interleave(heads // tensor.shape[1], dim=1)
        for name, tensor in inputs.items()
    }
    # Retain independent use-site casts and authored scalar Constant rounding.
    scale = torch.tensor(
        state.shape[-2] ** -0.5, device=state.device, dtype=torch.float32
    )
    query = (x["query"] * scale).float()
    gate = x["gate"].exp()
    left, right, coefficients = [], [], []
    for j in range(steps):
        left.append(x[f"key_{j}"].float())
        right.append(x[f"key_{j}"].float())
        coefficients.append(-x[f"beta_{j}"])
    writes = []
    for j in range(steps):
        if compact_writes:
            # Expanding a suffix into separate outer products creates extra
            # BF16 cast VJPs. Use independent tensor math for the compact
            # factor graph, including distinct casts at each action use site.
            key = x[f"key_{j}"].float()
            value = (x[f"beta_{j}"][..., None] * x[f"value_{j}"]).float()
            for later in range(j + 1, steps):
                action_left = x[f"key_{later}"].float()
                action_right = x[f"key_{later}"].float()
                scalar = -x[f"beta_{later}"]
                key = key + scalar[..., None] * action_left * (action_right * key).sum(
                    -1, keepdim=True
                )
            writes.append((key, value, ()))
            continue
        terms = [(j, ())]
        for later in range(j + 1, steps):
            terms += [
                (later, (*factors, later, (later, vector)))
                for vector, factors in tuple(terms)
            ]
        for vector, factors in terms:
            key = x[f"key_{vector}"].float()
            value = (x[f"beta_{j}"][..., None] * x[f"value_{j}"]).float()
            weights = []
            for factor in factors:
                if isinstance(factor, int):
                    weights.append(-x[f"beta_{factor}"])
                else:
                    a, b = factor
                    weights.append(
                        (x[f"key_{a}"].float() * x[f"key_{b}"].float()).sum(-1)
                    )
            writes.append((key, value, weights))
    output = []
    for t in range(query.shape[2]):
        state = state * gate[:, :, t, None, None]
        for j in range(steps):
            projection = (state * right[j][:, :, t, :, None]).sum(-2, keepdim=True)
            state = (
                state
                + coefficients[j][:, :, t, None, None]
                * left[j][:, :, t, :, None]
                * projection
            )
        for key, value, weights in writes:
            product = key[:, :, t, :, None] * value[:, :, t, None, :]
            for weight in weights:
                product = product * weight[:, :, t, None, None]
            state = state + product
        output.append((state * query[:, :, t, :, None]).sum(-2))
    return torch.stack(output, dim=2), state


def _oracle(inputs, state, offsets, steps, *, compact_writes=False):
    if offsets is None:
        return _dense_oracle(inputs, state, steps, compact_writes=compact_writes)
    outputs, finals = [], []
    boundaries = (
        offsets.tolist()
    )  # Test oracle only; production segmentation stays on device.
    for segment, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        if begin == end:
            finals.append(state[segment])
            continue
        output, final = _dense_oracle(
            {
                name: tensor[begin:end].transpose(0, 1)[None]
                for name, tensor in inputs.items()
            },
            state[segment : segment + 1],
            steps,
            compact_writes=compact_writes,
        )
        outputs.append(output[0].transpose(0, 1))
        finals.append(final[0])
    return torch.cat(outputs), torch.stack(finals)


def _invoke(compiled, inputs, state, offsets=None):
    return compiled(
        **inputs,
        initial_state=StateTuple(("memory",), (state,)),
        sequence_offsets=offsets,
        return_final_state=True,
    )


@pytest.mark.parametrize("dims", [(64, 128), (128, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["dense", "packed"])
def test_ordered_outputs_final_and_continuation(gpu_device, dims, dtype, mode):
    compiled, inputs, state, offsets = _fixture(gpu_device, dims, dtype, mode)
    actual = _invoke(compiled, inputs, state, offsets)
    output, final = _oracle(inputs, state, offsets, 2)
    tol = (
        dict(atol=2e-3, rtol=0.02)
        if dtype == torch.bfloat16
        else dict(atol=3e-6, rtol=3e-5)
    )
    for name in ("output", "second"):
        torch.testing.assert_close(
            actual.outputs[name],
            output.to(dtype if name == "output" else torch.float32),
            **tol,
        )
    torch.testing.assert_close(
        actual.final_state["memory"], final, atol=3e-6, rtol=3e-5
    )
    if mode == "dense":
        first = _invoke(
            compiled, {n: x[:, :, :63].contiguous() for n, x in inputs.items()}, state
        )
        tail = _invoke(
            compiled,
            {n: x[:, :, 63:].contiguous() for n, x in inputs.items()},
            first.final_state["memory"],
        )
        torch.testing.assert_close(tail.outputs["second"], output[:, :, 63:], **tol)
        torch.testing.assert_close(
            tail.final_state["memory"], final, atol=3e-6, rtol=3e-5
        )
    else:
        torch.testing.assert_close(
            actual.final_state["memory"][[0, 3]], state[[0, 3]], rtol=0, atol=0
        )


@pytest.mark.parametrize("dims", [(64, 128), (128, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["dense", "packed"])
@pytest.mark.parametrize("loss", ["both", "output", "final"])
def test_ordered_input_and_initial_gradients(gpu_device, dims, dtype, mode, loss):
    compiled, inputs, state, offsets = _fixture(
        gpu_device, dims, dtype, mode, length=35, backward=True
    )
    actual = _invoke(compiled, inputs, state, offsets)
    # The forward tests retain the independently expanded algebra. Gradients
    # must retain the compact factor graph's BF16 use-site cast boundaries.
    output, final = _oracle(inputs, state, offsets, 2, compact_writes=True)
    dy = torch.linspace(-1, 1, output.numel(), device=gpu_device).reshape_as(output)
    actual_loss = (actual.outputs["second"] * dy).sum() if loss != "final" else 0
    expected_loss = (output * dy).sum() if loss != "final" else 0
    if loss != "output":
        actual_loss += actual.final_state["memory"].square().sum() * 0.1
        expected_loss += final.square().sum() * 0.1
    leaves = (*inputs.values(), state)
    gradients = torch.autograd.grad(actual_loss, leaves)
    expected = torch.autograd.grad(expected_loss, leaves, allow_unused=True)
    for actual_gradient, reference, leaf in zip(
        gradients, expected, leaves, strict=True
    ):
        if reference is None:
            reference = torch.zeros_like(leaf)
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(
            actual_gradient,
            reference,
            atol=2e-5 if dtype == torch.float32 else 2e-4,
            rtol=2e-4 if dtype == torch.float32 else 0.02,
        )


@pytest.mark.parametrize("dims", [(64, 128), (128, 64)])
def test_three_ordered_updates_chunk_tail(gpu_device, dims):
    compiled, inputs, state, offsets = _fixture(
        gpu_device, dims, torch.float32, "dense", steps=3
    )
    actual = _invoke(compiled, inputs, state)
    output, final = _oracle(inputs, state, offsets, 3)
    torch.testing.assert_close(actual.outputs["second"], output, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(
        actual.final_state["memory"], final, atol=3e-6, rtol=3e-5
    )
