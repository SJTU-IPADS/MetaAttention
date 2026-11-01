"""Public affine fallback must retain small updates to near-identity matrices."""

import pytest
import torch

from attn_engine import (
    AlgorithmIR,
    Batch,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    ProductFactor,
    ProductInjection,
    RankOnePropagation,
    Sequence,
    StateContraction,
    StateSpec,
    StateTransition,
    StatefulOperator,
    StateTuple,
    Target,
    TensorInput,
)

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


@pytest.mark.parametrize("delta", (-1e-4, 1e-4))
@pytest.mark.parametrize("mode", ("dense", "packed"))
def test_affine_scan_retains_small_rank_one_updates_and_gradients(
    delta, mode, gpu_device
):
    key, value, head = FeatureRole("key"), FeatureRole("value"), HeadRole("head")
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=tuple(
                TensorInput(name, (Batch, head, Sequence, role), torch.float32)
                for name, role in (
                    ("query", key),
                    ("alpha", key),
                    ("beta", key),
                    ("key", key),
                    ("value", value),
                )
            ),
            states=(StateSpec("memory", (Batch, head, key, value)),),
            transition=StateTransition(
                "memory",
                propagations=(RankOnePropagation(key, Input("beta"), Input("alpha")),),
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(Input("key"), (key,)),
                            ProductFactor(Input("value"), (value,)),
                        )
                    ),
                ),
            ),
            readouts=(
                StateContraction(
                    "output", "memory", Input("query"), key, torch.float32
                ),
            ),
            head_mapping=HeadMapping({}),
        )
    )
    inputs = {
        name: torch.zeros(1, 1, 2, 64, device=gpu_device)
        for name in ("query", "alpha", "beta", "key", "value")
    }
    inputs["query"][..., 0] = 1
    inputs["alpha"][..., 0] = 1
    inputs["beta"][..., 0] = delta
    inputs = {name: tensor.requires_grad_() for name, tensor in inputs.items()}
    initial = torch.zeros(1, 1, 64, 64, device=gpu_device)
    initial[..., 0, 0] = 1
    initial.requires_grad_()
    # Independent residual recurrence keeps the small update separate from I.
    state = initial
    output = []
    for token in range(2):
        x = {name: tensor[:, :, token] for name, tensor in inputs.items()}
        projection = (x["alpha"].unsqueeze(-1) * state).sum(-2)
        state = state + x["beta"].unsqueeze(-1) * projection.unsqueeze(-2)
        state = state + x["key"].unsqueeze(-1) * x["value"].unsqueeze(-2)
        output.append((x["query"].unsqueeze(-1) * state).sum(-2))
    expected = torch.stack(output, dim=2)
    leaves = (*inputs.values(), initial)
    expected_gradients = torch.autograd.grad(
        expected.square().sum() + state.square().sum(), leaves, retain_graph=True
    )
    compiled = operator.compile(
        role_sizes={head: 1, key: 64, value: 64},
        target=Target(),
        mode=mode,
        optimized=False,
        backward=True,
    )
    assert (
        operator._plans[compiled.specialization].plan.candidate.lowerer
        == "tilelang-generic"
    )
    bindings = (
        inputs
        if mode == "dense"
        else {
            name: tensor[0].transpose(0, 1).contiguous()
            for name, tensor in inputs.items()
        }
    )
    offsets = (
        None
        if mode == "dense"
        else torch.tensor([0, 2], device=gpu_device, dtype=torch.int32)
    )
    result = compiled(
        **bindings,
        initial_state=StateTuple(("memory",), (initial,)),
        sequence_offsets=offsets,
        return_final_state=True,
    )
    expected = expected if mode == "dense" else expected[0].transpose(0, 1)
    torch.testing.assert_close(result.outputs["output"], expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        result.final_state["memory"], state, rtol=2e-5, atol=2e-6
    )
    gradients = torch.autograd.grad(
        result.outputs["output"].square().sum()
        + result.final_state["memory"].square().sum(),
        leaves,
    )
    for actual, reference in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
