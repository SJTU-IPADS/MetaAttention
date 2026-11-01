"""Previously unseen DSL compositions: real GPU output, carry, and continuation."""

from __future__ import annotations

import pytest
import torch

from attn_engine import StateTuple, Target
from testing.stateful_generalization_profiles import (
    FAMILIES,
    Mutation,
    check_result,
    make_inputs,
    make_operator,
    oracle,
)

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


@pytest.mark.parametrize(
    "family,level,seed",
    [
        pytest.param(
            family,
            level,
            seed,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="scalar decay plus value-axis scale returns an inaccurate final State; "
                "see docs/stateful-generalization-tests.md",
            )
            if (family, level, seed) == ("propagation", 1, 1701)
            else (),
        )
        for level, seed in [(1, 1701), (3, 2028)]
        for family in FAMILIES
    ],
)
def test_unseen_composition_output_state_and_continuation(
    gpu_device, family, level, seed
):
    operator, sizes = make_operator(
        Mutation(family, level, seed), heads=2, rows=64, columns=64
    )
    compiled = operator.compile(
        role_sizes=sizes, target=Target(), mode="dense", optimized=True
    )
    values, initial = make_inputs(
        operator, sizes, batch=2, length=65, seed=seed, device=gpu_device
    )
    expected, final = oracle(operator, sizes, values, initial)
    actual = compiled(
        **values,
        initial_state=StateTuple(("memory",), (initial,)),
        return_final_state=True,
    )
    check_result(actual, expected, final)
    # Cross a chunk boundary with a nonzero carried state; use the same compiled
    # operator at both runtime lengths rather than reconstructing the profile.
    split = 33
    first = compiled(
        **{name: tensor[:, :, :split].contiguous() for name, tensor in values.items()},
        initial_state=StateTuple(("memory",), (initial,)),
        return_final_state=True,
    )
    second = compiled(
        **{name: tensor[:, :, split:].contiguous() for name, tensor in values.items()},
        initial_state=StateTuple(
            ("memory",), (first.final_state["memory"].contiguous(),)
        ),
        return_final_state=True,
    )
    for name, expected_output in expected.items():
        combined = torch.cat((first.outputs[name], second.outputs[name]), dim=2)
        torch.testing.assert_close(
            combined.cpu().float(), expected_output, rtol=0.03, atol=0.001
        )
    torch.testing.assert_close(
        second.final_state["memory"].cpu(), final["memory"], rtol=0.02, atol=0.001
    )


@pytest.mark.parametrize("family", ["rank", "combined"])
def test_factored_windows_preserve_batch_tail_outputs_and_continuation(
    gpu_device, family
):
    operator, sizes = make_operator(
        Mutation(family, 1, 2028), heads=2, rows=64, columns=64
    )
    compiled = operator.compile(role_sizes=sizes, target=Target(), mode="dense")
    executable = operator._plans[compiled.specialization]
    assert executable.runtime_family == "factored-recurrence"
    length = executable.plan.workspace.inference_window_tokens + 7
    assert length > 7
    values, initial = make_inputs(
        operator, sizes, batch=2, length=length, seed=2028, device=gpu_device
    )
    saved = initial.clone()
    expected, final = oracle(operator, sizes, values, initial)
    kwargs = {"initial_state": StateTuple(("memory",), (initial,))}
    result = compiled(**values, **kwargs, return_final_state=True)
    check_result(result, expected, final)
    output_only = compiled(**values, **kwargs)
    assert output_only.final_state is None
    for actual, wanted in zip(output_only.outputs, result.outputs, strict=True):
        torch.testing.assert_close(actual, wanted)
    split = length - 8
    first = compiled(
        **{name: tensor[:, :, :split].contiguous() for name, tensor in values.items()},
        **kwargs,
        return_final_state=True,
    )
    last = compiled(
        **{name: tensor[:, :, split:].contiguous() for name, tensor in values.items()},
        initial_state=StateTuple(
            ("memory",), (first.final_state["memory"].contiguous(),)
        ),
        return_final_state=True,
    )
    for name, wanted in expected.items():
        torch.testing.assert_close(
            torch.cat((first.outputs[name], last.outputs[name]), 2).cpu().float(),
            wanted,
            rtol=0.03,
            atol=0.001,
        )
    torch.testing.assert_close(
        last.final_state["memory"].cpu(), final["memory"], rtol=0.02, atol=0.001
    )
    torch.testing.assert_close(initial, saved, rtol=0, atol=0)
