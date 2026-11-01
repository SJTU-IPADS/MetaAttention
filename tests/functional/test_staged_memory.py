"""Dense stage DAG inference carries States with bounded trajectory storage."""

import gc
from dataclasses import replace

import pytest
import torch

from attn_engine import StateTuple, Target
from testing import benchmark_stateful_plans as benchmark

pytestmark = [pytest.mark.functional, pytest.mark.gpu]
FAMILIES = ["preconditioned-kda", "preconditioned-gdn", "lightnet"]


def _compile(case, length):
    operator = case.factory(**case.kwargs(length))
    sizes = benchmark.role_sizes_for(operator, case, length)
    return (
        operator,
        sizes,
        operator.compile(role_sizes=sizes, target=Target(), mode="dense"),
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_long_staged_inference_fits_default_budget_and_bounded_peak(gpu_device, family):
    torch.manual_seed(123)
    case = replace(
        next(c for c in benchmark.CASES if c.name == family), batch=2, heads=16
    )
    length = 8192
    operator, _, compiled = _compile(case, length)
    inputs = benchmark.inputs_for(operator, case, length, gpu_device)
    with torch.inference_mode():
        compiled(**inputs, return_final_state=True)
        torch.cuda.synchronize()
        gc.collect()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = compiled(**inputs, return_final_state=True)
        torch.cuda.synchronize()
        incremental = torch.cuda.max_memory_allocated() - before
        # Releasing a result must release window scratch immediately, without
        # waiting for Python's cyclic garbage collector between invocations.
        del result
        assert torch.cuda.memory_allocated() <= before + (1 << 20)
        result = compiled(**inputs, return_final_state=True)
    assert incremental < 512 << 20
    assert result.outputs["output"].shape == (2, 16, length, 64)
    assert torch.isfinite(result.outputs["output"]).all()
    assert result.final_state.names == compiled.validation.state_names
    assert all(torch.isfinite(t).all() for t in result.final_state)


@pytest.mark.parametrize("family", FAMILIES)
def test_staged_windows_preserve_oracle_tail_continuation_and_graph(gpu_device, family):
    torch.manual_seed(42)
    case = replace(
        next(c for c in benchmark.CASES if c.name == family),
        batch=1,
        heads=2,
        dim_value=128,
    )
    length = 1025
    operator, sizes, compiled = _compile(case, length)
    inputs = benchmark.inputs_for(operator, case, length, gpu_device)
    initial = StateTuple(
        tuple(state.name for state in operator.algorithm.states),
        tuple(
            torch.full(
                (
                    case.batch,
                    sizes[state.head_role],
                    *(sizes[r] for r in state.feature_roles),
                ),
                0.1,
                device=gpu_device,
            )
            for state in operator.algorithm.states
        ),
    )
    saved = tuple(t.clone() for t in initial)
    expected_outputs, expected_states = benchmark._token_oracle(
        operator, inputs, sizes, initial_state=initial.as_dict()
    )
    with torch.inference_mode():
        actual = compiled(**inputs, initial_state=initial, return_final_state=True)
        first = compiled(
            **{n: v[:, :, :513].contiguous() for n, v in inputs.items()},
            initial_state=initial,
            return_final_state=True,
        )
        last = compiled(
            **{n: v[:, :, 513:].contiguous() for n, v in inputs.items()},
            initial_state=first.final_state,
            return_final_state=True,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled(
                **inputs, initial_state=initial, return_final_state=True
            )
        graph.replay()
        graph.replay()
    torch.testing.assert_close(
        actual.outputs["output"].cpu().float(),
        expected_outputs["output"],
        atol=0.05,
        rtol=0.02,
    )
    torch.testing.assert_close(
        captured.outputs["output"], actual.outputs["output"], atol=0, rtol=0
    )
    torch.testing.assert_close(
        torch.cat((first.outputs["output"], last.outputs["output"]), 2),
        actual.outputs["output"],
        atol=0.005,
        rtol=0.02,
    )
    for name, wanted in expected_states.items():
        torch.testing.assert_close(
            actual.final_state[name].cpu(), wanted, atol=2e-5, rtol=2e-4
        )
        for result in (last, captured):
            torch.testing.assert_close(
                result.final_state[name], actual.final_state[name], atol=2e-5, rtol=2e-4
            )
    for actual_state, wanted in zip(initial, saved, strict=True):
        torch.testing.assert_close(actual_state, wanted, atol=0, rtol=0)
