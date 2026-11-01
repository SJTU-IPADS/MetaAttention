"""Long dense inference bounds scratch independently of sequence length."""

import gc
from dataclasses import replace

import pytest
import torch

from attn_engine import Target
from testing import benchmark_stateful_plans as benchmark

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


@pytest.mark.parametrize("family", ["delta-rule", "kda", "delta-product"])
def test_long_factored_inference_fits_default_budget_and_bounded_peak(
    gpu_device, family
):
    case = replace(
        next(case for case in benchmark.CASES if case.name == family), batch=2, heads=16
    )
    length = 8192
    operator = case.factory(**case.kwargs(length))
    compiled = operator.compile(
        role_sizes=benchmark.role_sizes_for(operator, case, length),
        target=Target(),
        mode="dense",
        optimized=True,
    )
    torch.manual_seed(42)
    inputs = benchmark.inputs_for(operator, case, length, gpu_device)
    with torch.inference_mode():
        compiled(**inputs)
        torch.cuda.synchronize()
        gc.collect()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = compiled(**inputs, return_final_state=True)
        torch.cuda.synchronize()
        incremental = torch.cuda.max_memory_allocated() - before
    assert incremental < 512 << 20
    assert result.outputs["output"].shape == (2, 16, length, 64)
    assert result.final_state["memory"].shape == (2, 16, 64, 64)
    assert torch.isfinite(result.outputs["output"]).all()
    assert torch.isfinite(result.final_state["memory"]).all()
    if family == "delta-product":
        from tests.unit.test_delta_family_examples import _reference

        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            expected_output, expected_states = _reference(
                "delta_product",
                {name: value.cpu() for name, value in inputs.items()},
                {"memory": torch.zeros(2, 16, 64, 64)},
            )
        finally:
            torch.set_num_threads(previous_threads)
        verdict = benchmark._correctness(
            result.outputs.as_dict(),
            result.final_state.as_dict(),
            {"output": expected_output},
            expected_states,
            "final-state",
        )
        assert verdict["status"] == "VALID", verdict


@pytest.mark.parametrize("family", ["delta-rule", "kda", "oja", "delta-product"])
def test_factored_windows_preserve_readouts_state_tail_continuation_and_graph(
    gpu_device, family
):
    from attn_engine import AlgorithmIR, StateTuple, StatefulOperator

    from tests.unit.test_delta_family_examples import _case

    kind = {"delta-rule": "delta_rule", "delta-product": "delta_product"}.get(
        family, family
    )
    operator, sizes, inputs, initial = _case(
        kind, device=gpu_device, dim=64, dim_value=128
    )
    algorithm = operator.algorithm
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=algorithm.inputs,
            states=algorithm.states,
            transition=algorithm.transitions,
            readouts=(
                algorithm.readouts[0],
                replace(algorithm.readouts[0], name="second"),
            ),
            head_mapping=algorithm.head_mapping,
        )
    )
    length = 1025
    inputs = {
        name: value.repeat(1, 1, (length + 4) // 5, *([1] * (value.ndim - 3)))[
            :, :, :length
        ].contiguous()
        for name, value in inputs.items()
    }
    state = initial["memory"]
    state_before = state.clone()
    # Compare the same typed operator against its existing CPU reference path.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")
        reference = operator.compile(role_sizes=sizes, target=Target(), mode="dense")
        expected = reference(
            **{name: value.cpu() for name, value in inputs.items()},
            initial_state=StateTuple(("memory",), (state.cpu(),)),
            return_final_state=True,
        )
    compiled = reference
    with torch.inference_mode():
        actual = compiled(
            **inputs,
            initial_state=StateTuple(("memory",), (state,)),
            return_final_state=True,
        )
        first = compiled(
            **{name: value[:, :, :1001].contiguous() for name, value in inputs.items()},
            initial_state=StateTuple(("memory",), (state,)),
            return_final_state=True,
        )
        last = compiled(
            **{name: value[:, :, 1001:].contiguous() for name, value in inputs.items()},
            initial_state=first.final_state,
            return_final_state=True,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled(
                **inputs,
                initial_state=StateTuple(("memory",), (state,)),
                return_final_state=True,
            )
        graph.replay()
    for name in ("output", "second"):
        torch.testing.assert_close(
            actual.outputs[name].cpu(), expected.outputs[name], atol=2e-5, rtol=2e-4
        )
        torch.testing.assert_close(
            captured.outputs[name], actual.outputs[name], atol=2e-5, rtol=2e-4
        )
        torch.testing.assert_close(
            torch.cat((first.outputs[name], last.outputs[name]), dim=2),
            actual.outputs[name],
            atol=2e-5,
            rtol=2e-4,
        )
    for result in (actual, captured, last):
        torch.testing.assert_close(
            result.final_state["memory"].cpu(),
            expected.final_state["memory"],
            atol=2e-5,
            rtol=2e-4,
        )
    torch.testing.assert_close(state, state_before, atol=0, rtol=0)
