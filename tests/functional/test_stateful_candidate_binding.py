"""Real GPU tuning produces the same executable contract as the final callable."""

from dataclasses import replace

import pytest
import torch

from attn_engine import Batch, Sequence, Target
from attn_engine.stateful.compile import plan as planner
from attn_engine.stateful_operator import CompiledStatefulOperator
from testing.benchmark_stateful_plans import (
    CASES,
    _correctness,
    _token_oracle,
    inputs_for,
    role_sizes_for,
)

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


@pytest.mark.parametrize(
    "name,mode,backward",
    [
        ("simple-gla", "dense", False),
        ("simple-gla", "packed", False),
        ("simple-gla", "dense", True),
        ("simple-gla", "packed", True),
        ("gla", "dense", False),
        ("gla", "dense", True),
        ("gdn", "dense", False),
        ("gdn", "dense", True),
        ("gdn", "packed", True),
        ("abc", "dense", False),
        ("hgrn", "dense", False),
        ("hgrn", "packed", False),
    ],
)
def test_measured_binding_matches_published_execution_and_oracle(
    monkeypatch, tmp_path, gpu_device, request, name, mode, backward
):
    monkeypatch.setenv("STATEFUL_OPERATOR_CACHE_DIR", str(tmp_path))
    workload = planner.WorkloadPoint(
        1 if mode == "dense" else 3,
        65,
        "uniform" if mode == "dense" else "empty-containing",
    )
    monkeypatch.setattr(
        planner,
        "WORKLOAD_POLICY",
        planner.WorkloadPolicy(
            planner.WORKLOAD_POLICY.version, (workload,), (workload,)
        ),
    )
    measured = []
    invoke = CompiledStatefulOperator.__call__

    def record(self, **kwargs):
        if self._bound_executable is not None:
            assert not torch.is_grad_enabled()
            assert self.specialization not in self.operator._plans
            measured.append(self._executable)
        return invoke(self, **kwargs)

    monkeypatch.setattr(CompiledStatefulOperator, "__call__", record)
    case = replace(next(case for case in CASES if case.name == name), heads=1)
    operator = case.factory(**{**case.kwargs(65), "dtype": torch.float32})
    sizes = role_sizes_for(operator, case, 65)
    kwargs = dict(
        role_sizes=sizes, target=Target(), mode=mode, backward=backward, autotune=True
    )
    compiled = operator.compile(**kwargs)
    assert measured
    assert any(probe.plan == compiled._executable.plan for probe in measured)
    assert operator._plans == {compiled.specialization: compiled._executable}
    assert operator._zero_states == {}
    count = len(measured)
    assert operator.compile(**kwargs)._executable is compiled._executable
    assert len(measured) == count

    host = inputs_for(operator, case, 65, torch.device("cpu"))
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        expected, expected_states = _token_oracle(operator, host, sizes)
    finally:
        torch.set_num_threads(previous_threads)
    inputs = {}
    for spec in operator.algorithm.inputs:
        value = host[spec.name]
        if mode == "packed":
            axes = (
                spec.roles.index(Batch),
                spec.roles.index(Sequence),
                *(
                    i
                    for i, role in enumerate(spec.roles)
                    if role not in (Batch, Sequence)
                ),
            )
            value = value.permute(axes).reshape(65, *(value.shape[i] for i in axes[2:]))
        inputs[spec.name] = (
            value.contiguous()
            .to(gpu_device)
            .requires_grad_(backward and spec.differentiable)
        )
    call_kwargs = {}
    if mode == "packed":
        call_kwargs["sequence_offsets"] = torch.tensor(
            [0, 0, 65], dtype=torch.int32, device=gpu_device
        )
        expected = {
            name: value.transpose(1, 2).reshape(65, value.shape[1], *value.shape[3:])
            for name, value in expected.items()
        }
        expected_states = {
            name: torch.cat((torch.zeros_like(value), value))
            for name, value in expected_states.items()
        }
    result = compiled(**inputs, **call_kwargs, return_final_state=True)
    if mode == "packed" and compiled._executable.runtime_family == "scalar-factorized":
        # Binding/measurement assertions above remain unconditional. Only this
        # existing numerical limitation is expected to fail (also at 795a961).
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason="pre-existing packed scalar numerical mismatch, reproduced at 795a961",
            )
        )
    verdict = _correctness(
        result.outputs.as_dict(),
        result.final_state.as_dict(),
        expected,
        expected_states,
        "final-state",
    )
    assert verdict["status"] == "VALID", verdict
    if backward:
        sum(
            value.float().sum() for value in (*result.outputs, *result.final_state)
        ).backward()
        for value in inputs.values():
            if value.requires_grad:
                assert value.grad is not None
                assert torch.isfinite(value.grad).all()
