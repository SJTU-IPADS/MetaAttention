from __future__ import annotations

import pytest
import torch

from attn_engine import StateTuple, Target
from testing.stateful_composition_profiles import make_inputs, make_profile


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _allow_cpu_stateful_operator(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")


def _evaluate_input(expression, values, token):
    if expression.__class__.__name__ == "Input":
        return values[expression.name][:, :, token]
    if expression.__class__.__name__ == "Multiply":
        left = _evaluate_input(expression.left, values, token)
        right = _evaluate_input(expression.right, values, token)
        while left.ndim < right.ndim:
            left = left.unsqueeze(-1)
        while right.ndim < left.ndim:
            right = right.unsqueeze(-1)
        return left * right
    if expression.__class__.__name__ == "Exp":
        return _evaluate_input(expression.operand, values, token).exp()
    raise TypeError(f"unsupported oracle expression {type(expression).__name__}")


def _oracle(profile, inputs, *, initial):
    algorithm = profile.operator.algorithm
    state_spec = algorithm.states[0]
    row, column = state_spec.feature_roles
    heads = profile.role_sizes[state_spec.head_role]
    values = {}
    for spec in algorithm.inputs:
        value = inputs[spec.name].float()
        head_axis = next(
            (
                index
                for index, role in enumerate(spec.roles)
                if isinstance(role, type(state_spec.head_role))
            ),
            None,
        )
        if head_axis is not None:
            value = value.repeat_interleave(
                heads // value.shape[head_axis], dim=head_axis
            )
        values[spec.name] = value
    states = initial.float().clone()
    outputs = {readout.name: [] for readout in algorithm.readouts}
    length = next(value.shape[2] for value in values.values() if value.ndim == 4)
    for token in range(length):
        for node in algorithm.transitions[0].propagations:
            if node.__class__.__name__ == "AxisScale":
                factor = _evaluate_input(node.factor, values, token)
                if node.roles == (row,):
                    states = states * factor[:, :, :, None]
                elif node.roles == (column,):
                    states = states * factor[:, :, None, :]
                else:
                    states = states * factor
            else:
                left = _evaluate_input(node.left, values, token)
                right = _evaluate_input(node.right, values, token)
                coefficient = _evaluate_input(node.coefficient, values, token)
                contraction = (states * right[:, :, :, None]).sum(dim=2)
                states = (
                    states
                    + coefficient[:, :, None, None]
                    * left[:, :, :, None]
                    * contraction[:, :, None, :]
                )
        for injection in algorithm.transitions[0].injections:
            product = states.new_ones(states.shape)
            for factor in injection.factors:
                value = _evaluate_input(factor.expression, values, token)
                if factor.roles == (row,):
                    product = product * value[:, :, :, None]
                elif factor.roles == (column,):
                    product = product * value[:, :, None, :]
                else:
                    product = product * value[:, :, None, None]
            states = states + product
        for readout in algorithm.readouts:
            query = _evaluate_input(readout.operand, values, token)
            outputs[readout.name].append(
                torch.einsum("bhr,bhrc->bhc", query, states).to(readout.output_dtype)
            )
    return {name: torch.stack(items, dim=2) for name, items in outputs.items()}, states


def _compile(profile, *, optimized, target=None):
    compiled = profile.operator.compile(
        role_sizes=profile.role_sizes,
        target=Target(device_type="cpu") if target is None else target,
        mode="dense",
        optimized=optimized,
    )
    executable = profile.operator._plans[compiled.specialization]
    return compiled, executable


@pytest.mark.parametrize("kind", ("axis", "mixed", "injections", "rank-one"))
def test_unseen_composition_matches_independent_oracle_and_uses_parallel_plan(kind):
    profile = make_profile(kind)
    compiled, executable = _compile(profile, optimized=True)
    values = make_inputs(profile, batch=2, length=9, device=torch.device("cpu"))
    initial = 0.05 * torch.randn(2, 4, 64, 128)
    actual = compiled(
        **values,
        initial_state=StateTuple(("state",), (initial,)),
        return_final_state=True,
    )
    expected_outputs, expected_state = _oracle(profile, values, initial=initial)
    for name, expected in expected_outputs.items():
        torch.testing.assert_close(actual.outputs[name], expected, rtol=0.04, atol=0.1)
    assert actual.final_state is not None
    torch.testing.assert_close(
        actual.final_state["state"], expected_state, rtol=0.04, atol=0.1
    )

    if torch.cuda.is_available():
        plan = executable.plan
        assert plan.candidate_estimates
        selected = dict(plan.candidate_estimates)[plan.candidate.identity]
        assert selected.cost == min(
            estimate.cost for _, estimate in plan.candidate_estimates
        )
    assert executable.runtime_family != "stateful-reference"

    _, generic = _compile(profile, optimized=False)
    assert generic.plan.candidate.lowerer == "tilelang-generic"


def test_renaming_roles_and_inputs_does_not_change_selected_runtime_family():
    original = make_profile("mixed")
    renamed = make_profile("mixed", renamed=True)
    _, original_executable = _compile(original, optimized=True)
    _, renamed_executable = _compile(renamed, optimized=True)
    assert renamed_executable.runtime_family == original_executable.runtime_family
    assert (
        renamed_executable.plan.mathematical_plan.representation
        == original_executable.plan.mathematical_plan.representation
    )
