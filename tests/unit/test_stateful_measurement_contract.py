"""Behavioral regression tests for the Stateful performance measurement contract."""

from dataclasses import replace

import pytest
import torch

from testing.benchmark_stateful_plans import (
    CASES,
    _correctness,
    _split_inputs,
    _token_oracle,
    inputs_for,
    role_sizes_for,
)

pytestmark = pytest.mark.unit


def _workload(name, length=5):
    case = replace(
        next(case for case in CASES if case.name == name),
        batch=2,
        heads=2,
        dim=3,
        dim_value=2,
    )
    operator = case.factory(**case.kwargs(length))
    sizes = role_sizes_for(operator, case, length)
    torch.manual_seed(37)
    inputs = inputs_for(operator, case, length, torch.device("cpu"))
    return operator, sizes, inputs


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_linear_oracle_preserves_input_expression_rounding(dtype):
    operator, sizes, inputs = _workload("linear", length=3)
    inputs = {name: value.to(dtype) for name, value in inputs.items()}
    outputs, states = _token_oracle(operator, inputs, sizes)
    scale = torch.tensor(inputs["query"].shape[-1] ** -0.5)
    query = (inputs["query"].float() * scale).to(dtype).float()
    key = (inputs["key"].float() + torch.tensor(0.1)).to(dtype).float()
    value = inputs["value"].float()
    state = torch.zeros_like(states["memory"])
    expected = []
    for token in range(query.shape[2]):
        state = state + key[:, :, token, :, None] * value[:, :, token, None, :]
        expected.append((query[:, :, token, :, None] * state).sum(-2))
    torch.testing.assert_close(states["memory"], state, rtol=0, atol=0)
    torch.testing.assert_close(
        outputs["output"], torch.stack(expected, dim=2), rtol=0, atol=0
    )


def test_missing_final_state_is_invalid_only_for_state_contracts():
    expected = {"output": torch.tensor([[[[2.0]]]])}
    states = {"memory": torch.tensor([[[[3.0]]]])}
    assert (
        _correctness(expected, None, expected, states, "output-only")["status"]
        == "VALID"
    )
    assert (
        _correctness(expected, None, expected, states, "final-state")["status"]
        == "INVALID"
    )
    assert (
        _correctness(expected, None, expected, states, "continuation")["status"]
        == "INVALID"
    )


@pytest.mark.parametrize(
    "name",
    [
        "linear",
        "simple-gla",
        "gla",
        "gdn",
        "gsa",
        "based",
        "rebased",
        "rwkv6",
        "rwkv7",
        "hgrn",
        "delta-product",
    ],
)
def test_token_recurrence_continuation_preserves_all_states(name):
    operator, sizes, inputs = _workload(name)
    full_outputs, full_states = _token_oracle(operator, inputs, sizes)
    prefix_inputs, suffix_inputs = _split_inputs(operator, inputs, 2)
    prefix_outputs, prefix_states = _token_oracle(operator, prefix_inputs, sizes)
    suffix_outputs, suffix_states = _token_oracle(
        operator,
        suffix_inputs,
        sizes,
        initial_state=prefix_states,
    )
    for output_name, expected in full_outputs.items():
        torch.testing.assert_close(
            torch.cat(
                (prefix_outputs[output_name], suffix_outputs[output_name]), dim=2
            ),
            expected,
            rtol=0,
            atol=0,
        )
    for state_name, expected in full_states.items():
        torch.testing.assert_close(suffix_states[state_name], expected, rtol=0, atol=0)


def test_rank_one_oracle_applies_decay_before_removal():
    operator, sizes, inputs = _workload("gdn", length=1)
    initial = (
        torch.arange(1, 13, dtype=torch.float32)
        .reshape(2, 2, 3, 1)
        .expand(-1, -1, -1, 2)
        .clone()
    )
    _, actual = _token_oracle(
        operator, inputs, sizes, initial_state={"memory": initial}
    )
    key = inputs["key"][:, :, 0].float()
    value = inputs["value"][:, :, 0].float()
    beta = inputs["beta"][:, :, 0].float()
    decayed = initial * inputs["gate"][:, :, 0].float().exp()[:, :, None, None]
    removed = torch.einsum("bhk,bhkv->bhv", key, decayed)
    expected = decayed + beta[:, :, None, None] * key.unsqueeze(-1) * (
        value - removed
    ).unsqueeze(-2)
    torch.testing.assert_close(actual["memory"], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_based_oracle_matches_polynomial_attention(dtype):
    operator, sizes, inputs = _workload("based")
    inputs = {name: value.to(dtype) for name, value in inputs.items()}
    actual, _ = _token_oracle(operator, inputs, sizes)
    q, k, v = (inputs[name] for name in ("query", "key", "value"))
    q = (q.float() * torch.tensor(q.shape[-1] ** -0.5)).to(dtype)
    scores = torch.einsum("bhtk,bhsk->bhts", q.float(), k.float())
    if dtype == torch.float32:
        quadratic = scores.square()
    else:
        # Authored BF16 Outer nodes round before the FP32 State algebra.
        # Compare all causal token pairs independently of the token recurrence.
        q2 = (q.unsqueeze(-1) * q.unsqueeze(-2)).flatten(-2).float()
        k2 = (k.unsqueeze(-1) * k.unsqueeze(-2)).flatten(-2).float()
        quadratic = torch.einsum("bhtk,bhsk->bhts", q2, k2)
    weights = (1 + scores + 0.5 * quadratic).tril()
    expected = torch.einsum("bhts,bhsv->bhtv", weights, v.float()) / (
        weights.sum(-1, keepdim=True) + 1e-6
    )
    torch.testing.assert_close(actual["output"], expected, rtol=1e-6, atol=1e-6)


def test_delta_product_oracle_preserves_interleaved_writes_and_readout_timing():
    from examples.gated_delta_product import (
        gated_delta_product,
        STATE_HEADS,
        KEY,
        VALUE,
    )
    from attn_engine import AlgorithmIR, Readout, StatefulOperator

    operator = gated_delta_product(1, 1, 1, 2, 2, dtype=torch.float32)
    inputs = {
        "query": torch.tensor([[[[1.0, 0.0]]]]),
        "gate": torch.zeros(1, 1, 1),
        "key_0": torch.tensor([[[[1.0, 0.0]]]]),
        "key_1": torch.tensor([[[[0.5, 0.5]]]]),
        "value_0": torch.tensor([[[[1.0, 2.0]]]]),
        "value_1": torch.zeros(1, 1, 1, 2),
        "beta_0": torch.ones(1, 1, 1),
        "beta_1": torch.ones(1, 1, 1),
    }
    algorithm = operator.algorithm
    readout = algorithm.readouts[0]
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=algorithm.inputs,
            states=algorithm.states,
            transition=algorithm.transitions,
            readouts=(
                readout,
                Readout("prior", replace(readout.expression, prior=True)),
            ),
            head_mapping=algorithm.head_mapping,
        )
    )
    outputs, states = _token_oracle(
        operator, inputs, {STATE_HEADS: 1, KEY: 2, VALUE: 2}
    )
    expected_state = torch.tensor([[[[0.75, 1.5], [-0.25, -0.5]]]])
    torch.testing.assert_close(states["memory"], expected_state, rtol=0, atol=0)
    torch.testing.assert_close(outputs["output"], expected_state[:, :, :1, :] * 2**-0.5)
    torch.testing.assert_close(outputs["prior"], torch.zeros(1, 1, 1, 2))

    # A legacy grouped transition intentionally yields a different result.
    from attn_engine import StateTransition

    grouped = StatefulOperator(
        AlgorithmIR(
            inputs=algorithm.inputs,
            states=algorithm.states,
            transition=StateTransition(
                "memory",
                propagations=algorithm.transition.propagations,
                injections=algorithm.transition.injections,
            ),
            readouts=algorithm.readouts,
            head_mapping=algorithm.head_mapping,
        )
    )
    _, grouped_states = _token_oracle(
        grouped, inputs, {STATE_HEADS: 1, KEY: 2, VALUE: 2}
    )
    torch.testing.assert_close(
        grouped_states["memory"], torch.tensor([[[[1.0, 2.0], [0.0, 0.0]]]])
    )


@pytest.mark.parametrize("steps", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("nonzero_initial", [False, True])
def test_delta_product_oracle_matches_independent_residual_recurrence(
    steps, dtype, nonzero_initial
):
    from tests.unit.test_delta_family_examples import _case

    operator, sizes, inputs, initial = _case("delta_product", dtype=dtype, steps=steps)
    if not nonzero_initial:
        initial = {name: torch.zeros_like(state) for name, state in initial.items()}
    state = initial["memory"].clone()
    heads = state.shape[1]
    values = {
        name: value.repeat_interleave(heads // value.shape[1], dim=1)
        for name, value in inputs.items()
    }
    # Independent residual updates preserve the authored query's BF16 scaling
    # before FP32 contraction; the beta/value expression promotes to FP32.
    scale = torch.tensor(values["query"].shape[-1] ** -0.5)
    query = (values["query"].float() * scale).to(dtype).float()
    expected = []
    for token in range(query.shape[2]):
        state = state * values["gate"][:, :, token].exp()[..., None, None]
        for step in range(steps):
            key = values[f"key_{step}"][:, :, token].float()
            value = values[f"value_{step}"][:, :, token].float()
            beta = values[f"beta_{step}"][:, :, token].float()
            residual = value - torch.einsum("bhk,bhkv->bhv", key, state)
            state = state + key.unsqueeze(-1) * (
                beta.unsqueeze(-1) * residual
            ).unsqueeze(-2)
        expected.append(torch.einsum("bhk,bhkv->bhv", query[:, :, token], state))
    expected = torch.stack(expected, dim=2)
    expected_states = {"memory": state}
    outputs, states = _token_oracle(
        operator, inputs, sizes, initial_state=initial if nonzero_initial else None
    )
    torch.testing.assert_close(outputs["output"], expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        states["memory"], expected_states["memory"], rtol=1e-6, atol=1e-6
    )
