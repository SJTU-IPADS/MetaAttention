"""Contracts for the synthetic expression envelope and its independent oracle."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from attn_engine import StateTuple, Target
from attn_engine.stateful.compile.plan import WorkspacePolicy
from testing.benchmark_stateful_generalization import assess, benchmark, summarize
from testing.stateful_generalization_profiles import (
    FAMILIES,
    Mutation,
    check_result,
    complexity,
    make_inputs,
    make_operator,
    oracle,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("level", [0, 1, 3])
def test_generated_program_matches_independent_token_recurrence(
    monkeypatch, family, level
):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")
    torch.set_num_threads(1)
    operator, sizes = make_operator(
        Mutation(family, level, seed=2027), heads=2, rows=7, columns=11
    )
    values, initial = make_inputs(
        operator, sizes, batch=2, length=9, seed=43, device="cpu"
    )
    compiled = operator.compile(
        role_sizes=sizes, target=Target(device_type="cpu"), mode="dense"
    )
    actual = compiled(
        **values,
        initial_state=StateTuple(("memory",), (initial,)),
        return_final_state=True,
    )
    expected, final = oracle(operator, sizes, values, initial)
    check_result(actual, expected, final)
    # The invocation must not mutate a caller's continuation state.
    _, same_initial = make_inputs(
        operator, sizes, batch=2, length=9, seed=43, device="cpu"
    )
    torch.testing.assert_close(initial, same_initial, rtol=0, atol=0)


@pytest.mark.parametrize("family", FAMILIES)
def test_complexity_levels_are_monotonic_and_zero_level_is_shared(family):
    operators = [make_operator(Mutation(family, level))[0] for level in (0, 1, 2, 4)]
    counts = [complexity(operator) for operator in operators]
    assert all(
        b["expression_nodes"] > a["expression_nodes"]
        for a, b in zip(counts, counts[1:])
    )
    control, _ = make_operator(Mutation("expression", 0))
    assert operators[0].algorithm == control.algorithm
    at_four = counts[-1]
    assert at_four["writes"] == (5 if family in ("write", "combined") else 1)
    assert at_four["readouts"] == (5 if family in ("readout", "combined") else 1)
    assert at_four["rank_one_propagations"] == (
        4 if family in ("rank", "combined") else 0
    )


def test_seed_is_reproducible_and_new_inputs_preserve_control_workload():
    first, sizes = make_operator(Mutation("combined", 2, 17))
    second, _ = make_operator(Mutation("combined", 2, 17))
    different, _ = make_operator(Mutation("combined", 2, 19))
    assert first.algorithm == second.algorithm
    assert first.algorithm != different.algorithm
    control, _ = make_operator(Mutation("combined", 0, 17))
    variant_values, initial = make_inputs(
        first, sizes, batch=1, length=5, seed=17, device="cpu"
    )
    control_values, control_initial = make_inputs(
        control, sizes, batch=1, length=5, seed=17, device="cpu"
    )
    for name in control_values:
        torch.testing.assert_close(
            variant_values[name], control_values[name], rtol=0, atol=0
        )
    torch.testing.assert_close(initial, control_initial, rtol=0, atol=0)


def test_paired_ratio_uses_pair_medians_and_rejects_noise():
    result = assess(
        [1, 2, 4, 8], [1.2, 2.4, 4.8, 9.6], max_slowdown=1.25, max_jitter=0.1
    )
    assert result["stable"]
    assert result["variant_over_control"]["median"] == pytest.approx(1.2)
    result = assess([1] * 4, [1, 1, 1, 2], max_slowdown=1.25, max_jitter=0.1)
    assert result["within_latency_budget"]
    assert not result["repeatable"]
    assert not result["stable"]
    result = assess([1] * 4, [2] * 4, max_slowdown=1.25, max_jitter=0.1)
    assert not result["within_latency_budget"]
    assert not result["stable"]


def test_envelope_requires_all_seeds_and_stops_at_first_cliff():
    records = []
    for level in (0, 1, 2, 4):
        for seed in (17, 19):
            records.append(
                dict(
                    family="rank",
                    batch=1,
                    heads=2,
                    rows=7,
                    columns=11,
                    length=65,
                    seed=seed,
                    level=level,
                    status="OK",
                    stable=not (level == 2 and seed == 19),
                )
            )
    (summary,) = summarize(records, seeds=(17, 19))
    assert summary["stable_prefix_levels"] == [0, 1]
    assert summary["largest_stable_tested_level"] == 1
    assert summary["first_unstable_or_failed_level"] == 2
    records[5].update(status="REJECTED", stable=False)
    assert summarize(records, seeds=(17, 19))[0]["first_unstable_or_failed_level"] == 2
    # An omitted seed also closes the envelope.
    del records[5]
    assert summarize(records, seeds=(17, 19))[0]["first_unstable_or_failed_level"] == 2


def test_uncollected_requested_seed_does_not_advance_the_envelope():
    collected = dict(
        family="expression",
        batch=1,
        heads=4,
        rows=64,
        columns=64,
        length=512,
        seed=1701,
        level=0,
        status="OK",
        stable=True,
    )
    (summary,) = summarize([collected], seeds=(1701, 2028))
    assert summary["seeds"] == [1701, 2028]
    assert summary["stable_prefix_levels"] == []
    assert summary["largest_stable_tested_level"] is None
    assert summary["first_unstable_or_failed_level"] == 0
    assert summary["boundary_observed"] is True


def _benchmark_failure(monkeypatch, error):
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr("testing.benchmark_stateful_generalization._prepare", fail)
    return benchmark(
        Mutation("expression", 0, 1701),
        SimpleNamespace(batch=1, heads=4, rows=64, columns=64),
        512,
    )


def test_workspace_limit_is_rejected_before_timing(monkeypatch):
    policy = WorkspacePolicy(bytes_per_token=8, maximum_bytes=1)
    with pytest.raises(RuntimeError, match="resource envelope") as caught:
        policy.required_bytes(2)
    record = _benchmark_failure(monkeypatch, caught.value)
    assert record["status"] == "REJECTED"
    assert record["stable"] is False
    assert record["rejection"] == {
        "category": "RUNTIME_RESOURCE",
        "path": "workspace",
        "details": {"reason": str(caught.value)},
    }


def test_unrelated_runtime_failure_stays_an_error(monkeypatch):
    record = _benchmark_failure(monkeypatch, RuntimeError("CUDA driver lost"))
    assert record["status"] == "ERROR"
    assert record["stable"] is False
    assert "rejection" not in record
