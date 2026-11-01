"""Candidate ranking by a shape-aware estimate.

`select_candidate` must rank by what a plan actually costs for the concrete
Feature sizes and IR structure, not by a per-lowerer constant, so a Program
whose summary is diagonal or whose propagation is a rank-one contraction can
never fall back to a dense scan.
"""

from __future__ import annotations

import pytest

from attn_engine import FeatureRole, HeadRole, ProductFeatureRole, Target
from examples.based_attention import based_attention
from examples.gated_delta_rule import compositional_pressure_profile, gated_delta_rule
from examples.linear_attention import (
    dual_gate_gla_attention,
    gla_attention,
    linear_attention,
    simple_gla_attention,
)
from examples.rebased_attention import rebased_attention
from examples.hgrn import hgrn

pytestmark = pytest.mark.unit

HEADS = 8
DENSE_SCAN = {"tilelang-generic", "reference-stateful"}


@pytest.fixture(autouse=True)
def _allow_cpu_stateful_operator(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")


def _sizes(operator, rows: int, columns: int) -> dict:
    sizes: dict = {}
    for spec in (*operator.algorithm.inputs, *operator.algorithm.states):
        for role in spec.roles:
            if isinstance(role, HeadRole):
                sizes.setdefault(role, HEADS)
            elif isinstance(role, ProductFeatureRole):
                continue
            elif isinstance(role, FeatureRole):
                if role.name == "key_dim":
                    sizes[role] = rows
                elif role.name == "value_dim":
                    sizes[role] = columns
                elif role.name == "unit_dim":
                    sizes[role] = 1
                else:
                    sizes.setdefault(role, columns)
    for spec in (*operator.algorithm.inputs, *operator.algorithm.states):
        for role in spec.roles:
            if isinstance(role, ProductFeatureRole):
                sizes[role] = sizes[role.left] * sizes[role.right]
    return sizes


def _plan(factory, *, rows: int = 128, columns: int = 128, **options):
    head_kwargs = {"HQ": 1} if factory is gated_delta_rule else {}
    operator = factory(1, HEADS, 512, rows, columns, **head_kwargs)
    compiled = operator.compile(
        role_sizes=_sizes(operator, rows, columns),
        target=options.pop("target", None) or Target(),
        mode="dense",
        **options,
    )
    return compiled.operator._plans[compiled.specialization].plan


def test_feature_vector_gla_selects_vector_propagation():
    plan = _plan(gla_attention, rows=64, columns=64)
    assert plan.candidate.lowerer == "tilelang-vector-propagation"
    representations = {item.representation for item in plan.mathematical_plans}
    assert "vector-propagation" in representations
    assert "axis-scale" in representations


def test_product_feature_program_selects_chunked_runtime():
    for factory, dim in ((based_attention, 16), (rebased_attention, 16)):
        plan = _plan(factory, rows=dim, columns=64)
        assert plan.candidate.lowerer == "tilelang-product-feature-chunked"
        assert plan.candidate.lowerer not in DENSE_SCAN
        assert plan.mathematical_plan.representation == "product-feature-chunked"


def test_feature_vector_plan_remains_vector_candidate_for_chunked_runtime():
    plan = _plan(gla_attention, rows=64, columns=64)
    assert plan.candidate.lowerer == "tilelang-vector-propagation"
    assert plan.workspace.bytes_per_token == 0


def test_diagonal_decay_never_selects_a_dense_scan():
    plan = _plan(gla_attention)
    assert plan.candidate.lowerer not in DENSE_SCAN
    selected = dict(plan.candidate_estimates)[plan.candidate.identity]
    assert selected.cost == min(
        estimate.cost for _, estimate in plan.candidate_estimates
    )


def test_rank_one_propagation_never_selects_a_dense_scan():
    plan = _plan(gated_delta_rule)
    assert plan.candidate.lowerer == "tilelang-factored-recurrence"
    assert plan.candidate.lowerer not in DENSE_SCAN
    selected = dict(plan.candidate_estimates)[plan.candidate.identity]
    assert selected.cost == min(
        estimate.cost for _, estimate in plan.candidate_estimates
    )


def test_rank_one_metadata_records_symmetric_and_asymmetric_proofs():
    symmetric_operator = gated_delta_rule(1, HEADS, 65, 16, 16, HQ=1)
    symmetric_compiled = symmetric_operator.compile(
        role_sizes=_sizes(symmetric_operator, 16, 16), target=Target(), mode="dense"
    )
    symmetric_executable = symmetric_compiled.operator._plans[
        symmetric_compiled.specialization
    ]
    assert symmetric_executable.rank_one_factored is not None
    assert symmetric_executable.rank_one_factored.symmetric_rank_one is True
    assert symmetric_executable.symmetric_rank_one is True

    asymmetric = compositional_pressure_profile(B=1, H=HEADS, S=65, D=16, DV=16, HQ=1)
    compiled = asymmetric.compile(
        role_sizes=_sizes(asymmetric, 16, 16), target=Target(), mode="dense"
    )
    asymmetric_executable = compiled.operator._plans[compiled.specialization]
    assert asymmetric_executable.rank_one_factored is not None
    assert asymmetric_executable.rank_one_factored.symmetric_rank_one is False
    assert asymmetric_executable.symmetric_rank_one is False


def test_scalar_decay_with_constant_key_offset_keeps_factorized_selection():
    plan = _plan(simple_gla_attention)
    assert plan.candidate.lowerer == "tilelang-scalar-factorized"
    costs = [estimate.cost for _, estimate in plan.candidate_estimates]
    assert costs == sorted(costs)


def test_rank_one_identity_readout_uses_rank_one_diagonal_candidate():
    operator = hgrn(1, 1, 512, 64)
    compiled = operator.compile(
        role_sizes=_sizes(operator, 64, 1),
        target=Target(),
        mode="dense",
        optimized=True,
    )
    plan = compiled.operator._plans[compiled.specialization].plan
    assert plan.candidate.lowerer == "tilelang-rank-one-diagonal"
    assert plan.mathematical_plan.representation == "rank-one-diagonal"


@pytest.mark.parametrize(
    ("name", "representation"),
    [
        ("rwkv6", "native-diagonal"),
        ("gsa", "native-diagonal"),
        ("normalized-linear", "native-normalized"),
    ],
)
def test_reference_hotspots_select_typed_native_candidate(name, representation):
    from testing.benchmark_stateful_plans import CASES, role_sizes_for

    case = next(case for case in CASES if case.name == name)
    operator = case.factory(**case.kwargs(65))
    compiled = operator.compile(
        role_sizes=role_sizes_for(operator, case, 65),
        target=Target(),
        mode="dense",
        optimized=True,
    )
    plan = compiled.operator._plans[compiled.specialization].plan
    assert plan.candidate.lowerer not in DENSE_SCAN
    assert plan.mathematical_plan.representation == representation


def test_estimates_are_inspectable_with_all_three_components():
    plan = _plan(linear_attention)
    assert plan.candidate_estimates
    for identity, estimate in plan.candidate_estimates:
        assert identity
        assert estimate.bytes_per_token > 0
        assert estimate.flops_per_token > 0
        assert estimate.launches >= 1
    assert plan.candidate_estimates[0][0] == plan.candidate.identity


def test_infeasible_candidate_leaves_a_lower_ranked_plan():
    # an envelope below one candidate's per-token buffer removes it before
    # ranking while the remaining candidates still lower
    from attn_engine.stateful.compile.plan import (
        EstimationShape,
        LOWERERS,
        MathematicalPlan,
        select_candidate,
    )

    shape = EstimationShape.of(heads=8, feature_sizes=(128, 64), dense_summary=False)
    lowerers = {lowerer.name: lowerer for lowerer in LOWERERS}
    plans = {
        name: MathematicalPlan(name, lowerer.representations[0], (), ("t",))
        for name, lowerer in lowerers.items()
    }
    budget = 786_432  # fits the elementwise summary, not the generic identities
    lowered = {}
    estimates = {}
    for name in ("tilelang-generic", "tilelang-elementwise"):
        lowerer = lowerers[name]
        estimate = lowerer.estimate(plans[name], shape)
        estimates[name] = estimate
        assert estimate.workspace_bytes_per_token > 0
        candidate = lowerer.lower(plans[name], shape, maximum_workspace_bytes=budget)
        if candidate is not None:
            lowered[name] = candidate
    assert "tilelang-generic" not in lowered
    assert "tilelang-elementwise" in lowered
    selected = select_candidate(
        tuple(lowered.values()), {lowered[name]: estimates[name] for name in lowered}
    )
    assert selected is lowered["tilelang-elementwise"]


def test_optimization_disabled_selects_generic_execution_only():
    plan = _plan(gla_attention)
    assert plan.candidate.lowerer != "tilelang-generic"
    unoptimized = _plan(gla_attention, optimized=False)
    assert unoptimized.candidate.lowerer == "tilelang-generic"


def test_two_scaled_axes_lower_without_a_dense_summary():
    """Both axes scaled is still diagonal: one factor per axis, not a summary pair."""
    from attn_engine.stateful_operator import _dense_summary_required

    operator = dual_gate_gla_attention(1, HEADS, 32, 64, 64)
    assert _dense_summary_required(operator.algorithm) is False
    compiled = operator.compile(
        role_sizes=_sizes(operator, 64, 64),
        target=Target(),
        mode="dense",
        optimized=True,
    )
    executable = compiled.operator._plans[compiled.specialization]
    assert executable.dense_summary is False

    # a compilation that asks for a backward uses the same pair, whose VJP
    # contracts each factor along the axis the other spans
    backward_compiled = operator.compile(
        role_sizes=_sizes(operator, 64, 64),
        target=Target(),
        mode="dense",
        optimized=True,
        backward=True,
    )
    assert (
        backward_compiled.operator._plans[
            backward_compiled.specialization
        ].dense_summary
        is False
    )
