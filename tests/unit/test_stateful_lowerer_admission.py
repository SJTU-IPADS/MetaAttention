"""Lowerers own implementation admission; validation shares their structural facts."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from attn_engine import StatefulCompilationError, Target
from attn_engine.stateful.compile import plan as planner
from attn_engine.stateful.compile.capabilities import LoweringFacts
from attn_engine.stateful.lowering import chunked, structured
from attn_engine.stateful_operator import CompileOptions
from testing.benchmark_stateful_plans import CASES, role_sizes_for

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def cpu_contract(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")


def program(name, *, tail=False):
    case = replace(
        next(case for case in CASES if case.name == name),
        heads=1,
        **({"dim": 13, "dim_value": 19, "dim_slot": 11} if tail else {}),
    )
    operator = case.factory(**case.kwargs(65))
    return operator, role_sizes_for(operator, case, 65)


def compile_program(name, *, mode="dense", backward=False, optimized=True):
    operator, sizes = program(name)
    return operator.compile(
        role_sizes=sizes,
        target=Target(),
        mode=mode,
        backward=backward,
        optimized=optimized,
    )


@pytest.mark.parametrize("name", ["simple-gla", "gla", "abc", "gdn", "hgrn", "based"])
@pytest.mark.parametrize("mode", ["dense", "packed"])
@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("optimized", [False, True])
def test_recipe_is_built_once_and_reused_by_validation_and_binding(
    monkeypatch, name, mode, backward, optimized
):
    calls = []
    compile_recipe = chunked.compile_chunked_recipe

    def record(*args, **kwargs):
        recipe = compile_recipe(*args, **kwargs)
        calls.append(recipe)
        return recipe

    monkeypatch.setattr(chunked, "compile_chunked_recipe", record)
    compiled = compile_program(name, mode=mode, backward=backward, optimized=optimized)
    facts = compiled.compilation.lowering
    assert len(calls) == (1 if mode == "dense" else 0)
    assert facts.chunked_recipe is (calls[0] if calls else None)
    assert facts.chunked_recipe is (calls[0] if calls else None)
    assert compiled.compilation.validation.native_diagonal == (
        optimized and not backward and facts.native_diagonal is not None
    )
    assert len(calls) == (1 if mode == "dense" else 0)
    # Selected schedules may be copies; the capability proof stays unselected.
    if facts.native_diagonal is not None and name == "abc":
        assert facts.native_diagonal.fused_softmax
    assert compiled._executable.chunked_recipe == chunked.select_chunked_schedule(
        facts.chunked_recipe, compiled._executable.plan.candidate.lowerer
    )


def test_new_lowerer_contributes_its_requirement_without_planner_changes(monkeypatch):
    observed = []

    def permitted(facts):
        observed.append(facts)
        return facts.mode == "dense" and not facts.options.backward

    added = planner.CandidateLowerer(
        "test-implementation",
        (planner.PlanRepresentation.GENERIC_DENSE_AFFINE,),
        requires=permitted,
    )
    monkeypatch.setattr(planner, "LOWERERS", (*planner.LOWERERS, added))
    accepted = compile_program("simple-gla")
    candidates = accepted._executable.plan.lowered_candidates
    candidate = next(c for c in candidates if c.lowerer == added.name)
    assert (
        candidate.plan_identity
        == accepted._executable.plan.mathematical_plans[0].identity
    )
    assert candidate.identity in dict(accepted._executable.plan.candidate_estimates)
    for mode, backward in (("dense", True), ("packed", False)):
        rejected = compile_program("simple-gla", mode=mode, backward=backward)
        assert all(
            c.lowerer != added.name
            for c in rejected._executable.plan.lowered_candidates
        )
    assert observed
    assert all(isinstance(facts, LoweringFacts) for facts in observed)


@pytest.mark.parametrize(
    "name,lowerer,accepted",
    [
        ("gla", "tilelang-vector-propagation", True),
        ("simple-gla", "tilelang-vector-propagation", False),
        ("gla", "tilelang-native-diagonal-shared-scores", True),
        ("abc", "tilelang-native-diagonal-shared-scores", False),
        ("abc", "tilelang-native-diagonal-readout-cut", True),
        ("gla", "tilelang-native-diagonal-readout-cut", False),
        ("simple-gla", "tilelang-scalar-factorized", True),
    ],
)
def test_implementation_proposal_checks_the_program_proof(name, lowerer, accepted):
    compiled = compile_program(name)
    executable = compiled._executable
    implementation = next(item for item in planner.LOWERERS if item.name == lowerer)
    plan = planner.MathematicalPlan(
        "proof-test", implementation.representations[0], (), ("state-transition",)
    )
    result = implementation.propose(
        plan, executable.plan.estimation_shape, compiled.compilation.lowering
    )
    assert (result is not None) == accepted
    if accepted:
        candidate, estimate = result
        assert candidate.lowerer == lowerer
        assert estimate == implementation.estimate(
            plan, executable.plan.estimation_shape
        )


@pytest.mark.parametrize("family", ["vector-propagation", "scalar-factorized"])
@pytest.mark.parametrize("mode,backward", [("dense", True), ("packed", True)])
def test_inference_implementations_do_not_propose_for_backward(family, mode, backward):
    compiled = compile_program(
        "gla" if family == "vector-propagation" else "simple-gla",
        mode=mode,
        backward=backward,
    )
    implementation = next(
        item for item in planner.LOWERERS if item.runtime_family == family
    )
    plan = planner.MathematicalPlan(
        "proof-test", implementation.representations[0], (), ()
    )
    assert (
        implementation.propose(
            plan,
            compiled._executable.plan.estimation_shape,
            compiled.compilation.lowering,
        )
        is None
    )


def test_factored_backward_rejects_future_unsupported_operations():
    from attn_engine.stateful.compile.capabilities import factored_backward_supported

    analysis = SimpleNamespace(
        backward=object(),
        rank_one_factored=SimpleNamespace(
            operations=(SimpleNamespace(kind="future-operation"),)
        ),
    )
    facts = LoweringFacts(
        SimpleNamespace(_analysis=analysis),
        (),
        Target(),
        "dense",
        CompileOptions(backward=True),
    )
    assert not factored_backward_supported(facts)
    assert factored_backward_supported(replace(facts, options=CompileOptions()))


def test_each_requested_structural_proof_is_reused(monkeypatch):
    operator, sizes = program("gla")
    facts = LoweringFacts(
        operator._program, tuple(sizes.items()), Target(), "dense", CompileOptions()
    )
    calls = []

    def record(algorithm):
        calls.append(algorithm)
        return True, None

    monkeypatch.setitem(
        structured.RUNTIME_FAMILY_ELIGIBILITY, "vector-propagation", record
    )
    assert facts.eligibility("vector-propagation") == (True, None)
    assert facts.eligibility("vector-propagation") == (True, None)
    assert calls == [operator._program]
    assert facts.eligibility("unregistered") == (True, None)


def test_role_errors_precede_backend_recipe_work(monkeypatch):
    operator, sizes = program("gla")

    def unexpected(*args, **kwargs):
        pytest.fail("invalid roles reached backend recipe compilation")

    monkeypatch.setattr(chunked, "compile_chunked_recipe", unexpected)
    with pytest.raises(StatefulCompilationError) as error:
        operator.compile(role_sizes={}, target=Target(), mode="dense")
    assert error.value.category == "COMPILE_ROLE_SIZE"
    assert error.value.path == "role_sizes"


@pytest.mark.parametrize("backward,optimized", [(True, True), (False, False)])
def test_head_errors_precede_unused_dense_recipe_work(monkeypatch, backward, optimized):
    operator, sizes = program("gla")
    sizes[operator.algorithm.inputs[0].roles[1]] = 3

    def unexpected(*args, **kwargs):
        pytest.fail("head mismatch reached an unused recipe")

    monkeypatch.setattr(chunked, "compile_chunked_recipe", unexpected)
    with pytest.raises(StatefulCompilationError) as error:
        operator.compile(
            role_sizes=sizes,
            target=Target(),
            mode="dense",
            backward=backward,
            optimized=optimized,
        )
    assert error.value.category == "COMPILE_ROLE_SIZE"
    assert error.value.path == "head_mapping"


def test_native_feature_capability_keeps_existing_specialization_errors(monkeypatch):
    monkeypatch.delenv("STATEFUL_OPERATOR_TEST_CPU")
    for name, backward, optimized, accepted in (
        ("gla", False, True, True),
        ("gla", True, True, False),
        ("gla", False, False, False),
        ("simple-gla", False, True, False),
    ):
        operator, sizes = program(name, tail=True)
        if accepted:
            compiled = operator.compile(
                role_sizes=sizes,
                target=Target(),
                mode="dense",
                backward=backward,
                optimized=optimized,
            )
            assert compiled.compilation.validation.native_diagonal
        else:
            with pytest.raises(StatefulCompilationError) as error:
                operator.compile(
                    role_sizes=sizes,
                    target=Target(),
                    mode="dense",
                    backward=backward,
                    optimized=optimized,
                )
            assert error.value.category == "SPECIALIZATION"
            assert error.value.path == "role_sizes"


def test_proposal_applies_resource_gate_after_structural_admission():
    compiled = compile_program("simple-gla")
    implementation = next(
        item for item in planner.LOWERERS if item.name == "tilelang-generic"
    )
    plan = compiled._executable.plan.mathematical_plans[0]
    facts = replace(compiled.compilation.lowering, target=Target(workspace_bytes=0))
    assert (
        implementation.propose(plan, compiled._executable.plan.estimation_shape, facts)
        is None
    )


def test_unrequested_structural_proofs_do_not_run(monkeypatch):
    operator, sizes = program("simple-gla")
    facts = LoweringFacts(
        operator._program, tuple(sizes.items()), Target(), "packed", CompileOptions()
    )

    def unexpected(*args, **kwargs):
        pytest.fail("unused implementation proof was evaluated")

    monkeypatch.setitem(
        structured.RUNTIME_FAMILY_ELIGIBILITY, "vector-propagation", unexpected
    )
    monkeypatch.setattr(chunked, "compile_chunked_recipe", unexpected)
    assert facts.chunked_recipe is None
    assert facts.eligibility("unregistered") == (True, None)


@pytest.mark.parametrize(
    "profile",
    [
        None,
        SimpleNamespace(dense=None, injection_count=1, readout_count=1),
        SimpleNamespace(dense=object(), injection_count=2, readout_count=1),
        SimpleNamespace(dense=object(), injection_count=1, readout_count=2),
    ],
)
def test_scalar_admission_rejects_unpriced_fallbacks(profile):
    from attn_engine.stateful.compile.capabilities import scalar_factorized_supported

    facts = LoweringFacts(
        SimpleNamespace(_analysis=SimpleNamespace(scalar_factorized=profile)),
        (),
        Target(),
        "dense",
        CompileOptions(),
    )
    assert not scalar_factorized_supported(facts)


@pytest.mark.parametrize(
    "mode,target",
    [
        ("packed", Target()),
        ("dense", Target(device_type="cpu")),
    ],
)
def test_vector_admission_keeps_device_and_mode_limits(mode, target):
    from attn_engine.stateful.compile.capabilities import vector_propagation_supported

    facts = LoweringFacts(None, (), target, mode, CompileOptions())
    assert not vector_propagation_supported(facts)
