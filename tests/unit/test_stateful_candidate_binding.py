"""Published and tuning callables bind the same complete candidate contract."""

from dataclasses import asdict, replace

import pytest
import torch

from attn_engine import StatefulCompilationError, Target
from attn_engine.stateful.compile import benchmark, tuning
from attn_engine.stateful.compile.binding import CandidateBinding
from attn_engine.stateful.compile.plan import (
    ContextValue,
    WorkloadPoint,
    backward_context_family,
    kernel_configurations,
)
from attn_engine.stateful.lowering.chunked import select_chunked_schedule
from attn_engine.stateful.runtime.workspace import WorkspacePolicy
from attn_engine.stateful_operator import CompiledStatefulOperator
from testing.benchmark_stateful_plans import CASES, role_sizes_for

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def allow_cpu(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")
    monkeypatch.delenv("STATEFUL_OPERATOR_CACHE_DIR", raising=False)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1 << 40, 1 << 40))


@pytest.fixture
def bindings(monkeypatch):
    recorded = []
    bind = CandidateBinding.bind

    def record(self, candidate, config):
        recorded.append(self)
        return bind(self, candidate, config)

    monkeypatch.setattr(CandidateBinding, "bind", record)
    return recorded


def compile_case(name, *, mode="dense", backward=False, autotune=False):
    case = replace(next(case for case in CASES if case.name == name), heads=1)
    operator = case.factory(**case.kwargs(5))
    return operator.compile(
        role_sizes=role_sizes_for(operator, case, 5),
        target=Target(),
        mode=mode,
        backward=backward,
        autotune=autotune,
    )


@pytest.mark.parametrize(
    "name", ["linear", "simple-gla", "gla", "gdn", "oja", "hgrn", "abc", "based"]
)
@pytest.mark.parametrize("mode", ["dense", "packed"])
@pytest.mark.parametrize("backward", [False, True])
def test_every_candidate_binds_its_own_complete_contract(
    bindings, name, mode, backward
):
    compiled = compile_case(name, mode=mode, backward=backward)
    binding = bindings[0]
    published = compiled._executable
    original = asdict(published.plan)
    for candidate in binding.candidates:
        configs = kernel_configurations(candidate, binding.device_profile)
        for config in (configs[0], configs[-1]):
            executable = binding.bind(candidate, config)
            plan = executable.plan
            assert plan.candidate == candidate
            assert plan.mathematical_plan.identity == candidate.plan_identity
            assert plan.kernel_config.values == config
            assert executable._execution.executable is executable
            assert executable.rank_one_factored is (
                binding.rank_one_factored
                if candidate.lowerer == "tilelang-factored-recurrence"
                else None
            )
            assert executable.symmetric_rank_one == bool(
                executable.rank_one_factored is not None
                and executable.rank_one_factored.symmetric_rank_one
            )
            assert executable.chunked_recipe == select_chunked_schedule(
                binding.chunked_recipe, candidate.lowerer
            )
            assert plan.workspace == WorkspacePolicy.for_candidate(
                candidate,
                binding.estimates[candidate],
                compilation=compiled.compilation,
                algorithm=binding.program,
                rank_one_factored=binding.rank_one_factored,
            )
            assert executable.composition.forward_candidate == candidate
            assert executable.composition.backward_candidate == plan.backward_candidate
            assert executable.composition.context_contract == plan.context_contract
            if executable.backward_binding is not None:
                backward_binding = executable.backward_binding
                assert backward_binding.family == backward_context_family(
                    binding.program, candidate, binding.dense_summary
                )
                assert backward_binding.config == plan.backward_kernel_config.values
                assert plan.context_contract.required == tuple(
                    ContextValue(*value) for value in backward_binding.context_values
                )
            else:
                assert plan.backward_candidate is None
                assert plan.backward_kernel_config is None
                assert plan.context_contract.required == ()
    assert asdict(published.plan) == original
    assert (
        binding.bind(published.plan.candidate, published.plan.kernel_config.values)
        == published
    )


def test_binding_keeps_unselected_native_recipes(bindings):
    compile_case("abc")
    binding = bindings[0]
    native = binding.chunked_recipe.native_diagonal
    assert native.fused_softmax
    cut = next(c for c in binding.candidates if c.lowerer.endswith("readout-cut"))
    fused = next(
        c for c in binding.candidates if c.lowerer == "tilelang-native-diagonal"
    )
    cut_executable = binding.bind(cut, kernel_configurations(cut)[0])
    assert not cut_executable.chunked_recipe.native_diagonal.fused_softmax
    fused_executable = binding.bind(fused, kernel_configurations(fused)[0])
    assert fused_executable.chunked_recipe.native_diagonal.fused_softmax
    assert binding.chunked_recipe.native_diagonal is native


def test_binding_rejects_a_candidate_from_another_compilation(bindings):
    compile_case("simple-gla")
    binding = bindings[0]
    foreign = replace(binding.candidates[0], identity="foreign")
    with pytest.raises(ValueError, match="does not belong"):
        binding.bind(foreign, ())


@pytest.mark.parametrize("mode", ["dense", "packed"])
@pytest.mark.parametrize("backward", [False, True])
def test_trial_is_unpublished_and_uses_normal_validation(bindings, mode, backward):
    compiled = compile_case("simple-gla", mode=mode, backward=backward)
    binding = bindings[0]
    owner = compiled.operator
    plans = dict(owner._plans)
    specializations = set(owner._specializations)
    defaults = dict(owner._zero_states)
    candidate = compiled._executable.plan.candidate
    executable = binding.bind(candidate, kernel_configurations(candidate)[-1])
    trial = CompiledStatefulOperator._from_executable(
        owner, compiled.compilation, executable
    )
    assert trial.operator is owner
    assert trial._executable is executable
    assert trial.specialization not in owner._plans
    assert trial.workspace_bytes(
        10, sequence_count=2, sequence_length=5
    ) == executable.plan.workspace.required_bytes(
        10, sequence_count=2, sequence_length=5
    )
    with pytest.raises(StatefulCompilationError) as error:
        trial()
    assert error.value.category == "RUNTIME_BINDING"
    assert owner._plans == plans
    assert owner._specializations == specializations
    assert owner._zero_states == defaults


@pytest.mark.parametrize("mode", ["dense", "packed"])
def test_tuning_rejects_unsafe_workloads_before_allocating(monkeypatch, bindings, mode):
    compiled = compile_case("simple-gla", mode=mode)
    build = benchmark.invocation_factory(
        compiled.operator, compiled.compilation, bindings[0], torch.device("cpu")
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1, 1))

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("unsafe tuning workload allocated inputs")

    monkeypatch.setattr(torch, "full", unexpected_allocation)
    with pytest.raises(RuntimeError, match="safe resource envelope"):
        build(compiled._executable.plan.candidate, (), WorkloadPoint(2, 10, "uniform"))


@pytest.mark.parametrize("mode", ["dense", "packed"])
def test_failed_trial_is_discarded_without_mutating_live_handles(
    monkeypatch, bindings, mode
):
    compiled = compile_case("simple-gla", mode=mode)
    binding = bindings[0]
    plans = dict(compiled.operator._plans)
    generic = next(c for c in binding.candidates if c.lowerer == "tilelang-generic")
    specialized = compiled._executable.plan.candidate
    invoke = CompiledStatefulOperator.__call__

    def fail_specialized(self, **kwargs):
        if self._executable.plan.candidate == specialized:
            raise RuntimeError("trial execution failed")
        return invoke(self, **kwargs)

    monkeypatch.setattr(CompiledStatefulOperator, "__call__", fail_specialized)
    factory = benchmark.invocation_factory(
        compiled.operator, compiled.compilation, binding, torch.device("cpu")
    )

    def timer(call):
        call()
        return 1.0

    with pytest.warns(RuntimeWarning, match="discarded failed attempts"):
        result = tuning.tune_candidates(
            (generic, specialized),
            {c: kernel_configurations(c)[:1] for c in (generic, specialized)},
            (WorkloadPoint(2, 10, "uniform"),),
            factory,
            autotune=True,
            mode=mode,
            timer=timer,
        )
    assert result.winner == generic
    assert all(f.candidate == specialized.identity for f in result.failures)
    assert compiled.operator._plans == plans
    assert compiled.operator._zero_states == {}


@pytest.mark.parametrize("mode", ["dense", "packed"])
@pytest.mark.parametrize("backward", [False, True])
def test_autotuning_publishes_exactly_the_measured_binding_and_reuses_cache(
    monkeypatch, tmp_path, mode, backward
):
    from attn_engine.stateful.compile import plan as planner

    monkeypatch.setenv("STATEFUL_OPERATOR_CACHE_DIR", str(tmp_path))
    workload = WorkloadPoint(
        2, 10, "uniform" if mode == "dense" else "empty-containing"
    )
    monkeypatch.setattr(
        planner,
        "WORKLOAD_POLICY",
        planner.WorkloadPolicy(
            planner.WORKLOAD_POLICY.version, (workload,), (workload,)
        ),
    )
    invocation_factory = benchmark.invocation_factory
    tune_candidates = tuning.tune_candidates
    measured = []
    invoke = CompiledStatefulOperator.__call__

    def record_trial(self, **kwargs):
        assert not torch.is_grad_enabled()
        measured.append(self._executable)
        assert self.specialization not in self.operator._plans
        return invoke(self, **kwargs)

    def cpu_factory(operator, compilation, binding, device):
        return invocation_factory(operator, compilation, binding, torch.device("cpu"))

    def timer(call):
        call()
        return 1.0

    def cpu_tune(*args, **kwargs):
        return tune_candidates(*args, **kwargs, timer=timer)

    monkeypatch.setattr(CompiledStatefulOperator, "__call__", record_trial)
    monkeypatch.setattr(benchmark, "invocation_factory", cpu_factory)
    monkeypatch.setattr(tuning, "tune_candidates", cpu_tune)
    compiled = compile_case("simple-gla", mode=mode, backward=backward, autotune=True)
    assert measured
    assert any(trial == compiled._executable for trial in measured)
    assert list((tmp_path / "binaries").glob("*.json")) == [
        tmp_path / "binaries" / f"{compiled._executable.plan.binary_identity}.json"
    ]
    assert len(compiled.operator._plans) == 1
    assert compiled.operator._zero_states == {}
    count = len(measured)
    cached = compile_case("simple-gla", mode=mode, backward=backward, autotune=True)
    assert len(measured) == count
    assert cached._executable.plan == compiled._executable.plan
    assert cached._executable.backward_binding == compiled._executable.backward_binding
    assert cached._executable.runtime_family == compiled._executable.runtime_family


def test_workload_policy_changes_only_tuning_identity(bindings):
    compiled = compile_case("simple-gla")
    binding = replace(bindings[0], policy_version="another-policy")
    plan = compiled._executable.plan
    rebound = binding.bind(plan.candidate, plan.kernel_config.values)
    assert rebound.plan.tuning_identity != plan.tuning_identity
    assert rebound.plan.binary_identity == plan.binary_identity
    assert rebound.plan.kernel_config == plan.kernel_config
    assert rebound.plan.context_contract == plan.context_contract
    assert rebound.plan.workspace == plan.workspace


@pytest.mark.parametrize("mode", ["dense", "packed"])
def test_all_failed_trials_publish_neither_measurements_nor_artifacts(
    monkeypatch, tmp_path, bindings, mode
):
    compiled = compile_case("simple-gla", mode=mode)
    binding = bindings[0]
    monkeypatch.setenv("STATEFUL_OPERATOR_CACHE_DIR", str(tmp_path))
    factory = benchmark.invocation_factory(
        compiled.operator, compiled.compilation, binding, torch.device("cpu")
    )

    def fail(self, **kwargs):
        raise RuntimeError("no trial is executable")

    monkeypatch.setattr(CompiledStatefulOperator, "__call__", fail)
    candidate = compiled._executable.plan.candidate
    with pytest.raises(tuning.GenericBaselineError) as error:
        tuning.tune_candidates(
            (candidate,),
            {candidate: kernel_configurations(candidate)[:1]},
            (WorkloadPoint(2, 10, "uniform"),),
            factory,
            autotune=True,
            mode=mode,
            cache=tuning.MeasurementCache(tmp_path / "measurements.json"),
            timer=lambda call: 1.0,
        )
    assert error.value.failures
    assert not list(tmp_path.rglob("*.json"))
    assert compiled.operator._plans == {compiled.specialization: compiled._executable}
