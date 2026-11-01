"""Stage composition is checked through the public compile/invoke interface."""

import pytest
import torch

from attn_engine import (
    AlgorithmIR,
    AxisScale,
    Batch,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    ProductFactor,
    ProductInjection,
    Readout,
    ScanStage,
    ScanStageIR,
    Sequence,
    StageBinding,
    StageOutput,
    StateSpec,
    StateTerm,
    StateTransition,
    StagedStatefulOperator,
    StatefulCompilationError,
    StateTuple,
    Target,
    TensorInput,
)

pytestmark = pytest.mark.unit
H = HeadRole("heads")
F = FeatureRole("feature")


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")


def _stage(name, input_name, state_name, bindings=(), dtype=torch.float32, prior=False):
    algorithm = AlgorithmIR(
        (TensorInput(input_name, (Batch, H, Sequence, F), dtype),),
        (StateSpec(state_name, (Batch, H, F)),),
        StateTransition(
            state_name,
            propagations=(AxisScale(0.7),),
            injections=(ProductInjection((ProductFactor(Input(input_name), (F,)),)),),
        ),
        (
            Readout(
                "trajectory", StateTerm(state_name, 1, None, prior=prior), torch.float32
            ),
        ),
        HeadMapping(),
    )
    return ScanStage(name, algorithm, bindings)


def _program():
    # Deliberately author downstream first. State order stays declaration order.
    return StagedStatefulOperator(
        ScanStageIR(
            (
                _stage(
                    "b",
                    "derived",
                    "second",
                    (StageBinding("derived", "a", "trajectory"),),
                ),
                _stage("a", "x", "first"),
            ),
            (
                StageOutput("out", "b", "trajectory"),
                StageOutput("aux", "a", "trajectory"),
            ),
        )
    )


def _compile(operator, mode="dense", backward=True, workspace=1 << 30):
    return operator.compile(
        role_sizes={H: 2, F: 3},
        target=Target(device_type="cpu", workspace_bytes=workspace),
        mode=mode,
        backward=backward,
    )


@pytest.mark.parametrize("loss_kind", ["output", "final", "both"])
def test_composed_scans_gradients_continuation(loss_kind):
    torch.manual_seed(3)
    x = torch.randn(2, 2, 9, 3, requires_grad=True)
    initial = [torch.randn(2, 2, 3, requires_grad=True) for _ in range(2)]
    compiled = _compile(_program())
    result = compiled(
        x=x,
        initial_state=StateTuple(("second", "first"), initial),
        return_final_state=True,
    )
    b, a = initial
    aux, out = [], []
    for t in range(9):
        a = 0.7 * a + x[:, :, t]
        b = 0.7 * b + a
        aux.append(a)
        out.append(b)
    expected = torch.stack(out, 2)
    torch.testing.assert_close(result.outputs["out"], expected)
    torch.testing.assert_close(result.outputs["aux"], torch.stack(aux, 2))
    for actual, target in zip(result.final_state, (b, a), strict=True):
        torch.testing.assert_close(actual, target)
    loss = (
        sum(t.square().sum() for t in result.final_state)
        if loss_kind == "final"
        else result.outputs["out"].square().sum()
    )
    target = (
        b.square().sum() + a.square().sum()
        if loss_kind == "final"
        else expected.square().sum()
    )
    if loss_kind == "both":
        loss = loss + sum(t.square().sum() for t in result.final_state)
        target = target + b.square().sum() + a.square().sum()
    actual_grad = torch.autograd.grad(loss, (x, *initial), retain_graph=True)
    expected_grad = torch.autograd.grad(target, (x, *initial))
    for actual, wanted in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-4)
    first = compiled(
        x=x[:, :, :4].contiguous(),
        initial_state=StateTuple(("second", "first"), initial),
        return_final_state=True,
    )
    second = compiled(
        x=x[:, :, 4:].contiguous(),
        initial_state=first.final_state,
        return_final_state=True,
    )
    torch.testing.assert_close(
        torch.cat((first.outputs["out"], second.outputs["out"]), 2),
        result.outputs["out"],
    )
    for actual, wanted in zip(second.final_state, result.final_state, strict=True):
        torch.testing.assert_close(actual, wanted)


@pytest.mark.parametrize("offsets", [[0, 0, 3, 3, 8], [0, 0, 0]])
def test_packed_empty_segments(offsets):
    torch.manual_seed(4)
    x = torch.randn(offsets[-1], 2, 3, requires_grad=True)
    initial = [
        torch.randn(len(offsets) - 1, 2, 3, requires_grad=True) for _ in range(2)
    ]
    result = _compile(_program(), "packed")(
        x=x,
        initial_state=StateTuple(("second", "first"), initial),
        sequence_offsets=torch.tensor(offsets, dtype=torch.int32),
        return_final_state=True,
    )
    expected, finals_b, finals_a = [], [], []
    for seq, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        b, a = initial[0][seq], initial[1][seq]
        for token in range(start, end):
            a = 0.7 * a + x[token]
            b = 0.7 * b + a
            expected.append(b)
        finals_b.append(b)
        finals_a.append(a)
    oracle = torch.stack(expected) if expected else x
    torch.testing.assert_close(result.outputs["out"], oracle)
    torch.testing.assert_close(result.final_state[0], torch.stack(finals_b))
    torch.testing.assert_close(result.final_state[1], torch.stack(finals_a))
    gradients = torch.autograd.grad(
        sum(t.sum() for t in result.final_state) + result.outputs["out"].sum(),
        (x, *initial),
    )
    assert all(torch.isfinite(g).all() for g in gradients)


def test_binding_validation_and_cycle():
    a = _stage("a", "x", "first")
    with pytest.raises(StatefulCompilationError, match="IR_TYPE"):
        ScanStageIR(
            (a, _stage("b", "y", "second", (StageBinding("y", "a", "missing"),))),
            (StageOutput("out", "a", "trajectory"),),
        )
    with pytest.raises(StatefulCompilationError, match="IR_TYPE"):
        ScanStageIR(
            (
                a,
                _stage(
                    "b",
                    "y",
                    "second",
                    (StageBinding("y", "a", "trajectory"),),
                    torch.bfloat16,
                ),
            ),
            (StageOutput("out", "a", "trajectory"),),
        )
    with pytest.raises(StatefulCompilationError, match="IR_DEPENDENCY"):
        ScanStageIR(
            (
                _stage("a", "x", "first", (StageBinding("x", "b", "trajectory"),)),
                _stage("b", "y", "second", (StageBinding("y", "a", "trajectory"),)),
            ),
            (StageOutput("out", "a", "trajectory"),),
        )


def test_resource_and_input_contract():
    x = torch.ones(2, 2, 9, 3)
    with pytest.raises(StatefulCompilationError, match="RUNTIME_RESOURCE"):
        _compile(_program(), workspace=1000)(x=x)
    with pytest.raises(StatefulCompilationError, match="RUNTIME_BINDING"):
        _compile(_program())(x=x, derived=x)
    with pytest.raises(StatefulCompilationError, match="RUNTIME_GRAD"):
        _compile(_program(), backward=False)(x=x.requires_grad_())


def test_checkpoint_preserves_output_final_and_shared_gradients():
    torch.manual_seed(22)
    x = torch.randn(2, 2, 9, 3, requires_grad=True)
    initial = [torch.randn(2, 2, 3, requires_grad=True) for _ in range(2)]
    normal = _compile(_program())
    checked = _program().compile(
        role_sizes={H: 2, F: 3},
        target=Target(device_type="cpu"),
        mode="dense",
        backward=True,
        checkpoint=True,
    )
    a = normal(
        x=x,
        initial_state=StateTuple(("second", "first"), initial),
        return_final_state=True,
    )
    b = checked(
        x=x,
        initial_state=StateTuple(("second", "first"), initial),
        return_final_state=True,
    )
    for lhs, rhs in zip(
        (*a.outputs, *a.final_state), (*b.outputs, *b.final_state), strict=True
    ):
        torch.testing.assert_close(lhs, rhs)
    ga = torch.autograd.grad(
        sum(t.square().sum() for t in (*a.outputs, *a.final_state)), (x, *initial)
    )
    gb = torch.autograd.grad(
        sum(t.square().sum() for t in (*b.outputs, *b.final_state)), (x, *initial)
    )
    for lhs, rhs in zip(ga, gb, strict=True):
        torch.testing.assert_close(lhs, rhs)
    assert normal.specialization != checked.specialization


def test_bound_input_does_not_redeclare_another_stages_external_input():
    # The same spelling can denote an external FP32 input in one stage and an
    # internal FP32 trajectory in another; their role permutations differ.
    a = _stage("a", "x", "first")
    b = _stage("b", "x", "second", (StageBinding("x", "a", "trajectory"),))
    algorithm = AlgorithmIR(
        (TensorInput("x", (Batch, Sequence, H, F), torch.float32),),
        b.algorithm.states,
        b.algorithm.transitions,
        b.algorithm.readouts,
        HeadMapping(),
    )
    operator = StagedStatefulOperator(
        ScanStageIR(
            (a, ScanStage("b", algorithm, b.bindings)),
            (StageOutput("out", "b", "trajectory"),),
        )
    )
    x = torch.randn(2, 2, 9, 3)
    actual = _compile(operator)(x=x).outputs[0]
    expected = _compile(_program())(x=x).outputs["out"]
    torch.testing.assert_close(actual, expected)


def test_prior_trajectory_binding_keeps_single_feature_extent():
    stages = (
        _stage("a", "x", "first", prior=True),
        _stage("b", "derived", "second", (StageBinding("derived", "a", "trajectory"),)),
    )
    op = StagedStatefulOperator(
        ScanStageIR(stages, (StageOutput("out", "b", "trajectory"),))
    )
    compiled = op.compile(
        role_sizes={H: 2, F: 1},
        target=Target(device_type="cpu"),
        mode="dense",
        backward=True,
    )
    x = torch.ones(2, 2, 5, 1, requires_grad=True)
    result = compiled(x=x)
    a = torch.zeros(2, 2, 1)
    b = torch.zeros_like(a)
    expected = []
    for token in range(5):
        b = 0.7 * b + a
        a = 0.7 * a + x[:, :, token]
        expected.append(b)
    torch.testing.assert_close(result.outputs[0], torch.stack(expected, 2))
    torch.testing.assert_close(
        torch.autograd.grad(result.outputs[0].sum(), x)[0],
        torch.autograd.grad(torch.stack(expected, 2).sum(), x)[0],
    )


def test_dense_inference_windows_preserve_prior_exports_roles_and_continuation():
    torch.manual_seed(42)
    a = _stage("a", "x", "first", prior=True)
    b = _stage("b", "derived", "second", (StageBinding("derived", "a", "trajectory"),))
    permuted = AlgorithmIR(
        (TensorInput("x", (Batch, Sequence, H, F), torch.float32),),
        a.algorithm.states,
        a.algorithm.transitions,
        a.algorithm.readouts,
        HeadMapping(),
    )
    operator = StagedStatefulOperator(
        ScanStageIR(
            (b, ScanStage("a", permuted)),
            (
                StageOutput("out", "b", "trajectory"),
                StageOutput("aux", "a", "trajectory"),
            ),
        )
    )
    x = torch.randn(2, 1031, 2, 3)
    initial = StateTuple(
        ("second", "first"), (torch.randn(2, 2, 3), torch.randn(2, 2, 3))
    )
    saved = tuple(t.clone() for t in initial)
    compiled = _compile(operator, backward=False, workspace=150000)
    assert (
        compiled.workspace_bytes(2062, sequence_count=2, sequence_length=1031) <= 150000
    )
    result = compiled(x=x, initial_state=initial, return_final_state=True)
    b, a = initial
    expected, auxiliary = [], []
    for token in range(1031):
        b = 0.7 * b + a
        auxiliary.append(a)
        a = 0.7 * a + x[:, token]
        expected.append(b)
    torch.testing.assert_close(
        result.outputs["out"], torch.stack(expected, 2), atol=2e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        result.outputs["aux"], torch.stack(auxiliary, 2), atol=2e-6, rtol=1e-5
    )
    for actual, wanted in zip(result.final_state, (b, a), strict=True):
        torch.testing.assert_close(actual, wanted)
    first = compiled(
        x=x[:, :513].contiguous(), initial_state=initial, return_final_state=True
    )
    last = compiled(
        x=x[:, 513:].contiguous(),
        initial_state=first.final_state,
        return_final_state=True,
    )
    for name in ("out", "aux"):
        torch.testing.assert_close(
            torch.cat((first.outputs[name], last.outputs[name]), 2),
            result.outputs[name],
            atol=2e-6,
            rtol=1e-5,
        )
    for actual, wanted in zip(last.final_state, result.final_state, strict=True):
        torch.testing.assert_close(actual, wanted)
    for actual, wanted in zip(initial, saved, strict=True):
        torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
    # The full exported output alone exceeds this envelope; windowing cannot help.
    with pytest.raises(StatefulCompilationError, match="RUNTIME_RESOURCE"):
        _compile(operator, backward=False, workspace=90000)(x=x)


@pytest.mark.parametrize(
    "family", ["preconditioned-kda", "preconditioned-gdn", "lightnet"]
)
def test_long_staged_dense_inference_workspace(family):
    from dataclasses import replace
    from testing import benchmark_stateful_plans as benchmark

    case = replace(
        next(c for c in benchmark.CASES if c.name == family), batch=2, heads=16
    )
    operator = case.factory(**case.kwargs(8192))
    compiled = operator.compile(
        role_sizes=benchmark.role_sizes_for(operator, case, 8192),
        target=Target(),
        mode="dense",
    )
    assert (
        compiled.workspace_bytes(16384, sequence_count=2, sequence_length=8192)
        <= 1 << 30
    )


def test_staged_shift_keeps_sequence_history_at_window_boundary():
    from attn_engine import Shift

    stage = _stage("a", "x", "first")
    algorithm = stage.algorithm
    transition = StateTransition(
        "first",
        propagations=(AxisScale(0.7),),
        injections=(ProductInjection((ProductFactor(Shift(Input("x"), 1), (F,)),)),),
    )
    stage = ScanStage(
        "a",
        AlgorithmIR(
            algorithm.inputs,
            algorithm.states,
            transition,
            algorithm.readouts,
            algorithm.head_mapping,
        ),
    )
    operator = StagedStatefulOperator(
        ScanStageIR(
            (stage,),
            (StageOutput("out", "a", "trajectory"),),
        )
    )
    compiled = _compile(operator, backward=False)
    x = torch.ones(2, 2, 1025, 3)
    x[:, :, 1023] = 8
    result = compiled(x=x)
    state = torch.zeros(2, 2, 3)
    wanted = []
    for token in range(1025):
        state = 0.7 * state + (x[:, :, token - 1] if token else 0)
        wanted.append(state)
    torch.testing.assert_close(result.outputs["out"], torch.stack(wanted, 2))


@pytest.mark.parametrize("mode", ["dense", "packed"])
def test_checkpoint_does_not_discount_the_staged_workspace_contract(mode):
    normal = _compile(_program(), mode=mode)
    checked = _program().compile(
        role_sizes={H: 2, F: 3},
        target=Target(device_type="cpu"),
        mode=mode,
        backward=True,
        checkpoint=True,
    )
    kwargs = {"sequence_count": 2}
    if mode == "dense":
        kwargs["sequence_length"] = 1031
    assert checked.workspace_bytes(2062, **kwargs) == normal.workspace_bytes(
        2062, **kwargs
    )


def test_staged_workspace_rejection_precedes_child_execution(monkeypatch):
    compiled = _compile(_program(), backward=False, workspace=90000)

    def unexpected_execution(*args, **kwargs):
        pytest.fail("a rejected workspace must not execute a child scan")

    monkeypatch.setattr(compiled, "_execute_stages", unexpected_execution)
    with pytest.raises(StatefulCompilationError) as error:
        compiled(x=torch.ones(2, 2, 1031, 3))
    assert error.value.category == "RUNTIME_RESOURCE"
    assert error.value.path == "workspace"


def test_staged_windows_release_previous_outputs_before_preparing_next(monkeypatch):
    import weakref

    compiled = _compile(_program(), backward=False, workspace=150000)
    execute = compiled._execute_stages
    previous = []
    lengths = []

    def record_window(inputs, states, runtime, offsets, return_final):
        assert all(reference() is None for reference in previous)
        assert inputs["x"].is_contiguous()
        lengths.append(runtime.uniform_length)
        result = execute(inputs, states, runtime, offsets, return_final)
        previous[:] = [weakref.ref(value) for value in result.outputs]
        return result

    monkeypatch.setattr(compiled, "_execute_stages", record_window)
    result = compiled(x=torch.ones(2, 2, 1031, 3))
    assert len(lengths) > 1 and sum(lengths) == 1031
    assert result.final_state is None
    assert result.outputs["out"].shape == (2, 2, 1031, 3)
    assert all(reference() is None for reference in previous)
