"""Public invocation contracts shared by affine, LogAddExp and staged scans."""

import itertools
import json

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
    LogAddExpOperator,
    LogAddExpScanIR,
    ProductFactor,
    ProductInjection,
    Readout,
    ScanStage,
    ScanStageIR,
    Sequence,
    StageOutput,
    StateSpec,
    StateTerm,
    StateTransition,
    StateTuple,
    StatefulCompilationError,
    StatefulOperator,
    StagedStatefulOperator,
    Target,
    TensorInput,
)

pytestmark = pytest.mark.unit
H, F = HeadRole("heads"), FeatureRole("features")


@pytest.fixture(autouse=True)
def cpu_contract(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")


def _affine():
    return AlgorithmIR(
        (TensorInput("x", (Batch, H, Sequence, F), torch.float32),),
        (StateSpec("a", (Batch, H, F)),),
        StateTransition(
            "a",
            propagations=(AxisScale(0.5),),
            injections=(ProductInjection((ProductFactor(Input("x"), (F,)),)),),
        ),
        (Readout("out", StateTerm("a", 1, None)),),
        HeadMapping(),
    )


def _compile(kind, mode="dense", backward=False):
    affine = _affine()
    log = LogAddExpScanIR(TensorInput("x", (Batch, H, Sequence, F), torch.float32))
    if kind == "affine":
        operator = StatefulOperator(affine)
    elif kind == "log":
        operator = LogAddExpOperator(log)
    else:
        algorithm = log if kind == "staged-log" else affine
        output = "normalized_key" if kind == "staged-log" else "out"
        operator = StagedStatefulOperator(
            ScanStageIR(
                (ScanStage("scan", algorithm),), (StageOutput("out", "scan", output),)
            )
        )
    return operator.compile(
        role_sizes={H: 2, F: 3}, target=Target(), mode=mode, backward=backward
    )


@pytest.mark.parametrize("kind", ["affine", "log", "staged-affine", "staged-log"])
@pytest.mark.parametrize(
    "failure,category,path",
    [
        ("missing", "RUNTIME_BINDING", "inputs"),
        ("unknown", "RUNTIME_BINDING", "inputs"),
        ("tensor", "RUNTIME_BINDING", "inputs.x"),
        ("dtype", "RUNTIME_DTYPE", "inputs.x"),
        ("layout", "RUNTIME_LAYOUT", "inputs.x"),
        ("rank", "RUNTIME_SHAPE", "inputs.x"),
        ("head", "RUNTIME_SHAPE", "inputs.x"),
        ("feature", "RUNTIME_SHAPE", "inputs.x"),
        ("grad", "RUNTIME_GRAD", "inputs"),
        ("state_names", "RUNTIME_STATE", "initial_state"),
        ("state_shape", "RUNTIME_STATE", "state"),
        ("state_dtype", "RUNTIME_STATE", "state"),
        ("state_grad", "RUNTIME_GRAD", "inputs"),
    ],
)
def test_shared_structured_errors(kind, failure, category, path):
    op = _compile(kind)
    x = torch.ones(2, 2, 5, 3)
    kwargs = {"x": x}
    state_name = "z" if kind in ("log", "staged-log") else "a"
    if failure == "missing":
        kwargs = {}
    elif failure == "unknown":
        kwargs["extra"] = x
    elif failure == "tensor":
        kwargs["x"] = None
    elif failure == "dtype":
        kwargs["x"] = x.bfloat16()
    elif failure == "layout":
        kwargs["x"] = x.transpose(-1, -2)
    elif failure == "rank":
        kwargs["x"] = x.flatten()
    elif failure == "head":
        kwargs["x"] = torch.ones(2, 3, 5, 3)
    elif failure == "feature":
        kwargs["x"] = torch.ones(2, 2, 5, 4)
    elif failure == "grad":
        x.requires_grad_()
    else:
        state = torch.ones(2, 2, 3)
        if failure == "state_names":
            state_name = "wrong"
        elif failure == "state_shape":
            state = state[:1]
        elif failure == "state_dtype":
            state = state.bfloat16()
        elif failure == "state_grad":
            state.requires_grad_()
        kwargs["initial_state"] = StateTuple((state_name,), (state,))
    with pytest.raises(StatefulCompilationError) as error:
        op(**kwargs)
    assert error.value.category == category
    assert error.value.path == (
        f"initial_state.{state_name}" if path == "state" else path
    )
    json.dumps(error.value.as_dict())


@pytest.mark.parametrize("kind", ["affine", "log", "staged-affine", "staged-log"])
def test_packed_offsets_mode_and_default_state(kind):
    op = _compile(kind, mode="packed")
    x = torch.empty(0, 2, 3)
    with pytest.raises(StatefulCompilationError) as error:
        op(x=x, sequence_offsets=torch.tensor([0, 0], dtype=torch.int64))
    assert (error.value.category, error.value.path) == (
        "RUNTIME_OFFSETS",
        "sequence_offsets",
    )
    with pytest.raises(StatefulCompilationError) as error:
        op(x=torch.ones(2, 2, 1, 3))
    assert (error.value.category, error.value.path) == (
        "RUNTIME_MODE",
        "sequence_offsets",
    )
    result = op(
        x=x,
        sequence_offsets=torch.tensor([0, 0, 0], dtype=torch.int32),
        return_final_state=True,
    )
    expected = -torch.inf if kind in ("log", "staged-log") else 0.0
    torch.testing.assert_close(result.final_state[0], torch.full((2, 2, 3), expected))
    # A returned empty identity must not expose a mutable cached default State.
    result.final_state[0].fill_(42)
    again = op(
        x=x,
        sequence_offsets=torch.tensor([0, 0, 0], dtype=torch.int32),
        return_final_state=True,
    )
    torch.testing.assert_close(again.final_state[0], torch.full((2, 2, 3), expected))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("roles", list(itertools.permutations((Batch, H, Sequence, F))))
@pytest.mark.parametrize("warm", [False, True])
def test_log_dense_empty_preserves_physical_layout_and_identity(roles, warm, device):
    op = LogAddExpOperator(
        LogAddExpScanIR(TensorInput("x", roles, torch.float32))
    ).compile(role_sizes={H: 2, F: 3}, target=Target(), mode="dense", backward=True)
    sizes = {Batch: 2, H: 2, Sequence: 0, F: 3}
    x = torch.empty(*(sizes[r] for r in roles), device=device, requires_grad=True)
    prior = torch.randn(2, 2, 3, device=device, requires_grad=True) if warm else None
    result = op(
        x=x,
        initial_state=StateTuple(("z",), (prior,)) if warm else None,
        return_final_state=True,
    )
    assert all(t.shape == (2, 2, 0, 3) for t in result.outputs)
    expected = prior if warm else torch.full((2, 2, 3), -torch.inf, device=device)
    torch.testing.assert_close(result.final_state[0], expected)
    if warm:
        result.final_state[0].sum().backward()
        torch.testing.assert_close(prior.grad, torch.ones_like(prior))


@pytest.mark.parametrize("kind", ["affine", "staged-affine"])
def test_affine_still_rejects_dense_empty(kind):
    with pytest.raises(StatefulCompilationError) as error:
        _compile(kind)(x=torch.empty(2, 2, 0, 3))
    assert (error.value.category, error.value.path) == ("RUNTIME_SHAPE", "inputs.x")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("log_first", [False, True])
@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("mode,length", [("dense", 1), ("packed", 0)])
def test_mixed_stages_preserve_each_identity_head_shape_and_state_order(
    log_first, warm, mode, length, device
):
    shared_heads = HeadRole("shared_heads")
    log = LogAddExpScanIR(
        TensorInput("logits", (Batch, shared_heads, Sequence, F), torch.float32)
    )
    stages = [ScanStage("affine", _affine()), ScanStage("log", log)]
    if log_first:
        stages.reverse()
    op = StagedStatefulOperator(
        ScanStageIR(
            stages,
            (
                StageOutput("a", "affine", "out"),
                StageOutput("k", "log", "normalized_key"),
            ),
        )
    ).compile(
        role_sizes={H: 2, shared_heads: 1, F: 3},
        target=Target(),
        mode=mode,
        backward=True,
    )
    initial = {
        "a": torch.randn(2, 2, 3, device=device, requires_grad=True),
        "z": torch.randn(2, 1, 3, device=device, requires_grad=True),
    }
    names = ("z", "a") if log_first else ("a", "z")
    x = torch.zeros(2, 2, length, 3, device=device)
    logits = torch.zeros(2, 1, length, 3, device=device)
    kwargs = {}
    if mode == "packed":
        x, logits = (
            torch.empty(0, 2, 3, device=device),
            torch.empty(0, 1, 3, device=device),
        )
        kwargs["sequence_offsets"] = torch.tensor(
            [0, 0, 0], dtype=torch.int32, device=device
        )
    result = op(
        x=x,
        logits=logits,
        initial_state=StateTuple(names, tuple(initial[n] for n in names))
        if warm
        else None,
        return_final_state=True,
        **kwargs,
    )
    assert result.final_state.names == names
    a = initial["a"] if warm else torch.zeros(2, 2, 3, device=device)
    z = initial["z"] if warm else torch.full((2, 1, 3), -torch.inf, device=device)
    torch.testing.assert_close(result.final_state["a"], a * 0.5 if length else a)
    torch.testing.assert_close(
        result.final_state["z"],
        torch.logaddexp(z, torch.zeros_like(z)) if length else z,
    )
    if warm:
        sum(t.sum() for t in result.final_state).backward()
        assert all(
            t.grad is not None and torch.isfinite(t.grad).all()
            for t in initial.values()
        )
    if mode == "dense":
        # The LogAddExp child's identity does not grant affine dense-empty support.
        with pytest.raises(StatefulCompilationError) as error:
            op(
                x=torch.empty(2, 2, 0, 3, device=device),
                logits=torch.empty(2, 1, 0, 3, device=device),
            )
        assert error.value.category == "RUNTIME_SHAPE"
