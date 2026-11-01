"""Ordered affine steps and prior/current dependencies through public calls."""

import pytest
import torch

from attn_engine import (
    AlgorithmIR,
    AxisScale,
    Batch,
    Contract,
    Exp,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    ProductFactor,
    ProductInjection,
    Readout,
    Sequence,
    StateInjection,
    StateReference,
    StateSpec,
    StateTerm,
    StateTransition,
    StatefulCompilationError,
    StatefulOperator,
    StateTuple,
    Target,
    TensorInput,
)

pytestmark = pytest.mark.unit
H = HeadRole("heads")
K = FeatureRole("key")
V = FeatureRole("value")


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")


def _coupled(*, prior=True, second_axis=False, vector=False):
    roles = (Batch, H, K) if vector else (Batch, H, K, V)
    action = V if second_axis else K
    injection = (
        ProductInjection((ProductFactor(Input("x"), (K,)),))
        if vector
        else ProductInjection(
            (
                ProductFactor(Input("x"), (K,)),
                ProductFactor(Input("y"), (V,)),
            )
        )
    )
    return StatefulOperator(
        AlgorithmIR(
            inputs=(
                TensorInput("x", (Batch, H, Sequence, K), torch.float32),
                *(
                    ()
                    if vector
                    else (TensorInput("y", (Batch, H, Sequence, V), torch.float32),)
                ),
            ),
            states=(StateSpec("a", roles), StateSpec("b", roles)),
            transition=(
                StateTransition(
                    "a",
                    steps=(AxisScale(0.8), StateInjection(StateReference("b"), 0.3)),
                ),
                StateTransition(
                    "b",
                    steps=(
                        AxisScale(0.6),
                        injection,
                        StateInjection(
                            StateReference("a", prior=prior),
                            0.2,
                            role=action,
                            left=Input("y" if second_axis else "x"),
                            right=Input("y" if second_axis else "x"),
                        ),
                    ),
                ),
            ),
            readouts=(
                Readout(
                    "output",
                    StateTerm("a", 1 if vector else Input("x"), None if vector else K),
                    torch.float32,
                ),
                Readout(
                    "prior",
                    StateTerm(
                        "b",
                        1 if vector else Input("x"),
                        None if vector else K,
                        prior=True,
                    ),
                    torch.float32,
                ),
            ),
            head_mapping=HeadMapping(),
        )
    )


def _compile(operator, mode="dense", backward=False, workspace=1 << 30):
    roles = {H: 2, K: 3}
    if operator.algorithm.states[0].rank == 2:
        roles[V] = 5
    return operator.compile(
        role_sizes=roles,
        target=Target(device_type="cpu", workspace_bytes=workspace),
        mode=mode,
        backward=backward,
    )


@pytest.mark.parametrize(
    "second_axis,vector", [(False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize("loss_kind", ["output", "final", "both"])
def test_joint_summary_outputs_prior_reads_and_gradients(
    second_axis, vector, loss_kind
):
    torch.manual_seed(98)
    operator = _coupled(second_axis=second_axis, vector=vector)
    inputs = {"x": (torch.randn(2, 2, 7, 3) * 0.2).requires_grad_()}
    if not vector:
        inputs["y"] = (torch.randn(2, 2, 7, 5) * 0.2).requires_grad_()
    shape = (2, 2, 3) if vector else (2, 2, 3, 5)
    initial = [torch.randn(shape, requires_grad=True) for _ in range(2)]
    result = _compile(operator, backward=True)(
        **inputs, initial_state=StateTuple(("a", "b"), initial), return_final_state=True
    )
    a, b = initial
    outputs, priors = [], []
    for token in range(7):
        x = inputs["x"][:, :, token]
        old_a, old_b = a, b
        if vector:
            write = x
            correction = x * (x * old_a).sum(-1, keepdim=True)
        else:
            y = inputs["y"][:, :, token]
            write = x.unsqueeze(-1) * y.unsqueeze(-2)
            correction = (
                ((old_a @ y.unsqueeze(-1)) * y.unsqueeze(-2))
                if second_axis
                else x.unsqueeze(-1) * (x.unsqueeze(-2) @ old_a)
            )
        b = 0.6 * old_b + write + 0.2 * correction
        a = 0.8 * old_a + 0.3 * b
        outputs.append(a if vector else (x.unsqueeze(-2) @ a).squeeze(-2))
        priors.append(old_b if vector else (x.unsqueeze(-2) @ old_b).squeeze(-2))
    expected_output = torch.stack(outputs, dim=2)
    expected_prior = torch.stack(priors, dim=2)
    torch.testing.assert_close(
        result.outputs["output"], expected_output, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        result.outputs["prior"], expected_prior, rtol=2e-5, atol=2e-6
    )
    for name, expected in (("a", a), ("b", b)):
        torch.testing.assert_close(
            result.final_state[name], expected, rtol=2e-5, atol=2e-6
        )
    actual_loss = (
        result.outputs["output"].square().sum() + result.outputs["prior"].square().sum()
        if loss_kind != "final"
        else 0
    )
    expected_loss = (
        expected_output.square().sum() + expected_prior.square().sum()
        if loss_kind != "final"
        else 0
    )
    if loss_kind != "output":
        actual_loss = actual_loss + sum(
            value.square().sum() for value in result.final_state
        )
        expected_loss = expected_loss + a.square().sum() + b.square().sum()
    leaves = (*inputs.values(), *initial)
    actual_grad = torch.autograd.grad(actual_loss, leaves)
    expected_grad = torch.autograd.grad(expected_loss, leaves)
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, rtol=8e-5, atol=1e-5)


def test_current_state_cycle_is_rejected():
    with pytest.raises(StatefulCompilationError) as captured:
        _coupled(prior=False)
    assert captured.value.category == "IR_DEPENDENCY"
    assert set(captured.value.details["cycle"]) == {"a", "b"}


def test_steps_reject_grouped_operands_and_unknown_nodes():
    with pytest.raises(ValueError, match="not both"):
        StateTransition(
            "a",
            steps=(AxisScale(1),),
            injections=(ProductInjection((ProductFactor(Input("x"), (K,)),)),),
        )
    with pytest.raises(StatefulCompilationError) as captured:
        AlgorithmIR(
            inputs=(TensorInput("x", (Batch, H, Sequence, K), torch.float32),),
            states=(StateSpec("a", (Batch, H, K)),),
            transition=StateTransition("a", steps=(object(),)),
            readouts=(Readout("o", StateTerm("a", 1, None), torch.float32),),
            head_mapping=HeadMapping(),
        )
    assert captured.value.category == "IR_SCHEMA"


@pytest.mark.parametrize(
    "node",
    [
        AxisScale(Exp(StateTerm("a", Input("x"), K))),
        ProductInjection(
            (
                ProductFactor(
                    Contract(StateTerm("a", Input("x"), K), Input("y"), V), (K,)
                ),
                ProductFactor(Input("y"), (V,)),
            )
        ),
        StateInjection(StateReference("a", prior=True), StateTerm("a", Input("x"), K)),
    ],
)
def test_state_derived_coefficients_are_rejected_even_inside_contract(node):
    with pytest.raises(StatefulCompilationError) as captured:
        AlgorithmIR(
            inputs=(
                TensorInput("x", (Batch, H, Sequence, K), torch.float32),
                TensorInput("y", (Batch, H, Sequence, V), torch.float32),
            ),
            states=(StateSpec("a", (Batch, H, K, V)),),
            transition=StateTransition("a", steps=(node,)),
            readouts=(Readout("o", StateTerm("a", Input("x"), K), torch.float32),),
            head_mapping=HeadMapping(),
        )
    assert captured.value.category == "IR_TYPE"


def test_ordered_diagonal_write_then_scale_matches_recurrence():
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=(TensorInput("x", (Batch, H, Sequence, K), torch.float32),),
            states=(StateSpec("a", (Batch, H, K)),),
            transition=StateTransition(
                "a",
                steps=(
                    ProductInjection((ProductFactor(Input("x"), (K,)),)),
                    AxisScale(0.4),
                ),
            ),
            readouts=(Readout("o", StateTerm("a", 1, None), torch.float32),),
            head_mapping=HeadMapping(),
        )
    )
    x = torch.randn(1, 2, 9, 3)
    result = _compile(operator)(x=x)
    a = torch.zeros(1, 2, 3)
    expected = []
    for token in range(9):
        a = 0.4 * (a + x[:, :, token])
        expected.append(a)
    torch.testing.assert_close(result.outputs["o"], torch.stack(expected, dim=2))


def test_packed_joint_scan_preserves_empty_sequences_and_gradients():
    operator = _coupled()
    torch.manual_seed(103)
    inputs = {
        "x": (torch.randn(7, 2, 3) * 0.2).requires_grad_(),
        "y": (torch.randn(7, 2, 5) * 0.2).requires_grad_(),
    }
    initial = [torch.randn(4, 2, 3, 5, requires_grad=True) for _ in range(2)]
    offsets = torch.tensor([0, 2, 2, 7, 7], dtype=torch.int32)
    result = _compile(operator, mode="packed", backward=True)(
        **inputs,
        initial_state=StateTuple(("a", "b"), initial),
        sequence_offsets=offsets,
        return_final_state=True,
    )
    expected_outputs = {name: [] for name in result.outputs.names}
    final = {name: [] for name in ("a", "b")}
    dense = _compile(operator, backward=True)
    for index, (start, end) in enumerate(zip(offsets.tolist(), offsets.tolist()[1:])):
        if start == end:
            for name, value in zip(final, initial):
                final[name].append(value[index : index + 1])
            continue
        dense_result = dense(
            **{
                name: value[start:end].permute(1, 0, 2).unsqueeze(0).contiguous()
                for name, value in inputs.items()
            },
            initial_state=StateTuple(
                ("a", "b"), [value[index : index + 1].contiguous() for value in initial]
            ),
            return_final_state=True,
        )
        for name in expected_outputs:
            expected_outputs[name].append(
                dense_result.outputs[name].squeeze(0).permute(1, 0, 2)
            )
        for name in final:
            final[name].append(dense_result.final_state[name])
    expected_outputs = {
        name: torch.cat(values) for name, values in expected_outputs.items()
    }
    expected_final = {name: torch.cat(values) for name, values in final.items()}
    for name in expected_outputs:
        torch.testing.assert_close(
            result.outputs[name], expected_outputs[name], rtol=2e-5, atol=2e-6
        )
    for name in expected_final:
        torch.testing.assert_close(
            result.final_state[name], expected_final[name], rtol=2e-5, atol=2e-6
        )
    actual_loss = sum(x.square().sum() for x in (*result.outputs, *result.final_state))
    expected_loss = sum(
        x.square().sum() for x in (*expected_outputs.values(), *expected_final.values())
    )
    leaves = (*inputs.values(), *initial)
    for actual, expected in zip(
        torch.autograd.grad(actual_loss, leaves),
        torch.autograd.grad(expected_loss, leaves),
    ):
        torch.testing.assert_close(actual, expected, rtol=8e-5, atol=1e-5)


def test_order_identity_distinguishes_interleaved_writes():
    from examples.gated_delta_product import gated_delta_product

    operator = gated_delta_product(
        1, 2, 7, 3, 5, num_householder=12, dtype=torch.float32
    )
    transition = operator.algorithm.transition
    assert len(transition.ordered_steps) == 25
    assert len(transition.injections) == 12
    grouped = AlgorithmIR(
        inputs=operator.algorithm.inputs,
        states=operator.algorithm.states,
        transition=StateTransition(
            "memory",
            propagations=transition.propagations,
            injections=transition.injections,
        ),
        readouts=operator.algorithm.readouts,
        head_mapping=operator.algorithm.head_mapping,
    )
    assert operator.algorithm._structural_identity != grouped._structural_identity


def test_all_empty_packed_joint_scan_preserves_initial_state_gradients():
    operator = _coupled()
    initial = [torch.randn(3, 2, 3, 5, requires_grad=True) for _ in range(2)]
    result = _compile(operator, mode="packed", backward=True)(
        x=torch.empty(0, 2, 3),
        y=torch.empty(0, 2, 5),
        initial_state=StateTuple(("a", "b"), initial),
        sequence_offsets=torch.zeros(4, dtype=torch.int32),
        return_final_state=True,
    )
    assert result.outputs["output"].shape == (0, 2, 5)
    for actual, expected in zip(result.final_state, initial):
        torch.testing.assert_close(actual, expected)
    gradients = torch.autograd.grad(sum(x.sum() for x in result.final_state), initial)
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.ones_like(gradient))


def test_joint_axis_workspace_rejects_before_execution():
    operator = _coupled()
    compiled = _compile(operator, workspace=10_000)
    plan = operator._plans[compiled.specialization].plan
    assert plan.estimation_shape.joint_axis_size == 6
    with pytest.raises(StatefulCompilationError) as captured:
        compiled(x=torch.zeros(1, 2, 17, 3), y=torch.zeros(1, 2, 17, 5))
    assert captured.value.category == "RUNTIME_RESOURCE"


@pytest.mark.parametrize(
    "bad_reference", [StateReference("missing"), StateReference("a")]
)
def test_unknown_and_current_self_references_are_rejected(bad_reference):
    with pytest.raises(StatefulCompilationError) as captured:
        AlgorithmIR(
            inputs=(TensorInput("x", (Batch, H, Sequence, K), torch.float32),),
            states=(StateSpec("a", (Batch, H, K)),),
            transition=StateTransition("a", steps=(StateInjection(bad_reference),)),
            readouts=(Readout("o", StateTerm("a", 1, None), torch.float32),),
            head_mapping=HeadMapping(),
        )
    assert captured.value.category == (
        "IR_TYPE" if bad_reference.state == "missing" else "IR_DEPENDENCY"
    )


def test_coupled_mixed_axis_actions_are_rejected():
    from attn_engine import RankOnePropagation

    with pytest.raises(StatefulCompilationError, match="IR_TYPE"):
        AlgorithmIR(
            inputs=(
                TensorInput("x", (Batch, H, Sequence, K), torch.float32),
                TensorInput("y", (Batch, H, Sequence, V), torch.float32),
            ),
            states=(StateSpec("a", (Batch, H, K, V)), StateSpec("b", (Batch, H, K, V))),
            transition=(
                StateTransition(
                    "a",
                    steps=(
                        StateInjection(
                            StateReference("b", prior=True),
                            role=K,
                            left=Input("x"),
                            right=Input("x"),
                        ),
                    ),
                ),
                StateTransition(
                    "b", propagations=(RankOnePropagation(V, Input("y"), Input("y")),)
                ),
            ),
            readouts=(Readout("o", StateTerm("a", Input("x"), K), torch.float32),),
            head_mapping=HeadMapping(),
        )
