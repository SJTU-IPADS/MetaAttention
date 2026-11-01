"""Recovered structural proofs must reach the public executor, not just analysis."""

from dataclasses import replace

import pytest
import torch

from attn_engine import AlgorithmIR, Readout, StatefulCompilationError, Target
from testing import benchmark_stateful_plans as benchmark

pytestmark = pytest.mark.unit
RECOVERED = ("gdn2", "iplr", "dplr", "delta-product")


@pytest.mark.parametrize("family", RECOVERED)
def test_simple_readout_and_weighted_writes_recover_factored_candidate(family):
    case = next(case for case in benchmark.CASES if case.name == family)
    operator = case.factory(**case.kwargs(2048))
    compiled = operator.compile(
        role_sizes=benchmark.role_sizes_for(operator, case, 2048),
        target=Target(),
        mode="dense",
    )
    executable = operator._plans[compiled.specialization]
    assert executable.plan.candidate.lowerer == "tilelang-factored-recurrence"
    assert executable.rank_one_factored is not None
    assert (
        executable.plan.workspace.required_bytes(2048, sequence_length=2048) < 1 << 30
    )


@pytest.mark.parametrize("family", RECOVERED)
@pytest.mark.parametrize("options", [{"optimized": False}, {"mode": "packed"}])
def test_unoptimized_and_packed_programs_use_normalized_capabilities(family, options):
    case = next(case for case in benchmark.CASES if case.name == family)
    operator = case.factory(**case.kwargs(65))
    compiled = operator.compile(
        role_sizes=benchmark.role_sizes_for(operator, case, 65),
        target=Target(),
        **{"mode": "dense", **options},
    )
    executable = operator._plans[compiled.specialization]
    if family == "delta-product" and options.get("mode") == "packed":
        assert executable.rank_one_factored is not None
        assert executable.plan.candidate.lowerer == "tilelang-factored-recurrence"
    else:
        assert executable.rank_one_factored is None
        if family == "delta-product":
            assert executable.plan.candidate.lowerer == "reference-stateful"
        else:
            assert executable.plan.candidate.lowerer != "reference-stateful"


@pytest.mark.parametrize("kind", ["prior", "nonlinear"])
def test_readout_proof_rejects_prior_and_nonlinear_terms(kind):
    case = next(case for case in benchmark.CASES if case.name == "iplr")
    original = case.factory(**case.kwargs(65)).algorithm
    term = original.readouts[0].expression
    expression = replace(term, prior=True) if kind == "prior" else term * term
    algorithm = AlgorithmIR(
        inputs=original.inputs,
        states=original.states,
        transition=original.transitions,
        readouts=(Readout("output", expression, torch.bfloat16),),
        head_mapping=original.head_mapping,
    )
    assert algorithm._analysis.rank_one_factored is None


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ("gdn2", "iplr", "dplr", "delta_product"))
@pytest.mark.parametrize("length", (0, 17, 65))
@pytest.mark.parametrize("steps", (2, 3))
def test_recovered_path_matches_oracle_and_continuation(
    kind, length, steps, gpu_device
):
    from test_delta_family_examples import _case, _reference
    from attn_engine import StateTuple

    if steps == 3 and kind != "delta_product":
        pytest.skip("third-write coverage applies to Delta Product")
    operator, roles, inputs, initial = _case(
        kind,
        device=gpu_device,
        dim=64,
        dim_value=128,
        dtype=torch.bfloat16,
        steps=steps,
    )
    # Repeat the seeded fixture to cross chunk boundaries, including a tail.
    inputs = {
        name: value.repeat(1, 1, (length + 4) // 5, *([1] * (value.ndim - 3)))[
            :, :, :length
        ].contiguous()
        for name, value in inputs.items()
    }
    compiled = operator.compile(role_sizes=roles, target=Target(), mode="dense")
    assert (
        operator._plans[compiled.specialization].plan.candidate.lowerer
        == "tilelang-factored-recurrence"
    )
    state = StateTuple(tuple(initial), tuple(initial.values()))
    if not length:
        with pytest.raises(StatefulCompilationError) as error:
            compiled(**inputs, initial_state=state, return_final_state=True)
        assert error.value.category == "RUNTIME_SHAPE"
        return
    result = compiled(**inputs, initial_state=state, return_final_state=True)
    output, expected = _reference(kind, inputs, initial, steps=steps)
    torch.testing.assert_close(
        result.outputs["output"].float(), output, rtol=0.06, atol=0.01
    )
    torch.testing.assert_close(
        result.final_state["memory"], expected["memory"], rtol=0.02, atol=0.002
    )
    split = length // 2
    prefix = compiled(
        **{n: x[:, :, :split].contiguous() for n, x in inputs.items()},
        initial_state=state,
        return_final_state=True,
    )
    suffix = compiled(
        **{n: x[:, :, split:].contiguous() for n, x in inputs.items()},
        initial_state=prefix.final_state,
        return_final_state=True,
    )
    torch.testing.assert_close(
        suffix.outputs["output"],
        result.outputs["output"][:, :, split:],
        rtol=0.06,
        atol=0.01,
    )
    torch.testing.assert_close(
        suffix.final_state["memory"],
        result.final_state["memory"],
        rtol=0.02,
        atol=0.002,
    )


@pytest.mark.parametrize("family", RECOVERED)
def test_cpu_compilation_preserves_authored_ir_and_uses_normalized_program(
    family, monkeypatch
):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")
    case = next(case for case in benchmark.CASES if case.name == family)
    operator = case.factory(**case.kwargs(65))
    compiled = operator.compile(
        role_sizes=benchmark.role_sizes_for(operator, case, 65),
        target=Target(device_type="cpu"),
        mode="dense",
    )
    from attn_engine import StateContraction

    assert compiled.operator is operator
    assert isinstance(operator.algorithm.readouts[0], Readout)
    assert isinstance(operator._program.readouts[0], StateContraction)


def test_weighted_injection_training_selects_compiled_factored_backward():
    case = next(case for case in benchmark.CASES if case.name == "delta-product")
    operator = case.factory(**case.kwargs(65))
    compiled = operator.compile(
        role_sizes=benchmark.role_sizes_for(operator, case, 65),
        target=Target(),
        mode="dense",
        backward=True,
    )
    executable = operator._plans[compiled.specialization]
    assert executable.plan.candidate.lowerer == "tilelang-factored-recurrence"
    assert executable.backward_binding is not None
    assert executable.plan.backward_candidate is not None


@pytest.mark.gpu
def test_weighted_outer_product_keeps_authored_multiplication_order(gpu_device):
    from attn_engine import (
        Batch,
        FeatureRole,
        HeadMapping,
        HeadRole,
        Input,
        ProductFactor,
        ProductInjection,
        RankOnePropagation,
        Sequence,
        StateSpec,
        StateTerm,
        StateTransition,
        StatefulOperator,
        TensorInput,
    )

    key, value, head = FeatureRole("key"), FeatureRole("value"), HeadRole("head")
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=(
                TensorInput("query", (Batch, head, Sequence, key), torch.float32),
                TensorInput("key", (Batch, head, Sequence, key), torch.float32),
                TensorInput("value", (Batch, head, Sequence, value), torch.float32),
                TensorInput("first", (Batch, head, Sequence), torch.float32),
                TensorInput("second", (Batch, head, Sequence), torch.float32),
            ),
            states=(StateSpec("memory", (Batch, head, key, value)),),
            transition=StateTransition(
                "memory",
                propagations=(RankOnePropagation(key, Input("key"), Input("key"), 0),),
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(Input("key"), (key,)),
                            ProductFactor(Input("value"), (value,)),
                            ProductFactor(Input("first"), ()),
                            ProductFactor(Input("second"), ()),
                        )
                    ),
                ),
            ),
            readouts=(
                Readout(
                    "output", StateTerm("memory", Input("query"), key), torch.float32
                ),
            ),
            head_mapping=HeadMapping({}),
        )
    )
    compiled = operator.compile(
        role_sizes={head: 1, key: 64, value: 64}, target=Target(), mode="dense"
    )
    assert (
        operator._plans[compiled.specialization].plan.candidate.lowerer
        == "tilelang-factored-recurrence"
    )
    # Folding first into the right vector overflows; the authored outer product
    # followed by first and second stays finite. Folding into BF16 is also forbidden.
    query = torch.zeros((1, 1, 17, 64), device=gpu_device)
    query[..., 0] = 1
    inputs = {
        "query": query,
        "key": torch.full_like(query, 1e-20),
        "value": torch.full_like(query, 1e20),
        "first": torch.full((1, 1, 17), 1e20, device=gpu_device),
        "second": torch.full((1, 1, 17), 1e-20, device=gpu_device),
    }
    result = compiled(**inputs, return_final_state=True)
    expected = torch.zeros((1, 1, 64, 64), device=gpu_device)
    outputs = []
    for token in range(17):
        update = (
            inputs["key"][:, :, token, :, None] * inputs["value"][:, :, token, None, :]
        )
        update = update * inputs["first"][:, :, token, None, None]
        update = update * inputs["second"][:, :, token, None, None]
        expected = expected + update
        outputs.append(expected[..., 0, :])
    torch.testing.assert_close(
        result.outputs["output"], torch.stack(outputs, dim=2), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        result.final_state["memory"], expected, rtol=1e-6, atol=1e-6
    )


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ("gdn2", "iplr", "dplr", "delta_product"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_recovered_backward_crosses_chunks_and_batches(kind, dtype, gpu_device):
    from test_delta_family_examples import _case, _reference
    from attn_engine import StateTuple

    operator, roles, inputs, initial = _case(
        kind, device=gpu_device, dim=64, dim_value=128, dtype=dtype
    )
    inputs = {
        name: value.repeat(2, 1, 4, *([1] * (value.ndim - 3)))[:, :, :17].contiguous()
        for name, value in inputs.items()
    }
    # Bound the independent low-rank vectors, as in the long-sequence benchmark.
    # Repeating unrestricted random IPLR maps can amplify BF16 gradient rounding.
    for name in ("alpha", "beta"):
        if name in inputs:
            inputs[name] = (inputs[name] * 0.25).to(dtype=inputs[name].dtype)
    inputs = {name: value.requires_grad_() for name, value in inputs.items()}
    initial = {
        name: value.repeat(2, 1, 1, 1).contiguous().requires_grad_()
        for name, value in initial.items()
    }
    compiled = operator.compile(
        role_sizes=roles, target=Target(), mode="dense", backward=True
    )
    assert operator._plans[compiled.specialization].backward_binding is not None
    actual = compiled(
        **inputs,
        initial_state=StateTuple(tuple(initial), tuple(initial.values())),
        return_final_state=True,
    )
    output, states = _reference(kind, inputs, initial)
    torch.testing.assert_close(
        actual.outputs["output"].float(), output, rtol=0.06, atol=0.01
    )
    torch.testing.assert_close(
        actual.final_state["memory"], states["memory"], rtol=0.02, atol=0.002
    )
    weights = torch.linspace(-1, 1, output.numel(), device=gpu_device).reshape_as(
        output
    )
    loss = (
        actual.outputs["output"].float() * weights
    ).sum() + 0.1 * actual.final_state["memory"].square().sum()
    expected_loss = (output * weights).sum() + 0.1 * states["memory"].square().sum()
    leaves = (*inputs.values(), *initial.values())
    actual_gradients = torch.autograd.grad(loss, leaves)
    expected_gradients = torch.autograd.grad(expected_loss, leaves)
    rtol, atol = (0.08, 0.02) if dtype == torch.bfloat16 else (0.002, 0.0001)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=rtol, atol=atol
        )
