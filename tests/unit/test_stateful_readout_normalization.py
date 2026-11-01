"""Equivalent authored readouts share analyzed structure and execution plans."""

import pytest
import torch

from attn_engine import AlgorithmIR, Readout, StateContraction, StatefulOperator, Target
from testing import benchmark_stateful_plans as benchmark

pytestmark = pytest.mark.unit


def _pair(family):
    case = next(c for c in benchmark.CASES if c.name == family)
    original = case.factory(**case.kwargs(65))
    algorithm = original.algorithm
    readouts = []
    for readout in algorithm.readouts:
        if isinstance(readout, Readout):
            term = readout.expression
            readout = StateContraction(
                readout.name, term.state, term.operand, term.role, readout.output_dtype
            )
        readouts.append(readout)
    return (
        case,
        original,
        StatefulOperator(
            AlgorithmIR(
                inputs=algorithm.inputs,
                states=algorithm.states,
                transition=algorithm.transitions,
                readouts=readouts,
                head_mapping=algorithm.head_mapping,
            )
        ),
    )


@pytest.mark.parametrize("family", ("gdn2", "iplr", "dplr", "delta-product"))
@pytest.mark.parametrize(
    "options", ({}, {"optimized": False}, {"backward": True}, {"mode": "packed"})
)
def test_equivalent_readouts_have_identical_analyzed_plans(family, options):
    case, authored, contraction = _pair(family)
    assert (
        authored.algorithm._analysis.simplified.identity
        == contraction.algorithm._analysis.simplified.identity
    )
    assert isinstance(authored.algorithm.readouts[0], Readout)
    assert isinstance(
        authored.algorithm._analysis.normalized.readouts[0], StateContraction
    )
    compiled = [
        operator.compile(
            role_sizes=benchmark.role_sizes_for(operator, case, 65),
            target=Target(),
            **{"mode": "dense", **options},
        )
        for operator in (authored, contraction)
    ]
    plans = [call.operator._plans[call.specialization].plan for call in compiled]
    assert plans[0].semantic_identity == plans[1].semantic_identity
    assert plans[0].mathematical_plans == plans[1].mathematical_plans
    assert plans[0].lowered_candidates == plans[1].lowered_candidates
    assert plans[0].candidate == plans[1].candidate
    assert plans[0].binary_identity == plans[1].binary_identity
    assert compiled[0].specialization == compiled[1].specialization
    assert plans[0].backward_candidate == plans[1].backward_candidate
    assert plans[0].context_contract == plans[1].context_contract
    assert (
        authored.algorithm._analysis.backward
        == contraction.algorithm._analysis.backward
    )
    assert (
        authored.algorithm._analysis.structural_features
        == contraction.algorithm._analysis.structural_features
    )
    assert (
        authored.algorithm._analysis.reference_required
        == contraction.algorithm._analysis.reference_required
    )
    assert (
        authored.algorithm._structural_identity
        != contraction.algorithm._structural_identity
    )


@pytest.mark.parametrize("kind", ("gdn2", "iplr", "dplr", "delta_product"))
@pytest.mark.parametrize("mode", ("dense", "packed"))
@pytest.mark.parametrize("device", ("cpu", pytest.param("cuda", marks=pytest.mark.gpu)))
def test_equivalent_readouts_execute_and_differentiate(kind, mode, device, monkeypatch):
    from test_delta_family_examples import _case, _reference
    from attn_engine import StateTuple

    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    original, roles, inputs, initial = _case(
        kind,
        device=device,
        dim=64 if device == "cuda" else 3,
        dim_value=128 if device == "cuda" else 5,
    )
    algorithm = original.algorithm
    term = algorithm.readouts[0].expression
    canonical = StatefulOperator(
        AlgorithmIR(
            inputs=algorithm.inputs,
            states=algorithm.states,
            transition=algorithm.transitions,
            readouts=(
                StateContraction(
                    "output", term.state, term.operand, term.role, torch.float32
                ),
            ),
            head_mapping=algorithm.head_mapping,
        )
    )
    inputs = {name: value.requires_grad_() for name, value in inputs.items()}
    initial = {name: value.requires_grad_() for name, value in initial.items()}
    output, states = _reference(kind, inputs, initial)
    expected = output if mode == "dense" else output[0].transpose(0, 1)
    offsets = (
        torch.tensor([0, 5], dtype=torch.int32, device=device)
        if mode == "packed"
        else None
    )
    values = (
        inputs
        if mode == "dense"
        else {
            name: value[0].transpose(0, 1).contiguous()
            for name, value in inputs.items()
        }
    )
    leaves = (*inputs.values(), *initial.values())
    expected_loss = expected.square().sum() + states["memory"].square().sum()
    expected_gradients = torch.autograd.grad(expected_loss, leaves, retain_graph=True)
    first_result = None
    first_gradients = None
    for operator in (original, canonical):
        call = operator.compile(
            role_sizes=roles,
            target=Target(device_type=device),
            mode=mode,
            backward=True,
        )
        result = call(
            **values,
            initial_state=StateTuple(tuple(initial), tuple(initial.values())),
            sequence_offsets=offsets,
            return_final_state=True,
        )
        # Packed programs use the existing affine scan's fast GEMM; its
        # TF32 rounding differs from the FP32 token recurrence oracle.
        tolerance = (
            {"rtol": 0.01, "atol": 1e-3}
            if device == "cuda" and mode == "packed"
            else {"rtol": 2e-4, "atol": 2e-5}
            if device == "cuda"
            else {"rtol": 2e-5, "atol": 1e-6}
        )
        torch.testing.assert_close(result.outputs["output"], expected, **tolerance)
        torch.testing.assert_close(
            result.final_state["memory"], states["memory"], **tolerance
        )
        gradients = torch.autograd.grad(
            result.outputs["output"].square().sum()
            + result.final_state["memory"].square().sum(),
            leaves,
            retain_graph=True,
        )
        for actual, reference in zip(gradients, expected_gradients):
            gradient_tolerance = (
                {"rtol": 0.08, "atol": 0.02}
                if device == "cuda" and mode == "packed"
                else {"rtol": 5e-4, "atol": 5e-5}
                if device == "cuda"
                else {"rtol": 1e-4, "atol": 2e-6}
            )
            torch.testing.assert_close(actual, reference, **gradient_tolerance)
        if first_result is None:
            first_result, first_gradients = result, gradients
        else:
            # Equivalent frontends must produce identical execution, regardless
            # of the selected kernel's tolerance against a recurrence oracle.
            torch.testing.assert_close(
                result.outputs["output"], first_result.outputs["output"], rtol=0, atol=0
            )
            torch.testing.assert_close(
                result.final_state["memory"],
                first_result.final_state["memory"],
                rtol=0,
                atol=0,
            )
            for actual, first in zip(gradients, first_gradients):
                torch.testing.assert_close(actual, first, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ("prior", "nonlinear", "state_operand", "rank_one"))
def test_general_readouts_keep_their_expression_semantics(kind):
    from dataclasses import replace
    from test_stateful_v2 import KEY, VALUE, _normalized_linear_attention
    from attn_engine import Input, StateTerm

    original = _normalized_linear_attention()
    term = StateTerm("memory", Input("query"), KEY)
    expression = {
        "prior": replace(term, prior=True),
        "nonlinear": term * term,
        "state_operand": StateTerm(
            "memory", StateTerm("memory", Input("value"), VALUE), KEY
        ),
        "rank_one": StateTerm("normalizer", Input("query"), KEY),
    }[kind]
    readout = Readout("output", expression, torch.float32)
    algorithm = AlgorithmIR(
        inputs=original.inputs,
        states=original.states,
        transition=original.transitions,
        readouts=(readout,),
        head_mapping=original.head_mapping,
    )
    assert algorithm._analysis.normalized.readouts == (readout,)
    assert algorithm._analysis.reference_required
    assert algorithm._analysis.rank_one_factored is None


def test_multiple_states_normalize_using_each_readouts_state_roles(monkeypatch):
    from test_stateful_v2 import (
        KEY,
        VALUE,
        STATE_HEADS,
        ROLE_SIZES,
        _normalized_linear_attention,
    )
    from attn_engine import (
        Input,
        ProductFactor,
        ProductInjection,
        StateSpec,
        StateTerm,
        StateTransition,
    )

    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")
    original = _normalized_linear_attention()
    states = (
        original.states[0],
        StateSpec("transposed", (*original.states[0].roles[:2], VALUE, KEY)),
    )
    transitions = (
        original.transitions[0],
        StateTransition(
            "transposed",
            injections=(
                ProductInjection(
                    (
                        ProductFactor(Input("value"), (VALUE,)),
                        ProductFactor(Input("key"), (KEY,)),
                    )
                ),
            ),
        ),
    )
    readouts = (
        Readout("first", StateTerm("memory", Input("query"), KEY), torch.float32),
        Readout(
            "second", StateTerm("transposed", Input("value"), VALUE), torch.float32
        ),
    )
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=original.inputs,
            states=states,
            transition=transitions,
            readouts=readouts,
            head_mapping=original.head_mapping,
        )
    )
    assert all(
        isinstance(readout, StateContraction) for readout in operator._program.readouts
    )
    assert operator._program._analysis.reference_required
    call = operator.compile(
        role_sizes=ROLE_SIZES, target=Target(device_type="cpu"), mode="dense"
    )
    generator = torch.Generator().manual_seed(11)
    inputs = {
        "query": torch.randn(
            1, ROLE_SIZES[STATE_HEADS], 4, ROLE_SIZES[KEY], generator=generator
        ),
        "key": torch.randn(
            1, ROLE_SIZES[STATE_HEADS], 4, ROLE_SIZES[KEY], generator=generator
        ),
        "value": torch.randn(
            1, ROLE_SIZES[STATE_HEADS], 4, ROLE_SIZES[VALUE], generator=generator
        ),
    }
    result = call(**inputs, return_final_state=True)
    memory = (inputs["key"].unsqueeze(-1) * inputs["value"].unsqueeze(-2)).cumsum(2)
    torch.testing.assert_close(
        result.outputs["first"], (inputs["query"].unsqueeze(-1) * memory).sum(-2)
    )
    torch.testing.assert_close(
        result.outputs["second"], (inputs["value"].unsqueeze(-2) * memory).sum(-1)
    )
    torch.testing.assert_close(result.final_state["memory"], memory[:, :, -1])
    torch.testing.assert_close(
        result.final_state["transposed"], memory[:, :, -1].transpose(-1, -2)
    )
