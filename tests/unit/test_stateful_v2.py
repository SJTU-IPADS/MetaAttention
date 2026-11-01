"""Schema-v2 capability tests: several States, rank-1 States, readout expressions.

Every test crosses the public seam (construct Algorithm IR, compile, invoke) and
compares against an independent FP32 token recurrence. The reference recurrences
are written from the mathematics, not from the compiler.
"""

from __future__ import annotations

import pytest
import torch

from attn_engine import (
    AlgorithmIR,
    AxisScale,
    Batch,
    Clamp,
    Constant,
    Exp,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    ProductFactor,
    ProductInjection,
    Readout,
    Reciprocal,
    Sequence,
    Shift,
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

KEY = FeatureRole("key_dim")
VALUE = FeatureRole("value_dim")
QUERY_HEADS = HeadRole("query_heads")
KEY_HEADS = HeadRole("key_heads")
VALUE_HEADS = HeadRole("value_heads")
STATE_HEADS = HeadRole("state_heads")
ROLE_SIZES = {
    QUERY_HEADS: 1,
    KEY_HEADS: 1,
    VALUE_HEADS: 1,
    STATE_HEADS: 1,
    KEY: 3,
    VALUE: 2,
}


@pytest.fixture(autouse=True)
def _allow_cpu_stateful_operator(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")


def _compile(operator, *, mode="dense", backward=False, role_sizes=None):
    return operator.compile(
        role_sizes=ROLE_SIZES if role_sizes is None else role_sizes,
        target=Target(),
        mode=mode,
        optimized=True,
        backward=backward,
    )


def _normalized_linear_attention(dtype=torch.float32) -> AlgorithmIR:
    """Based-style numerator/denominator program: two States and one division."""
    return AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
        ),
        states=(
            StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),
            StateSpec("normalizer", (Batch, STATE_HEADS, KEY)),
        ),
        transition=(
            StateTransition(
                "memory",
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(Input("key"), (KEY,)),
                            ProductFactor(Input("value"), (VALUE,)),
                        )
                    ),
                ),
            ),
            StateTransition(
                "normalizer",
                injections=(ProductInjection((ProductFactor(Input("key"), (KEY,)),)),),
            ),
        ),
        readouts=(
            Readout(
                "output",
                StateTerm("memory", Input("query"), KEY)
                / (StateTerm("normalizer", Input("query"), KEY) + Constant(1e-3)),
                torch.float32,
            ),
        ),
        head_mapping=HeadMapping(
            {
                QUERY_HEADS: STATE_HEADS,
                KEY_HEADS: STATE_HEADS,
                VALUE_HEADS: STATE_HEADS,
            }
        ),
    )


def _normalized_reference(inputs, initial_memory=None, initial_normalizer=None):
    query, key, value = (inputs[name].float() for name in ("query", "key", "value"))
    memory = (
        torch.zeros(query.shape[0], query.shape[1], KEY_SIZE, VALUE_SIZE)
        if initial_memory is None
        else initial_memory.float().clone()
    )
    normalizer = (
        torch.zeros(query.shape[0], query.shape[1], KEY_SIZE)
        if initial_normalizer is None
        else initial_normalizer.float().clone()
    )
    outputs = []
    for token in range(query.shape[2]):
        memory = memory + torch.einsum(
            "bhf,bhg->bhfg", key[..., token, :], value[..., token, :]
        )
        normalizer = normalizer + key[..., token, :]
        numerator = torch.einsum("bhf,bhfg->bhg", query[..., token, :], memory)
        denominator = torch.einsum("bhf,bhf->bh", query[..., token, :], normalizer)
        outputs.append(numerator / (denominator.unsqueeze(-1) + 1e-3))
    return torch.stack(outputs, dim=2), memory, normalizer


KEY_SIZE = ROLE_SIZES[KEY]
VALUE_SIZE = ROLE_SIZES[VALUE]


@pytest.fixture
def normalized_inputs():
    generator = torch.Generator().manual_seed(11)
    length = 4
    return {
        "query": torch.randn(1, 1, length, KEY_SIZE, generator=generator),
        "key": torch.randn(1, 1, length, KEY_SIZE, generator=generator),
        "value": torch.randn(1, 1, length, VALUE_SIZE, generator=generator),
    }


def test_two_states_with_division_readout_matches_reference(normalized_inputs):
    compiled = _compile(StatefulOperator(_normalized_linear_attention()))
    result = compiled(**normalized_inputs, return_final_state=True)
    expected, expected_memory, expected_normalizer = _normalized_reference(
        normalized_inputs
    )
    torch.testing.assert_close(result.outputs["output"], expected, rtol=1e-5, atol=1e-6)
    assert result.final_state is not None
    torch.testing.assert_close(
        result.final_state["memory"], expected_memory, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        result.final_state["normalizer"], expected_normalizer, rtol=1e-5, atol=1e-6
    )
    assert result.final_state.names == ("memory", "normalizer")


def test_two_state_program_carries_gradients_from_the_readout(
    normalized_inputs,
):
    inputs = {
        name: tensor.clone().requires_grad_(True)
        for name, tensor in normalized_inputs.items()
    }
    initial_memory = torch.randn(1, 1, KEY_SIZE, VALUE_SIZE, requires_grad=True)
    initial_normalizer = torch.randn(1, 1, KEY_SIZE, requires_grad=True)
    compiled = _compile(StatefulOperator(_normalized_linear_attention()), backward=True)
    result = compiled(
        **inputs,
        initial_state=StateTuple(
            ("memory", "normalizer"), (initial_memory, initial_normalizer)
        ),
    )
    weights = torch.randn_like(result.outputs["output"])
    result.outputs["output"].mul(weights).sum().backward()

    reference_inputs = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in normalized_inputs.items()
    }
    reference_memory = initial_memory.detach().clone().requires_grad_(True)
    reference_normalizer = initial_normalizer.detach().clone().requires_grad_(True)
    expected, _, _ = _normalized_reference(
        reference_inputs, reference_memory, reference_normalizer
    )
    expected.mul(weights).sum().backward()

    for name in ("query", "key", "value"):
        torch.testing.assert_close(
            inputs[name].grad, reference_inputs[name].grad, rtol=1e-4, atol=1e-5
        )
    torch.testing.assert_close(
        initial_memory.grad, reference_memory.grad, rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        initial_normalizer.grad, reference_normalizer.grad, rtol=1e-4, atol=1e-5
    )


def test_packed_two_state_program_preserves_empty_sequence_state():
    generator = torch.Generator().manual_seed(5)
    lengths = (2, 0, 2)
    total = sum(lengths)
    inputs = {
        "query": torch.randn(total, 1, KEY_SIZE, generator=generator),
        "key": torch.randn(total, 1, KEY_SIZE, generator=generator),
        "value": torch.randn(total, 1, VALUE_SIZE, generator=generator),
    }
    operator = StatefulOperator(_normalized_linear_attention())
    compiled = operator.compile(
        role_sizes=ROLE_SIZES, target=Target(), mode="packed", optimized=True
    )
    offsets = torch.tensor([0, 2, 2, 4], dtype=torch.int32)
    initial_memory = torch.zeros(3, 1, KEY_SIZE, VALUE_SIZE)
    initial_normalizer = torch.zeros(3, 1, KEY_SIZE)
    result = compiled(
        **inputs,
        sequence_offsets=offsets,
        initial_state=StateTuple(
            ("memory", "normalizer"), (initial_memory, initial_normalizer)
        ),
        return_final_state=True,
    )
    assert result.outputs["output"].shape == (total, 1, VALUE_SIZE)
    assert result.final_state is not None
    # the empty middle sequence keeps its initial State
    torch.testing.assert_close(result.final_state["memory"][1], initial_memory[1])
    torch.testing.assert_close(
        result.final_state["normalizer"][1], initial_normalizer[1]
    )
    # each packed sequence matches its own dense execution
    dense = _compile(StatefulOperator(_normalized_linear_attention()))
    for index, (start, length) in enumerate(((0, 2), (2, 0), (2, 2))):
        if length == 0:
            continue
        slice_inputs = {
            name: tensor[start : start + length]
            .transpose(0, 1)
            .unsqueeze(0)
            .contiguous()
            for name, tensor in inputs.items()
        }
        dense_result = dense(**slice_inputs)
        torch.testing.assert_close(
            result.outputs["output"][start : start + length]
            .transpose(0, 1)
            .unsqueeze(0),
            dense_result.outputs["output"],
            rtol=1e-5,
            atol=1e-6,
        )


def test_rank1_state_identity_readout_matches_reference():
    """HGRN-style: the output is the State itself, with no contraction."""
    generator = torch.Generator().manual_seed(3)
    length = 5
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=(
                TensorInput(
                    "value", (Batch, VALUE_HEADS, Sequence, VALUE), torch.float32
                ),
                TensorInput("gate", (Batch, STATE_HEADS, Sequence), torch.float32),
            ),
            states=(StateSpec("hidden", (Batch, STATE_HEADS, VALUE)),),
            transition=StateTransition(
                "hidden",
                propagations=(AxisScale(Exp(Input("gate"))),),
                injections=(
                    ProductInjection((ProductFactor(Input("value"), (VALUE,)),)),
                ),
            ),
            readouts=(
                Readout(
                    "output",
                    StateTerm("hidden", Constant(1.0), None),
                    torch.float32,
                ),
            ),
            head_mapping=HeadMapping(
                {VALUE_HEADS: STATE_HEADS, STATE_HEADS: STATE_HEADS}
            ),
        )
    )
    inputs = {
        "value": torch.randn(1, 1, length, VALUE_SIZE, generator=generator),
        "gate": torch.randn(1, 1, length, generator=generator),
    }
    result = _compile(
        operator, role_sizes={VALUE_HEADS: 1, STATE_HEADS: 1, VALUE: VALUE_SIZE}
    )(**inputs, return_final_state=True)

    hidden = torch.zeros(1, 1, VALUE_SIZE)
    outputs = []
    for token in range(length):
        hidden = hidden * inputs["gate"][..., token].exp().unsqueeze(-1)
        hidden = hidden + inputs["value"][..., token, :]
        outputs.append(hidden.clone())
    expected = torch.stack(outputs, dim=2)
    torch.testing.assert_close(result.outputs["output"], expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        result.final_state["hidden"], hidden, rtol=1e-5, atol=1e-6
    )


def test_token_shift_expression_matches_reference():
    """RWKV-style: the injection reads the previous token's key."""
    generator = torch.Generator().manual_seed(7)
    length = 5
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=(
                TensorInput(
                    "query", (Batch, QUERY_HEADS, Sequence, KEY), torch.float32
                ),
                TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), torch.float32),
                TensorInput(
                    "value", (Batch, VALUE_HEADS, Sequence, VALUE), torch.float32
                ),
                TensorInput("mix", (Batch, STATE_HEADS, Sequence, KEY), torch.float32),
            ),
            states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
            transition=StateTransition(
                "memory",
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(
                                Shift(Input("key"), 1) * Input("mix")
                                + Input("key") * (1 - Input("mix")),
                                (KEY,),
                            ),
                            ProductFactor(Input("value"), (VALUE,)),
                        )
                    ),
                ),
            ),
            readouts=(
                Readout(
                    "output",
                    StateTerm("memory", Input("query"), KEY),
                    torch.float32,
                ),
            ),
            head_mapping=HeadMapping(
                {
                    QUERY_HEADS: STATE_HEADS,
                    KEY_HEADS: STATE_HEADS,
                    VALUE_HEADS: STATE_HEADS,
                }
            ),
        )
    )
    inputs = {
        "query": torch.randn(1, 1, length, KEY_SIZE, generator=generator),
        "key": torch.randn(1, 1, length, KEY_SIZE, generator=generator),
        "value": torch.randn(1, 1, length, VALUE_SIZE, generator=generator),
        "mix": torch.rand(1, 1, length, KEY_SIZE, generator=generator),
    }
    result = _compile(operator)(**inputs)

    memory = torch.zeros(1, 1, KEY_SIZE, VALUE_SIZE)
    outputs = []
    previous = torch.zeros(1, 1, KEY_SIZE)
    for token in range(length):
        mixed = previous * inputs["mix"][..., token, :] + inputs["key"][
            ..., token, :
        ] * (1 - inputs["mix"][..., token, :])
        memory = memory + torch.einsum(
            "bhf,bhg->bhfg", mixed, inputs["value"][..., token, :]
        )
        outputs.append(
            torch.einsum("bhf,bhfg->bhg", inputs["query"][..., token, :], memory)
        )
        previous = inputs["key"][..., token, :]
    expected = torch.stack(outputs, dim=2)
    torch.testing.assert_close(result.outputs["output"], expected, rtol=1e-5, atol=1e-6)


def test_clamp_and_reciprocal_expressions_match_reference():
    generator = torch.Generator().manual_seed(13)
    length = 3
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=(
                TensorInput(
                    "query", (Batch, QUERY_HEADS, Sequence, KEY), torch.float32
                ),
                TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), torch.float32),
                TensorInput(
                    "value", (Batch, VALUE_HEADS, Sequence, VALUE), torch.float32
                ),
            ),
            states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
            transition=StateTransition(
                "memory",
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(Input("key"), (KEY,)),
                            ProductFactor(Input("value"), (VALUE,)),
                        )
                    ),
                ),
            ),
            readouts=(
                Readout(
                    "output",
                    StateTerm("memory", Input("query"), KEY)
                    * Reciprocal(
                        Clamp(Input("value"), 0.0, 0.5).sqrt() + Constant(1.0)
                    ),
                    torch.float32,
                ),
            ),
            head_mapping=HeadMapping(
                {
                    QUERY_HEADS: STATE_HEADS,
                    KEY_HEADS: STATE_HEADS,
                    VALUE_HEADS: STATE_HEADS,
                }
            ),
        )
    )
    inputs = {
        "query": torch.randn(1, 1, length, KEY_SIZE, generator=generator),
        "key": torch.randn(1, 1, length, KEY_SIZE, generator=generator),
        "value": torch.randn(1, 1, length, VALUE_SIZE, generator=generator),
    }
    result = _compile(operator)(**inputs)

    memory = torch.zeros(1, 1, KEY_SIZE, VALUE_SIZE)
    outputs = []
    for token in range(length):
        memory = memory + torch.einsum(
            "bhf,bhg->bhfg",
            inputs["key"][..., token, :],
            inputs["value"][..., token, :],
        )
        operand = inputs["query"][..., token, :]
        scale = 1.0 / (inputs["value"][..., token, :].clamp(0.0, 0.5).sqrt() + 1.0)
        outputs.append(torch.einsum("bhf,bhfg->bhg", operand, memory) * scale)
    expected = torch.stack(outputs, dim=2)
    torch.testing.assert_close(result.outputs["output"], expected, rtol=1e-5, atol=1e-6)


def test_states_must_share_one_head_role():
    with pytest.raises(StatefulCompilationError) as captured:
        AlgorithmIR(
            inputs=(
                TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), torch.float32),
            ),
            states=(
                StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),
                StateSpec("normalizer", (Batch, KEY_HEADS, KEY)),
            ),
            transition=(
                StateTransition(
                    "memory",
                    injections=(
                        ProductInjection(
                            (
                                ProductFactor(Input("key"), (KEY,)),
                                ProductFactor(Constant(1.0), (VALUE,)),
                            )
                        ),
                    ),
                ),
                StateTransition(
                    "normalizer",
                    injections=(
                        ProductInjection((ProductFactor(Input("key"), (KEY,)),)),
                    ),
                ),
            ),
            readouts=(StateTerm("memory", Input("key"), KEY),),
            head_mapping=HeadMapping({KEY_HEADS: STATE_HEADS}),
        )
    assert captured.value.category in ("IR_SCHEMA", "IR_TYPE")


def test_transition_rejects_state_terms():
    with pytest.raises(StatefulCompilationError) as captured:
        AlgorithmIR(
            inputs=(
                TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), torch.float32),
            ),
            states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
            transition=StateTransition(
                "memory",
                propagations=(
                    AxisScale(StateTerm("memory", Input("key"), KEY), (KEY,)),
                ),
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(Input("key"), (KEY,)),
                            ProductFactor(Constant(1.0), (VALUE,)),
                        )
                    ),
                ),
            ),
            readouts=(
                Readout(
                    "output", StateTerm("memory", Input("key"), KEY), torch.float32
                ),
            ),
            head_mapping=HeadMapping({KEY_HEADS: STATE_HEADS}),
        )
    assert captured.value.category == "IR_TYPE"


def test_readout_rejects_unknown_state_and_identity_on_matrix_state():
    with pytest.raises(StatefulCompilationError) as unknown:
        AlgorithmIR(
            inputs=(
                TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), torch.float32),
            ),
            states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
            transition=StateTransition(
                "memory",
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(Input("key"), (KEY,)),
                            ProductFactor(Constant(1.0), (VALUE,)),
                        )
                    ),
                ),
            ),
            readouts=(
                Readout("output", StateTerm("mem", Input("key"), KEY), torch.float32),
            ),
            head_mapping=HeadMapping({KEY_HEADS: STATE_HEADS}),
        )
    assert unknown.value.category == "IR_TYPE"

    with pytest.raises(StatefulCompilationError) as identity:
        AlgorithmIR(
            inputs=(
                TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), torch.float32),
            ),
            states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
            transition=StateTransition(
                "memory",
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(Input("key"), (KEY,)),
                            ProductFactor(Constant(1.0), (VALUE,)),
                        )
                    ),
                ),
            ),
            readouts=(
                Readout(
                    "output", StateTerm("memory", Constant(1.0), None), torch.float32
                ),
            ),
            head_mapping=HeadMapping({KEY_HEADS: STATE_HEADS}),
        )
    assert identity.value.category == "IR_TYPE"


def test_clamp_bounds_are_validated():
    with pytest.raises(ValueError):
        Clamp(Input("key"), 2.0, 1.0)
