"""Schema-v3 vocabulary tests: product Features, reductions, softmax, prior reads.

Each test crosses the public seam (construct Algorithm IR, compile, invoke). The
typing rules pinned here are what make the additions safe: an `Outer` operand must
carry the Feature its product axis names, a `Contract` or `Softmax` must name a
Feature its operand actually produces, a Feature-free Injection factor may declare
an axis it does not carry, and a prior-State term reads the State before the
current token's transition.
"""

from __future__ import annotations

import pytest
import torch

from attn_engine import (
    AlgorithmIR,
    AxisScale,
    Batch,
    Constant,
    Contract,
    Exp,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    Outer,
    ProductFactor,
    ProductInjection,
    ProductFeatureRole,
    Readout,
    Sequence,
    Softmax,
    StateSpec,
    StateTerm,
    StateTransition,
    StatefulCompilationError,
    StatefulOperator,
    Target,
    TensorInput,
)

pytestmark = pytest.mark.unit

KEY = FeatureRole("key_dim")
VALUE = FeatureRole("value_dim")
UNIT = FeatureRole("unit_dim")
QUERY_HEADS = HeadRole("query_heads")
KEY_HEADS = HeadRole("key_heads")
VALUE_HEADS = HeadRole("value_heads")
STATE_HEADS = HeadRole("state_heads")
PAIR = ProductFeatureRole(KEY, KEY)

ROLE_SIZES = {
    KEY: 2,
    VALUE: 3,
    UNIT: 1,
    PAIR: 4,
    QUERY_HEADS: 2,
    KEY_HEADS: 2,
    VALUE_HEADS: 2,
    STATE_HEADS: 2,
}


@pytest.fixture(autouse=True)
def _allow_cpu_stateful_operator(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")


def _inputs():
    generator = torch.Generator().manual_seed(0)
    return {
        "query": torch.randn(1, 2, 4, 2, generator=generator),
        "key": torch.randn(1, 2, 4, 2, generator=generator),
        "value": torch.randn(1, 2, 4, 3, generator=generator),
    }


def _pair_term(state="numerator", *, prior=False, times=1.0):
    return (
        StateTerm(
            state,
            Outer(Input("query"), Input("query"), PAIR),
            PAIR,
            prior=prior,
        )
        * times
    )


def _count_term():
    return StateTerm("count", Constant(1.0), None)


def _operator(readout, *, with_count=True, transition=None):
    states = [StateSpec("numerator", (Batch, STATE_HEADS, PAIR, VALUE))]
    if with_count:
        states.append(StateSpec("count", (Batch, STATE_HEADS, UNIT)))
    transitions = [
        StateTransition(
            "numerator",
            injections=(
                ProductInjection(
                    (
                        ProductFactor(Outer(Input("key"), Input("key"), PAIR), (PAIR,)),
                        ProductFactor(Input("value"), (VALUE,)),
                    )
                ),
            ),
        ),
        StateTransition(
            "count",
            injections=(ProductInjection((ProductFactor(Constant(1.0), (UNIT,)),)),),
        ),
    ]
    if transition is not None:
        transitions[0] = transition
    if not with_count:
        transitions = transitions[:1]
    return StatefulOperator(
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
            states=tuple(states),
            transition=tuple(transitions),
            readouts=(readout,),
            head_mapping=HeadMapping(
                {
                    QUERY_HEADS: STATE_HEADS,
                    KEY_HEADS: STATE_HEADS,
                    VALUE_HEADS: STATE_HEADS,
                }
            ),
        )
    )


def _compile(operator):
    return operator.compile(
        role_sizes=ROLE_SIZES, target=Target(), mode="dense", optimized=True
    )


def test_product_feature_role_names_its_operands():
    assert PAIR.name == "key_dim_x_key_dim"
    assert PAIR.left is KEY and PAIR.right is KEY
    with pytest.raises(TypeError):
        ProductFeatureRole(KEY, "key_dim")


def test_outer_requires_an_operand_carrying_each_named_feature():
    with pytest.raises(StatefulCompilationError) as captured:
        _operator(
            Readout(
                "output",
                StateTerm("numerator", Outer(Input("value"), Input("key"), PAIR), PAIR),
                torch.float32,
            )
        )
    assert captured.value.category == "IR_TYPE"


def test_outer_rejects_a_plain_feature_role():
    with pytest.raises(TypeError):
        Outer(Input("key"), Input("key"), KEY)


def test_contract_requires_an_operand_carrying_the_contracted_feature():
    with pytest.raises(StatefulCompilationError) as captured:
        _operator(
            Readout(
                "output",
                _pair_term() + Contract(Input("value"), Constant(1.0), KEY),
                torch.float32,
            )
        )
    assert captured.value.category == "IR_TYPE"


def test_feature_free_contract_coefficient_compiles_and_runs():
    readout = Readout(
        "output",
        _pair_term()
        + Contract(Input("query") * Input("key"), Constant(1.0), KEY) * Input("value"),
        torch.float32,
    )
    output = _compile(_operator(readout))(**_inputs()).outputs["output"]
    assert output.shape == (1, 2, 4, 3)
    assert torch.isfinite(output).all()


def test_softmax_requires_the_named_axis_and_normalizes_it():
    with pytest.raises(StatefulCompilationError) as captured:
        _operator(
            Readout(
                "output",
                Softmax(
                    StateTerm(
                        "numerator", Outer(Input("query"), Input("query"), PAIR), PAIR
                    ),
                    PAIR,
                ),
                torch.float32,
            )
        )
    assert captured.value.category == "IR_TYPE"

    softmax = Softmax(
        StateTerm("numerator", Outer(Input("query"), Input("query"), PAIR), PAIR),
        VALUE,
    )
    output = _compile(_operator(Readout("output", softmax, torch.float32)))(
        **_inputs()
    ).outputs["output"]
    assert output.shape == (1, 2, 4, 3)
    # the softmax distributes over the VALUE axis, so every token's rows sum to 1
    assert torch.allclose(output.sum(-1), torch.ones_like(output.sum(-1)), atol=1e-5)


def test_feature_free_injection_factor_broadcasts_the_declared_axis():
    # the `count` State receives a constant that broadcasts along UNIT: a token count
    compiled = _compile(_operator(Readout("output", _count_term(), torch.float32)))
    output = compiled(**_inputs()).outputs["output"]
    assert output.shape == (1, 2, 4, 1)
    # a post-transition readout sees the running count, so token t reads t + 1
    assert torch.allclose(
        output[:, :, :, 0],
        torch.arange(1, 5, dtype=output.dtype).expand(1, 2, 4),
    )

    # an expression that carries a State Feature the factor does not declare is rejected
    with pytest.raises(StatefulCompilationError) as captured:
        _operator(
            Readout("output", _pair_term() + _count_term(), torch.float32),
            transition=StateTransition(
                "numerator",
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(
                                Outer(Input("key"), Input("key"), PAIR), (PAIR,)
                            ),
                            ProductFactor(
                                Input("value")
                                * Outer(Input("key"), Input("key"), PAIR),
                                (VALUE,),
                            ),
                        )
                    ),
                ),
            ),
        )
    assert captured.value.category == "IR_TYPE"


def test_scaled_transition_requires_a_declared_feature():
    with pytest.raises(StatefulCompilationError) as captured:
        _operator(
            Readout("output", _count_term(), torch.float32),
            transition=StateTransition(
                "numerator",
                propagations=(AxisScale(Exp(Input("slot")), (KEY,)),),
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(
                                Outer(Input("key"), Input("key"), PAIR), (PAIR,)
                            ),
                            ProductFactor(Input("value"), (VALUE,)),
                        )
                    ),
                ),
            ),
        )
    assert captured.value.category in ("IR_SCHEMA", "IR_TYPE")


def test_prior_state_term_reads_the_state_before_the_transition():
    # an identity readout of a matrix State stays invalid, with or without `prior`
    with pytest.raises(StatefulCompilationError) as captured:
        _operator(
            Readout(
                "output",
                StateTerm("numerator", Constant(1.0), None, prior=True),
                torch.float32,
            )
        )
    assert captured.value.category == "IR_TYPE"

    inputs = _inputs()
    prior = _compile(
        _operator(Readout("output", _pair_term(prior=True), torch.float32))
    )
    post = _compile(_operator(Readout("output", _pair_term(), torch.float32)))
    prior_output = prior(**inputs).outputs["output"]
    post_output = post(**inputs).outputs["output"]

    # the first token has no history, so only the post-transition readout sees it
    assert torch.allclose(
        prior_output[:, :, 0], torch.zeros_like(prior_output[:, :, 0])
    )
    assert not torch.allclose(
        post_output[:, :, 0], torch.zeros_like(post_output[:, :, 0])
    )
