"""Weighted writes retain vector/scalar roles and authored association."""

from dataclasses import replace

import pytest
import torch

from attn_engine import (
    AlgorithmIR,
    AxisScale,
    Batch,
    Contract,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    ProductFactor,
    ProductInjection,
    RankOnePropagation,
    Readout,
    Sequence,
    StateContraction,
    StateSpec,
    StateTerm,
    StateTransition,
    StatefulOperator,
    TensorInput,
)

pytestmark = pytest.mark.unit
KEY, VALUE = FeatureRole("key_dim"), FeatureRole("value_dim")
HEAD, SOURCE = HeadRole("state_heads"), HeadRole("source_heads")


@pytest.fixture(autouse=True)
def _allow_cpu(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")


def weighted_operator(order=(-1, -2, 0, 1), *, dtype=torch.float32):
    factors = (
        ProductFactor(Input("key"), (KEY,)),
        ProductFactor(Input("value") * Input("write"), (VALUE,)),
        ProductFactor(Input("weight"), ()),
        ProductFactor(Contract(Input("key"), Input("prediction"), KEY), ()),
    )
    # Order codes match the recipe but the fixture authors ordinary DSL nodes.
    indices = {-1: 0, -2: 1, 0: 2, 1: 3}
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, SOURCE, Sequence, KEY), dtype),
            TensorInput("key", (Batch, SOURCE, Sequence, KEY), dtype),
            TensorInput("prediction", (Batch, SOURCE, Sequence, KEY), dtype),
            TensorInput("value", (Batch, HEAD, Sequence, VALUE), dtype),
            TensorInput("write", (Batch, HEAD, Sequence, VALUE), dtype),
            TensorInput("decay", (Batch, HEAD, Sequence, KEY), torch.float32),
            TensorInput("weight", (Batch, HEAD, Sequence), torch.float32),
        ),
        states=(StateSpec("memory", (Batch, HEAD, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            propagations=(
                AxisScale(Input("decay"), (KEY,)),
                RankOnePropagation(KEY, Input("key"), Input("prediction"), -0.2),
            ),
            injections=(
                ProductInjection(tuple(factors[indices[code]] for code in order)),
                ProductInjection((factors[0], factors[1])),
            ),
        ),
        readouts=(
            StateContraction("output", "memory", Input("query"), KEY, torch.float32),
        ),
        head_mapping=HeadMapping({SOURCE: HEAD}),
    )
    return StatefulOperator(algorithm)


@pytest.mark.parametrize("order", [(-1, -2, 0, 1), (0, -2, 1, -1), (-2, 0, -1, 1)])
def test_weighted_recipe_preserves_factor_positions(order):
    operator = weighted_operator(order)
    recipe = operator.algorithm._analysis.rank_one_factored
    assert recipe is not None
    first, plain = recipe.injections
    assert first.factor_order == order
    assert first.left_factor == order.index(-1)
    assert first.right_factor == order.index(-2)
    assert first.scalar_factors == tuple(order.index(code) for code in (0, 1))
    assert plain.factor_order == (-1, -2)
    assert recipe.injection_scalar_count == 2


def test_matrix_factor_stays_outside_lowrank_recipe():
    algorithm = weighted_operator().algorithm
    injection = ProductInjection((ProductFactor(Input("weight"), (KEY, VALUE)),))
    changed = AlgorithmIR(
        inputs=algorithm.inputs,
        states=algorithm.states,
        transition=replace(algorithm.transition, injections=(injection,)),
        readouts=algorithm.readouts,
        head_mapping=algorithm.head_mapping,
    )
    assert changed._analysis.rank_one_factored is None


def test_state_term_readout_normalizes_to_weighted_factored_recipe():
    algorithm = weighted_operator().algorithm
    changed = AlgorithmIR(
        inputs=algorithm.inputs,
        states=algorithm.states,
        transition=algorithm.transition,
        readouts=(
            Readout("output", StateTerm("memory", Input("query"), KEY), torch.float32),
        ),
        head_mapping=algorithm.head_mapping,
    )
    assert isinstance(changed.readouts[0], Readout)
    assert isinstance(changed._analysis.normalized.readouts[0], StateContraction)
    assert changed._analysis.rank_one_factored == algorithm._analysis.rank_one_factored
    assert (
        changed._analysis.simplified.identity == algorithm._analysis.simplified.identity
    )


@pytest.mark.parametrize("backward", [False, True])
def test_weighted_plan_charges_scalar_factor_workspace(backward):
    from attn_engine import Target

    operator = weighted_operator()
    sizes = {HEAD: 2, SOURCE: 1, KEY: 64, VALUE: 128}
    weighted = operator.compile(
        role_sizes=sizes, target=Target(), mode="dense", backward=backward
    )
    algorithm = operator.algorithm
    plain = StatefulOperator(
        AlgorithmIR(
            inputs=algorithm.inputs,
            states=algorithm.states,
            transition=replace(
                algorithm.transition,
                injections=(algorithm.transition.injections[1],) * 2,
            ),
            readouts=algorithm.readouts,
            head_mapping=algorithm.head_mapping,
        )
    )
    unweighted = plain.compile(
        role_sizes=sizes, target=Target(), mode="dense", backward=backward
    )
    weighted_plan = operator._plans[weighted.specialization].plan
    plain_plan = plain._plans[unweighted.specialization].plan
    assert weighted_plan.candidate.lowerer == "tilelang-factored-recurrence"
    assert (
        weighted_plan.workspace.bytes_per_token > plain_plan.workspace.bytes_per_token
    )
    assert weighted_plan.workspace.required_bytes(2048, sequence_length=2048) < 1 << 30


def test_packed_workspace_charges_scalar_gradients_and_partials():
    from attn_engine.stateful.lowering.packed import packed_workspace_bound

    args = (52, 3, 2, 7, 19)
    plain = packed_workspace_bound(*args, operations=1, injections=2, readouts=1)
    weighted = packed_workspace_bound(
        *args, operations=1, injections=2, injection_scalars=3, readouts=1
    )
    scalar_elements = 52 * 2 * 3
    assert (
        weighted["packed_factor_elements"] - plain["packed_factor_elements"]
        == scalar_elements
    )
    assert (
        weighted["result_gradient_elements"] - plain["result_gradient_elements"]
        == scalar_elements
    )
    assert (
        weighted["partial_gradient_elements"] - plain["partial_gradient_elements"]
        == 2 * scalar_elements
    )
