"""Role-directed orientation is a compiler proof, independent of model names."""

from dataclasses import replace

import pytest

from attn_engine import (
    AlgorithmIR,
    Input,
    RankOnePropagation,
    Target,
)
from tests.unit.test_delta_family_examples import _case

pytestmark = pytest.mark.unit


def _changed(original, **kwargs):
    return AlgorithmIR(
        inputs=original.inputs,
        states=original.states,
        transition=kwargs.get("transition", original.transitions),
        readouts=kwargs.get("readouts", original.readouts),
        head_mapping=original.head_mapping,
    )


def test_right_axis_update_has_explicit_orientation_and_write_order():
    operator, _, _, _ = _case("oja")
    recipe = operator.algorithm._analysis.rank_one_factored
    assert recipe.transpose_state
    assert recipe.readout_axis == 1
    assert tuple(op.kind for op in recipe.operations) == ("row-scale", "rank-one")
    assert recipe.injections[0].left_factor == 1
    assert recipe.injections[0].right_factor == 0
    assert recipe.injections[0].factor_order == (-2, -1)
    same_axis = _changed(
        operator.algorithm,
        readouts=(
            replace(
                operator.algorithm.readouts[0],
                role=operator.algorithm.states[0].feature_roles[1],
                operand=Input("value"),
            ),
        ),
    )
    assert same_axis._analysis.rank_one_factored.transpose_state
    assert same_axis._analysis.rank_one_factored.readout_axis == 0


def test_mixed_update_or_readout_axes_keep_semantic_fallback():
    operator, _, _, _ = _case("oja")
    algorithm = operator.algorithm
    first, second = algorithm.states[0].feature_roles
    mixed_updates = _changed(
        algorithm,
        transition=replace(
            algorithm.transition,
            propagations=(
                *algorithm.transition.propagations,
                RankOnePropagation(first, Input("key"), Input("key"), 0.1),
            ),
        ),
    )
    assert mixed_updates._analysis.rank_one_factored is None
    mixed_readouts = _changed(
        algorithm,
        readouts=(
            *algorithm.readouts,
            replace(
                algorithm.readouts[0], name="other", role=second, operand=Input("value")
            ),
        ),
    )
    assert mixed_readouts._analysis.rank_one_factored is None


@pytest.mark.parametrize("dims", [(64, 128), (128, 64)])
@pytest.mark.parametrize("mode", ["dense", "packed"])
@pytest.mark.parametrize("backward", [False, True])
def test_orientation_reaches_executor_and_prices_internal_rows(dims, mode, backward):
    operator, roles, _, _ = _case("oja", dim=dims[0], dim_value=dims[1])
    compiled = operator.compile(
        role_sizes=roles, target=Target(), mode=mode, backward=backward
    )
    executable = operator._plans[compiled.specialization]
    assert executable.plan.candidate.lowerer == "tilelang-factored-recurrence"
    assert executable.rank_one_factored.transpose_state
    assert executable.rank_one_factored.readout_axis == 1
    assert (
        executable.backward_binding is not None
        if backward
        else executable.backward_binding is None
    )
    if not backward:
        heads, rows, columns = 2, dims[1], dims[0]
        assert executable.plan.workspace.bytes_per_chunk == 4 * heads * (
            rows * rows + 2 * rows * columns + columns
        )
    assert (
        executable.plan.workspace.required_bytes(2048, sequence_length=2048) < 1 << 30
    )
