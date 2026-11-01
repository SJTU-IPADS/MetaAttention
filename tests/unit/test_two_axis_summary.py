"""Composition rules for Axis Scale summaries.

Whole-State and per-axis scales all multiply the State elementwise, so a Program
that carries both must keep every authored factor: an accumulated whole-State
scale folds into whichever per-axis factor appears, and the generic family's
backward must read the same context the forward produced.
"""

from __future__ import annotations

import torch

from attn_engine import AxisScale, Exp, FeatureRole, HeadRole, Input, Target
from attn_engine.stateful.compile.plan import LoweredCandidate, backward_context_family
from attn_engine.stateful_operator import _evaluate
from examples.linear_attention import dual_gate_gla_attention

HEADS, TOKENS, KEY, VALUE = 2, 6, 3, 4


class _State:
    def __init__(self, features):
        self.feature_roles = features


def _summary(propagations, values):
    from attn_engine.stateful.lowering.structured import _axis_scale_summary

    key_role, value_role = FeatureRole("key_dim"), FeatureRole("value_dim")
    return _axis_scale_summary(
        propagations,
        evaluate=_evaluate,
        values=values,
        state_spec=_State((key_role, value_role)),
        token_count=TOKENS,
        heads=HEADS,
        left_size=KEY,
        right_size=VALUE,
        device=torch.device("cpu"),
        dense=False,
    )


def _values():
    return {
        "whole": torch.rand(TOKENS, HEADS, 1, 1),
        "rows": torch.rand(TOKENS, HEADS, KEY, 1),
        "columns": torch.rand(TOKENS, HEADS, 1, VALUE),
    }


def _whole_scale():
    return AxisScale(Exp(Input("whole")), ())


def _row_scale():
    return AxisScale(Exp(Input("rows")), (FeatureRole("key_dim"),))


def test_whole_state_scale_composes_with_a_first_feature_scale():
    values = _values()
    for propagations in (
        (_whole_scale(), _row_scale()),
        (_row_scale(), _whole_scale()),
    ):
        left, right, rows, columns = _summary(propagations, values)
        assert left is None and right is None and columns is None
        expected = values["whole"].exp() * values["rows"].exp()
        torch.testing.assert_close(rows, expected)


def test_whole_state_scale_composes_with_both_axis_scales():
    values = _values()
    left, right, rows, columns = _summary(
        (
            _whole_scale(),
            _row_scale(),
            AxisScale(Exp(Input("columns")), (FeatureRole("value_dim"),)),
        ),
        values,
    )
    assert left is None and right is None
    # the whole-State factor is applied once: it folds into the first per-axis
    # factor that appears, and the elementwise product stays w * r * c
    torch.testing.assert_close(rows, values["whole"].exp() * values["rows"].exp())
    torch.testing.assert_close(columns, values["columns"].exp())


def test_generic_two_axis_backward_reads_the_elementwise_context():
    operator = dual_gate_gla_attention(1, HEADS, TOKENS, KEY, VALUE)
    candidate = LoweredCandidate("c", "p", "tilelang-generic", "forward")
    assert (
        backward_context_family(operator.algorithm, candidate, False) == "elementwise"
    )
    assert backward_context_family(operator.algorithm, candidate, True) == "dense"


def test_generic_two_axis_compilation_selects_a_consistent_backward_context():
    """The compiled plan's context family must match what the forward produces."""
    size = 64
    operator = dual_gate_gla_attention(1, HEADS, TOKENS, size, size)
    state = operator.algorithm.states[0]
    sizes: dict = {state.head_role: HEADS}
    for role in state.feature_roles:
        sizes[role] = size
    for spec in operator.algorithm.inputs:
        for role in spec.roles:
            if isinstance(role, HeadRole):
                sizes.setdefault(role, HEADS)
    plan = operator.compile(
        role_sizes=sizes,
        target=Target(),
        mode="dense",
        optimized=False,
        backward=True,
    )
    executable = operator._plans[plan.specialization]
    assert executable.plan.candidate.lowerer == "tilelang-generic"
    assert executable.dense_summary is False
    assert executable.backward_binding is not None
    assert executable.backward_binding.family == "elementwise"
