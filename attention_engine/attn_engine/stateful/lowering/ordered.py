"""Affine summaries for interleaved writes and coupled State transitions.

Loops here visit authored steps and States, never sequence tokens. All token
maps compose with the same segmented parallel scans as independent States.
"""

from __future__ import annotations

import torch

from ...stateful_operator import (
    AxisScale,
    ProductInjection,
    RankOnePropagation,
    StateInjection,
    _evaluate,
    _segmented_affine_scan,
    _segmented_elementwise_scan,
)
from .reference import _affine_summary, _injection_terms, _matrix_view, _role_vector


def scan_ordered_state(
    algorithm,
    state,
    *,
    values,
    initial,
    segment,
    role_sizes,
    heads,
    token_count,
    device,
):
    """Compose every affine step, propagating each earlier write when required."""
    transition = next(t for t in algorithm.transitions if t.state == state.name)
    view = _matrix_view(state)
    rows = role_sizes[state.feature_roles[0]]
    columns = role_sizes[state.feature_roles[1]] if state.rank == 2 else 1
    shape = (token_count, heads, rows, columns)

    def evaluate(expr, bound):
        return _evaluate(expr, bound, segment=segment)

    dense = any(
        isinstance(step, RankOnePropagation) for step in transition.ordered_steps
    )
    left, right, scale = _affine_summary(
        (),
        view=view,
        evaluate=evaluate,
        values=values,
        token_count=token_count,
        heads=heads,
        left_size=rows,
        right_size=columns,
        device=device,
        dense=dense,
    )
    bias = torch.zeros(shape, device=device)
    for node in transition.ordered_steps:
        if isinstance(node, ProductInjection):
            bias = bias + _injection_terms(
                (node,),
                evaluate=evaluate,
                values=values,
                feature_roles=view.feature_roles,
                token_count=token_count,
                heads=heads,
                left_size=rows,
                right_size=columns,
            )
        else:
            step_left, step_right, step_scale = _affine_summary(
                (node,),
                view=view,
                evaluate=evaluate,
                values=values,
                token_count=token_count,
                heads=heads,
                left_size=rows,
                right_size=columns,
                device=device,
                dense=dense,
            )
            if dense:
                left = step_left @ left
                right = right @ step_right
                bias = step_left @ bias @ step_right
            else:
                scale = step_scale * scale
                bias = step_scale * bias
    initial_matrix = initial.unsqueeze(-1) if state.rank == 1 else initial
    if dense:
        left, right, bias = _segmented_affine_scan(left, right, bias, segment)
        result = left @ initial_matrix[segment.long()] @ right + bias
    else:
        scale, bias = _segmented_elementwise_scan(scale, bias, segment)
        result = scale * initial_matrix[segment.long()] + bias
    return result.squeeze(-1) if state.rank == 1 else result


def scan_coupled_states(
    algorithm,
    *,
    values,
    initial_states,
    segment,
    role_sizes,
    heads,
    token_count,
    device,
):
    """Stack compatible States on their shared action axis and scan jointly.

    Each completed Transition is a block row map from the joint Prior State.
    A current reference substitutes that source map; a prior reference selects
    the corresponding identity block. This constructs the transition DAG's
    affine summary without a token-local interpreter or a vec(K*V) transform.
    """
    axis = algorithm._analysis.joint_axis
    state = algorithm.states[0]
    active_role = state.feature_roles[axis]
    width = role_sizes[active_role]
    passive = role_sizes[state.feature_roles[1 - axis]] if state.rank == 2 else 1
    size = len(algorithm.states) * width
    identity = torch.eye(size, device=device).expand(token_count, heads, size, size)
    zero = torch.zeros(token_count, heads, width, passive, device=device)
    prior = {
        spec.name: (identity[..., index * width : (index + 1) * width, :], zero)
        for index, spec in enumerate(algorithm.states)
    }
    completed = {}
    transitions = {t.state: t for t in algorithm.transitions}

    def evaluate(expr):
        return _evaluate(expr, values, segment=segment).float()

    def scalar(expression):
        value = evaluate(expression)
        return value.reshape(token_count, heads, 1, 1) if value.numel() > 1 else value

    def vector(expression):
        value = evaluate(expression)
        return _role_vector(value, token_count, heads)

    def to_active(value):
        if state.rank == 1:
            return value.unsqueeze(-1)
        return value.transpose(-1, -2) if axis == 1 else value

    def injection(node):
        view = _matrix_view(state)
        value = _injection_terms(
            (node,),
            evaluate=lambda expr, bound: _evaluate(expr, bound, segment=segment),
            values=values,
            feature_roles=view.feature_roles,
            token_count=token_count,
            heads=heads,
            left_size=role_sizes[state.feature_roles[0]],
            right_size=role_sizes[state.feature_roles[1]] if state.rank == 2 else 1,
        )
        return value.transpose(-1, -2) if axis == 1 else value

    for name in algorithm._analysis.transition_order:
        transform, bias = prior[name]
        for node in transitions[name].ordered_steps:
            if isinstance(node, ProductInjection):
                bias = bias + injection(node)
            elif isinstance(node, AxisScale):
                scale = (
                    vector(node.factor).unsqueeze(-1)
                    if node.roles
                    else scalar(node.factor)
                )
                transform, bias = scale * transform, scale * bias
            elif isinstance(node, RankOnePropagation):
                left, right = vector(node.left), vector(node.right)
                coefficient = scalar(node.coefficient)
                transform = transform + coefficient * left.unsqueeze(-1) * (
                    right.unsqueeze(-2) @ transform
                )
                bias = bias + coefficient * left.unsqueeze(-1) * (
                    right.unsqueeze(-2) @ bias
                )
            elif isinstance(node, StateInjection):
                source_transform, source_bias = (
                    prior if node.source.prior else completed
                )[node.source.state]
                coefficient = scalar(node.coefficient)
                if node.role is None:
                    transform = transform + coefficient * source_transform
                    bias = bias + coefficient * source_bias
                else:
                    left, right = vector(node.left), vector(node.right)
                    transform = transform + coefficient * left.unsqueeze(-1) * (
                        right.unsqueeze(-2) @ source_transform
                    )
                    bias = bias + coefficient * left.unsqueeze(-1) * (
                        right.unsqueeze(-2) @ source_bias
                    )
        completed[name] = transform, bias
    transform = torch.cat([completed[s.name][0] for s in algorithm.states], dim=-2)
    bias = torch.cat([completed[s.name][1] for s in algorithm.states], dim=-2)
    initial = torch.cat(
        [to_active(initial_states[s.name]) for s in algorithm.states], dim=-2
    )
    offset = 1
    while offset < token_count:
        old_transform, old_bias = transform, bias
        valid = (segment[offset:] == segment[:-offset]).view(-1, 1, 1, 1)
        transform = torch.cat(
            (
                old_transform[:offset],
                torch.where(
                    valid,
                    old_transform[offset:] @ old_transform[:-offset],
                    old_transform[offset:],
                ),
            )
        )
        bias = torch.cat(
            (
                old_bias[:offset],
                torch.where(
                    valid,
                    old_transform[offset:] @ old_bias[:-offset] + old_bias[offset:],
                    old_bias[offset:],
                ),
            )
        )
        offset *= 2
    result = transform @ initial[segment.long()] + bias
    output = {}
    for index, spec in enumerate(algorithm.states):
        value = result[..., index * width : (index + 1) * width, :]
        if state.rank == 1:
            value = value.squeeze(-1)
        elif axis == 1:
            value = value.transpose(-1, -2)
        output[spec.name] = value
    return output
