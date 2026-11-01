"""Reference lowering for independent and coupled affine Stateful Programs.

Schema v2 admits several independent States, rank-1 States, readouts expressed
as arithmetic over State Terms, and expression nodes such as ``Divide`` or
``Shift``. This module executes such a Program with the same maths the
specialized lowerings use:

* every State's per-token Transition is summarized as an affine map
  (``scale``/``left``/``right`` plus an additive ``bias``) built from the typed
  Algorithm IR;
* the summaries are composed by the segmented scans shared with the rest of the
  compiler, so execution stays parallel over the Sequence instead of falling
  back to a token loop;
* readouts evaluate the authored expression against prior/current States.

Schema v4 also composes ordered writes and coupled block-axis summaries through
``ordered.py``, preserving a parallel segmented scan for both forms.

The lowering is deliberately reference-grade: it is the semantic baseline for
v2 Programs and the oracle that specialized kernels must match. Kernels that
reuse the same summaries can replace it without changing Algorithm IR.
"""

from __future__ import annotations

from typing import Any, Mapping
import math

import torch

from ...stateful_operator import (
    Batch,
    FeatureRole,
    HeadRole,
    One,
    Readout,
    Sequence,
    StateContraction,
    StateTerm,
    StatefulCompilationError,
    _evaluate,
    _segmented_affine_scan,
    _segmented_elementwise_scan,
    _squeeze_singleton_features,
)


def feature_layout(
    algorithm: Any,
) -> tuple[tuple[FeatureRole, ...], dict[FeatureRole, int]]:
    """Ordered union of every State Feature role, and its axis in a token tensor.

    The value layout is ``[tokens, heads, *roles]`` with at least two axes, so a
    Program that uses a single Feature keeps a trailing singleton axis and one
    that uses more Feature roles keeps one axis each.
    """
    order: list[FeatureRole] = []
    for spec in (*algorithm.inputs, *algorithm.states):
        for role in spec.roles:
            if isinstance(role, FeatureRole) and role not in order:
                order.append(role)
    return tuple(order), {role: 2 + index for index, role in enumerate(order)}


def state_head_role(algorithm: Any) -> HeadRole:
    """The single Head role every State of a v2 Program must share."""
    roles = {state.head_role for state in algorithm.states}
    if len(roles) != 1:
        raise StatefulCompilationError(
            "IR_SCHEMA",
            "states",
            {
                "expected": "States share one Head role",
                "actual": sorted(str(role) for role in roles),
            },
        )
    return next(iter(roles))


def normalize_inputs(
    algorithm: Any,
    role_sizes: Mapping[Any, int],
    inputs: Mapping[str, torch.Tensor],
    *,
    packed: bool,
    sequence_count: int,
    token_count: int,
    uniform_length: int | None,
    state_heads: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Lay every Input out as token-major ``[tokens, state_heads, *features]``.

    Feature axes follow the Program's Feature order; an absent Feature is a
    singleton axis so a State Term can still locate and contract the role it
    needs.
    """
    order, _ = feature_layout(algorithm)
    result: dict[str, torch.Tensor] = {}
    for spec in algorithm.inputs:
        tensor = inputs[spec.name]
        head_role = next(
            (role for role in spec.roles if isinstance(role, HeadRole)), None
        )
        present = tuple(role for role in spec.roles if isinstance(role, FeatureRole))
        if packed and Sequence in spec.roles:
            physical = tuple(
                role for role in spec.roles if role not in (Batch, Sequence)
            )
            value = tensor
            value = _move_head_axis(value, tuple(physical), head_role)
        elif Sequence in spec.roles:
            feature_axes = [spec.roles.index(role) for role in present]
            value = tensor.permute(
                spec.roles.index(Batch),
                spec.roles.index(Sequence),
                *([spec.roles.index(head_role)] if head_role is not None else []),
                *feature_axes,
            )
            trailing = [tensor.shape[axis] for axis in feature_axes]
            value = (
                value.reshape(
                    token_count, *value.shape[2 : -len(trailing) or None], *trailing
                )
                if trailing
                else value.reshape(token_count, *value.shape[2:])
            )
        else:
            singleton = (
                spec.roles.index(Batch)
                if Batch in spec.roles
                else spec.roles.index(One)
            )
            feature_axes = [spec.roles.index(role) for role in present]
            order_axes = [singleton]
            if head_role is not None:
                order_axes.append(spec.roles.index(head_role))
            order_axes.extend(feature_axes)
            value = tensor.permute(*order_axes)
            trailing = [tensor.shape[axis] for axis in feature_axes]
            value = (
                value.reshape(1, *value.shape[1 : -len(trailing) or None], *trailing)
                if trailing
                else value.reshape(1, *value.shape[1:])
            )
            value = value.expand(sequence_count, *value.shape[1:])
            value = value.repeat_interleave(uniform_length or 1, dim=0)
        if head_role is not None:
            source_heads = value.shape[1]
            if state_heads % source_heads:
                raise StatefulCompilationError(
                    "RUNTIME_SHAPE",
                    f"inputs.{spec.name}",
                    {"heads": source_heads, "state_heads": state_heads},
                )
            if source_heads != state_heads:
                value = value.repeat_interleave(state_heads // source_heads, dim=1)
        else:
            value = value.unsqueeze(1).expand(
                -1, state_heads, *([1] * (value.ndim - 1))
            )
        # scatter the present features into the Program-wide Feature order
        feature_part = value.shape[2:]
        expanded = []
        cursor = 0
        for role in order:
            if role in present:
                expanded.append(feature_part[cursor])
                cursor += 1
            else:
                expanded.append(1)
        while len(expanded) < 2:
            expanded.append(1)
        value = value.reshape(value.shape[0], value.shape[1], *expanded)
        result[spec.name] = value.to(device)
    return result


def _move_head_axis(
    value: torch.Tensor, physical: tuple[Any, ...], head_role: HeadRole | None
) -> torch.Tensor:
    if head_role is None:
        return value.unsqueeze(1)
    index = 1 + physical.index(head_role)
    if index != 1:
        value = value.movedim(index, 1)
    return value


def execute_reference(
    algorithm: Any,
    *,
    values: Mapping[str, torch.Tensor],
    initial_states: Mapping[str, torch.Tensor],
    segment: torch.Tensor,
    role_sizes: Mapping[Any, int],
    heads: int,
    token_count: int,
    device: torch.device,
    return_final_state: bool,
    sequence_length: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    """Run one invocation of a v2 Program through the reference lowering."""
    order, axes = feature_layout(algorithm)
    state_values: dict[str, torch.Tensor] = {}
    prior_values: dict[str, torch.Tensor] = {}
    if algorithm._analysis.joint_axis is not None:
        from .ordered import scan_coupled_states

        state_values = scan_coupled_states(
            algorithm,
            values=values,
            initial_states=initial_states,
            segment=segment,
            role_sizes=role_sizes,
            heads=heads,
            token_count=token_count,
            device=device,
        )
    for state in algorithm.states:
        if state.name not in state_values:
            state_values[state.name] = _scan_state(
                algorithm,
                state,
                values=values,
                initial=initial_states[state.name],
                segment=segment,
                role_sizes=role_sizes,
                feature_axes=axes,
                heads=heads,
                token_count=token_count,
                device=device,
            )
        prior_values[state.name] = _prior_state(
            state_values[state.name], initial_states[state.name], segment
        )
    state_specs = {state.name: state for state in algorithm.states}
    outputs: dict[str, torch.Tensor] = {}
    for readout in algorithm.readouts:
        if isinstance(readout, StateContraction):
            expression = StateTerm(readout.state, readout.operand, readout.role)
        elif isinstance(readout, Readout):
            expression = readout.expression
        else:
            raise StatefulCompilationError(
                "IR_SCHEMA", "readouts", {"node": type(readout).__name__}
            )
        output = _evaluate(
            expression,
            values,
            states=state_values,
            state_specs=state_specs,
            feature_axes=axes,
            segment=segment,
            squeeze_features=True,
            prior_states=prior_values,
        )
        outputs[readout.name] = output.to(readout.output_dtype)
    final_states = None
    if return_final_state:
        final_states = {
            name: _final_state(state_values[name], segment, initial_states[name])
            for name in state_values
        }
    return outputs, final_states


def _scan_state(
    algorithm: Any,
    state: Any,
    *,
    values: Mapping[str, torch.Tensor],
    initial: torch.Tensor,
    segment: torch.Tensor,
    role_sizes: Mapping[Any, int],
    feature_axes: Mapping[Any, int],
    heads: int,
    token_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Compose one State's per-token affine summary and apply it in parallel."""
    transition = next(
        item for item in algorithm.transitions if item.state == state.name
    )
    if transition.steps is not None:
        from .ordered import scan_ordered_state

        return scan_ordered_state(
            algorithm,
            state,
            values=values,
            initial=initial,
            segment=segment,
            role_sizes=role_sizes,
            heads=heads,
            token_count=token_count,
            device=device,
        )
    view = _matrix_view(state)
    initial_matrix = initial.unsqueeze(-1) if state.rank == 1 else initial
    transition = next(
        item for item in algorithm.transitions if item.state == state.name
    )
    left_size = role_sizes[view.feature_roles[0]]
    right_size = 1 if state.rank == 1 else role_sizes[view.feature_roles[1]]
    # only a rank-one contraction composes a non-diagonal update; a single-feature
    # Axis Scale stays one elementwise factor per axis, as the v1 families do
    dense = any(
        type(node).__name__ == "RankOnePropagation" for node in transition.propagations
    )
    evaluate = lambda expression, bound: _evaluate(  # noqa: E731
        expression, bound, feature_axes=None, segment=segment
    )
    left, right, elementwise = _affine_summary(
        transition.propagations,
        view=view,
        evaluate=evaluate,
        values=values,
        token_count=token_count,
        heads=heads,
        left_size=left_size,
        right_size=right_size,
        device=device,
        dense=dense,
    )
    bias = _injection_terms(
        transition.injections,
        evaluate=evaluate,
        values=values,
        feature_roles=view.feature_roles,
        token_count=token_count,
        heads=heads,
        left_size=left_size,
        right_size=right_size,
    )
    if state.rank == 1 and not dense:
        # A promoted vector State scales every Feature with the same per-token
        # factor, so the scan tensors must share one matrix shape.
        elementwise = elementwise.expand(token_count, heads, left_size, 1)
    if bias is None:
        bias = torch.zeros(token_count, heads, left_size, right_size, device=device)
    if dense:
        prefix_left, prefix_right, prefix_bias = _segmented_affine_scan(
            left, right, bias, segment
        )
        states = (
            prefix_left @ initial_matrix[segment.long()] @ prefix_right + prefix_bias
        )
    else:
        streamed = None
        requires_grad = initial.requires_grad or any(
            tensor.requires_grad for tensor in values.values()
        )
        if device.type == "cuda" and not requires_grad:
            # one streaming kernel instead of the host-driven doubling scan, which
            # costs a pass per halving and materializes token-sized intermediates;
            # a differentiable call keeps the torch scan so autograd still works
            from .scan.elementwise import tilelang_elementwise_prefix

            streamed = tilelang_elementwise_prefix(
                elementwise.contiguous(),
                bias,
                segment.to(torch.int32),
                initial_matrix,
                config=(("block_tokens", 128),),
            )
        if streamed is None:
            prefix_scale, prefix_bias = _segmented_elementwise_scan(
                elementwise, bias, segment
            )
            streamed = prefix_scale * initial_matrix[segment.long()] + prefix_bias
        states = streamed
    return states.squeeze(-1) if state.rank == 1 else states


def _role_vector(value: torch.Tensor, token_count: int, heads: int) -> torch.Tensor:
    """Reduce a value carrying exactly one Feature role to ``[tokens, heads, size]``."""
    if value.ndim < 3 or sum(size != 1 for size in value.shape[2:]) > 1:
        raise StatefulCompilationError(
            "IR_TYPE",
            "expression",
            {"reason": "operand must carry exactly one Feature role"},
        )
    # A logical Feature of extent one is still a vector. Squeezing all
    # singleton axes would erase it; an explicit extent also admits no tokens.
    size = math.prod(value.shape[2:])
    return value.float().reshape(token_count, heads, size)


def _injection_terms(
    injections,
    *,
    evaluate,
    values: Mapping[str, torch.Tensor],
    feature_roles: tuple[Any, ...],
    token_count: int,
    heads: int,
    left_size: int,
    right_size: int,
) -> torch.Tensor | None:
    """Materialize additive Injections as ``[tokens, heads, left, right]``.

    A factor carrying the first State Feature (including a Product Feature role)
    lands on the left axis and a factor carrying the second lands on the right
    axis; a factor without Feature roles broadcasts over the whole State. Rank-1
    States use the same builder through their promoted matrix view.
    """
    result: torch.Tensor | None = None
    for injection in injections:
        product: torch.Tensor | None = None
        for factor in injection.factors:
            value = evaluate(factor.expression, values).float()
            roles = tuple(factor.roles)
            if not roles:
                shaped = (
                    value.reshape(1, 1, 1, 1)
                    if value.ndim == 0
                    else value.reshape(token_count, heads, 1, 1)
                )
            elif len(roles) == 1:
                squeezed = _squeeze_singleton_features(value)
                if squeezed.ndim == 0:
                    # a literal Constant broadcasts along every declared role
                    shaped = squeezed.reshape(1, 1, 1, 1)
                elif squeezed.ndim == 2:
                    # feature-free per token: broadcast along the declared role
                    shaped = squeezed.reshape(token_count, heads, 1, 1)
                elif roles[0] == feature_roles[0]:
                    shaped = _role_vector(value, token_count, heads).reshape(
                        token_count, heads, left_size, 1
                    )
                elif len(feature_roles) > 1 and roles[0] == feature_roles[1]:
                    shaped = _role_vector(value, token_count, heads).reshape(
                        token_count, heads, 1, right_size
                    )
                else:
                    raise StatefulCompilationError(
                        "IR_TYPE",
                        "transitions.injections",
                        {"factor_role": str(roles[0])},
                    )
            else:
                raise StatefulCompilationError(
                    "IR_TYPE",
                    "transitions.injections",
                    {
                        "factor_roles": [str(role) for role in roles],
                        "expected": "one Feature role per factor",
                    },
                )
            product = shaped if product is None else product * shaped
        assert product is not None
        result = product if result is None else result + product
    if result is None:
        return None
    return torch.broadcast_to(
        result, (token_count, heads, left_size, right_size)
    ).contiguous()


def _affine_summary(
    propagations,
    *,
    view: Any,
    evaluate,
    values: Mapping[str, torch.Tensor],
    token_count: int,
    heads: int,
    left_size: int,
    right_size: int,
    device: torch.device,
    dense: bool,
):
    """Turn ordered Propagations into ``(left, right, elementwise)`` summaries.

    This mirrors the single ordered pass the specialized lowerings use: Axis
    Scale acts on the Feature role it names and Rank-One composition keeps the
    authored order, so a noncommuting program cannot be silently reordered.
    """
    from ...stateful_operator import AxisScale, Identity, RankOnePropagation

    left = (
        torch.eye(left_size, device=device)
        .expand(token_count, heads, left_size, left_size)
        .contiguous()
        if dense
        else None
    )
    right = (
        torch.eye(right_size, device=device)
        .expand(token_count, heads, right_size, right_size)
        .contiguous()
        if dense
        else None
    )
    elementwise = None if dense else torch.ones(token_count, heads, 1, 1, device=device)
    for node in propagations:
        if isinstance(node, Identity):
            continue
        if isinstance(node, AxisScale):
            factor = evaluate(node.factor, values).float()
            if len(node.roles) == 2 or not node.roles:
                # a whole-State scale broadcasts over both axes
                scale = torch.broadcast_to(
                    factor.reshape(*([1] * (4 - factor.ndim)), *factor.shape),
                    (token_count, heads, 1, 1),
                ).float()
                if dense:
                    left = scale * left
                else:
                    elementwise = scale if elementwise is None else elementwise * scale
            elif node.roles[0] == view.feature_roles[0]:
                if left is None:
                    # one factor per row rather than a [rows, rows] matrix
                    row_factor = _role_vector(factor, token_count, heads).reshape(
                        token_count, heads, left_size, 1
                    )
                    elementwise = (
                        row_factor if elementwise is None else elementwise * row_factor
                    )
                else:
                    left = (
                        torch.diag_embed(_role_vector(factor, token_count, heads))
                        @ left
                    )
            elif node.roles[0] == view.feature_roles[1]:
                if right is None:
                    column_factor = _role_vector(factor, token_count, heads).reshape(
                        token_count, heads, 1, right_size
                    )
                    elementwise = (
                        column_factor
                        if elementwise is None
                        else elementwise * column_factor
                    )
                else:
                    right = right @ torch.diag_embed(
                        _role_vector(factor, token_count, heads)
                    )
            else:
                raise StatefulCompilationError(
                    "IR_TYPE", "propagations", {"role": str(node.roles[0])}
                )
            continue
        if not isinstance(node, RankOnePropagation):
            raise StatefulCompilationError(
                "IR_SCHEMA", "propagations", {"node": type(node).__name__}
            )
        if not dense:
            raise StatefulCompilationError(
                "IR_SCHEMA", "propagations", {"reason": "rank-one needs dense factors"}
            )
        left_vector = _role_vector(evaluate(node.left, values), token_count, heads)
        right_vector = _role_vector(evaluate(node.right, values), token_count, heads)
        raw_coefficient = evaluate(node.coefficient, values).float()
        coefficient = raw_coefficient.reshape(
            *raw_coefficient.shape, *([1] * max(0, 4 - raw_coefficient.ndim))
        )[..., :1, :1]
        if node.role == view.feature_roles[0]:
            update = torch.eye(left_size, device=device).expand(
                token_count, heads, left_size, left_size
            ) + coefficient * left_vector.unsqueeze(-1) * right_vector.unsqueeze(-2)
            left = update @ left
        else:
            update = torch.eye(right_size, device=device).expand(
                token_count, heads, right_size, right_size
            ) + coefficient * right_vector.unsqueeze(-1) * left_vector.unsqueeze(-2)
            right = right @ update
    return left, right, elementwise


def _matrix_view(state: Any):
    """Present a rank-1 State as a matrix with a singleton second Feature."""
    if state.rank == 2:
        return state
    from ...stateful_operator import StateSpec

    return StateSpec(
        state.name,
        (
            Batch,
            state.head_role,
            state.feature_roles[0],
            FeatureRole(f"{state.name}__unit"),
        ),
    )


def _prior_state(
    states: torch.Tensor, initial: torch.Tensor, segment: torch.Tensor
) -> torch.Tensor:
    """State before each token's Transition, per logical sequence."""
    prior = torch.empty_like(states)
    if states.shape[0] == 0:
        return prior
    prior[0] = initial[segment[0].long()]
    if states.shape[0] > 1:
        same = (segment[1:] == segment[:-1]).view(-1, *([1] * (states.ndim - 1)))
        prior[1:] = torch.where(same, states[:-1], initial[segment[1:].long()])
    return prior


def _final_state(
    states: torch.Tensor, segment: torch.Tensor, initial: torch.Tensor
) -> torch.Tensor:
    if states.shape[0] == 0:
        return initial.clone()
    sequence_count = initial.shape[0]
    lengths = torch.bincount(segment.long(), minlength=sequence_count)
    last = (torch.cumsum(lengths, dim=0) - 1).clamp_min(0)
    gathered = states[last]
    empty = (lengths == 0).view(-1, *([1] * (initial.ndim - 1)))
    return torch.where(empty, initial, gathered)
