"""Dependency proof for ordered, input-driven affine State updates."""

from __future__ import annotations

from ...stateful_operator import (
    AxisScale,
    RankOnePropagation,
    StateInjection,
    StatefulCompilationError,
)


def transition_schedule(algorithm):
    """Return a stable topological order and the joint action axis, if coupled.

    Prior references introduce no current-token edge. Current references always
    select a completed Transition; cycles (including self edges) are rejected.
    The first coupled lowering stacks compatible States along one Feature axis,
    preserving the other matrix axis instead of flattening matrix elements.
    """
    states = {state.name: state for state in algorithm.states}
    dependencies = {state.name: set() for state in algorithm.states}
    coupled = False
    action_roles = set()
    for index, transition in enumerate(algorithm.transitions):
        target = states[transition.state]
        for step_index, node in enumerate(transition.ordered_steps):
            path = f"transitions[{index}].steps[{step_index}]"
            if isinstance(node, StateInjection):
                coupled = True
                source = states.get(node.source.state)
                if source is None:
                    raise StatefulCompilationError(
                        "IR_TYPE",
                        f"{path}.source",
                        {"unknown_state": node.source.state},
                    )
                if source.roles != target.roles:
                    raise StatefulCompilationError(
                        "IR_TYPE",
                        f"{path}.source",
                        {"expected": "source and target State roles match"},
                    )
                if not node.source.prior:
                    dependencies[target.name].add(source.name)
                if node.role is not None:
                    action_roles.add(node.role)
            elif isinstance(node, RankOnePropagation):
                action_roles.add(node.role)
            elif isinstance(node, AxisScale):
                action_roles.update(node.roles)

    pending = dict(dependencies)
    order = []
    while pending:
        ready = [name for name, deps in pending.items() if not deps]
        if not ready:
            raise StatefulCompilationError(
                "IR_DEPENDENCY",
                "transitions",
                {"cycle": {name: sorted(deps) for name, deps in pending.items()}},
            )
        for name in ready:
            order.append(name)
            del pending[name]
        for deps in pending.values():
            deps.difference_update(ready)
    if not coupled:
        return tuple(order), None
    first = algorithm.states[0]
    if any(state.roles != first.roles for state in algorithm.states):
        raise StatefulCompilationError(
            "IR_TYPE", "states", {"expected": "coupled States share ordered roles"}
        )
    if len(action_roles) > 1 or not action_roles <= set(first.feature_roles):
        raise StatefulCompilationError(
            "IR_TYPE",
            "transitions",
            {
                "expected": "coupled affine actions on one shared Feature axis",
                "roles": sorted(map(str, action_roles)),
            },
        )
    axis = first.feature_roles.index(next(iter(action_roles))) if action_roles else 0
    return tuple(order), axis
