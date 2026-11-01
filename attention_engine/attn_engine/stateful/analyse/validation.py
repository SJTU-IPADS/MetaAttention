"""Structural validation for the Algorithm IR, including ordered affine steps.

The module validates shapes of the authored program only: which nodes the
language admits, which roles they carry, and whether every State is driven by
exactly one Transition. Typing, Affine-Summary derivation, and Backward IR stay
in ``attn_engine.stateful.analyse.algorithm``.
"""

from __future__ import annotations

from ...stateful_operator import (
    Add,
    AlgorithmIR,
    AxisScale,
    Constant,
    Divide,
    Contract,
    Outer,
    Softmax,
    Abs,
    Cast,
    Clamp,
    Exp,
    Identity,
    Input,
    InputExpression,
    Log,
    Multiply,
    Negate,
    RankOnePropagation,
    Readout,
    Reciprocal,
    Shift,
    Sigmoid,
    Sqrt,
    StateContraction,
    StateTerm,
    StateInjection,
    ProductInjection,
    StatefulCompilationError,
    _reject_duplicate_names,
)


def validate_algorithm_structure(algorithm: AlgorithmIR) -> None:
    _reject_duplicate_names(algorithm.inputs, "inputs")
    _reject_duplicate_names(algorithm.states, "states")
    if not algorithm.states:
        raise StatefulCompilationError(
            "IR_SCHEMA", "states", {"expected": "at least one"}
        )
    transitions = algorithm.transitions
    if len(transitions) != len(algorithm.states):
        raise StatefulCompilationError(
            "IR_SCHEMA",
            "transitions",
            {
                "states": [state.name for state in algorithm.states],
                "transitions": len(transitions),
            },
        )
    declared = {state.name for state in algorithm.states}
    seen: set[str] = set()
    for index, transition in enumerate(transitions):
        path = f"transitions[{index}].state"
        if transition.state not in declared:
            raise StatefulCompilationError(
                "IR_TYPE",
                path,
                {"unknown_state": transition.state, "states": sorted(declared)},
            )
        if transition.state in seen:
            raise StatefulCompilationError(
                "IR_SCHEMA", path, {"duplicate_transition": transition.state}
            )
        seen.add(transition.state)
    if not algorithm.readouts:
        raise StatefulCompilationError(
            "IR_SCHEMA", "readouts", {"expected": "at least one"}
        )
    for index, readout in enumerate(algorithm.readouts):
        if isinstance(readout, (StateContraction, Readout)):
            continue
        raise StatefulCompilationError(
            "IR_SCHEMA", f"readouts[{index}]", {"node": type(readout).__name__}
        )
    _reject_duplicate_names(algorithm.readouts, "readouts")


def validate_transition_structure(algorithm: AlgorithmIR) -> None:
    states = {state.name: state for state in algorithm.states}
    for index, transition in enumerate(algorithm.transitions):
        state = states[transition.state]
        features = set(state.feature_roles)
        for step_index, step in enumerate(transition.ordered_steps):
            if not isinstance(
                step,
                (
                    Identity,
                    AxisScale,
                    RankOnePropagation,
                    ProductInjection,
                    StateInjection,
                ),
            ):
                raise StatefulCompilationError(
                    "IR_SCHEMA",
                    f"transitions[{index}].steps[{step_index}]",
                    {"node": type(step).__name__},
                )
            if isinstance(step, StateInjection):
                path = f"transitions[{index}].steps[{step_index}]"
                if step.role is not None and step.role not in features:
                    raise StatefulCompilationError("IR_TYPE", f"{path}.role", {})
                for name in ("coefficient", "left", "right"):
                    expression = getattr(step, name)
                    if expression is not None:
                        _reject_state_terms(
                            expression, f"{path}.{name}", "State Injection"
                        )
        for node_index, propagation in enumerate(transition.propagations):
            path = f"transitions[{index}].propagations[{node_index}]"
            if isinstance(propagation, AxisScale):
                if not set(propagation.roles) <= features:
                    raise StatefulCompilationError(
                        "IR_TYPE",
                        f"{path}.roles",
                        {"roles": [str(role) for role in propagation.roles]},
                    )
            elif isinstance(propagation, RankOnePropagation):
                if propagation.role not in features:
                    raise StatefulCompilationError(
                        "IR_TYPE", f"{path}.role", {"role": str(propagation.role)}
                    )
            elif not isinstance(propagation, Identity):
                raise StatefulCompilationError(
                    "IR_SCHEMA", path, {"node": type(propagation).__name__}
                )
            for expression_path, expression in _propagation_expressions(
                propagation, path
            ):
                _reject_state_terms(expression, expression_path, "Transition")
        for injection_index, injection in enumerate(transition.injections):
            path = f"transitions[{index}].injections[{injection_index}]"
            if not isinstance(injection, ProductInjection):
                raise StatefulCompilationError(
                    "IR_SCHEMA",
                    path,
                    {"expected": "ProductInjection; use steps for StateInjection"},
                )
            covered = {role for factor in injection.factors for role in factor.roles}
            if not features <= covered:
                raise StatefulCompilationError(
                    "IR_TYPE",
                    path,
                    {"missing_roles": sorted(str(role) for role in features - covered)},
                )
            for factor_index, factor in enumerate(injection.factors):
                factor_path = f"{path}.factors[{factor_index}].expression"
                _reject_state_terms(factor.expression, factor_path, "Injection")
    for index, readout in enumerate(algorithm.readouts):
        state_names = set(states)
        if isinstance(readout, StateContraction):
            if (
                readout.state not in state_names
                or readout.role not in states[readout.state].feature_roles
            ):
                raise StatefulCompilationError(
                    "IR_TYPE",
                    f"readouts[{index}]",
                    {"state": readout.state, "role": str(readout.role)},
                )
            continue
        for term_path, term in _state_terms(
            readout.expression, f"readouts[{index}].expression"
        ):
            if term.state not in state_names:
                raise StatefulCompilationError(
                    "IR_TYPE",
                    term_path,
                    {"unknown_state": term.state, "states": sorted(state_names)},
                )
            state = states[term.state]
            if term.role is not None and term.role not in state.feature_roles:
                raise StatefulCompilationError(
                    "IR_TYPE",
                    term_path,
                    {
                        "state": term.state,
                        "role": str(term.role),
                        "features": [str(role) for role in state.feature_roles],
                    },
                )
            if term.role is None and state.rank != 1:
                raise StatefulCompilationError(
                    "IR_TYPE",
                    term_path,
                    {"identity_readout": "requires a rank-1 State"},
                )


def _propagation_expressions(propagation, path: str):
    if isinstance(propagation, AxisScale):
        yield f"{path}.factor", propagation.factor
    elif isinstance(propagation, RankOnePropagation):
        yield f"{path}.left", propagation.left
        yield f"{path}.right", propagation.right
        yield f"{path}.coefficient", propagation.coefficient


def _reject_state_terms(expression: InputExpression, path: str, context: str) -> None:
    for term_path, term in _state_terms(expression, path):
        raise StatefulCompilationError(
            "IR_TYPE",
            term_path,
            {
                "state": term.state,
                "expected": f"{context} expression without State Terms",
            },
        )


def _state_terms(expression: InputExpression, path: str):
    if isinstance(expression, StateTerm):
        yield path, expression
        yield from _state_terms(expression.operand, f"{path}.operand")
        return
    if isinstance(expression, (Input, Constant)):
        return
    if isinstance(expression, Negate):
        yield from _state_terms(expression.operand, f"{path}.operand")
        return
    if isinstance(
        expression,
        (Abs, Cast, Exp, Log, Sigmoid, Sqrt, Reciprocal, Clamp, Shift, Softmax),
    ):
        yield from _state_terms(expression.operand, f"{path}.operand")
        return
    if isinstance(expression, (Add, Multiply, Divide, Outer, Contract)):
        yield from _state_terms(expression.left, f"{path}.left")
        yield from _state_terms(expression.right, f"{path}.right")
        return
    return
