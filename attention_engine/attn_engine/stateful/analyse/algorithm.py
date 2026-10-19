from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ...stateful_operator import AlgorithmIR, InputExpression


class ExecutionClass(Enum):
    SCANABLE = "scanable"
    RECURRENT = "recurrent"


@dataclass(frozen=True)
class ScalarExpressionIR:
    """One-input expression recipe accepted by the raw dense lowering."""

    input_name: str
    operation: str
    scale: float = 1.0
    is_log_scale: bool = False
    offset: float = 0.0
    # Nested FP32 multiplies execute from the Input outward, without folding.
    scale_steps: tuple[float, ...] = ()
    # BF16 input arithmetic must round at each authored expression node.
    expression_nodes: tuple[tuple[Any, ...], ...] = ()


@dataclass(frozen=True)
class DenseInputIR:
    """Physical axes needed to load one dense Tensor Input directly."""

    name: str
    batch_axis: int
    head_axis: int
    sequence_axis: int
    feature_axis: int | None


@dataclass(frozen=True)
class DenseScalarFactorizedIR:
    """Target-independent recipe for one-launch dense scalar lowering."""

    propagation: tuple[DenseInputIR, ScalarExpressionIR]
    left_factor: tuple[DenseInputIR, ScalarExpressionIR]
    right_factor: tuple[DenseInputIR, ScalarExpressionIR]
    readout: tuple[DenseInputIR, ScalarExpressionIR]


@dataclass(frozen=True)
class ScalarFactorizedIR:
    """Structural proof for scalar propagation with rank-one injections."""

    injection_count: int
    readout_count: int
    propagation_count: int
    dense: DenseScalarFactorizedIR | None


@dataclass(frozen=True)
class FactoredOperationIR:
    """One authored Transition step the factored kernel replays per token."""

    kind: str


@dataclass(frozen=True)
class WeightedOuterProductIR:
    """Two vector factors and feature-free factors in authored product order.

    ``factor_order`` uses -1/-2 for the first/second State Feature vectors and
    nonnegative local indices for scalar factors. No coefficients are folded
    into vectors, so FP32 product association and expression rounding survive.
    """

    left_factor: int
    right_factor: int
    scalar_factors: tuple[int, ...]
    factor_order: tuple[int, ...]


@dataclass(frozen=True)
class RankOneFactoredIR:
    """Structural proof for one ordered matrix-State recurrence.

    ``operations`` preserves authored order and ``readout_roles`` records every
    independent contraction fused by the inference chunk schedule.
    ``symmetric_rank_one`` proves that every rank-one propagation uses the same
    feature expression on both sides of its contraction. ``transpose_state``
    maps the public second-axis updates onto internal rows; ``readout_axis``
    identifies the homogeneous readout reduction axis in that internal layout.
    Mixed update axes or mixed readout axes retain the general affine fallback.
    """

    operations: tuple[FactoredOperationIR, ...]
    injection_count: int
    readout_roles: tuple[Any, ...]
    symmetric_rank_one: bool = False
    injections: tuple[WeightedOuterProductIR, ...] = ()
    transpose_state: bool = False
    readout_axis: int = 0
    injection_propagations: tuple[tuple[int, ...], ...] = ()

    @property
    def injection_scalar_count(self) -> int:
        return sum(len(term.scalar_factors) for term in self.injections)


@dataclass(frozen=True)
class TypedExpression:
    expression: InputExpression
    dtype: torch.dtype
    roles: tuple[Any, ...]


@dataclass(frozen=True)
class AffineSummaryIR:
    """Canonical semantic summary for vec(State): M @ state + additive."""

    state: str
    transform: str = "dense"
    additive: str = "state"
    states: tuple[str, ...] = ()


@dataclass(frozen=True)
class VJPStep:
    path: str
    operation: str
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class BackwardIR:
    output_cotangents: tuple[str, ...]
    final_state_cotangent: str
    reverse_steps: tuple[VJPStep, ...]
    input_gradients: tuple[str, ...]
    initial_state_gradient: str
    state_cotangents: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class SimplifiedIR:
    """Deterministic mandatory-safe optimization view."""

    identity: str
    rewrites: tuple[str, ...]


@dataclass(frozen=True)
class StructuralFeatures:
    axis_scale: bool
    vector_propagation: bool
    product_injection: bool
    rank_one: bool
    shared_state_contraction: bool


@dataclass(frozen=True)
class Analysis:
    execution_class: ExecutionClass
    affine_summary: AffineSummaryIR
    backward: BackwardIR
    expression_types: tuple[tuple[str, TypedExpression], ...]
    scalar_factorized: ScalarFactorizedIR | None
    rank_one_factored: RankOneFactoredIR | None
    simplified: SimplifiedIR
    structural_features: StructuralFeatures
    canonical_identity: str
    normalized: AlgorithmIR
    reference_required: bool
    transition_order: tuple[str, ...]
    joint_axis: int | None


def analyze_algorithm(algorithm: AlgorithmIR) -> Analysis:
    """Validate typing and derive target-independent forward/backward semantics."""
    from ...stateful_operator import (
        Add,
        AxisScale,
        Abs,
        Cast,
        Clamp,
        Constant,
        Divide,
        Contract,
        Exp,
        FeatureRole,
        Identity,
        Input,
        Log,
        Outer,
        Multiply,
        Negate,
        ProductInjection,
        RankOnePropagation,
        Reciprocal,
        Shift,
        Sigmoid,
        Softmax,
        Sqrt,
        StateContraction,
        StateTerm,
        StateInjection,
        StatefulCompilationError,
        _canonical_identity,
    )
    from .transitions import transition_schedule

    transition_order, joint_axis = transition_schedule(algorithm)
    input_types = {item.name: (item.dtype, item.roles) for item in algorithm.inputs}
    states = {state.name: state for state in algorithm.states}
    transitions = {transition.state: transition for transition in algorithm.transitions}
    ordered = tuple((state, transitions[state.name]) for state in algorithm.states)
    known_features = {
        role
        for spec in (*algorithm.inputs, *algorithm.states)
        for role in spec.roles
        if isinstance(role, FeatureRole)
    }
    state = algorithm.states[0]
    feature_roles = state.feature_roles
    types: list[tuple[str, TypedExpression]] = []

    def _type_outer(expression, path: str, recurse) -> TypedExpression:
        left = recurse(expression.left, f"{path}.left")
        right = recurse(expression.right, f"{path}.right")
        for name, typed, expected in (
            ("left", left, expression.role.left),
            ("right", right, expression.role.right),
        ):
            features = tuple(
                role for role in typed.roles if isinstance(role, FeatureRole)
            )
            if features != (expected,):
                raise StatefulCompilationError(
                    "IR_TYPE",
                    f"{path}.{name}",
                    {
                        "expected_feature": str(expected),
                        "operand_features": _role_names(features),
                    },
                )
        heads = tuple(role for role in left.roles if not isinstance(role, FeatureRole))
        typed = TypedExpression(expression, torch.float32, (*heads, expression.role))
        types.append((path, typed))
        return typed

    def _type_contract_or_softmax(expression, path: str, recurse) -> TypedExpression:
        if isinstance(expression, Softmax):
            operand = recurse(expression.operand, f"{path}.operand")
            features = tuple(
                role for role in operand.roles if isinstance(role, FeatureRole)
            )
            if features != (expression.role,):
                raise StatefulCompilationError(
                    "IR_TYPE",
                    f"{path}.operand",
                    {
                        "expected_feature": str(expression.role),
                        "operand_features": _role_names(features),
                    },
                )
            typed = TypedExpression(expression, torch.float32, operand.roles)
            types.append((path, typed))
            return typed
        left = recurse(expression.left, f"{path}.left")
        right = recurse(expression.right, f"{path}.right")
        for name, typed in (("left", left), ("right", right)):
            features = tuple(
                role for role in typed.roles if isinstance(role, FeatureRole)
            )
            # A Contract operand either carries exactly the contracted role or no
            # Feature at all, in which case it broadcasts: that is how a
            # Feature-axis reduction of a single expression is authored.
            if features not in ((), (expression.role,)):
                raise StatefulCompilationError(
                    "IR_TYPE",
                    f"{path}.{name}",
                    {
                        "expected_feature": str(expression.role),
                        "operand_features": _role_names(features),
                    },
                )
        heads = tuple(role for role in left.roles if not isinstance(role, FeatureRole))
        typed = TypedExpression(expression, torch.float32, heads)
        types.append((path, typed))
        return typed

    def type_expression(expression: InputExpression, path: str) -> TypedExpression:
        if isinstance(expression, Input):
            try:
                dtype, roles = input_types[expression.name]
            except KeyError as error:
                raise StatefulCompilationError(
                    "IR_TYPE", path, {"unknown_input": expression.name}
                ) from error
            typed = TypedExpression(
                expression,
                dtype,
                _canonical_roles(roles, algorithm),
            )
        elif isinstance(expression, Constant):
            typed = TypedExpression(expression, expression.dtype, ())
        elif isinstance(expression, StateTerm):
            spec = states.get(expression.state)
            if spec is None:
                raise StatefulCompilationError(
                    "IR_TYPE", path, {"unknown_state": expression.state}
                )
            operand = type_expression(expression.operand, f"{path}.operand")
            feature = tuple(
                role for role in operand.roles if role in spec.feature_roles
            )
            if expression.role is None:
                if spec.rank != 1 or feature:
                    raise StatefulCompilationError(
                        "IR_TYPE",
                        path,
                        {
                            "identity_readout": "requires a rank-1 State and a State-free operand"
                        },
                    )
                produced = (*spec.feature_roles,)
            else:
                if feature != (expression.role,):
                    raise StatefulCompilationError(
                        "IR_TYPE",
                        path,
                        {
                            "expected_feature": str(expression.role),
                            "operand_features": _role_names(feature),
                        },
                    )
                produced = tuple(
                    role for role in spec.feature_roles if role != expression.role
                )
            kept_heads = tuple(
                role for role in operand.roles if not isinstance(role, FeatureRole)
            )
            typed = TypedExpression(expression, torch.float32, (*kept_heads, *produced))
        elif isinstance(expression, (Outer, Contract, Softmax)):
            named = (
                (expression.role.left, expression.role.right)
                if isinstance(expression, Outer)
                else (expression.role,)
            )
            for role in named:
                if role not in known_features:
                    raise StatefulCompilationError(
                        "IR_TYPE", path, {"unknown_feature": str(role)}
                    )
            if isinstance(expression, Outer):
                return _type_outer(expression, path, type_expression)
            return _type_contract_or_softmax(expression, path, type_expression)
        elif isinstance(expression, (Add, Multiply, Divide)):
            left = type_expression(expression.left, f"{path}.left")
            right = type_expression(expression.right, f"{path}.right")
            if not _broadcastable(left.roles, right.roles, algorithm):
                raise StatefulCompilationError(
                    "IR_TYPE",
                    path,
                    {
                        "left_roles": _role_names(left.roles),
                        "right_roles": _role_names(right.roles),
                    },
                )
            dtype = (
                torch.float32
                if isinstance(expression, Divide)
                else _promote(left.dtype, right.dtype)
            )
            typed = TypedExpression(
                expression,
                dtype,
                _broadcast_roles(left.roles, right.roles, algorithm),
            )
        elif isinstance(expression, Cast):
            operand = type_expression(expression.operand, f"{path}.operand")
            typed = TypedExpression(expression, expression.dtype, operand.roles)
        elif isinstance(expression, (Negate, Shift)):
            operand = type_expression(expression.operand, f"{path}.operand")
            typed = TypedExpression(expression, operand.dtype, operand.roles)
        elif isinstance(expression, (Abs, Exp, Log, Sigmoid, Sqrt, Reciprocal, Clamp)):
            operand = type_expression(expression.operand, f"{path}.operand")
            typed = TypedExpression(expression, torch.float32, operand.roles)
        else:
            raise StatefulCompilationError(
                "IR_TYPE", path, {"node": type(expression).__name__}
            )
        types.append((path, typed))
        return typed

    reverse: list[VJPStep] = []
    transition_indices = {t.state: i for i, t in enumerate(algorithm.transitions)}
    if joint_axis is not None:
        ordered = tuple((states[name], transitions[name]) for name in transition_order)
    for state_spec, transition in ordered:
        base = (
            "transition"
            if len(ordered) == 1
            else f"transitions[{transition_indices[state_spec.name]}]"
        )
        nodes = (
            tuple(
                (f"{base}.steps[{i}]", node) for i, node in enumerate(transition.steps)
            )
            if transition.steps is not None
            else (
                *(
                    (f"{base}.propagations[{i}]", node)
                    for i, node in enumerate(transition.propagations)
                ),
                *(
                    (f"{base}.injections[{i}]", node)
                    for i, node in enumerate(transition.injections)
                ),
            )
        )
        for path, node in nodes:
            if isinstance(node, Identity):
                continue
            if isinstance(node, AxisScale):
                typed = type_expression(node.factor, f"{path}.factor")
                expected = tuple(
                    role for role in typed.roles if role in state_spec.feature_roles
                )
                if set(expected) != set(node.roles):
                    raise StatefulCompilationError(
                        "IR_TYPE",
                        f"{path}.roles",
                        {
                            "factor_features": _role_names(expected),
                            "declared": _role_names(node.roles),
                        },
                    )
                reverse.append(VJPStep(path, "axis_scale", _input_names(node.factor)))
            elif isinstance(node, RankOnePropagation):
                left = type_expression(node.left, f"{path}.left")
                right = type_expression(node.right, f"{path}.right")
                coefficient = type_expression(node.coefficient, f"{path}.coefficient")
                for name, typed in (("left", left), ("right", right)):
                    if tuple(
                        role for role in typed.roles if role in state_spec.feature_roles
                    ) != (node.role,):
                        raise StatefulCompilationError(
                            "IR_TYPE",
                            f"{path}.{name}",
                            {"expected_feature": str(node.role)},
                        )
                if any(role in state_spec.feature_roles for role in coefficient.roles):
                    raise StatefulCompilationError(
                        "IR_TYPE",
                        f"{path}.coefficient",
                        {"expected": "feature scalar"},
                    )
                reverse.append(
                    VJPStep(
                        path,
                        "rank_one",
                        _input_names(node.left, node.right, node.coefficient),
                    )
                )
            elif isinstance(node, ProductInjection):
                for factor_index, factor in enumerate(node.factors):
                    typed = type_expression(
                        factor.expression, f"{path}.factors[{factor_index}].expression"
                    )
                    actual = tuple(
                        role for role in typed.roles if role in state_spec.feature_roles
                    )
                    # An expression may broadcast along declared axes, but may
                    # not carry a State Feature axis it does not declare.
                    if not set(actual) <= set(factor.roles):
                        raise StatefulCompilationError(
                            "IR_TYPE",
                            f"{path}.factors[{factor_index}].roles",
                            {
                                "expression_features": _role_names(actual),
                                "declared": _role_names(factor.roles),
                            },
                        )
                reverse.append(
                    VJPStep(
                        path,
                        "product_injection",
                        tuple(
                            name
                            for factor in node.factors
                            for name in _input_names(factor.expression)
                        ),
                    )
                )
            elif isinstance(node, StateInjection):
                coefficient = type_expression(node.coefficient, f"{path}.coefficient")
                if any(isinstance(role, FeatureRole) for role in coefficient.roles):
                    raise StatefulCompilationError(
                        "IR_TYPE",
                        f"{path}.coefficient",
                        {"expected": "feature scalar"},
                    )
                if node.role is not None:
                    for name in ("left", "right"):
                        typed = type_expression(getattr(node, name), f"{path}.{name}")
                        if tuple(
                            role
                            for role in typed.roles
                            if isinstance(role, FeatureRole)
                        ) != (node.role,):
                            raise StatefulCompilationError(
                                "IR_TYPE",
                                f"{path}.{name}",
                                {"expected_feature": str(node.role)},
                            )
                version = "prior" if node.source.prior else "current"
                reverse.append(
                    VJPStep(
                        path,
                        "state_injection",
                        (
                            f"{node.source.state}.{version}",
                            *_input_names(node.coefficient, node.left, node.right),
                        ),
                    )
                )
            else:
                raise StatefulCompilationError(
                    "RECURRENT", path, {"node": type(node).__name__}
                )

    for index, readout in enumerate(algorithm.readouts):
        path = f"readouts[{index}]"
        if isinstance(readout, StateContraction):
            spec = states.get(readout.state)
            if spec is None:
                raise StatefulCompilationError(
                    "IR_TYPE", path, {"unknown_state": readout.state}
                )
            operand = type_expression(readout.operand, f"{path}.operand")
            actual = tuple(role for role in operand.roles if role in spec.feature_roles)
            if actual != (readout.role,):
                raise StatefulCompilationError(
                    "IR_TYPE",
                    f"{path}.operand",
                    {"expected_feature": str(readout.role)},
                )
            reverse.append(
                VJPStep(path, "state_contraction", _input_names(readout.operand))
            )
        else:
            type_expression(readout.expression, f"{path}.expression")
            reverse.append(VJPStep(path, "readout", _input_names(readout.expression)))

    normalized_readouts = _normalize_readouts(algorithm)
    if normalized_readouts != algorithm.readouts:
        from ...stateful_operator import AlgorithmIR

        normalized = AlgorithmIR(
            inputs=algorithm.inputs,
            states=algorithm.states,
            transition=algorithm.transitions,
            readouts=normalized_readouts,
            head_mapping=algorithm.head_mapping,
        )
        return replace(
            normalized._analysis,
            canonical_identity=_canonical_identity(algorithm),
            expression_types=tuple(types),
            normalized=normalized,
        )

    reverse.extend(
        VJPStep(f"head_mapping[{index}]", "head_reduce", (str(source),))
        for index, (source, _) in enumerate(algorithm.head_mapping.mappings)
    )
    for state_index, (state_spec, _) in enumerate(reversed(ordered)):
        base = (
            "transition"
            if len(ordered) == 1
            else f"transitions[{transition_indices[state_spec.name]}]"
        )
        reverse.append(
            VJPStep(
                f"{base}.state",
                "state_recurrence",
                (state_spec.name, f"d_{state_spec.name}_final"),
            )
        )
    backward = BackwardIR(
        tuple(readout.name for readout in algorithm.readouts),
        f"d_{state.name}_final",
        tuple(reversed(reverse)),
        tuple(item.name for item in algorithm.inputs if item.differentiable),
        f"d_{state.name}_initial",
        tuple(
            (
                state_spec.name,
                f"d_{state_spec.name}_final",
                f"d_{state_spec.name}_initial",
            )
            for state_spec in algorithm.states
        ),
    )
    scalar_factorized = None
    dense_scalar = None
    primary_transition = algorithm.transitions[0]
    v1_shape = (
        len(algorithm.states) == 1
        and algorithm.states[0].rank == 2
        and algorithm.transitions[0].steps is None
        and all(isinstance(readout, StateContraction) for readout in algorithm.readouts)
    )
    scalar_propagations = v1_shape and all(
        isinstance(node, Identity) or isinstance(node, AxisScale) and not node.roles
        for node in primary_transition.propagations
    )
    factorized_injections = (
        v1_shape
        and bool(primary_transition.injections)
        and all(
            len(injection.factors) == 2
            and {role for factor in injection.factors for role in factor.roles}
            == set(feature_roles)
            and all(len(factor.roles) == 1 for factor in injection.factors)
            for injection in primary_transition.injections
        )
    )
    same_orientation = (
        v1_shape
        and bool(algorithm.readouts)
        and all(readout.role == feature_roles[0] for readout in algorithm.readouts)
    )
    if scalar_propagations and factorized_injections and same_orientation:
        if len(primary_transition.propagations) == 0:
            propagation = (
                DenseInputIR("__identity_propagation__", 0, 1, 2, None),
                ScalarExpressionIR("__identity_propagation__", "identity"),
            )
        elif len(primary_transition.propagations) == 1:
            propagation_node = primary_transition.propagations[0]
            propagation = _dense_scalar_binding(
                propagation_node.factor, algorithm, feature_role=None
            )
        else:
            propagation = None
        if len(primary_transition.injections) == 1 and len(algorithm.readouts) == 1:
            injection = primary_transition.injections[0]
            left_factor = next(
                factor
                for factor in injection.factors
                if factor.roles == (feature_roles[0],)
            )
            right_factor = next(
                factor
                for factor in injection.factors
                if factor.roles == (feature_roles[1],)
            )
            left = _dense_scalar_binding(
                left_factor.expression, algorithm, feature_role=feature_roles[0]
            )
            right = _dense_scalar_binding(
                right_factor.expression, algorithm, feature_role=feature_roles[1]
            )
            readout = _dense_scalar_binding(
                algorithm.readouts[0].operand,
                algorithm,
                feature_role=feature_roles[0],
            )
            if all(
                recipe is not None for recipe in (propagation, left, right, readout)
            ):
                dense_scalar = DenseScalarFactorizedIR(
                    propagation, left, right, readout
                )
        scalar_factorized = ScalarFactorizedIR(
            len(primary_transition.injections),
            len(algorithm.readouts),
            len(primary_transition.propagations),
            dense_scalar,
        )
    rank_one_factored = rank_one_factored_eligible(algorithm)
    canonical_identity = _canonical_identity(algorithm)
    simplified_expressions = tuple(
        (path, simplify_expression(typed.expression)) for path, typed in types
    )
    simplified_identity = _canonical_identity(_simplified_identity_payload(algorithm))
    original_expression_identity = _canonical_identity(
        tuple((path, typed.expression) for path, typed in types)
    )
    simplified_rewrites = (
        ("safe-zero-one",)
        if _canonical_identity(simplified_expressions) != original_expression_identity
        else ()
    )
    simplified = SimplifiedIR(simplified_identity, simplified_rewrites)
    all_propagations = tuple(
        node for transition in algorithm.transitions for node in transition.propagations
    )
    readout_states: set[str] = set()
    for readout in algorithm.readouts:
        if isinstance(readout, StateContraction):
            readout_states.add(readout.state)
        else:
            readout_states.update(
                term.state for _, term in _all_state_terms(readout.expression)
            )
    propagation_factor = (
        primary_transition.propagations[0].factor
        if (
            v1_shape
            and len(primary_transition.propagations) == 1
            and isinstance(primary_transition.propagations[0], AxisScale)
        )
        else None
    )
    vector_propagation = (
        v1_shape
        and len(algorithm.transitions) == 1
        and len(primary_transition.propagations) == 1
        and isinstance(primary_transition.propagations[0], AxisScale)
        and isinstance(propagation_factor, Exp)
        and isinstance(propagation_factor.operand, Input)
        and primary_transition.propagations[0].roles == (feature_roles[0],)
        and factorized_injections
        and len(primary_transition.injections) == 1
        and same_orientation
    )
    structural_features = StructuralFeatures(
        axis_scale=any(isinstance(node, AxisScale) for node in all_propagations),
        vector_propagation=vector_propagation,
        product_injection=any(
            transition.injections for transition in algorithm.transitions
        ),
        rank_one=any(isinstance(node, RankOnePropagation) for node in all_propagations),
        shared_state_contraction=len(algorithm.readouts) > 1
        and len(readout_states) == 1,
    )
    return Analysis(
        ExecutionClass.SCANABLE,
        AffineSummaryIR(
            state.name,
            "joint-axis" if joint_axis is not None else "dense",
            "state",
            tuple(s.name for s in algorithm.states),
        ),
        backward,
        tuple(types),
        scalar_factorized,
        rank_one_factored,
        simplified,
        structural_features,
        canonical_identity,
        algorithm,
        _requires_reference(algorithm),
        transition_order,
        joint_axis,
    )


def _requires_reference(algorithm: AlgorithmIR) -> bool:
    """Prove whether the normalized program fits the matrix-summary executor.

    This describes semantic capabilities, not authored readout spellings. A
    separate factored proof may admit a compiled schedule for weighted writes.
    """
    from ...stateful_operator import Shift, StateContraction, _expressions

    if any(transition.steps is not None for transition in algorithm.transitions):
        return True
    if len(algorithm.states) != 1 or algorithm.states[0].rank != 2:
        return True
    if any(not isinstance(readout, StateContraction) for readout in algorithm.readouts):
        return True
    if any(isinstance(expression, Shift) for expression in _expressions(algorithm)):
        return True
    # The matrix-summary path assumes two feature factors. General products
    # retain reference binding/broadcast semantics; the factored proof
    # separately admits weighted outer writes.
    features = algorithm.states[0].feature_roles
    for injection in algorithm.transitions[0].injections:
        if len(injection.factors) != 2 or {
            factor.roles for factor in injection.factors
        } != {
            (features[0],),
            (features[1],),
        }:
            return True
    return False


def _normalize_readouts(algorithm: AlgorithmIR) -> tuple[Any, ...]:
    """Canonicalize proved post-transition matrix contractions after typing.

    The operand DAG and output cast are preserved. General arithmetic, prior
    reads and operands depending on another State remain expression readouts.
    This pass is independent of target, optimization flags and kernel family.
    """
    from ...stateful_operator import Readout, StateContraction, StateTerm

    states = {state.name: state for state in algorithm.states}
    readouts = []
    for readout in algorithm.readouts:
        if isinstance(readout, Readout):
            term = readout.expression
            if (
                isinstance(term, StateTerm)
                and not term.prior
                and states[term.state].rank == 2
                and term.role in states[term.state].feature_roles
                and not any(_all_state_terms(term.operand))
            ):
                readout = StateContraction(
                    readout.name,
                    term.state,
                    term.operand,
                    term.role,
                    readout.output_dtype,
                )
        readouts.append(readout)
    return tuple(readouts)


def _promote(left: torch.dtype, right: torch.dtype) -> torch.dtype:
    return torch.float32 if torch.float32 in (left, right) else torch.bfloat16


def simplify_expression(expression):
    """Apply representation-safe identities without reassociating arithmetic."""
    from ...stateful_operator import (
        Add,
        Abs,
        Cast,
        Clamp,
        Constant,
        Contract,
        Divide,
        Exp,
        Input,
        Log,
        Multiply,
        Outer,
        Negate,
        Reciprocal,
        Shift,
        Sigmoid,
        Softmax,
        Sqrt,
        StateTerm,
    )

    if isinstance(expression, (Input, Constant, StateTerm)):
        if isinstance(expression, StateTerm):
            return StateTerm(
                expression.state,
                simplify_expression(expression.operand),
                expression.role,
                prior=expression.prior,
            )
        return expression
    if isinstance(expression, Cast):
        # Retain even constant casts: rounding precedes subsequent arithmetic.
        return Cast(simplify_expression(expression.operand), expression.dtype)
    if isinstance(expression, Abs):
        return Abs(simplify_expression(expression.operand))
    if isinstance(expression, Negate):
        operand = simplify_expression(expression.operand)
        if isinstance(operand, Constant):
            return Constant(-operand.value, operand.dtype)
        return Negate(operand)
    if isinstance(expression, Exp):
        operand = simplify_expression(expression.operand)
        if isinstance(operand, Constant) and operand.value == 0:
            return Constant(1.0, operand.dtype)
        return Exp(operand)
    if isinstance(expression, (Log, Sigmoid, Sqrt, Reciprocal)):
        operand = simplify_expression(expression.operand)
        if isinstance(operand, Constant):
            folded = _fold_unary(expression, operand)
            if folded is not None:
                return folded
        return type(expression)(operand)
    if isinstance(expression, Clamp):
        operand = simplify_expression(expression.operand)
        if isinstance(operand, Constant):
            return Constant(
                min(max(operand.value, expression.minimum), expression.maximum),
                operand.dtype,
            )
        return Clamp(operand, expression.minimum, expression.maximum)
    if isinstance(expression, Shift):
        return Shift(simplify_expression(expression.operand), expression.lag)
    if isinstance(expression, Softmax):
        return Softmax(simplify_expression(expression.operand), expression.role)
    if isinstance(expression, Outer):
        return Outer(
            simplify_expression(expression.left),
            simplify_expression(expression.right),
            expression.role,
        )
    if isinstance(expression, Contract):
        return Contract(
            simplify_expression(expression.left),
            simplify_expression(expression.right),
            expression.role,
        )
    if isinstance(expression, (Add, Multiply, Divide)):
        left = simplify_expression(expression.left)
        right = simplify_expression(expression.right)
        if isinstance(expression, Add):
            if _is_constant(left, 0):
                return right
            if _is_constant(right, 0):
                return left
            return Add(left, right)
        if isinstance(expression, Divide):
            if _is_constant(right, 1):
                return left
            return Divide(left, right)
        if _is_constant(left, 1):
            return right
        if _is_constant(right, 1):
            return left
        return Multiply(left, right)
    raise TypeError(f"unsupported expression: {type(expression).__name__}")


def _fold_unary(expression, constant):
    """Fold a transcendental or reciprocal of a non-negative literal Constant."""
    from ...stateful_operator import Constant, Log, Reciprocal, Sigmoid, Sqrt

    value = constant.value
    if isinstance(expression, (Log, Sqrt, Reciprocal)) and value <= 0:
        return None
    try:
        if isinstance(expression, Log):
            folded = math.log(value)
        elif isinstance(expression, Sigmoid):
            folded = 1.0 / (1.0 + math.exp(-value))
        elif isinstance(expression, Sqrt):
            folded = math.sqrt(value)
        else:
            folded = 1.0 / value
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(folded):
        return None
    return Constant(folded, constant.dtype)


def _all_state_terms(expression):
    """Yield ``(path, StateTerm)`` for every State reference in an expression."""
    from ...stateful_operator import (
        Add,
        Abs,
        Cast,
        Clamp,
        Constant,
        Contract,
        Divide,
        Exp,
        Input,
        Log,
        Multiply,
        Outer,
        Negate,
        Reciprocal,
        Shift,
        Sigmoid,
        Softmax,
        Sqrt,
        StateTerm,
    )

    if isinstance(expression, StateTerm):
        yield "state", expression
        yield from _all_state_terms(expression.operand)
        return
    if isinstance(expression, (Input, Constant)):
        return
    if isinstance(
        expression,
        (Negate, Abs, Cast, Exp, Log, Sigmoid, Sqrt, Reciprocal, Clamp, Shift, Softmax),
    ):
        yield from _all_state_terms(expression.operand)
        return
    if isinstance(expression, (Add, Multiply, Divide, Outer, Contract)):
        yield from _all_state_terms(expression.left)
        yield from _all_state_terms(expression.right)
        return
    return


def _simplified_identity_payload(algorithm):
    """Build a canonical identity payload without recursively validating a new IR."""
    from ...stateful_operator import (
        AxisScale,
        ProductFactor,
        RankOnePropagation,
        Readout,
        StateContraction,
    )

    def simplify_transition(transition):
        if transition.steps is not None:
            # Keep the semantic order and StateReference versions in identity.
            return transition
        propagations = tuple(
            replace(node, factor=simplify_expression(node.factor))
            if isinstance(node, AxisScale)
            else replace(
                node,
                left=simplify_expression(node.left),
                right=simplify_expression(node.right),
                coefficient=simplify_expression(node.coefficient),
            )
            if isinstance(node, RankOnePropagation)
            else node
            for node in transition.propagations
        )
        injections = tuple(
            replace(
                injection,
                factors=tuple(
                    ProductFactor(simplify_expression(factor.expression), factor.roles)
                    for factor in injection.factors
                ),
            )
            for injection in transition.injections
        )
        return replace(transition, propagations=propagations, injections=injections)

    transitions = tuple(
        simplify_transition(transition) for transition in algorithm.transitions
    )
    readouts = tuple(
        replace(readout, operand=simplify_expression(readout.operand))
        if isinstance(readout, StateContraction)
        else replace(readout, expression=simplify_expression(readout.expression))
        if isinstance(readout, Readout)
        else readout
        for readout in algorithm.readouts
    )
    return (
        algorithm.inputs,
        algorithm.states,
        transitions,
        readouts,
        algorithm.head_mapping,
    )


def _is_constant(expression, value: float) -> bool:
    from ...stateful_operator import Constant

    return isinstance(expression, Constant) and expression.value == value


def _broadcastable(
    left: tuple[Any, ...], right: tuple[Any, ...], algorithm: AlgorithmIR
) -> bool:
    from ...stateful_operator import HeadRole

    left_heads = {
        _mapped_head(role, algorithm) for role in left if isinstance(role, HeadRole)
    }
    right_heads = {
        _mapped_head(role, algorithm) for role in right if isinstance(role, HeadRole)
    }
    return not left_heads or not right_heads or left_heads == right_heads


def _broadcast_roles(
    left: tuple[Any, ...], right: tuple[Any, ...], algorithm: AlgorithmIR
) -> tuple[Any, ...]:
    from ...stateful_operator import HeadRole, One

    combined = left + tuple(role for role in right if role not in left)
    result: list[Any] = []
    for role in combined:
        if role is One:
            continue
        mapped = _mapped_head(role, algorithm) if isinstance(role, HeadRole) else role
        if mapped not in result:
            result.append(mapped)
    return tuple(result) or combined


def _canonical_roles(roles: tuple[Any, ...], algorithm: AlgorithmIR) -> tuple[Any, ...]:
    from ...stateful_operator import HeadRole

    return tuple(
        _mapped_head(role, algorithm) if isinstance(role, HeadRole) else role
        for role in roles
    )


def _mapped_head(role: Any, algorithm: AlgorithmIR) -> Any:
    return algorithm.head_mapping.target_for(role) or role


def _role_names(roles: tuple[Any, ...]) -> list[str]:
    return [str(role) for role in roles]


def _input_names(*expressions: InputExpression) -> tuple[str, ...]:
    from ...stateful_operator import (
        Add,
        Abs,
        Cast,
        Clamp,
        Contract,
        Divide,
        Exp,
        Input,
        Log,
        Multiply,
        Outer,
        Negate,
        Reciprocal,
        Shift,
        Sigmoid,
        Softmax,
        Sqrt,
        StateTerm,
    )

    result: list[str] = []
    for expression in expressions:
        if isinstance(expression, Input):
            result.append(expression.name)
        elif isinstance(expression, (Add, Multiply, Divide, Outer, Contract)):
            result.extend(_input_names(expression.left, expression.right))
        elif isinstance(
            expression,
            (
                Negate,
                Abs,
                Cast,
                Exp,
                Log,
                Sigmoid,
                Sqrt,
                Reciprocal,
                Clamp,
                Shift,
                Softmax,
            ),
        ):
            result.extend(_input_names(expression.operand))
        elif isinstance(expression, StateTerm):
            result.extend(_input_names(expression.operand))
    return tuple(result)


def rank_one_factored_eligible(algorithm: AlgorithmIR) -> RankOneFactoredIR | None:
    """Prove an ordered matrix recurrence is factorizable.

    The proof preserves authored operation order. Scale operations may target
    either feature axis; injections and readouts remain independent terms over
    the same State.
    """
    from ...stateful_operator import (
        AxisScale,
        RankOnePropagation,
        Shift,
        StateContraction,
        _expressions,
    )

    from ...stateful_operator import StateInjection, ProductInjection

    if any(
        isinstance(node, StateInjection)
        for transition in algorithm.transitions
        for node in transition.ordered_steps
    ):
        return None
    if any(isinstance(expression, Shift) for expression in _expressions(algorithm)):
        return None
    if len(algorithm.states) != 1 or algorithm.states[0].rank != 2:
        return None
    if len(algorithm.transitions) != 1:
        return None
    state = algorithm.states[0]
    first_role, second_role = state.feature_roles
    update_roles = {
        node.role
        for node in algorithm.transitions[0].propagations
        if isinstance(node, RankOnePropagation)
    }
    if len(update_roles) != 1:
        return None
    update_role = next(iter(update_roles))
    if update_role not in state.feature_roles:
        return None
    transpose_state = update_role == second_role
    if transpose_state:
        first_role, second_role = second_role, first_role
    readouts = algorithm.readouts
    if not readouts or not all(
        isinstance(readout, StateContraction) for readout in readouts
    ):
        return None
    if any(readout.state != state.name for readout in readouts):
        return None
    readout_roles = tuple(readout.role for readout in readouts)
    if len(set(readout_roles)) != 1:
        return None
    readout_role = readout_roles[0]
    if readout_role not in (first_role, second_role):
        return None
    readout_axis = int(readout_role == second_role)
    operations: list[FactoredOperationIR] = []
    for node in algorithm.transitions[0].propagations:
        if isinstance(node, AxisScale):
            if node.roles and not set(node.roles).issubset({first_role, second_role}):
                return None
            if len(node.roles) > 1:
                return None
            kind = (
                "scale"
                if not node.roles
                else ("row-scale" if node.roles[0] == first_role else "column-scale")
            )
            operations.append(FactoredOperationIR(kind))
        elif isinstance(node, RankOnePropagation):
            if node.role != first_role:
                return None
            operations.append(FactoredOperationIR("rank-one"))
        else:
            return None
    if not operations or not any(
        operation.kind == "rank-one" for operation in operations
    ):
        return None
    injections = algorithm.transitions[0].injections
    if not injections:
        return None
    writes: list[WeightedOuterProductIR] = []
    for injection in injections:
        left = right = None
        scalars: list[int] = []
        order: list[int] = []
        for index, factor in enumerate(injection.factors):
            roles = tuple(factor.roles)
            if roles == (first_role,) and left is None:
                left = index
                order.append(-1)
            elif roles == (second_role,) and right is None:
                right = index
                order.append(-2)
            elif not roles:
                order.append(len(scalars))
                scalars.append(index)
            else:
                return None
        if left is None or right is None:
            return None
        writes.append(WeightedOuterProductIR(left, right, tuple(scalars), tuple(order)))
    symmetric_rank_one = all(
        node.left == node.right
        for node in algorithm.transitions[0].propagations
        if isinstance(node, RankOnePropagation)
    )
    propagation_positions = [
        index
        for index, node in enumerate(algorithm.transitions[0].ordered_steps)
        if isinstance(node, (AxisScale, RankOnePropagation))
    ]
    return RankOneFactoredIR(
        tuple(operations),
        len(injections),
        readout_roles,
        symmetric_rank_one,
        tuple(writes),
        transpose_state,
        readout_axis,
        tuple(
            tuple(
                propagation_index
                for propagation_index, position in enumerate(propagation_positions)
                if position > write_index
            )
            for write_index, node in enumerate(algorithm.transitions[0].ordered_steps)
            if isinstance(node, ProductInjection)
        )
        if algorithm.transitions[0].steps is not None
        else (),
    )


def _dense_scalar_binding(
    expression: InputExpression,
    algorithm: AlgorithmIR,
    *,
    feature_role: Any | None,
    require_exp: bool = False,
) -> tuple[DenseInputIR, ScalarExpressionIR] | None:
    from ...stateful_operator import (
        Add,
        Batch,
        Constant,
        Exp,
        HeadRole,
        Input,
        Multiply,
        Negate,
        Sequence,
    )

    authored_expression = expression
    operation = "identity"
    scale = 1.0
    offset = 0.0
    is_log_scale = False
    if isinstance(expression, Add):
        if isinstance(expression.left, Constant):
            offset += expression.left.value
            expression = expression.right
        elif isinstance(expression.right, Constant):
            offset += expression.right.value
            expression = expression.left
        else:
            return None
    if isinstance(expression, Exp):
        expression = expression.operand
        is_log_scale = True
    elif require_exp:
        return None
    if isinstance(expression, Negate):
        expression = expression.operand
        scale = -scale
    constants = []
    while isinstance(expression, Multiply):
        if isinstance(expression.left, Constant):
            constants.append(expression.left)
            expression = expression.right
        elif isinstance(expression.right, Constant):
            constants.append(expression.right)
            expression = expression.left
        else:
            return None
    scale_steps = ()
    if len(constants) > 1:
        # The raw kernels evaluate in FP32. A BF16 intermediate would require
        # an explicit rounding step before the next multiply.
        if any(constant.dtype != torch.float32 for constant in constants):
            return None
        scale_steps = tuple(constant.value for constant in reversed(constants))
        if scale == -1.0:
            scale_steps += (-1.0,)
        scale = 1.0
    elif constants:
        scale *= constants[0].value
    if not isinstance(expression, Input):
        return None
    spec = next(item for item in algorithm.inputs if item.name == expression.name)
    if Batch not in spec.roles or Sequence not in spec.roles:
        return None
    head_axis = next(
        (index for index, role in enumerate(spec.roles) if isinstance(role, HeadRole)),
        None,
    )
    if head_axis is None:
        return None
    expected_roles = (
        (Batch, spec.roles[head_axis], Sequence)
        if feature_role is None
        else (Batch, spec.roles[head_axis], Sequence, feature_role)
    )
    if spec.roles != expected_roles:
        return None
    present_features = tuple(
        role for role in algorithm.states[0].feature_roles if role in spec.roles
    )
    if present_features != (() if feature_role is None else (feature_role,)):
        return None
    feature_axis = None if feature_role is None else spec.roles.index(feature_role)
    binding = DenseInputIR(
        spec.name,
        spec.roles.index(Batch),
        head_axis,
        spec.roles.index(Sequence),
        feature_axis,
    )
    expression_nodes = ()
    if spec.dtype == torch.bfloat16:
        from ..lowering.expression.dense_expression import compile_dense_expression

        dense = compile_dense_expression(
            authored_expression, {spec.name: spec}, feature_role
        )
        if dense is None:
            return None
        expression_nodes = dense.loader.nodes
    return binding, ScalarExpressionIR(
        spec.name, operation, scale, is_log_scale, offset, scale_steps, expression_nodes
    )
