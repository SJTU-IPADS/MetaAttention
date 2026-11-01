from __future__ import annotations

import os
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from typing import (
    Any,
    Iterable,
    Iterator,
    Mapping,
    Sequence as TypingSequence,
)

import torch

SCHEMA_VERSION = "stateful-operator-ir/5"
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


@dataclass(frozen=True)
class _SingletonRole:
    name: str

    def __str__(self) -> str:
        return self.name


Batch = _SingletonRole("batch")
Sequence = _SingletonRole("sequence")
One = _SingletonRole("one")


@dataclass(frozen=True)
class HeadRole:
    name: str

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"invalid Head role name {self.name!r}")

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class FeatureRole:
    name: str

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"invalid Feature role name {self.name!r}")

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, init=False)
class ProductFeatureRole(FeatureRole):
    """A Feature role whose axis is the flattened product of two Feature roles.

    Second-order feature maps such as Based's ``k (x) k`` state are declared with
    this role instead of a hand-made name, so the compiler can check that
    ``size(role) == size(left) * size(right)`` and know how an :class:`Outer`
    node lays its result out.
    """

    left: FeatureRole
    right: FeatureRole

    def __init__(self, left: FeatureRole, right: FeatureRole) -> None:
        if not isinstance(left, FeatureRole) or not isinstance(right, FeatureRole):
            raise TypeError("Product Feature role requires two Feature roles")
        object.__setattr__(self, "name", f"{left.name}_x_{right.name}")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)


LogicalRole = _SingletonRole | HeadRole | FeatureRole


class StatefulCompilationError(Exception):
    """Stable public failure from Stateful Operator validation or specialization."""

    def __init__(
        self,
        category: str,
        path: str,
        details: Mapping[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        self.category = category
        self.code = category
        self.path = path
        self.details = dict(details or {})
        super().__init__(message or f"{category} at {path}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "path": self.path,
            "details": _canonical(self.details),
        }


class _NamedTensorTuple:
    __slots__ = ("_names", "_values")

    def __init__(
        self, names: TypingSequence[str], values: TypingSequence[torch.Tensor]
    ) -> None:
        if len(names) != len(values):
            raise ValueError("names and values must have equal length")
        if len(set(names)) != len(names):
            raise ValueError("names must be unique")
        object.__setattr__(self, "_names", tuple(names))
        object.__setattr__(self, "_values", tuple(values))

    @classmethod
    def _unchecked(
        cls, names: tuple[str, ...], values: tuple[torch.Tensor, ...]
    ) -> _NamedTensorTuple:
        result = cls.__new__(cls)
        object.__setattr__(result, "_names", names)
        object.__setattr__(result, "_values", values)
        return result

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def values(self) -> tuple[torch.Tensor, ...]:
        return self._values

    def __getitem__(self, key: int | str) -> torch.Tensor:
        if isinstance(key, int):
            return self._values[key]
        try:
            return self._values[self._names.index(key)]
        except ValueError as error:
            raise KeyError(key) from error

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def as_dict(self) -> dict[str, torch.Tensor]:
        return dict(zip(self._names, self._values))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.as_dict()!r})"


class NamedOutputTuple(_NamedTensorTuple):
    pass


class StateTuple(_NamedTensorTuple):
    pass


@dataclass(frozen=True)
class ExecutionResult:
    outputs: NamedOutputTuple
    final_state: StateTuple | None = None


class InputExpression:
    def __add__(self, other: InputExpression | float | int) -> Add:
        return Add(self, as_expression(other))

    def __radd__(self, other: InputExpression | float | int) -> Add:
        return Add(as_expression(other), self)

    def __sub__(self, other: InputExpression | float | int) -> Add:
        return Add(self, Negate(as_expression(other)))

    def __rsub__(self, other: InputExpression | float | int) -> Add:
        return Add(as_expression(other), Negate(self))

    def __mul__(self, other: InputExpression | float | int) -> Multiply:
        return Multiply(self, as_expression(other))

    def __rmul__(self, other: InputExpression | float | int) -> Multiply:
        return Multiply(as_expression(other), self)

    def __truediv__(self, other: InputExpression | float | int) -> Divide:
        return Divide(self, as_expression(other))

    def __rtruediv__(self, other: InputExpression | float | int) -> Divide:
        return Divide(as_expression(other), self)

    def __neg__(self) -> Negate:
        return Negate(self)

    def exp(self) -> Exp:
        return Exp(self)

    def log(self) -> Log:
        return Log(self)

    def abs(self) -> Abs:
        return Abs(self)

    def to(self, dtype: torch.dtype) -> Cast:
        return Cast(self, dtype)

    def sigmoid(self) -> Sigmoid:
        return Sigmoid(self)

    def sqrt(self) -> Sqrt:
        return Sqrt(self)

    def reciprocal(self) -> Reciprocal:
        return Reciprocal(self)

    def clamp(self, minimum: float, maximum: float) -> Clamp:
        return Clamp(self, minimum, maximum)

    def outer(
        self, other: InputExpression | float | int, role: ProductFeatureRole
    ) -> Outer:
        return Outer(self, other, role)

    def contract(
        self, other: InputExpression | float | int, role: FeatureRole
    ) -> Contract:
        return Contract(self, other, role)

    def softmax(self, role: FeatureRole) -> Softmax:
        return Softmax(self, role)


@dataclass(frozen=True)
class Input(InputExpression):
    name: str


@dataclass(frozen=True, init=False)
class Constant(InputExpression):
    value: float
    dtype: torch.dtype
    bits: int

    def __init__(self, value: float | int, dtype: torch.dtype = torch.float32) -> None:
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError("constants support only torch.float32 and torch.bfloat16")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("constants reject NaN and infinity")
        rounded = torch.tensor(numeric, dtype=dtype)
        integer_dtype = torch.int32 if dtype == torch.float32 else torch.int16
        object.__setattr__(self, "value", float(rounded.item()))
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "bits", int(rounded.view(integer_dtype).item()))


@dataclass(frozen=True)
class Add(InputExpression):
    left: InputExpression
    right: InputExpression

    def __post_init__(self) -> None:
        left, right = self.left, self.right
        if _identity(left) > _identity(right):
            object.__setattr__(self, "left", right)
            object.__setattr__(self, "right", left)


@dataclass(frozen=True)
class Multiply(InputExpression):
    left: InputExpression
    right: InputExpression

    def __post_init__(self) -> None:
        left, right = self.left, self.right
        if _identity(left) > _identity(right):
            object.__setattr__(self, "left", right)
            object.__setattr__(self, "right", left)


@dataclass(frozen=True)
class Negate(InputExpression):
    operand: InputExpression


@dataclass(frozen=True)
class Exp(InputExpression):
    operand: InputExpression


@dataclass(frozen=True)
class Divide(InputExpression):
    """Quotient of two expressions; computed in FP32 like every reciprocal form."""

    left: InputExpression
    right: InputExpression


@dataclass(frozen=True)
class Cast(InputExpression):
    operand: InputExpression
    dtype: torch.dtype

    def __post_init__(self):
        if self.dtype not in _SUPPORTED_DTYPES:
            raise ValueError("Cast supports FP32 and BF16")


@dataclass(frozen=True)
class Abs(InputExpression):
    operand: InputExpression


@dataclass(frozen=True)
class Log(InputExpression):
    operand: InputExpression


@dataclass(frozen=True)
class Sigmoid(InputExpression):
    operand: InputExpression


@dataclass(frozen=True)
class Sqrt(InputExpression):
    operand: InputExpression


@dataclass(frozen=True)
class Reciprocal(InputExpression):
    operand: InputExpression


@dataclass(frozen=True, init=False)
class Clamp(InputExpression):
    """Finite two-sided clamp; the bounds are compile-time constants."""

    operand: InputExpression
    minimum: float
    maximum: float

    def __init__(
        self, operand: InputExpression | float | int, minimum: float, maximum: float
    ) -> None:
        low, high = float(minimum), float(maximum)
        if not (math.isfinite(low) and math.isfinite(high)) or low > high:
            raise ValueError("Clamp bounds must be finite and ordered")
        object.__setattr__(self, "operand", as_expression(operand))
        object.__setattr__(self, "minimum", low)
        object.__setattr__(self, "maximum", high)


@dataclass(frozen=True, init=False)
class Shift(InputExpression):
    """Value of ``operand`` ``lag`` tokens earlier inside the same sequence.

    Tokens whose lag crosses a sequence boundary read the typed zero of
    ``operand``.
    """

    operand: InputExpression
    lag: int

    def __init__(self, operand: InputExpression | float | int, lag: int) -> None:
        steps = int(lag)
        if isinstance(lag, bool) or steps < 1:
            raise ValueError("Shift requires a positive integer lag")
        object.__setattr__(self, "operand", as_expression(operand))
        object.__setattr__(self, "lag", steps)


@dataclass(frozen=True, init=False)
class Outer(InputExpression):
    """Flattened outer product of two Feature-carrying expressions.

    The result carries :class:`ProductFeatureRole` ``role`` in row-major order:
    element ``i * size(right) + j`` is ``left[i] * right[j]``.
    """

    left: InputExpression
    right: InputExpression
    role: ProductFeatureRole

    def __init__(
        self,
        left: InputExpression | float | int,
        right: InputExpression | float | int,
        role: ProductFeatureRole,
    ) -> None:
        if not isinstance(role, ProductFeatureRole):
            raise TypeError("Outer requires a Product Feature role")
        object.__setattr__(self, "left", as_expression(left))
        object.__setattr__(self, "right", as_expression(right))
        object.__setattr__(self, "role", role)


@dataclass(frozen=True, init=False)
class Contract(InputExpression):
    """Feature-axis contraction of two expressions carrying the same role."""

    left: InputExpression
    right: InputExpression
    role: FeatureRole

    def __init__(
        self,
        left: InputExpression | float | int,
        right: InputExpression | float | int,
        role: FeatureRole,
    ) -> None:
        if not isinstance(role, FeatureRole):
            raise TypeError("Contract requires a Feature role")
        object.__setattr__(self, "left", as_expression(left))
        object.__setattr__(self, "right", as_expression(right))
        object.__setattr__(self, "role", role)


@dataclass(frozen=True, init=False)
class Softmax(InputExpression):
    """Softmax over the axis of one Feature role."""

    operand: InputExpression
    role: FeatureRole

    def __init__(
        self, operand: InputExpression | float | int, role: FeatureRole
    ) -> None:
        if not isinstance(role, FeatureRole):
            raise TypeError("Softmax requires a Feature role")
        object.__setattr__(self, "operand", as_expression(operand))
        object.__setattr__(self, "role", role)


@dataclass(frozen=True, init=False)
class StateTerm(InputExpression):
    """Contraction of one named State inside a readout expression.

    ``role`` selects the contracted Feature role; the term carries the State's
    other Feature role. ``role=None`` reads a rank-1 State itself without
    contraction, which is how identity readouts such as HGRN's ``o = h`` are
    expressed. ``prior=True`` reads the State from before the current token's
    Transition, which is what a readout such as RWKV-6's
    ``o = (h_{t-1} + u (x) k v) q`` needs. Transitions are driven by Inputs only,
    so a State Term never appears inside a Transition.
    """

    state: str
    operand: InputExpression
    role: FeatureRole | None
    prior: bool

    def __init__(
        self,
        state: str,
        operand: InputExpression | float | int,
        role: FeatureRole | None,
        *,
        prior: bool = False,
    ) -> None:
        if not isinstance(state, str) or not state:
            raise TypeError("State Term requires a State name")
        if role is not None and not isinstance(role, FeatureRole):
            raise TypeError("State Term role must be a Feature role or None")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "operand", as_expression(operand))
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "prior", bool(prior))


def as_expression(value: InputExpression | float | int) -> InputExpression:
    if isinstance(value, InputExpression):
        return value
    if isinstance(value, (int, float)):
        return Constant(value)
    raise TypeError(
        f"expected an input expression or numeric constant, got {type(value).__name__}"
    )


@dataclass(frozen=True)
class TensorInput:
    name: str
    roles: tuple[LogicalRole, ...]
    dtype: torch.dtype
    differentiable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", tuple(self.roles))
        if not self.name.isidentifier():
            raise ValueError(f"invalid tensor input name {self.name!r}")
        if not self.roles or len(set(self.roles)) != len(self.roles):
            raise ValueError(
                f"tensor input '{self.name}' requires unique logical roles"
            )
        if not all(
            isinstance(role, (_SingletonRole, HeadRole, FeatureRole))
            for role in self.roles
        ):
            raise TypeError("Tensor Input roles must be typed logical roles")
        if self.dtype not in _SUPPORTED_DTYPES:
            raise ValueError("Tensor Inputs support only BF16 and FP32")


@dataclass(frozen=True, init=False)
class StateSpec:
    """One recurrent State: a matrix with two ordered Features, or a vector.

    A rank-2 State carries two distinct ordered Feature roles; a rank-1 State
    (``Batch, Head, Feature``) carries a single Feature role and is how scalar
    accumulators such as a linear-attention denominator are declared.
    """

    name: str
    roles: tuple[_SingletonRole, HeadRole, FeatureRole, ...]

    def __init__(self, name: str, roles: TypingSequence[LogicalRole]) -> None:
        roles = tuple(roles)
        if (
            len(roles) not in (3, 4)
            or roles[0] != Batch
            or not isinstance(roles[1], HeadRole)
            or not all(isinstance(role, FeatureRole) for role in roles[2:])
        ):
            raise ValueError("State roles are Batch, one Head, and one or two Features")
        if len(roles) == 4 and roles[2] == roles[3]:
            raise ValueError("State Feature roles must be distinct and ordered")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "roles", roles)

    @property
    def rank(self) -> int:
        return len(self.roles) - 2

    @property
    def head_role(self) -> HeadRole:
        return self.roles[1]

    @property
    def feature_roles(self) -> tuple[FeatureRole, ...]:
        return tuple(self.roles[2:])


@dataclass(frozen=True)
class Identity:
    pass


@dataclass(frozen=True, init=False)
class AxisScale:
    factor: InputExpression
    roles: tuple[FeatureRole, ...]

    def __init__(
        self,
        factor: InputExpression | float | int,
        roles: TypingSequence[FeatureRole] = (),
    ) -> None:
        roles = tuple(roles)
        if len(set(roles)) != len(roles) or not all(
            isinstance(role, FeatureRole) for role in roles
        ):
            raise TypeError("Axis Scale roles must be unique Feature roles")
        object.__setattr__(self, "factor", as_expression(factor))
        object.__setattr__(self, "roles", roles)


@dataclass(frozen=True, init=False)
class RankOnePropagation:
    role: FeatureRole
    left: InputExpression
    right: InputExpression
    coefficient: InputExpression

    def __init__(
        self,
        role: FeatureRole,
        left: InputExpression,
        right: InputExpression,
        coefficient: InputExpression | float | int = 1.0,
    ) -> None:
        if not isinstance(role, FeatureRole):
            raise TypeError("Rank-One propagation role must be a Feature role")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "left", as_expression(left))
        object.__setattr__(self, "right", as_expression(right))
        object.__setattr__(self, "coefficient", as_expression(coefficient))


@dataclass(frozen=True, init=False)
class ProductFactor:
    expression: InputExpression
    roles: tuple[FeatureRole, ...]

    def __init__(
        self, expression: InputExpression, roles: TypingSequence[FeatureRole] = ()
    ) -> None:
        roles = tuple(roles)
        if len(set(roles)) != len(roles) or not all(
            isinstance(role, FeatureRole) for role in roles
        ):
            raise TypeError("Product factor roles must be unique Feature roles")
        object.__setattr__(self, "expression", as_expression(expression))
        object.__setattr__(self, "roles", roles)


@dataclass(frozen=True)
class ProductInjection:
    factors: tuple[ProductFactor, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "factors", tuple(self.factors))
        if not self.factors:
            raise ValueError("Product Injection requires at least one factor")


@dataclass(frozen=True, init=False)
class StateContraction:
    name: str
    state: str
    operand: InputExpression
    role: FeatureRole
    output_dtype: torch.dtype

    def __init__(
        self,
        name: str,
        state: str,
        operand: InputExpression,
        role: FeatureRole,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if not isinstance(role, FeatureRole):
            raise TypeError("State Contraction role must be a Feature role")
        if output_dtype not in _SUPPORTED_DTYPES:
            raise ValueError("State Contraction output supports only BF16 and FP32")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "operand", as_expression(operand))
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "output_dtype", output_dtype)


@dataclass(frozen=True, init=False)
class Readout:
    """A named output produced by an arbitrary expression over States and Inputs.

    The canonical single-role form is :class:`StateContraction`. ``Readout`` is
    the general form used when an output combines several States, divides by a
    denominator State, adds a non-State term, or reads a State without
    contraction. Its expression may contain :class:`StateTerm` nodes, which
    reference post-transition State values.
    """

    name: str
    expression: InputExpression
    output_dtype: torch.dtype

    def __init__(
        self,
        name: str,
        expression: InputExpression | float | int,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("Readout requires a name")
        if output_dtype not in _SUPPORTED_DTYPES:
            raise ValueError("Readout output supports only BF16 and FP32")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "expression", as_expression(expression))
        object.__setattr__(self, "output_dtype", output_dtype)


@dataclass(frozen=True, init=False)
class HeadMapping:
    mappings: tuple[tuple[HeadRole, HeadRole], ...]

    def __init__(
        self,
        mappings: Mapping[HeadRole, HeadRole]
        | TypingSequence[tuple[HeadRole, HeadRole]] = (),
    ) -> None:
        values = mappings.items() if isinstance(mappings, Mapping) else mappings
        pairs = tuple(values)
        if not all(
            isinstance(source, HeadRole) and isinstance(target, HeadRole)
            for source, target in pairs
        ):
            raise TypeError("Head Mapping requires typed Head roles")
        if len({source for source, _ in pairs}) != len(pairs):
            raise ValueError("Head Mapping source roles must be unique")
        object.__setattr__(
            self, "mappings", tuple(sorted(pairs, key=lambda pair: pair[0].name))
        )

    def target_for(self, source: HeadRole) -> HeadRole | None:
        return dict(self.mappings).get(source)


@dataclass(frozen=True)
class StateReference:
    """Prior token State, or the complete current-token value of another State.

    Current references induce a Transition dependency, never an implicit read
    of a partially updated State. Within a Transition, each step acts on that
    Transition's running value.
    """

    state: str
    prior: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, str) or not self.state.isidentifier():
            raise ValueError("State Reference requires a State name")
        if not isinstance(self.prior, bool):
            raise TypeError("State Reference prior must be a boolean")


@dataclass(frozen=True, init=False)
class StateInjection:
    """Add an input-driven linear action on a referenced State.

    Without ``role``, add ``coefficient * source``. With ``role``, add
    ``coefficient * left * contract(right, source)`` along that Feature axis.
    Unlike RankOnePropagation, this action has no implicit identity term.
    Source and destination have identical logical roles. Coefficients and
    vectors are State-free, keeping the joint Transition affine.
    """

    source: StateReference
    coefficient: InputExpression
    role: FeatureRole | None
    left: InputExpression | None
    right: InputExpression | None

    def __init__(
        self,
        source: StateReference,
        coefficient: InputExpression | float | int = 1.0,
        *,
        role: FeatureRole | None = None,
        left: InputExpression | float | int | None = None,
        right: InputExpression | float | int | None = None,
    ) -> None:
        if not isinstance(source, StateReference):
            raise TypeError("State Injection requires a StateReference")
        if role is not None and not isinstance(role, FeatureRole):
            raise TypeError("State Injection role must be a Feature role")
        if (role is None and (left is not None or right is not None)) or (
            role is not None and (left is None or right is None)
        ):
            raise ValueError(
                "State Injection rank-one action requires role, left and right"
            )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "coefficient", as_expression(coefficient))
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "left", None if left is None else as_expression(left))
        object.__setattr__(
            self, "right", None if right is None else as_expression(right)
        )


@dataclass(frozen=True, init=False)
class StateTransition:
    state: str
    propagations: tuple[Identity | AxisScale | RankOnePropagation, ...]
    injections: tuple[ProductInjection, ...]
    steps: (
        tuple[
            Identity
            | AxisScale
            | RankOnePropagation
            | ProductInjection
            | StateInjection,
            ...,
        ]
        | None
    )

    def __init__(
        self,
        state: str,
        *,
        propagations: TypingSequence[Identity | AxisScale | RankOnePropagation] = (),
        injections: TypingSequence[ProductInjection] = (),
        steps: TypingSequence[
            Identity
            | AxisScale
            | RankOnePropagation
            | ProductInjection
            | StateInjection
        ]
        | None = None,
    ) -> None:
        propagations, injections = tuple(propagations), tuple(injections)
        if steps is not None:
            if propagations or injections:
                raise ValueError("use steps or propagations/injections, not both")
            steps = tuple(node for node in steps if not isinstance(node, Identity))
            propagations = tuple(
                node
                for node in steps
                if isinstance(node, (AxisScale, RankOnePropagation))
            )
            injections = tuple(
                node for node in steps if isinstance(node, ProductInjection)
            )
        object.__setattr__(self, "state", state)
        object.__setattr__(
            self,
            "propagations",
            tuple(node for node in propagations if not isinstance(node, Identity)),
        )
        object.__setattr__(self, "injections", injections)
        object.__setattr__(self, "steps", steps)

    @property
    def ordered_steps(self):
        """Semantic step order, including the legacy grouped spelling."""
        return (
            self.steps
            if self.steps is not None
            else (*self.propagations, *self.injections)
        )


@dataclass(frozen=True, init=False)
class AlgorithmIR:
    inputs: tuple[TensorInput, ...]
    states: tuple[StateSpec, ...]
    transitions: tuple[StateTransition, ...]
    readouts: tuple[StateContraction | Readout, ...]
    head_mapping: HeadMapping

    def __init__(
        self,
        inputs: TypingSequence[TensorInput],
        states: TypingSequence[StateSpec],
        transition: StateTransition | TypingSequence[StateTransition],
        readouts: TypingSequence[StateContraction | Readout],
        head_mapping: HeadMapping,
    ) -> None:
        transitions = (
            (transition,)
            if isinstance(transition, StateTransition)
            else tuple(transition)
        )
        object.__setattr__(self, "inputs", tuple(inputs))
        object.__setattr__(self, "states", tuple(states))
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "readouts", tuple(readouts))
        object.__setattr__(self, "head_mapping", head_mapping)
        self._validate()

    @property
    def transition(self) -> StateTransition:
        """The single Transition; only valid for one-State programs."""
        if len(self._transitions) != 1:
            raise StatefulCompilationError(
                "IR_SCHEMA",
                "transitions",
                {"expected": "single State program", "actual": len(self._transitions)},
            )
        return self._transitions[0]

    @property
    def _transitions(self) -> tuple[StateTransition, ...]:
        return self.transitions

    def _validate(self) -> None:
        from .stateful.analyse.validation import (
            validate_algorithm_structure,
            validate_transition_structure,
        )
        from .stateful.analyse.algorithm import analyze_algorithm

        validate_algorithm_structure(self)
        validate_transition_structure(self)
        object.__setattr__(self, "_analysis", analyze_algorithm(self))

    @property
    def _structural_identity(self) -> str:
        return self._analysis.canonical_identity


@dataclass(frozen=True)
class Target:
    device_type: str = "cuda"
    product_family: str = "H20"
    capability: tuple[int, int] = (9, 0)
    workspace_bytes: int = 1 << 30


@dataclass(frozen=True)
class CompileOptions:
    optimized: bool = True
    autotune: bool = False
    backward: bool = False


@dataclass(frozen=True)
class _InputValidation:
    """Immutable declaration metadata used by every compiled invocation."""

    name: str
    roles: tuple[LogicalRole, ...]
    dtype: torch.dtype
    differentiable: bool
    sequence_axis: int | None
    packed_roles: tuple[LogicalRole, ...]


@dataclass(frozen=True)
class _StateValidation:
    """Immutable State role metadata used by runtime shape validation."""

    name: str
    head_role: HeadRole
    feature_roles: tuple[FeatureRole, ...]


@dataclass(frozen=True)
class _ValidationRecipe:
    """Compiled structural contract; it contains no runtime tensor data."""

    expected_input_names: tuple[str, ...]
    state_names: tuple[str, ...]
    output_names: tuple[str, ...]
    inputs: tuple[_InputValidation, ...]
    states: tuple[_StateValidation, ...]
    reference_required: bool
    native_diagonal: bool
    head_mapping: tuple[tuple[HeadRole, HeadRole], ...]


@dataclass(frozen=True)
class _Compilation:
    role_sizes: tuple[tuple[LogicalRole, int], ...]
    target: Target
    mode: str
    options: CompileOptions
    validation: _ValidationRecipe


@dataclass(frozen=True)
class _Runtime:
    packed: bool
    device: torch.device
    sequence_count: int
    token_count: int
    uniform_length: int | None
    state_heads: int
    feature_sizes: tuple[int, ...]
    offsets: torch.Tensor | None


class CompiledStatefulOperator:
    """A metadata-specialized Stateful Operator callable."""

    def __init__(self, operator, compilation, specialization):
        self.operator = operator
        self.compilation = compilation
        self.specialization = specialization

    def workspace_bytes(self, token_count, *, sequence_count=1, sequence_length=None):
        """Query the same resource gate used by invocation and Stage composition."""
        return self._workspace_schedule(
            token_count,
            sequence_count=sequence_count,
            sequence_length=sequence_length,
        ).required_bytes

    def _workspace_schedule(
        self, token_count, *, sequence_count=1, sequence_length=None
    ):
        policy = self.operator._plans[self.specialization].plan.workspace
        try:
            return policy.schedule(
                token_count,
                sequence_count=sequence_count,
                sequence_length=sequence_length,
            )
        except RuntimeError as error:
            raise StatefulCompilationError(
                "RUNTIME_RESOURCE", "workspace", {"reason": str(error)}
            ) from error

    def __call__(
        self,
        *,
        initial_state: StateTuple | None = None,
        sequence_offsets: torch.Tensor | None = None,
        return_final_state: bool = False,
        **inputs: torch.Tensor,
    ) -> ExecutionResult:
        return self.operator._execute_compiled(
            self,
            initial_state=initial_state,
            sequence_offsets=sequence_offsets,
            return_final_state=return_final_state,
            **inputs,
        )


class StatefulOperator:
    def __init__(self, algorithm: AlgorithmIR) -> None:
        self.algorithm = algorithm
        self._specializations: set[str] = set()
        self._plans: dict[str, Any] = {}
        self._zero_states: dict[tuple[Any, ...], torch.Tensor] = {}

    @property
    def _program(self) -> AlgorithmIR:
        """Typed, normalized program shared by planning and every executor."""
        return self.algorithm._analysis.normalized

    @property
    def _structural_identity(self) -> str:
        return hashlib.sha256(
            json.dumps(
                (SCHEMA_VERSION, self._program._structural_identity), sort_keys=True
            ).encode()
        ).hexdigest()

    def compile(
        self,
        *,
        role_sizes: Mapping[HeadRole | FeatureRole, int],
        target: Target,
        mode: str,
        optimized: bool = True,
        autotune: bool = False,
        backward: bool = False,
    ) -> CompiledStatefulOperator:
        """Compile one fixed-role implementation for dynamic runtime workloads."""
        compilation = self._validate_compilation(
            role_sizes,
            target,
            mode,
            CompileOptions(optimized, autotune, backward),
        )
        specialization = self._compilation_key(compilation)
        self._specializations.add(specialization)
        self._compile_specialization(specialization, compilation)
        return CompiledStatefulOperator(self, compilation, specialization)

    def _compile_specialization(self, specialization, compilation):
        if specialization in self._plans:
            return self._plans[specialization]
        from .stateful.compile.plan import compile_plan

        state = self._program.states[0]
        role_sizes = dict(compilation.role_sizes)
        needs_dense = _dense_summary_required(self._program)
        plan = compile_plan(
            mode=compilation.mode,
            source_identity=self._program._analysis.simplified.identity,
            feature_sizes=tuple(role_sizes[role] for role in state.feature_roles),
            target=compilation.target,
            backward_ir=(
                self._program._analysis.backward
                if compilation.options.backward
                else None
            ),
            dense_summary=needs_dense,
            scalar_factorized=self._program._analysis.scalar_factorized,
            rank_one_factored=self._program._analysis.rank_one_factored,
            optimized=compilation.options.optimized,
            autotune=compilation.options.autotune,
            structural_features=self._program._analysis.structural_features,
            operator=self,
            compilation=compilation,
        )
        self._plans[specialization] = plan
        return plan

    def __call__(
        self,
        *,
        initial_state: StateTuple | None = None,
        sequence_offsets: torch.Tensor | None = None,
        return_final_state: bool = False,
        **inputs: torch.Tensor,
    ) -> ExecutionResult:
        raise RuntimeError(
            "StatefulOperator must be explicitly compiled before invocation; "
            "call operator.compile(...) and invoke the returned handle"
        )

    def _execute_compiled(
        self,
        compiled: CompiledStatefulOperator,
        *,
        initial_state: StateTuple | None = None,
        sequence_offsets: torch.Tensor | None = None,
        return_final_state: bool = False,
        **inputs: torch.Tensor,
    ) -> ExecutionResult:
        from .stateful.runtime.validation import validate_input_names

        validation = compiled.compilation.validation
        validate_input_names(validation.expected_input_names, inputs)
        runtime, role_sizes = self._validate_runtime(
            inputs,
            sequence_offsets,
            compiled.compilation.role_sizes,
            validation,
        )
        expected_packed = compiled.compilation.mode == "packed"
        if runtime.packed != expected_packed:
            raise StatefulCompilationError(
                "RUNTIME_MODE",
                "sequence_offsets",
                {"expected": compiled.compilation.mode},
            )
        self._validate_target(runtime.device, compiled.compilation.target)
        self._validate_aliases(inputs)
        self._validate_fixed_roles(role_sizes, compiled.compilation)
        initial_states = self._validate_state(
            initial_state,
            runtime,
            role_sizes,
            validation,
            cache_default=not (runtime.token_count == 0 and return_final_state),
        )
        requires_grad_state = any(
            tensor.requires_grad for tensor in initial_states.values()
        )
        requires_grad_inputs = any(tensor.requires_grad for tensor in inputs.values())
        if not compiled.compilation.options.backward and (
            torch.is_grad_enabled() and (requires_grad_state or requires_grad_inputs)
        ):
            raise StatefulCompilationError(
                "RUNTIME_GRAD", "inputs", {"backward": "not compiled"}
            )
        executable = self._plans[compiled.specialization]
        schedule = compiled._workspace_schedule(
            runtime.token_count,
            sequence_count=runtime.sequence_count,
            sequence_length=runtime.uniform_length,
        )
        return executable.execute(
            runtime=runtime,
            role_sizes=role_sizes,
            inputs=inputs,
            initial_states=initial_states,
            return_final_state=return_final_state,
            validation=validation,
            schedule=schedule,
            requires_grad=requires_grad_state or requires_grad_inputs,
        )

    def _validate_compilation(
        self,
        role_sizes: Mapping[HeadRole | FeatureRole, int],
        target: Target,
        mode: str,
        options: CompileOptions,
    ) -> _Compilation:
        if not isinstance(target, Target):
            raise StatefulCompilationError("COMPILE_TARGET", "target", {})
        if mode not in ("dense", "packed"):
            raise StatefulCompilationError(
                "COMPILE_MODE", "mode", {"expected": ["dense", "packed"]}
            )
        declared = {
            role
            for spec in self._program.inputs
            for role in spec.roles
            if isinstance(role, (HeadRole, FeatureRole))
        }
        declared.update(
            role
            for state in self._program.states
            for role in state.roles
            if isinstance(role, (HeadRole, FeatureRole))
        )
        actual = set(role_sizes)
        invalid = [
            role for role in actual if not isinstance(role, (HeadRole, FeatureRole))
        ]
        if invalid:
            raise StatefulCompilationError(
                "COMPILE_ROLE_SIZE", "role_sizes", {"runtime_roles": invalid}
            )
        missing = declared - actual
        unknown = actual - declared
        if missing or unknown:
            raise StatefulCompilationError(
                "COMPILE_ROLE_SIZE",
                "role_sizes",
                {
                    "missing": sorted(map(str, missing)),
                    "unknown": sorted(map(str, unknown)),
                },
            )
        for role, size in role_sizes.items():
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise StatefulCompilationError(
                    "COMPILE_ROLE_SIZE", f"role_sizes.{role}", {"size": size}
                )
        state = self._program.states[0]
        head_roles = {spec.head_role for spec in self._program.states}
        if len(head_roles) != 1:
            raise StatefulCompilationError(
                "IR_SCHEMA",
                "states",
                {
                    "expected": "States share one Head role",
                    "actual": sorted(str(role) for role in head_roles),
                },
            )
        _verify_product_roles(dict(role_sizes), "role_sizes")
        # The {64,128} pair is a capability of the specialized matrix kernels.
        # Native diagonal lowering has its own tiled kernels and supports
        # arbitrary feature extents, so use its structural recipe as the
        # capability proof here rather than treating every non-reference IR as
        # a matrix-kernel program.
        native_diagonal = (
            options.optimized
            and not options.backward
            and _native_diagonal_available(self._program, role_sizes, mode=mode)
        )
        if (
            target.device_type == "cuda"
            and not self._program._analysis.reference_required
            and not native_diagonal
            and os.environ.get("STATEFUL_OPERATOR_TEST_CPU") != "1"
        ):
            for spec in self._program.states:
                feature_sizes = tuple(role_sizes[role] for role in spec.feature_roles)
                if any(size not in (64, 128) for size in feature_sizes):
                    raise StatefulCompilationError(
                        "SPECIALIZATION",
                        "role_sizes",
                        {
                            "state": spec.name,
                            "feature_sizes": feature_sizes,
                            "supported": [64, 128],
                        },
                    )
        state_heads = role_sizes[state.head_role]
        for source, mapped_target in self._program.head_mapping.mappings:
            if mapped_target != state.head_role or state_heads % role_sizes[source]:
                raise StatefulCompilationError(
                    "COMPILE_ROLE_SIZE", "head_mapping", {"source": source}
                )
        ordered = tuple(sorted(role_sizes.items(), key=lambda item: _identity(item[0])))
        validation = _ValidationRecipe(
            expected_input_names=tuple(spec.name for spec in self._program.inputs),
            state_names=tuple(spec.name for spec in self._program.states),
            output_names=tuple(readout.name for readout in self._program.readouts),
            inputs=tuple(
                _InputValidation(
                    spec.name,
                    spec.roles,
                    spec.dtype,
                    spec.differentiable,
                    spec.roles.index(Sequence) if Sequence in spec.roles else None,
                    tuple(role for role in spec.roles if role not in (Batch, Sequence)),
                )
                for spec in self._program.inputs
            ),
            states=tuple(
                _StateValidation(spec.name, spec.head_role, spec.feature_roles)
                for spec in self._program.states
            ),
            reference_required=self._program._analysis.reference_required,
            native_diagonal=native_diagonal,
            head_mapping=self._program.head_mapping.mappings,
        )
        return _Compilation(ordered, target, mode, options, validation)

    def _compilation_key(self, compilation: _Compilation) -> str:
        return _canonical_identity(
            (
                self._structural_identity,
                compilation.role_sizes,
                compilation.target,
                compilation.mode,
                compilation.options,
            )
        )

    @staticmethod
    def _validate_fixed_roles(
        actual: Mapping[LogicalRole, int], compilation: _Compilation
    ) -> None:
        for role, expected in compilation.role_sizes:
            if actual.get(role) != expected:
                raise StatefulCompilationError(
                    "RUNTIME_SHAPE",
                    "inputs",
                    {"role": role, "expected": expected, "actual": actual.get(role)},
                )

    def _validate_runtime(
        self,
        inputs: Mapping[str, torch.Tensor],
        offsets: torch.Tensor | None,
        declared_roles: tuple[tuple[LogicalRole, int], ...],
        validation: _ValidationRecipe,
    ) -> tuple[_Runtime, dict[str, int]]:
        packed = offsets is not None
        device: torch.device | None = None
        # Fixed Roles are validated against the compiled contract; a Role carried
        # only by a State (never by an Input) must still resolve.
        # Compilation already restricts these immutable pairs to Head/Feature
        # Roles. Make one mutable extent map; all input checks below still run.
        role_sizes: dict[LogicalRole, int] = dict(declared_roles)
        token_count: int | None = None
        uniform_length: int | None = None
        for spec in validation.inputs:
            tensor = inputs[spec.name]
            path = f"inputs.{spec.name}"
            if not isinstance(tensor, torch.Tensor):
                raise StatefulCompilationError(
                    "RUNTIME_BINDING", path, {"expected": "torch.Tensor"}
                )
            if tensor.dtype != spec.dtype:
                raise StatefulCompilationError(
                    "RUNTIME_DTYPE",
                    path,
                    {"expected": str(spec.dtype), "actual": str(tensor.dtype)},
                )
            if not tensor.is_contiguous():
                raise StatefulCompilationError(
                    "RUNTIME_LAYOUT", path, {"expected": "contiguous"}
                )
            if not spec.differentiable and tensor.requires_grad:
                raise StatefulCompilationError(
                    "RUNTIME_GRAD", path, {"differentiable": False}
                )
            if device is None:
                device = tensor.device
            elif tensor.device != device:
                raise StatefulCompilationError("RUNTIME_DEVICE", "inputs", {})
            if packed and spec.sequence_axis is None:
                raise StatefulCompilationError(
                    "SPECIALIZATION",
                    path,
                    {"reason": "packed non-sequence inputs are not supported"},
                )
            if packed and spec.sequence_axis is not None:
                physical_roles = spec.packed_roles
                if tensor.ndim != len(physical_roles) + 1:
                    raise StatefulCompilationError(
                        "RUNTIME_SHAPE", path, {"mode": "packed"}
                    )
                token_count = tensor.shape[0] if token_count is None else token_count
                if tensor.shape[0] != token_count:
                    raise StatefulCompilationError(
                        "RUNTIME_SHAPE", path, {"role": "sequence"}
                    )
                dimensions = zip(physical_roles, tensor.shape[1:])
            else:
                if tensor.ndim != len(spec.roles):
                    raise StatefulCompilationError(
                        "RUNTIME_SHAPE", path, {"expected_ndim": len(spec.roles)}
                    )
                dimensions = zip(spec.roles, tensor.shape)
                if spec.sequence_axis is not None:
                    length = tensor.shape[spec.sequence_axis]
                    uniform_length = (
                        length if uniform_length is None else uniform_length
                    )
                    if length != uniform_length or length <= 0:
                        raise StatefulCompilationError(
                            "RUNTIME_SHAPE", path, {"role": "sequence"}
                        )
            for role, size in dimensions:
                if isinstance(role, _SingletonRole):
                    if role not in (Batch, Sequence) and size != 1:
                        raise StatefulCompilationError(
                            "RUNTIME_SHAPE", path, {"role": str(role), "expected": 1}
                        )
                    if role not in (Batch, Sequence):
                        continue
                previous = role_sizes.setdefault(role, size)
                if previous != size:
                    raise StatefulCompilationError(
                        "RUNTIME_SHAPE",
                        path,
                        {"role": role, "sizes": [previous, size]},
                    )
        assert device is not None
        if packed:
            assert offsets is not None
            token_count = token_count or 0
            if (
                offsets.dtype != torch.int32
                or offsets.ndim != 1
                or not offsets.is_contiguous()
                or offsets.device != device
                or offsets.numel() < 1
            ):
                raise StatefulCompilationError(
                    "RUNTIME_OFFSETS",
                    "sequence_offsets",
                    {"expected": "device-local contiguous int32 vector"},
                )
            torch._assert_async(offsets[0] == 0, "sequence_offsets must start at zero")
            torch._assert_async(
                offsets[-1] == token_count,
                "sequence_offsets must end at total token count",
            )
            torch._assert_async(
                torch.all(offsets[1:] >= offsets[:-1]),
                "sequence_offsets must be nondecreasing",
            )
            sequence_count = offsets.shape[0] - 1
            role_sizes[Batch] = sequence_count
        else:
            if uniform_length is None:
                raise StatefulCompilationError(
                    "RUNTIME_SHAPE", "inputs", {"sequence": "required"}
                )
            sequence_count = role_sizes[Batch]
            token_count = sequence_count * uniform_length
        state_spec = validation.states[0]
        try:
            state_heads = role_sizes[state_spec.head_role]
            feature_sizes = tuple(role_sizes[role] for role in state_spec.feature_roles)
        except KeyError as error:
            raise StatefulCompilationError(
                "RUNTIME_SHAPE", "states[0].roles", {"missing_role": error.args[0]}
            ) from error
        for spec in validation.states:
            try:
                sizes = tuple(role_sizes[role] for role in spec.feature_roles)
            except KeyError as error:
                raise StatefulCompilationError(
                    "RUNTIME_SHAPE",
                    f"states.{spec.name}",
                    {"missing_role": error.args[0]},
                ) from error
            if (
                device.type == "cuda"
                and not validation.reference_required
                and not validation.native_diagonal
                and any(size not in (64, 128) for size in sizes)
            ):
                raise StatefulCompilationError(
                    "SPECIALIZATION",
                    "states[0].roles",
                    {
                        "state": spec.name,
                        "feature_sizes": sizes,
                        "supported": [64, 128],
                    },
                )
        _verify_product_roles(role_sizes, "inputs")
        for source, target in validation.head_mapping:
            if target != state_spec.head_role or source not in role_sizes:
                raise StatefulCompilationError(
                    "IR_TYPE", "head_mapping", {"source": source, "target": target}
                )
            if state_heads % role_sizes[source]:
                raise StatefulCompilationError(
                    "RUNTIME_SHAPE", "head_mapping", {"source": source}
                )
        return (
            _Runtime(
                packed,
                device,
                sequence_count,
                token_count,
                uniform_length,
                state_heads,
                tuple(feature_sizes),
                offsets,
            ),
            role_sizes,
        )

    @staticmethod
    def _validate_target(device: torch.device, target: Target) -> None:
        from .stateful.runtime.validation import validate_target

        validate_target(device, target)

    @staticmethod
    def _validate_aliases(inputs: Mapping[str, torch.Tensor]) -> None:
        from .stateful.runtime.validation import validate_aliases

        validate_aliases(inputs)

    def _validate_state(
        self,
        state: StateTuple | None,
        runtime: _Runtime,
        role_sizes: Mapping[LogicalRole, int],
        validation: _ValidationRecipe,
        *,
        cache_default: bool,
    ) -> dict[str, torch.Tensor]:
        """Validate initial State for every declared State and return them by name."""
        expected = validation.state_names
        if state is not None and (
            not isinstance(state, StateTuple) or state.names != expected
        ):
            raise StatefulCompilationError(
                "RUNTIME_STATE", "initial_state", {"expected": list(expected)}
            )
        states: dict[str, torch.Tensor] = {}
        for index, spec in enumerate(validation.states):
            shape = (
                runtime.sequence_count,
                role_sizes[spec.head_role],
                *(role_sizes[role] for role in spec.feature_roles),
            )
            if state is None:
                states[spec.name] = self._zero_state(
                    runtime, spec.name, shape, cache_default
                )
                continue
            tensor = state[index]
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != torch.float32
                or tensor.device != runtime.device
                or not tensor.is_contiguous()
            ):
                raise StatefulCompilationError(
                    "RUNTIME_STATE",
                    f"initial_state.{spec.name}",
                    {
                        "shape": shape,
                        "dtype": "torch.float32",
                        "device": str(runtime.device),
                    },
                )
            states[spec.name] = tensor
        return states

    def _zero_state(
        self, runtime: _Runtime, name: str, shape: tuple[int, ...], cache_default: bool
    ) -> torch.Tensor:
        if not cache_default:
            return torch.zeros(shape, device=runtime.device, dtype=torch.float32)
        key = (runtime.device, name, shape)
        cached = self._zero_states.get(key)
        if cached is None:
            cached = torch.zeros(shape, device=runtime.device, dtype=torch.float32)
            self._zero_states[key] = cached
        return cached

    def _specialization_key(
        self,
        inputs: Mapping[str, torch.Tensor],
        state: torch.Tensor,
        runtime: _Runtime,
        return_final_state: bool,
    ) -> str:
        active = sorted(
            name
            for name, tensor in inputs.items()
            if torch.is_grad_enabled() and tensor.requires_grad
        )
        if torch.is_grad_enabled() and state.requires_grad:
            active.append(f"state:{self._program.states[0].name}")
        metadata = [
            (name, tuple(tensor.shape), str(tensor.dtype), tensor.device.type)
            for name, tensor in sorted(inputs.items())
        ]
        device_index = (
            runtime.device.index
            if runtime.device.index is not None
            else torch.cuda.current_device()
            if runtime.device.type == "cuda"
            else None
        )
        offset_metadata = (
            None
            if runtime.offsets is None
            else (tuple(runtime.offsets.shape), str(runtime.offsets.dtype))
        )
        payload = (
            self._structural_identity,
            metadata,
            runtime.packed,
            offset_metadata,
            active,
            return_final_state,
            ("cuda-h20-sm90", device_index),
        )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _token_major(value: torch.Tensor, runtime: _Runtime) -> torch.Tensor:
    """Materialize a per-token factor in the token-major layout the kernels read.

    Runtime inputs are already normalized to token-major (a batch-major flattened
    view for dense sequences), so this only makes the buffer contiguous, at one
    declared point, rather than inside the kernel.
    """
    return value.contiguous()


def _segmented_affine_scan(
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor,
    segment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offset = 1
    token_count = left.shape[0]
    while offset < token_count:
        old_left, old_right, old_bias = left, right, bias
        valid = segment[offset:] == segment[:-offset]
        combined_left = old_left[offset:] @ old_left[:-offset]
        combined_right = old_right[:-offset] @ old_right[offset:]
        combined_bias = (
            old_left[offset:] @ old_bias[:-offset] @ old_right[offset:]
            + old_bias[offset:]
        )
        mask = valid.view(-1, 1, 1, 1)
        left = torch.cat(
            (old_left[:offset], torch.where(mask, combined_left, old_left[offset:]))
        )
        right = torch.cat(
            (old_right[:offset], torch.where(mask, combined_right, old_right[offset:]))
        )
        bias = torch.cat(
            (old_bias[:offset], torch.where(mask, combined_bias, old_bias[offset:]))
        )
        offset *= 2
    return left, right, bias


def _segmented_elementwise_scan(
    scale: torch.Tensor,
    bias: torch.Tensor,
    segment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    offset = 1
    token_count = scale.shape[0]
    while offset < token_count:
        old_scale, old_bias = scale, bias
        valid = (segment[offset:] == segment[:-offset]).view(-1, 1, 1, 1)
        combined_scale = old_scale[offset:] * old_scale[:-offset]
        combined_bias = old_scale[offset:] * old_bias[:-offset] + old_bias[offset:]
        scale = torch.cat(
            (old_scale[:offset], torch.where(valid, combined_scale, old_scale[offset:]))
        )
        bias = torch.cat(
            (old_bias[:offset], torch.where(valid, combined_bias, old_bias[offset:]))
        )
        offset *= 2
    return scale, bias


def _native_diagonal_available(
    algorithm: AlgorithmIR,
    role_sizes: Mapping[LogicalRole, int],
    *,
    mode: str,
) -> bool:
    """Return whether the selected dense lowering has native diagonal support."""
    if mode != "dense" or not hasattr(algorithm, "transitions"):
        return False
    from .stateful.lowering.chunked import compile_chunked_recipe

    recipe = compile_chunked_recipe(algorithm, dict(role_sizes))
    return recipe is not None and recipe.native_diagonal is not None


def _dense_summary_required(algorithm: AlgorithmIR) -> bool:
    """Whether the per-token summary needs dense (left, right) factors.

    A rank-one propagation composes a non-diagonal update, so its summary stays
    dense until the chunk-summary scan lands. Axis Scales are diagonal: one
    feature axis lowers to one elementwise factor and two axes lower to one
    factor each, which the prefix kernel composes per element.
    """
    for transition in algorithm.transitions:
        for node in transition.ordered_steps:
            if isinstance(node, (RankOnePropagation, StateInjection)):
                return True
    return False


def _verify_product_roles(role_sizes: Mapping[LogicalRole, int], path: str) -> None:
    """Every Product Feature role axis must equal the product of its factors."""
    for role, size in role_sizes.items():
        if not isinstance(role, ProductFeatureRole):
            continue
        left = role_sizes.get(role.left)
        right = role_sizes.get(role.right)
        if left is None or right is None:
            raise StatefulCompilationError(
                "IR_TYPE",
                path,
                {
                    "product_role": role.name,
                    "missing_factors": [
                        str(item)
                        for item, value in ((role.left, left), (role.right, right))
                        if value is None
                    ],
                },
            )
        if left * right != size:
            raise StatefulCompilationError(
                "RUNTIME_SHAPE",
                f"{path}.{role.name}",
                {"expected": left * right, "actual": size},
            )


def _evaluate(
    expression: InputExpression,
    values: Mapping[str, torch.Tensor],
    *,
    states: Mapping[str, torch.Tensor] | None = None,
    state_specs: Mapping[str, Any] | None = None,
    feature_axes: Mapping[Any, int] | None = None,
    segment: torch.Tensor | None = None,
    squeeze_features: bool = False,
    prior_states: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Evaluate an Input Expression; State Terms need the State binding.

    ``squeeze_features`` selects the readout value space, where an Input carries
    only the Feature axes it actually declares, so an input term and a State Term
    result can be combined arithmetically.
    """
    if isinstance(expression, Input):
        bound = values[expression.name]
        if squeeze_features:
            value = _squeeze_singleton_features(bound)
            # Readouts retain one output Feature axis. Feature-free per-token
            # scalars must broadcast over that axis, not over heads or tokens.
            return value.unsqueeze(-1) if value.ndim == 2 else value
        return bound
    if isinstance(expression, Constant):
        device = _expression_device(values, states)
        return torch.full((), expression.value, device=device, dtype=expression.dtype)
    if isinstance(expression, Add):
        return _evaluate(
            expression.left,
            values,
            states=states,
            state_specs=state_specs,
            feature_axes=feature_axes,
            segment=segment,
            squeeze_features=squeeze_features,
            prior_states=prior_states,
        ) + _evaluate(
            expression.right,
            values,
            states=states,
            state_specs=state_specs,
            feature_axes=feature_axes,
            segment=segment,
            squeeze_features=squeeze_features,
            prior_states=prior_states,
        )
    if isinstance(expression, Multiply):
        return _evaluate(
            expression.left,
            values,
            states=states,
            state_specs=state_specs,
            feature_axes=feature_axes,
            segment=segment,
            squeeze_features=squeeze_features,
            prior_states=prior_states,
        ) * _evaluate(
            expression.right,
            values,
            states=states,
            state_specs=state_specs,
            feature_axes=feature_axes,
            segment=segment,
            squeeze_features=squeeze_features,
            prior_states=prior_states,
        )
    if isinstance(expression, Divide):
        return _evaluate(
            expression.left,
            values,
            states=states,
            state_specs=state_specs,
            feature_axes=feature_axes,
            segment=segment,
            squeeze_features=squeeze_features,
            prior_states=prior_states,
        ) / _evaluate(
            expression.right,
            values,
            states=states,
            state_specs=state_specs,
            feature_axes=feature_axes,
            segment=segment,
            squeeze_features=squeeze_features,
            prior_states=prior_states,
        )
    if isinstance(expression, Negate):
        return -_evaluate(
            expression.operand,
            values,
            states=states,
            state_specs=state_specs,
            feature_axes=feature_axes,
            segment=segment,
            squeeze_features=squeeze_features,
            prior_states=prior_states,
        )
    if isinstance(expression, Cast):
        return _evaluate(
            expression.operand,
            values,
            states=states,
            state_specs=state_specs,
            feature_axes=feature_axes,
            segment=segment,
            squeeze_features=squeeze_features,
            prior_states=prior_states,
        ).to(expression.dtype)
    if isinstance(expression, (Abs, Exp, Log, Sigmoid, Sqrt, Reciprocal, Clamp)):
        operand = _evaluate(
            expression.operand,
            values,
            states=states,
            state_specs=state_specs,
            feature_axes=feature_axes,
            segment=segment,
            squeeze_features=squeeze_features,
            prior_states=prior_states,
        ).float()
        if isinstance(expression, Abs):
            return operand.abs()
        if isinstance(expression, Exp):
            return operand.exp()
        if isinstance(expression, Log):
            return operand.log()
        if isinstance(expression, Sigmoid):
            return operand.sigmoid()
        if isinstance(expression, Sqrt):
            return operand.sqrt()
        if isinstance(expression, Reciprocal):
            return operand.reciprocal()
        return operand.clamp(expression.minimum, expression.maximum)
    if isinstance(expression, Shift):
        if segment is None:
            raise StatefulCompilationError(
                "IR_TYPE", "expression.Shift", {"reason": "missing sequence segment"}
            )
        operand = _evaluate(
            expression.operand,
            values,
            states=states,
            state_specs=state_specs,
            feature_axes=feature_axes,
            segment=segment,
            squeeze_features=squeeze_features,
            prior_states=prior_states,
        )
        return _shift_within_segment(operand, expression.lag, segment)
    if isinstance(expression, StateTerm):
        if states is None or state_specs is None:
            raise StatefulCompilationError(
                "IR_TYPE", "expression.StateTerm", {"reason": "missing State binding"}
            )
        spec = state_specs[expression.state]
        operand = _evaluate(
            expression.operand,
            values,
            states=states,
            state_specs=state_specs,
            feature_axes=feature_axes,
            segment=segment,
            squeeze_features=squeeze_features,
            prior_states=prior_states,
        )
        binding = states
        if expression.prior:
            if prior_states is None:
                raise StatefulCompilationError(
                    "IR_TYPE",
                    "expression.StateTerm",
                    {"prior": "no prior State binding"},
                )
            binding = prior_states
        return _contract_state_term(
            operand, spec, expression.role, binding[expression.state], feature_axes
        )
    if isinstance(expression, Outer):
        left = _evaluate(
            expression.left,
            values,
            states=states,
            state_specs=state_specs,
            feature_axes=feature_axes,
            segment=segment,
            squeeze_features=True,
            prior_states=prior_states,
        )
        right = _evaluate(
            expression.right,
            values,
            states=states,
            state_specs=state_specs,
            feature_axes=feature_axes,
            segment=segment,
            squeeze_features=True,
            prior_states=prior_states,
        )
        expanded = left.float().unsqueeze(-1) * right.float().unsqueeze(-2)
        return expanded.reshape(*left.shape[:-1], left.shape[-1] * right.shape[-1])
    if isinstance(expression, Contract):
        left = _evaluate(
            expression.left,
            values,
            states=states,
            state_specs=state_specs,
            feature_axes=feature_axes,
            segment=segment,
            squeeze_features=True,
            prior_states=prior_states,
        )
        right = _evaluate(
            expression.right,
            values,
            states=states,
            state_specs=state_specs,
            feature_axes=feature_axes,
            segment=segment,
            squeeze_features=True,
            prior_states=prior_states,
        )
        # keep one singleton axis so the result broadcasts against a
        # Feature-carrying term, exactly like a rank-1 State Term
        contracted = (left.float() * right.float()).sum(dim=-1, keepdim=True)
        if not squeeze_features and feature_axes is None:
            # The single-matrix evaluator keeps both State axes, including
            # feature-free reductions. Readout evaluation uses vector space.
            return contracted.unsqueeze(-1)
        return contracted
    if isinstance(expression, Softmax):
        operand = _evaluate(
            expression.operand,
            values,
            states=states,
            state_specs=state_specs,
            feature_axes=feature_axes,
            segment=segment,
            squeeze_features=True,
            prior_states=prior_states,
        )
        return torch.softmax(operand.float(), dim=-1)
    raise StatefulCompilationError(
        "IR_TYPE", "expression", {"node": type(expression).__name__}
    )


def _expression_device(
    values: Mapping[str, torch.Tensor], states: Mapping[str, torch.Tensor] | None
) -> torch.device:
    for tensor in values.values():
        return tensor.device
    if states:
        for tensor in states.values():
            return tensor.device
    raise StatefulCompilationError(
        "IR_TYPE", "expression.Constant", {"reason": "no bound Tensor"}
    )


def _squeeze_singleton_features(value: torch.Tensor) -> torch.Tensor:
    """Drop Feature axes that carry a single element (readout value space)."""
    if value.ndim <= 2:
        return value
    kept = [axis for axis in range(2, value.ndim) if value.shape[axis] != 1]
    order = [0, 1, *kept, *(axis for axis in range(2, value.ndim) if axis not in kept)]
    return value.permute(*order).reshape(
        value.shape[0], value.shape[1], *[value.shape[axis] for axis in kept]
    )


def _shift_within_segment(
    operand: torch.Tensor, lag: int, segment: torch.Tensor
) -> torch.Tensor:
    """Shift a token-major tensor backwards, zero-filling sequence starts."""
    token_count = operand.shape[0]
    shifted = torch.zeros_like(operand)
    if token_count <= lag:
        return shifted
    source = slice(0, token_count - lag)
    destination = slice(lag, token_count)
    same = segment[destination] == segment[source]
    mask = same.view(-1, *([1] * (operand.ndim - 1)))
    shifted[destination] = torch.where(
        mask, operand[source], torch.zeros_like(operand[source])
    )
    return shifted


def _contract_state_term(
    operand: torch.Tensor,
    state_spec: Any,
    role: Any,
    state_value: torch.Tensor,
    feature_axes: Mapping[Any, int] | None,
) -> torch.Tensor:
    """Contract one State Feature role (or read a rank-1 State) from an operand."""
    features = state_spec.feature_roles
    if role is None:
        if any(size != 1 for size in operand.shape[2:]):
            raise StatefulCompilationError(
                "IR_TYPE",
                "expression.StateTerm",
                {"identity_readout": "requires a State-free operand"},
            )
        scalars = operand.reshape(
            *operand.shape[:2], *([1] * (state_value.ndim - 2))
        ).float()
        return state_value.float() * scalars
    axes = feature_axes or {}
    if operand.ndim < 4:
        axis = operand.ndim - 1
    else:
        axis = axes.get(role)
    if axis is None:
        raise StatefulCompilationError(
            "IR_TYPE", "expression.StateTerm", {"role": str(role)}
        )
    moved = operand.movedim(axis, 2)
    if any(size != 1 for size in moved.shape[3:]):
        raise StatefulCompilationError(
            "IR_TYPE",
            "expression.StateTerm",
            {"operand_features": [str(feature) for feature in axes]},
        )
    reduced = moved.reshape(moved.shape[0], moved.shape[1], moved.shape[2]).float()
    state_float = state_value.float()
    if state_spec.rank == 1:
        return torch.einsum("thf,thf->th", reduced, state_float).unsqueeze(-1)
    if role == features[0]:
        return torch.einsum("thf,thfv->thv", reduced, state_float)
    return torch.einsum("thv,thfv->thf", reduced, state_float)


def _feature_vector(
    value: torch.Tensor, role: FeatureRole, state: StateSpec
) -> torch.Tensor:
    if role == state.feature_roles[0]:
        return value[..., :, 0]
    return value[..., 0, :]


def _head_role(
    roles: TypingSequence[LogicalRole], features: TypingSequence[FeatureRole]
) -> HeadRole | None:
    return next(
        (role for role in roles if isinstance(role, HeadRole)),
        None,
    )


def _reject_duplicate_names(items: Iterable[Any], path: str) -> None:
    seen: set[str] = set()
    for item in items:
        if item.name in seen:
            raise StatefulCompilationError("IR_SCHEMA", path, {"duplicate": item.name})
        seen.add(item.name)


def _compile_factory_profile(
    operator: StatefulOperator,
    *,
    query_heads: int,
    state_heads: int,
    key_size: int,
    value_size: int,
    key_heads: int | None = None,
    value_heads: int | None = None,
    mode: str = "dense",
    autotune: bool = False,
    backward: bool = False,
) -> CompiledStatefulOperator:
    roles = {
        role.name: role
        for spec in (*operator.algorithm.inputs, *operator.algorithm.states)
        for role in spec.roles
        if isinstance(role, (HeadRole, FeatureRole))
    }
    return operator.compile(
        role_sizes={
            roles["query_heads"]: query_heads,
            roles["key_heads"]: key_heads if key_heads is not None else state_heads,
            roles["value_heads"]: value_heads
            if value_heads is not None
            else state_heads,
            roles["state_heads"]: state_heads,
            roles["key_dim"]: key_size,
            roles["value_dim"]: value_size,
        },
        target=Target(),
        mode=mode,
        autotune=autotune,
        backward=backward,
    )


def _expressions(value: Any) -> Iterable[InputExpression]:
    if isinstance(value, InputExpression):
        yield value
    if is_dataclass(value):
        for field in fields(value):
            yield from _expressions(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _expressions(item)


def _canonical(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if is_dataclass(value):
        return {
            "node": type(value).__name__,
            **{
                field.name: _canonical(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _canonical_identity(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_canonical(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _identity(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"))
