"""Compiled elementwise expression DAGs for Stateful Operator lowerings.

The module deliberately has a small seam: :func:`compile_expression` accepts an IR
expression (or a named collection of expressions) and returns a callable accepting
leaf bindings.  Expressions are walked once, structurally common nodes are shared,
and each invocation evaluates the resulting topological order.  StateTerm nodes
are materialisation leaves: their transition/readout semantics belong to the
caller and are never re-evaluated here.

The reference implementation uses native Torch elementwise operations.  This is
important for CPU, autograd, arbitrary strides, and for preserving Torch's dtype
rounding at every authored node.  CUDA lowerers can replace the private execution
adapter with a TileLang launch without changing this interface; no intermediate
expression tensor is required by the evaluator itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ....stateful_operator import (
    Add,
    Abs,
    Cast,
    Clamp,
    Constant,
    Contract,
    Divide,
    Exp,
    Input,
    InputExpression,
    Log,
    Multiply,
    Negate,
    Outer,
    Reciprocal,
    Shift,
    Sigmoid,
    Softmax,
    Sqrt,
    StateTerm,
)


_UNARY = (Abs, Cast, Negate, Exp, Log, Sigmoid, Sqrt, Reciprocal, Clamp, Softmax)
_BINARY = (Add, Multiply, Divide)


@dataclass(frozen=True)
class _Node:
    expression: Any
    children: tuple[int, ...]
    kind: str


class CompiledExpression:
    """Reusable expression evaluator with structural CSE.

    ``CompiledExpression`` is intentionally callable and exposes only read-only
    diagnostics.  ``bound`` maps Input names to either tensors or symbolic
    binding descriptors (objects exposing ``materialize``).  ``state_terms``
    maps StateTerm objects to already materialized tensors; state names are also
    accepted as a convenience when a state occurs once.  The latter is not a
    State evaluator -- it is explicitly a leaf binding.
    """

    __slots__ = (
        "_roots",
        "_nodes",
        "_output_dtype",
        "_kernel_identity",
        "_name_roots",
        "_leaf_names",
        "_state_leaves",
        "_materialization_cuts",
    )

    def __init__(
        self,
        expression: InputExpression
        | Mapping[str, InputExpression]
        | Sequence[InputExpression],
        *,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        self._output_dtype = output_dtype
        self._nodes: list[_Node] = []
        # Equality/hash of IR dataclasses is semantic, rather than object based.
        # This makes separately constructed but identical subexpressions CSE too.
        indices: dict[Any, int] = {}

        if isinstance(expression, Mapping):
            self._name_roots = tuple(expression.keys())
            roots_expr = tuple(expression.values())
        elif isinstance(expression, Sequence) and not isinstance(
            expression, (str, bytes)
        ):
            self._name_roots = None
            roots_expr = tuple(expression)
        else:
            self._name_roots = None
            roots_expr = (expression,)

        def visit(node: Any) -> int:
            try:
                existing = indices.get(node)
            except TypeError as error:
                raise TypeError(f"expression node is not hashable: {node!r}") from error
            if existing is not None:
                return existing
            if isinstance(node, Input):
                children: tuple[int, ...] = ()
                kind = "input"
            elif isinstance(node, Constant):
                children = ()
                kind = "constant"
            elif isinstance(node, StateTerm):
                # Deliberately do not visit operand: the entire term is a leaf.
                children = ()
                kind = "state"
            elif isinstance(node, (Outer, Contract)):
                children = (visit(node.left), visit(node.right))
                kind = type(node).__name__.lower()
            elif isinstance(node, Shift):
                children = (visit(node.operand),)
                kind = "shift"
            elif isinstance(node, _BINARY):
                children = (visit(node.left), visit(node.right))
                kind = type(node).__name__.lower()
            elif isinstance(node, _UNARY):
                children = (visit(node.operand),)
                kind = type(node).__name__.lower()
            else:
                raise TypeError(
                    f"unsupported expression node {type(node).__name__}; "
                    "expression DAG supports pointwise nodes, Outer, Contract, "
                    "Shift, Constant, Input, and materialized StateTerm leaves"
                )
            index = len(self._nodes)
            self._nodes.append(_Node(node, children, kind))
            indices[node] = index
            return index

        self._roots = tuple(visit(root) for root in roots_expr)
        self._kernel_identity = (
            tuple((node.kind, node.expression, node.children) for node in self._nodes),
            self._roots,
        )
        self._leaf_names = tuple(
            node.expression.name for node in self._nodes if node.kind == "input"
        )
        self._state_leaves = tuple(
            node.expression for node in self._nodes if node.kind == "state"
        )
        # A DAG node with multiple consumers is retained until all consumers have
        # run.  Native Torch may still reuse storage; this count is the resource
        # contract used by lowerers when deciding a materialization cut.
        users = [0] * len(self._nodes)
        for node in self._nodes:
            for child in node.children:
                users[child] += 1
        self._materialization_cuts = tuple(
            index for index, count in enumerate(users) if count > 1
        )

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def leaf_names(self) -> tuple[str, ...]:
        return self._leaf_names

    @property
    def state_leaves(self) -> tuple[StateTerm, ...]:
        return self._state_leaves

    @property
    def materialization_cuts(self) -> tuple[int, ...]:
        return self._materialization_cuts

    @property
    def fused(self) -> bool:
        """Whether this DAG has no reduction/materialization node.

        All nodes accepted here are pointwise and can be emitted as one TileLang
        elementwise chain by a CUDA adapter.  State terms are leaves, not cuts.
        """
        return True

    def __call__(
        self,
        bound: Mapping[str, Any] | Any,
        state_terms: Mapping[Any, Any] | None = None,
        output_dtype: torch.dtype | None = None,
        segment: torch.Tensor | None = None,
    ) -> Any:
        bindings = _as_mapping(bound)
        states = _as_mapping(state_terms) if state_terms is not None else {}
        if any(
            isinstance(getattr(source, "tensor", source), torch.Tensor)
            and getattr(source, "tensor", source).is_cuda
            for source in (*bindings.values(), *states.values())
        ):
            from .expression_kernels import launch_expression

            fused = launch_expression(
                self, bindings, states, output_dtype or self._output_dtype
            )
            if fused is not None:
                if self._name_roots is not None:
                    values = fused if isinstance(fused, tuple) else (fused,)
                    return dict(zip(self._name_roots, values, strict=True))
                return fused
        values: list[torch.Tensor] = [None] * len(self._nodes)  # type: ignore[list-item]
        for index, node in enumerate(self._nodes):
            expr = node.expression
            if node.kind == "input":
                try:
                    source = bindings[expr.name]
                except KeyError as error:
                    raise KeyError(f"missing Input binding {expr.name!r}") from error
                values[index] = _binding_tensor(source)
            elif node.kind == "constant":
                device = _first_device(bindings, states)
                values[index] = torch.full(
                    (), expr.value, dtype=expr.dtype, device=device
                )
            elif node.kind == "state":
                source = _state_binding(states, expr)
                if source is None:
                    raise KeyError(
                        f"missing materialized StateTerm binding for {expr.state!r}"
                    )
                # State readout materialization is an FP32 seam.  Do not let a
                # BF16 transport binding introduce a rounding point downstream.
                values[index] = _binding_tensor(source).float()
            elif node.kind == "outer":
                left, right = values[node.children[0]], values[node.children[1]]
                values[index] = (left.unsqueeze(-1) * right.unsqueeze(-2)).flatten(-2)
            elif node.kind == "contract":
                left, right = values[node.children[0]], values[node.children[1]]
                values[index] = (left.float() * right.float()).sum(-1, keepdim=True)
            elif node.kind == "shift":
                value = values[node.children[0]]
                lag = expr.lag
                shifted = torch.zeros_like(value)
                if segment is None:
                    if value.ndim and value.shape[0] > lag:
                        shifted[lag:] = value[:-lag]
                elif value.ndim and value.shape[0] > lag:
                    same = segment[lag:] == segment[:-lag]
                    shifted[lag:] = torch.where(
                        same.reshape(-1, *([1] * (value.ndim - 1))),
                        value[:-lag],
                        torch.zeros_like(value[:-lag]),
                    )
                values[index] = shifted
            elif node.kind == "softmax":
                values[index] = values[node.children[0]].float().softmax(-1)
            elif node.kind in ("add", "multiply", "divide"):
                left, right = values[node.children[0]], values[node.children[1]]
                values[index] = {
                    "add": torch.add,
                    "multiply": torch.mul,
                    "divide": torch.div,
                }[node.kind](left, right)
            else:
                values[index] = _apply_unary(expr, values[node.children[0]])
        result = tuple(values[root] for root in self._roots)
        if output_dtype is None:
            output_dtype = self._output_dtype
        if output_dtype is not None:
            result = tuple(value.to(output_dtype) for value in result)
        if self._name_roots is not None:
            return dict(zip(self._name_roots, result, strict=True))
        if len(result) == 1:
            return result[0]
        return result


def compile_expression(
    expression: InputExpression
    | Mapping[str, InputExpression]
    | Sequence[InputExpression],
    *,
    output_dtype: torch.dtype | None = None,
) -> CompiledExpression:
    """Compile one or more pointwise expressions into a shared typed DAG."""
    return CompiledExpression(expression, output_dtype=output_dtype)


def evaluate_expression(
    expression: InputExpression
    | Mapping[str, InputExpression]
    | Sequence[InputExpression],
    bound: Mapping[str, Any] | Any,
    *,
    state_terms: Mapping[Any, Any] | None = None,
    output_dtype: torch.dtype | None = None,
    segment: torch.Tensor | None = None,
) -> Any:
    """Evaluate an expression through the same public compiler seam."""
    return compile_expression(expression, output_dtype=output_dtype)(
        bound, state_terms, segment=segment
    )


def make_expression_evaluator(
    expression: InputExpression
    | Mapping[str, InputExpression]
    | Sequence[InputExpression],
    *,
    output_dtype: torch.dtype | None = None,
) -> CompiledExpression:
    """Build the callable seam used by chunked input/readout lowerings."""
    return compile_expression(expression, output_dtype=output_dtype)


def evaluate_readout(
    expression: InputExpression
    | Mapping[str, InputExpression]
    | Sequence[InputExpression],
    bound: Mapping[str, Any] | Any,
    state_terms: Mapping[Any, Any] | None = None,
    *,
    output_dtype: torch.dtype | None = None,
    segment: torch.Tensor | None = None,
) -> Any:
    return compile_expression(expression, output_dtype=output_dtype)(
        bound, state_terms, segment=segment
    )


def _as_mapping(value: Any) -> Mapping[Any, Any]:
    if isinstance(value, Mapping):
        return value
    as_dict = getattr(value, "as_dict", None)
    if as_dict is not None:
        return as_dict()
    raise TypeError("expression bindings must be a mapping or BindingSet")


def _binding_tensor(source: Any) -> torch.Tensor:
    if isinstance(source, torch.Tensor):
        return source
    materialize = getattr(source, "materialize", None)
    if materialize is not None:
        # Preserve the authored leaf dtype. Pointwise Torch operations then
        # retain per-node rounding (notably BF16 + BF16).
        dtype = getattr(source, "dtype", torch.float32)
        return materialize(dtype=dtype)
    raise TypeError("leaf binding must be a Tensor or binding descriptor")


def _state_binding(states: Mapping[Any, Any], expression: StateTerm) -> Any:
    if expression in states:
        return states[expression]
    # Supporting a name key is useful for callers with one occurrence.  If a
    # State has multiple differently contracted terms, object keys are required.
    return states.get(expression.state)


def _first_device(
    bindings: Mapping[Any, Any], states: Mapping[Any, Any]
) -> torch.device:
    for source in (*bindings.values(), *states.values()):
        if isinstance(source, torch.Tensor):
            return source.device
        tensor = getattr(source, "tensor", None)
        if isinstance(tensor, torch.Tensor):
            return tensor.device
    raise ValueError("Constant evaluation requires at least one tensor binding")


def _apply_unary(expression: Any, value: torch.Tensor) -> torch.Tensor:
    # Transcendentals and reciprocal are authored as FP32 operations in the IR.
    if isinstance(expression, Cast):
        return value.to(expression.dtype)
    if isinstance(expression, Abs):
        return value.float().abs()
    if isinstance(expression, Negate):
        return -value
    operand = value.float()
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
    if isinstance(expression, Clamp):
        return operand.clamp(expression.minimum, expression.maximum)
    if isinstance(expression, Softmax):
        return operand.softmax(-1)
    raise TypeError(f"unsupported unary expression {type(expression).__name__}")


__all__ = [
    "CompiledExpression",
    "compile_expression",
    "evaluate_expression",
    "make_expression_evaluator",
    "evaluate_readout",
]
