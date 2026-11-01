"""Authored pointwise expressions represented for TileLang lowering."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
from typing import Any
import torch
from ...runtime.bindings import DenseRawBindingRecipe, compile_dense_binding


@dataclass(frozen=True)
class DenseLoader:
    """Immutable expression IR consumed by native TileLang macros.

    ``nodes`` contains operation records in topological order.  Keeping the IR
    rather than dynamically generating source preserves authored node rounding
    and lets each lowering select its own TileLang load primitive.
    """

    nodes: tuple[tuple[Any, ...], ...]
    root: int
    feature: Any
    cache_key: str


@dataclass(frozen=True)
class DenseExpression:
    loader: DenseLoader
    sources: tuple[Any, ...]
    feature: Any
    bindings: tuple[DenseRawBindingRecipe, ...]

    def bind(self, inputs, heads):
        pointers, strides, groups = [], [], []
        for binding in self.bindings:
            raw, source_strides, group = binding.bind(inputs[binding.name], heads)
            pointers.append(raw)
            strides.extend(source_strides)
            groups.append(group)
        return self.loader, tuple(pointers), tuple(strides), tuple(groups)


def compile_dense_expression(
    expression, specs, feature, *, factors=(), local_values=None
):
    from ....stateful_operator import (
        Add,
        Abs,
        Cast,
        Clamp,
        Constant,
        Divide,
        Exp,
        Input,
        Log,
        Multiply,
        Negate,
        Reciprocal,
        Sigmoid,
        Sqrt,
    )

    sources, records, nodes = [], [], {}

    def visit(node):
        if node in nodes:
            value = nodes[node]
            if value is None:
                raise ValueError("cyclic expression")
            return value
        nodes[node] = None
        if local_values is not None and node in local_values:
            name, dtype = local_values[node]
            records.append(("local", name))
            result = (len(records) - 1, dtype, False)
        elif isinstance(node, Input):
            spec = specs[node.name]
            from ....stateful_operator import FeatureRole

            if any(
                isinstance(role, FeatureRole) and role != feature for role in spec.roles
            ):
                raise ValueError("expression carries a different Feature")
            if spec not in sources:
                sources.append(spec)
            records.append(("input", node.name))
            index = len(records) - 1
            result = (index, spec.dtype, False)
        elif isinstance(node, Constant):
            records.append(("constant", node.value, node.dtype))
            index = len(records) - 1
            result = (index, node.dtype, True)
        elif isinstance(node, (Add, Multiply, Divide)):
            left, ld, ls = visit(node.left)
            right, rd, rs = visit(node.right)
            dtype = (
                rd
                if ls and not rs
                else ld
                if rs and not ls
                else torch.promote_types(ld, rd)
            )
            symbol = (
                "+"
                if isinstance(node, Add)
                else "*"
                if isinstance(node, Multiply)
                else "/"
            )
            records.append(("binary", left, right, dtype, symbol))
            index = len(records) - 1
            result = (index, dtype, ls and rs)
        elif isinstance(node, Cast):
            operand, _, scalar = visit(node.operand)
            records.append(("cast", operand, node.dtype))
            result = (len(records) - 1, node.dtype, scalar)
        elif isinstance(
            node, (Abs, Negate, Exp, Log, Sigmoid, Sqrt, Reciprocal, Clamp)
        ):
            operand, dtype, scalar = visit(node.operand)
            if isinstance(node, Negate):
                records.append(("unary", operand, "neg"))
                index = len(records) - 1
                result = (index, dtype, scalar)
            else:
                kind = {
                    Abs: "abs",
                    Exp: "exp",
                    Log: "log",
                    Sigmoid: "sigmoid",
                    Sqrt: "sqrt",
                    Reciprocal: "reciprocal",
                    Clamp: "clamp",
                }[type(node)]
                records.append(
                    (
                        "unary",
                        operand,
                        kind,
                        getattr(node, "minimum", None),
                        getattr(node, "maximum", None),
                    )
                )
                index = len(records) - 1
                result = (index, torch.float32, scalar)
        else:
            raise ValueError("expression requires a materialization cut")
        nodes[node] = result
        return result

    try:
        root, _, _ = visit(expression)
        root_index = root
        for factor in factors:
            other, _, _ = visit(factor)
            records.append(("factor", root_index, other))
            root_index = len(records) - 1
    except ValueError:
        return None
    digest = hashlib.sha256(repr((records, root_index, feature)).encode()).hexdigest()
    loader = DenseLoader(tuple(records), root_index, feature, digest)
    return DenseExpression(
        loader,
        tuple(sources),
        feature,
        tuple(compile_dense_binding(spec, feature) for spec in sources),
    )
