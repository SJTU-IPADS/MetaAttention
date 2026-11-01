"""Prove a two-contraction softmax region with typed local pointwise cuts."""

from dataclasses import dataclass

import torch

from ..expression.dense_expression import compile_dense_expression


@dataclass(frozen=True)
class SoftmaxRegion:
    scores: object
    weights: object
    # A vector State may be read before and after its update in one region.
    # Bind each temporal view separately, but advance each State only once.
    vectors: tuple[object, ...]


def compile_softmax_region(expression, algorithm, specs):
    from ....stateful_operator import Constant, Softmax, StateTerm

    states = {state.name: state for state in algorithm.states}
    if not isinstance(expression, StateTerm) or states[expression.state].rank != 2:
        return None

    def leaves(node, kind):
        if isinstance(node, kind):
            return (node,)
        if hasattr(node, "left"):
            return leaves(node.left, kind) + leaves(node.right, kind)
        if hasattr(node, "operand"):
            return leaves(node.operand, kind)
        return ()

    softmaxes = tuple(dict.fromkeys(leaves(expression.operand, Softmax)))
    if len(softmaxes) != 1:
        return None
    softmax = softmaxes[0]
    terms = tuple(dict.fromkeys(leaves(softmax.operand, StateTerm)))
    contractions = tuple(term for term in terms if states[term.state].rank == 2)
    if len(contractions) != 1:
        return None
    first = contractions[0]
    fs, ss = states[first.state], states[expression.state]
    if (
        first.state == expression.state
        or first.role != fs.feature_roles[0]
        or expression.role != ss.feature_roles[0]
        or fs.feature_roles[1] != ss.feature_roles[0]
        or softmax.role != fs.feature_roles[1]
    ):
        return None
    feature = softmax.role
    vectors = tuple(
        dict.fromkeys(
            term
            for term in leaves(expression.operand, StateTerm)
            if states[term.state].rank == 1
        )
    )
    for term in vectors:
        if (
            states[term.state].feature_roles != (feature,)
            or term.role is not None
            or not isinstance(term.operand, Constant)
            or term.operand.value != 1
        ):
            return None
    bindings = {term: (f"vector{i}", torch.float32) for i, term in enumerate(vectors)}
    scores = compile_dense_expression(
        softmax.operand,
        specs,
        feature,
        local_values={**bindings, first: ("scores", torch.float32)},
    )
    weights = compile_dense_expression(
        expression.operand,
        specs,
        feature,
        local_values={**bindings, softmax: ("probability", torch.float32)},
    )
    query = compile_dense_expression(first.operand, specs, first.role)
    if scores is None or weights is None or query is None:
        return None
    return first, query, SoftmaxRegion(scores, weights, vectors)
