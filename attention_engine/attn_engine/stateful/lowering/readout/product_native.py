"""Typed identity polynomial normalization over explicit public States."""

from dataclasses import dataclass

import torch

from ..expression.dense_expression import compile_dense_expression


@dataclass(frozen=True)
class ProductPair:
    numerator: str
    denominator: str
    key: tuple
    query: tuple
    value: object
    rows: int
    right: int
    prior: bool


@dataclass(frozen=True)
class NativeProductNormalized:
    readout: object
    expression: object
    pairs: tuple[ProductPair, ...]
    terms: tuple
    columns: int


def native_product_resources(recipe, sizes, heads):
    """Return ``(bytes/token, bytes/chunk/sequence)`` including scan scratch.

    Charge B*T tokens and B*(ceil(T/64)+1) chunks. Three full State copies
    conservatively cover retained summaries, long-level carries, recursive
    groups and requested finals; inputs are raw views and readouts are tiles.
    """
    output = heads * recipe.columns * recipe.readout.output_dtype.itemsize
    elements = sum(pair.rows * (recipe.columns + 1) for pair in recipe.pairs)
    return output, 12 * heads * elements


def native_product_estimate(recipe, sizes, heads):
    """Conservative TileLang cost metadata for the shared chunked planner.

    Launches represent a multi-chunk direct scan: summary + one paired scan
    per term + readout. Recursive scan levels add launches; output-only with
    one chunk uses just the readout. GEMM work includes the three-TF32 macro.
    """
    from ..diagonal.diagonal_native import NativeDiagonalResources

    output, chunk = native_product_resources(recipe, sizes, heads)
    tiles = (recipe.columns + 15) // 16
    base = sum(
        pair.rows
        if len(pair.key) == 1
        else pair.rows // pair.right + pair.right
        if pair.key
        else 0
        for pair in recipe.pairs
    )
    rows = sum(pair.rows for pair in recipe.pairs)
    # Denominator occupies one extra GEMM lane in each value tile. Count the
    # physical padded 32 columns, not merely the 16 visible output columns.
    gemm_work = 2 * rows * (recipe.columns + 1) + tiles * (
        2 * rows * 32 + 2 * 64 * base + 2 * 64 * 32 * len(recipe.pairs)
    )
    live = 4 * (
        2 * 64 * 64
        + 2 * 64 * 32 * len(recipe.pairs)
        + 64 * 32
        + 32 * 64
        + 32 * 32
        + 64 * 32
        + 64 * 64
    )
    return NativeDiagonalResources(
        output,
        chunk,
        2 + len(recipe.pairs),
        3 * heads * gemm_work,
        0,
        0,
        0,
        live,
        False,
    )


def compile_native_product_normalized(algorithm, readouts, sizes):
    """Prove matched zero/base/product identity terms, without naming models.

    The readout tree is retained verbatim: sums, scalar coefficients and epsilon
    are emitted in authored order, rather than collapsed into a polynomial.
    """
    from ....stateful_operator import (
        Add,
        Constant,
        Divide,
        Identity,
        Multiply,
        Outer,
        ProductFeatureRole,
        StateTerm,
    )

    if sizes is None or len(readouts) != 1:
        return None
    readout, expression = readouts[0]
    if not isinstance(expression, Divide):
        return None
    specs = {s.name: s for s in algorithm.inputs}
    states = {s.name: s for s in algorithm.states}
    transitions = {t.state: t for t in algorithm.transitions}

    def collect(node):
        if isinstance(node, StateTerm):
            return [node]
        if isinstance(node, Constant):
            return []
        if isinstance(node, Add):
            a, b = collect(node.left), collect(node.right)
            return None if a is None or b is None else a + b
        if isinstance(node, Multiply) and (
            isinstance(node.left, Constant) or isinstance(node.right, Constant)
        ):
            return collect(node.right if isinstance(node.left, Constant) else node.left)
        return None

    numerators, denominators = collect(expression.left), collect(expression.right)
    if not numerators or not denominators or len(numerators) != len(denominators):
        return None
    pairs, terms, used = [], [], set()
    columns = None
    for numerator in numerators:
        matches = [
            d
            for d in denominators
            if (d.role, d.operand, d.prior)
            == (numerator.role, numerator.operand, numerator.prior)
            and d.state not in used
        ]
        if len(matches) != 1:
            return None
        denominator = matches[0]
        if numerator.state in used or numerator.state == denominator.state:
            return None
        ns, ds = states[numerator.state], states[denominator.state]
        zero = numerator.role is None
        if ds.rank != 1 or ns.rank != (1 if zero else 2):
            return None
        if zero:
            if (
                not isinstance(numerator.operand, Constant)
                or numerator.operand.value != 1.0
                or sizes[ds.feature_roles[0]] != 1
            ):
                return None
            row_role = ds.feature_roles[0]
        else:
            row_role = ns.feature_roles[0]
            if numerator.role != row_role or ds.feature_roles != (row_role,):
                return None
        value_role = ns.feature_roles[-1]
        if columns is not None and columns != sizes[value_role]:
            return None
        columns = sizes[value_role]
        nt, dt = transitions[ns.name], transitions[ds.name]
        if any(
            not isinstance(p, Identity) for t in (nt, dt) for p in t.propagations
        ) or any(len(t.injections) != 1 for t in (nt, dt)):
            return None
        nf = {f.roles: f.expression for f in nt.injections[0].factors}
        df = {f.roles: f.expression for f in dt.injections[0].factors}
        if (
            len(nf) != len(nt.injections[0].factors)
            or len(df) != 1
            or set(nf) != ({(value_role,)} if zero else {(row_role,), (value_role,)})
        ):
            return None
        if set(df) != {(row_role,)}:
            return None
        key = df[(row_role,)]
        if zero:
            if not isinstance(key, Constant) or key.value != 1.0:
                return None
            key_parts, query_parts, right = (), (), 0
        else:
            if nf[(row_role,)] != key:
                return None
            if isinstance(row_role, ProductFeatureRole):
                if not isinstance(key, Outer) or not isinstance(
                    numerator.operand, Outer
                ):
                    return None
                if key.role != row_role or numerator.operand.role != row_role:
                    return None
                roles = (row_role.left, row_role.right)
                key_parts = tuple(
                    compile_dense_expression(e, specs, r)
                    for e, r in zip((key.left, key.right), roles)
                )
                query_parts = tuple(
                    compile_dense_expression(e, specs, r)
                    for e, r in zip(
                        (numerator.operand.left, numerator.operand.right), roles
                    )
                )
                right = sizes[row_role.right]
            else:
                key_parts = (compile_dense_expression(key, specs, row_role),)
                query_parts = (
                    compile_dense_expression(numerator.operand, specs, row_role),
                )
                right = 0
        value = compile_dense_expression(nf[(value_role,)], specs, value_role)
        if value is None or any(e is None for e in (*key_parts, *query_parts)):
            return None
        pairs.append(
            ProductPair(
                ns.name,
                ds.name,
                key_parts,
                query_parts,
                value,
                1 if zero else sizes[row_role],
                right,
                numerator.prior,
            )
        )
        terms.append((numerator, denominator))
        used.update((ns.name, ds.name))
    if used != set(states) or not any(p.right for p in pairs):
        return None
    return NativeProductNormalized(
        readout, expression, tuple(pairs), tuple(terms), columns
    )


def execute_native_product_normalized(
    recipe,
    inputs,
    initial_states,
    *,
    sequence_count,
    sequence_length,
    heads,
    return_final_state,
):
    from ..diagonal.identity_scan import identity_hierarchical_carry_pair
    from .product_native_kernels import (
        launch_product_summary,
        launch_product_normalized,
    )

    chunks = (sequence_length + 63) // 64
    initials, summaries, finals = [], [], {}
    for pair in recipe.pairs:
        num = initial_states[pair.numerator]
        den = initial_states[pair.denominator]
        num = num.unsqueeze(-2) if not pair.key else num
        den = den.unsqueeze(-1)
        initials.append((num, den))
        if chunks == 1 and not return_final_state:
            summaries.append(
                (
                    num.reshape(sequence_count * heads, 1, pair.rows, recipe.columns),
                    den.reshape(sequence_count * heads, 1, pair.rows, 1),
                )
            )
        else:
            summaries.append(
                (
                    torch.empty(
                        (sequence_count * heads, chunks, pair.rows, recipe.columns),
                        device=num.device,
                        dtype=torch.float32,
                    ),
                    torch.empty(
                        (sequence_count * heads, chunks, pair.rows, 1),
                        device=den.device,
                        dtype=torch.float32,
                    ),
                )
            )
    if chunks > 1 or return_final_state:
        launch_product_summary(
            recipe, inputs, summaries, sequence_length=sequence_length, heads=heads
        )
        carries = []
        for pair, (num, den), (bias, den_bias) in zip(
            recipe.pairs, initials, summaries
        ):
            final = torch.empty_like(num) if return_final_state else num
            den_final = torch.empty_like(den) if return_final_state else den
            carry, den_carry = identity_hierarchical_carry_pair(
                bias,
                den_bias,
                num,
                den,
                final,
                den_final,
                write_final=return_final_state,
            )
            carries.append((carry, den_carry))
            if return_final_state:
                finals[pair.numerator] = final.squeeze(-2) if not pair.key else final
                finals[pair.denominator] = den_final.squeeze(-1)
        del summaries, bias, den_bias
    else:
        carries = summaries
    output = torch.empty(
        sequence_count * sequence_length,
        heads,
        recipe.columns,
        device=initials[0][0].device,
        dtype=recipe.readout.output_dtype,
    )
    launch_product_normalized(
        recipe, inputs, carries, output, sequence_length=sequence_length, heads=heads
    )
    return {recipe.readout.name: output}, finals if return_final_state else None
