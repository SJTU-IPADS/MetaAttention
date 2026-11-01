"""Dense diagonal-State lowering with FP32 chunk carries and typed readouts.

Identity recurrence uses chunk GEMMs, synthesizing Product Features in tiles.
Other diagonal transitions keep State in distributed registers for one chunk;
dependent readouts consume contracted vectors, never per-token matrix States.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch

# the chunk length the recurrence kernel and the published workspace bound share
CHUNK_TOKENS = 64


def _chunked_state(
    *,
    gate_rows: torch.Tensor,
    gate_columns: torch.Tensor | None,
    key: torch.Tensor,
    value: torch.Tensor | None,
    query: torch.Tensor,
    initial_state: torch.Tensor,
    output_dtype: torch.dtype,
    chunk_tokens: int = CHUNK_TOKENS,
    log_gate_rows: bool = False,
    unit_gate_columns: bool = False,
    scalar_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a diagonal recurrence through distributed chunk replay."""
    from .diagonal.diagonal_kernels import prepare_state, read_state

    if chunk_tokens != CHUNK_TOKENS:
        raise ValueError("diagonal recurrence uses 64-token chunks")
    promoted = initial_state.dim() == 3
    initial = initial_state.unsqueeze(-1) if promoted else initial_state
    batch, heads, rows, columns = initial.shape
    tokens = key.shape[0]
    length = tokens // batch
    if tokens != batch * length:
        raise ValueError("dense tokens must divide evenly between sequences")
    if value is None:
        value = torch.ones(
            tokens, heads, columns, device=key.device, dtype=torch.float32
        )
    rows_gate = gate_rows.reshape(tokens, heads, -1).float()
    if log_gate_rows:
        rows_gate = rows_gate.exp()
    rows_gate = rows_gate.expand(tokens, heads, rows)
    columns_gate = None if unit_gate_columns else gate_columns
    carry, final, factors = prepare_state(
        rows_gate=rows_gate,
        columns_gate=columns_gate,
        key=key,
        value=value,
        initial=initial,
        sequence_length=length,
        return_final_state=True,
    )
    output = read_state(carry, factors, query, sequence_length=length)
    output = (
        output.reshape(batch, length, heads, columns)
        .permute(0, 2, 1, 3)
        .to(output_dtype)
    )
    return output, final.squeeze(-1) if promoted else final


def chunked_elementwise_dense(
    *,
    gate_rows: tuple[torch.Tensor, ...],
    gate_columns: tuple[torch.Tensor | None, ...],
    key: tuple[torch.Tensor | None, ...],
    value: tuple[torch.Tensor, ...],
    query: tuple[torch.Tensor, ...],
    initial_state: tuple[torch.Tensor, ...],
    output_dtype: torch.dtype,
    chunk_tokens: int = CHUNK_TOKENS,
    log_gate_rows: tuple[bool, ...] = (),
    unit_gate_columns: tuple[bool, ...] = (),
    scalar_gate: tuple[bool, ...] = (),
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Run every diagonal recurrence through the chunked schedule."""
    state_count = len(initial_state)
    per_state = (gate_rows, gate_columns, key, value, query)
    if any(len(entry) != state_count for entry in per_state):
        raise ValueError("every per-State argument must name the same States")
    if log_gate_rows and len(log_gate_rows) != state_count:
        raise ValueError("log_gate_rows must name every State when provided")
    if unit_gate_columns and len(unit_gate_columns) != state_count:
        raise ValueError("unit_gate_columns must name every State when provided")
    if scalar_gate and len(scalar_gate) != state_count:
        raise ValueError("scalar_gate must name every State when provided")
    outputs: list[torch.Tensor] = []
    finals: list[torch.Tensor] = []
    for index in range(state_count):
        output, final = _chunked_state(
            gate_rows=gate_rows[index],
            gate_columns=gate_columns[index],
            key=key[index],
            value=value[index],
            query=query[index],
            initial_state=initial_state[index],
            output_dtype=output_dtype,
            chunk_tokens=chunk_tokens,
            log_gate_rows=bool(log_gate_rows and log_gate_rows[index]),
            unit_gate_columns=bool(unit_gate_columns and unit_gate_columns[index]),
            scalar_gate=bool(scalar_gate and scalar_gate[index]),
        )
        outputs.append(output)
        finals.append(final)
    return tuple(outputs), tuple(finals)


def _contracted_terms(expression: Any) -> list[Any] | None:
    """Collect nested State Terms; reject unimplemented expression nodes."""
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
        Negate,
        Outer,
        Reciprocal,
        Sigmoid,
        Softmax,
        Sqrt,
        StateTerm,
    )

    if isinstance(expression, StateTerm):
        dependencies = _contracted_terms(expression.operand)
        return None if dependencies is None else dependencies + [expression]
    if isinstance(expression, (Input, Constant)):
        return []
    if isinstance(expression, (Add, Multiply, Divide, Outer, Contract)):
        left = _contracted_terms(expression.left)
        right = _contracted_terms(expression.right)
        return None if left is None or right is None else left + right
    if isinstance(
        expression,
        (Abs, Cast, Negate, Exp, Log, Sigmoid, Sqrt, Reciprocal, Clamp, Softmax),
    ):
        return _contracted_terms(expression.operand)
    return None


def _axis_factors(
    propagations: Any,
    *,
    view: Any,
    evaluate: Any,
    values: Mapping[str, torch.Tensor],
    token_count: int,
    heads: int,
    left_size: int,
    right_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """A diagonal Transition's per-axis decay factors, kept separate.

    ``_affine_summary`` returns their outer product, which costs
    ``token_count * heads * rows * columns`` elements -- 268 MB for a 64x64 State
    at T=2048, and 1 GiB for the 256x64 one `rebased` carries. The chunked kernels
    take one gate per axis, so that product is pure overhead here. ``None`` means a
    propagation is not a diagonal scale, which sends the caller to the reference.
    """
    from ...stateful_operator import AxisScale, Identity

    from .reference import _role_vector

    def per_token(expression: Any, size: int | None) -> torch.Tensor | None:
        factor = evaluate(expression, values).float()
        if factor.numel() == 1:
            return factor.reshape(1, 1, 1).expand(token_count, heads, size or 1)
        if size is None:
            broadcast = factor.reshape(token_count, heads, -1)
            return broadcast if broadcast.shape[-1] == 1 else None
        return _role_vector(factor, token_count, heads).reshape(
            token_count, heads, size
        )

    rows = None
    columns = None
    for node in propagations:
        if isinstance(node, Identity):
            continue
        if not isinstance(node, AxisScale):
            return None
        roles = tuple(node.roles)
        whole = len(roles) == 2 or not roles
        if whole:
            scale = per_token(node.factor, None)
            if scale is None:
                return None
            rows = scale if rows is None else rows * scale
            continue
        if roles[0] == view.feature_roles[0]:
            row_factor = per_token(node.factor, left_size)
            if row_factor is None:
                return None
            rows = row_factor if rows is None else rows * row_factor
        elif roles[0] == view.feature_roles[-1]:
            if right_size == 1:
                continue
            column_factor = per_token(node.factor, right_size)
            if column_factor is None:
                return None
            columns = column_factor if columns is None else columns * column_factor
        else:
            return None
    return rows, columns


def _readout_expression(readout: Any) -> Any:
    from ...stateful_operator import StateContraction, StateTerm

    return (
        StateTerm(readout.state, readout.operand, readout.role)
        if isinstance(readout, StateContraction)
        else readout.expression
    )


def _expression_features(expression: Any, algorithm: Any) -> tuple[Any, ...] | None:
    """Prove the squeezed value carries at most one Feature axis."""
    from ...stateful_operator import (
        Constant,
        Contract,
        FeatureRole,
        Input,
        Outer,
        StateTerm,
    )

    if isinstance(expression, Constant):
        return ()
    if isinstance(expression, Input):
        spec = next(
            (item for item in algorithm.inputs if item.name == expression.name), None
        )
        if spec is None:
            return None
        roles = tuple(role for role in spec.roles if isinstance(role, FeatureRole))
        return roles if len(roles) <= 1 else None
    if isinstance(expression, StateTerm):
        spec = next(
            (item for item in algorithm.states if item.name == expression.state), None
        )
        if spec is None:
            return None
        operand = _expression_features(expression.operand, algorithm)
        if expression.role is None:
            return spec.feature_roles if operand == () and spec.rank == 1 else None
        if operand != (expression.role,):
            return None
        return tuple(role for role in spec.feature_roles if role != expression.role)
    if hasattr(expression, "left"):
        left = _expression_features(expression.left, algorithm)
        right = _expression_features(expression.right, algorithm)
        if left is None or right is None:
            return None
        if isinstance(expression, Outer):
            return (
                (expression.role,)
                if left == (expression.role.left,) and right == (expression.role.right,)
                else None
            )
        if isinstance(expression, Contract):
            return (
                ()
                if left in ((), (expression.role,))
                and right in ((), (expression.role,))
                else None
            )
        # The reference readout value space also aligns its single surviving
        # Feature positionally (including an extent-one authored unit Feature).
        return left or right
    if hasattr(expression, "operand"):
        return _expression_features(expression.operand, algorithm)
    return None


def chunked_states_eligible(algorithm: Any, *, packed: bool = False) -> bool:
    """Static proof for dense diagonal recurrence with dependent readout passes."""
    from ...stateful_operator import AxisScale, Identity, ProductFeatureRole

    if packed or any(
        transition.steps is not None for transition in algorithm.transitions
    ):
        return False
    specs = {state.name: state for state in algorithm.states}
    modes: dict[str, set[bool]] = {name: set() for name in specs}
    for readout in algorithm.readouts:
        terms = _contracted_terms(_readout_expression(readout))
        if terms is None:
            return False
        if _expression_features(_readout_expression(readout), algorithm) is None:
            return False
        for term in terms:
            state = specs.get(term.state)
            if state is None:
                return False
            if term.role is None:
                if state.rank != 1:
                    return False
            elif term.role != state.feature_roles[0]:
                return False
            modes[state.name].add(term.role is None)
    if any(len(mode) > 1 for mode in modes.values()):
        return False
    for state in algorithm.states:
        transition = next(
            (item for item in algorithm.transitions if item.state == state.name), None
        )
        if transition is None or not transition.injections:
            return False
        multiple = len(transition.injections) > 1
        if multiple and (
            state.rank != 2
            or any(isinstance(role, ProductFeatureRole) for role in state.feature_roles)
        ):
            return False
        for node in transition.propagations:
            if not isinstance(node, (AxisScale, Identity)):
                return False
            if isinstance(node, AxisScale):
                if _contracted_terms(node.factor) != []:
                    return False
                features = _expression_features(node.factor, algorithm)
                if features is None or (features and features != tuple(node.roles)):
                    return False
                if any(role not in state.feature_roles for role in node.roles):
                    return False
                if len(node.roles) > 1 and features:
                    return False
        for injection in transition.injections:
            seen: set[Any] = set()
            scalar_count = 0
            for factor in injection.factors:
                roles = tuple(factor.roles)
                scalar_count += not roles
                if scalar_count > (0 if multiple else 1):
                    return False
                if len(roles) > 1 or any(
                    role not in state.feature_roles or role in seen for role in roles
                ):
                    return False
                if _contracted_terms(factor.expression) != []:
                    return False
                features = _expression_features(factor.expression, algorithm)
                if features is None or (features and features != roles):
                    return False
                seen.update(roles)
            if seen != set(state.feature_roles):
                return False
    return True


def chunked_states_workspace(
    algorithm: Any, role_sizes: Mapping[Any, int], heads: int
) -> tuple[int, int]:
    """Conservative ``(bytes/token, bytes/chunk/sequence)`` scratch bound.

    Charge B*T times the first and B*(ceil(T/64)+1) times the second, including
    final States. Covers factor evaluation, contracted FP32 outputs and authored
    arithmetic. Identity Product Features are synthesized inside GEMM tiles.
    """
    from ...stateful_operator import Identity, ProductFeatureRole

    per_token = 0
    per_chunk = 0
    for state in algorithm.states:
        dims = [role_sizes[role] for role in state.feature_roles]
        elements = dims[0] * (dims[1] if len(dims) == 2 else 1)
        transition = next(
            item for item in algorithm.transitions if item.state == state.name
        )
        identity = all(isinstance(node, Identity) for node in transition.propagations)
        per_chunk += 4 * heads * elements * (2 if identity else 3)
        factor_dims = [
            role_sizes[role.left] + role_sizes[role.right]
            if identity and isinstance(role, ProductFeatureRole)
            else role_sizes[role]
            for role in state.feature_roles
        ]
        per_token += 32 * heads * sum(factor_dims)
    for readout in algorithm.readouts:
        for term in _contracted_terms(_readout_expression(readout)) or ():
            state = next(
                state for state in algorithm.states if state.name == term.state
            )
            output = (
                role_sizes[state.feature_roles[-1]]
                if state.rank == 2 or term.role is None
                else 1
            )
            per_token += 64 * heads * output
    return per_token, per_chunk


@dataclass(frozen=True)
class DenseInputRecipe:
    name: str
    axes: tuple[int, ...]
    feature_axes: tuple[int, ...]
    feature_positions: tuple[int | None, ...]
    head_axis: int | None
    sequence: bool


@dataclass(frozen=True)
class ChunkedRecipe:
    specs: Mapping[str, Any]
    transitions: Mapping[str, Any]
    identity_read: frozenset[str]
    readouts: tuple[tuple[Any, Any], ...]
    normalized_readout: tuple[Any, Any, Any, float] | None
    bonus_readout: tuple[Any, Any, Any, float, Any, Any] | None
    rank_one: Any | None
    native_normalized: Any | None
    native_diagonal: Any | None
    native_product_normalized: Any | None
    inputs: tuple[Any, ...]
    feature_order: tuple[Any, ...]
    input_only: frozenset[int]
    expression_slots: Mapping[int, int]


def _product_terms(expression: Any) -> list[Any]:
    from ...stateful_operator import Multiply

    if isinstance(expression, Multiply):
        return _product_terms(expression.left) + _product_terms(expression.right)
    return [expression]


def _contains_expression(expression: Any, target: Any) -> bool:
    if expression == target:
        return True
    children = []
    if hasattr(expression, "left"):
        children.extend((expression.left, expression.right))
    elif hasattr(expression, "operand"):
        children.append(expression.operand)
    return any(_contains_expression(child, target) for child in children)


def _bonus_readout(readout: Any, expression: Any, algorithm: Any):
    """Prove the typed prior-State plus current-token bonus region."""
    from ...stateful_operator import (
        Add,
        AxisScale,
        Batch,
        Constant,
        Contract,
        FeatureRole,
        HeadRole,
        Input,
        Multiply,
        Sequence,
        StateTerm,
    )

    if not isinstance(expression, Add):
        return None
    branches = (expression.left, expression.right)
    state_terms = [branch for branch in branches if isinstance(branch, StateTerm)]
    if len(state_terms) != 1:
        return None
    state_term = state_terms[0]
    if not state_term.prior or state_term.role is None:
        return None
    state = next(
        (item for item in algorithm.states if item.name == state_term.state), None
    )
    if state is None or state.rank != 2 or state_term.role != state.feature_roles[0]:
        return None
    other = branches[0] if branches[1] is state_term else branches[1]
    if not isinstance(other, Multiply):
        return None
    contraction = next(
        (item for item in (other.left, other.right) if isinstance(item, Contract)), None
    )
    if contraction is None:
        return None
    value_expression = other.right if other.left is contraction else other.left
    if (
        not isinstance(value_expression, Input)
        or contraction.role != state.feature_roles[0]
    ):
        return None
    if not isinstance(contraction.right, Constant) or contraction.right.value != 1.0:
        return None
    if not _contains_expression(contraction.left, state_term.operand):
        return None
    query_terms = _product_terms(state_term.operand)
    if len(query_terms) != 2 or not any(
        isinstance(item, Constant) for item in query_terms
    ):
        return None
    query_input = next((item for item in query_terms if isinstance(item, Input)), None)
    query_constant = next(
        (item for item in query_terms if isinstance(item, Constant)), None
    )
    if (
        query_input is None
        or query_constant is None
        or query_constant.dtype != torch.float32
    ):
        return None
    query_scale = query_constant.value
    remaining = _product_terms(contraction.left)
    for term in query_terms:
        try:
            remaining.remove(term)
        except ValueError:
            return None
    if len(remaining) != 2 or not all(isinstance(item, Input) for item in remaining):
        return None
    specs = {spec.name: spec for spec in algorithm.inputs}
    transitions = {item.state: item for item in algorithm.transitions}
    transition = transitions.get(state.name)
    if transition is None or len(transition.injections) != 1:
        return None
    factors = transition.injections[0].factors
    key_expression = next(
        (
            factor.expression
            for factor in factors
            if factor.roles == (state.feature_roles[0],)
        ),
        None,
    )
    value_transition = next(
        (
            factor.expression
            for factor in factors
            if factor.roles == (state.feature_roles[1],)
        ),
        None,
    )
    if not isinstance(key_expression, Input) or value_transition != value_expression:
        return None
    if key_expression not in remaining:
        return None
    bonus_expression = next(item for item in remaining if item != key_expression)
    if not isinstance(bonus_expression, Input):
        return None
    if len(transition.propagations) != 1 or not isinstance(
        transition.propagations[0], AxisScale
    ):
        return None
    if transition.propagations[0].roles != (state.feature_roles[0],):
        return None
    for name, feature in (
        (query_input.name, state.feature_roles[0]),
        (key_expression.name, state.feature_roles[0]),
        (bonus_expression.name, state.feature_roles[0]),
        (value_expression.name, state.feature_roles[1]),
    ):
        spec = specs[name]
        if Batch not in spec.roles or Sequence not in spec.roles:
            return None
        if tuple(role for role in spec.roles if isinstance(role, FeatureRole)) != (
            feature,
        ):
            return None
        if not any(isinstance(role, HeadRole) for role in spec.roles):
            return None
    return (
        readout,
        state_term,
        query_input,
        query_scale,
        bonus_expression,
        key_expression,
    )


def _normalized_readout(readout: Any, expression: Any, algorithm: Any):
    """Prove two input-only row contractions and preserve the authored constant."""
    from ...stateful_operator import Add, Constant, Divide, StateTerm

    if not isinstance(expression, Divide):
        return None
    numerator, denominator = expression.left, expression.right
    if not isinstance(numerator, StateTerm) or numerator.role is None:
        return None
    if not isinstance(denominator, Add):
        return None
    if isinstance(denominator.left, StateTerm) and isinstance(
        denominator.right, Constant
    ):
        state_term, constant = denominator.left, denominator.right
    elif isinstance(denominator.right, StateTerm) and isinstance(
        denominator.left, Constant
    ):
        state_term, constant = denominator.right, denominator.left
    else:
        return None
    if state_term.role != numerator.role or _contracted_terms(numerator.operand) != []:
        return None
    if state_term.operand != numerator.operand:
        return None
    specs = {state.name: state for state in algorithm.states}
    first, second = specs[numerator.state], specs[state_term.state]
    if first.rank != 2 or second.rank != 1:
        return None
    if first.feature_roles[0] != second.feature_roles[0]:
        return None
    return readout, numerator, state_term, constant.value


def compile_chunked_recipe(algorithm: Any, role_sizes=None) -> ChunkedRecipe | None:
    """Bind immutable State/readout structure once, independent of runtime B/T."""
    if not chunked_states_eligible(algorithm):
        return None
    from .scan.rank1 import compile_rank_one
    from .readout.native import compile_native_normalized
    from .readout.product_native import compile_native_product_normalized
    from .diagonal.diagonal_native import compile_native_diagonal
    from .reference import feature_layout
    from ...stateful_operator import Batch, FeatureRole, HeadRole, Sequence

    feature_order = feature_layout(algorithm)[0]
    bindings = []
    for spec in algorithm.inputs:
        roles = spec.roles
        head_axis = next(
            (index for index, role in enumerate(roles) if isinstance(role, HeadRole)),
            None,
        )
        features = tuple(role for role in roles if isinstance(role, FeatureRole))
        sequence = Sequence in roles
        axes = (roles.index(Batch), roles.index(Sequence)) if sequence else (0,)
        axes += (head_axis,) if head_axis is not None else ()
        axes += tuple(roles.index(role) for role in features)
        bindings.append(
            DenseInputRecipe(
                spec.name,
                axes,
                tuple(roles.index(role) for role in features),
                tuple(
                    roles.index(role) if role in features else None
                    for role in feature_order
                ),
                head_axis,
                sequence,
            )
        )

    readouts = tuple(
        (readout, _readout_expression(readout)) for readout in algorithm.readouts
    )
    normalized = (
        _normalized_readout(*readouts[0], algorithm) if len(readouts) == 1 else None
    )
    bonus = _bonus_readout(*readouts[0], algorithm) if len(readouts) == 1 else None
    native = compile_native_normalized(algorithm, normalized)
    if native is not None:
        from .readout.readout_regions import readout_region_resources

        state = next(state for state in algorithm.states if state.name == native.memory)
        if role_sizes is None:
            native = None
        else:
            rows, columns = (role_sizes[role] for role in state.feature_roles)
            if not readout_region_resources(
                (1, 1, rows, columns), (1, 1, rows, 1), softmax=False
            )["eligible"]:
                native = None
    product = compile_native_product_normalized(algorithm, readouts, role_sizes)
    diagonal = compile_native_diagonal(algorithm, readouts, bonus, role_sizes)
    # The materialized-factor executor and normalized pair admit one injection.
    # Multiple injections require the complete raw-loader/readout proof.
    if (
        product is None
        and diagonal is None
        and any(len(item.injections) > 1 for item in algorithm.transitions)
    ):
        return None
    # Dependency classification and semantic CSE happen once. Runtime caches
    # use integer slots, avoiding recursive IR hashing on each invocation.
    from ...stateful_operator import StateTerm

    slots, canonical, input_only = {}, {}, set()

    def visit(expression):
        key = id(expression)
        if key in slots:
            return key in input_only
        children = ()
        if hasattr(expression, "left"):
            children = (expression.left, expression.right)
        elif hasattr(expression, "operand"):
            children = (expression.operand,)
        dependencies = tuple(visit(child) for child in children)
        independent = not isinstance(expression, StateTerm) and all(dependencies)
        slots[key] = canonical.setdefault(expression, len(canonical))
        if independent:
            input_only.add(key)
        return independent

    for transition in algorithm.transitions:
        for propagation in transition.propagations:
            if hasattr(propagation, "factor"):
                visit(propagation.factor)
        for injection in transition.injections:
            for factor in injection.factors:
                visit(factor.expression)
    for _, expression in readouts:
        visit(expression)

    return ChunkedRecipe(
        MappingProxyType({state.name: state for state in algorithm.states}),
        MappingProxyType(
            {transition.state: transition for transition in algorithm.transitions}
        ),
        frozenset(
            term.state
            for _, expression in readouts
            for term in (_contracted_terms(expression) or ())
            if term.role is None
        ),
        readouts,
        normalized,
        bonus,
        compile_rank_one(algorithm),
        native,
        diagonal,
        product,
        tuple(bindings),
        feature_order,
        frozenset(input_only),
        MappingProxyType(slots),
    )


def select_chunked_schedule(recipe, lowerer):
    """Bind a compiler-selected schedule; runtime lengths never select a plan."""
    if recipe is None or recipe.native_diagonal is None:
        return recipe
    from dataclasses import replace

    native = recipe.native_diagonal
    return replace(
        recipe,
        native_diagonal=replace(
            native,
            shared_scores=lowerer
            in (
                "tilelang-chunked-shared-scores",
                "tilelang-native-diagonal-shared-scores",
            ),
            fused_softmax=native.fused_softmax
            and lowerer
            not in (
                "tilelang-chunked-readout-cut",
                "tilelang-native-diagonal-readout-cut",
            ),
        ),
    )


def bind_chunked_inputs(recipe: ChunkedRecipe, inputs, runtime):
    """Bind dense values without reference segments or per-call IR analysis."""

    if recipe.native_product_normalized is not None:
        return inputs
    if recipe.native_normalized is not None:
        return inputs
    if recipe.native_diagonal is not None:
        return inputs

    values = {}
    for binding in recipe.inputs:
        tensor = inputs[binding.name]
        if binding.sequence:
            value = tensor.permute(binding.axes)
            feature_shape = tuple(tensor.shape[axis] for axis in binding.feature_axes)
            head_shape = (
                (tensor.shape[binding.head_axis],)
                if binding.head_axis is not None
                else ()
            )
            value = value.reshape(runtime.token_count, *head_shape, *feature_shape)
        else:
            # Non-sequence inputs use a stride-zero token broadcast until a
            # consumer explicitly needs a contiguous materialization.
            value = tensor.permute(binding.axes).unsqueeze(1)
            value = value.expand(
                runtime.sequence_count, runtime.uniform_length, *value.shape[2:]
            )
            value = value.reshape(runtime.token_count, *value.shape[2:])
        if binding.head_axis is None:
            value = value.unsqueeze(1)
        source_heads = value.shape[1]
        if source_heads != runtime.state_heads:
            if source_heads == 1:
                value = value.expand(-1, runtime.state_heads, *value.shape[2:])
            else:
                value = value.repeat_interleave(
                    runtime.state_heads // source_heads, dim=1
                )
        shape = [
            tensor.shape[axis] if axis is not None else 1
            for axis in binding.feature_positions
        ]
        while len(shape) < 2:
            shape.append(1)
        values[binding.name] = value.reshape(
            runtime.token_count, runtime.state_heads, *shape
        )
    return values


def execute_chunked_states(
    algorithm: Any,
    *,
    values: Mapping[str, torch.Tensor],
    initial_states: Mapping[str, torch.Tensor],
    segment: torch.Tensor | None,
    role_sizes: Mapping[Any, int],
    heads: int,
    token_count: int,
    device: torch.device,
    return_final_state: bool,
    sequence_length: int | None = None,
    recipe: ChunkedRecipe | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None] | None:
    """Dense inference with boundary carries, GEMMs and dependent readout passes."""
    if device.type != "cuda" or sequence_length is None or sequence_length <= 0:
        return None
    if any(
        tensor.requires_grad for tensor in (*values.values(), *initial_states.values())
    ):
        return None
    if recipe is None:
        recipe = compile_chunked_recipe(algorithm, role_sizes)
        if recipe is not None and recipe.native_product_normalized is not None:
            # Direct internal callers provide canonical token-major values;
            # only the public binder supplies the authored raw-input ABI.
            from dataclasses import replace

            recipe = replace(recipe, native_product_normalized=None)
    if recipe is None:
        return None
    return bind_chunked_executor(recipe)(
        algorithm,
        values=values,
        initial_states=initial_states,
        segment=segment,
        role_sizes=role_sizes,
        heads=heads,
        token_count=token_count,
        device=device,
        return_final_state=return_final_state,
        sequence_length=sequence_length,
    )


def bind_chunked_executor(recipe):
    """Bind the proven native/materialized adapter once per compiled candidate."""
    from functools import partial

    if recipe.native_product_normalized is not None:
        from .readout.product_native import execute_native_product_normalized

        return partial(
            _execute_native_states,
            execute_native_product_normalized,
            recipe.native_product_normalized,
        )
    if recipe.native_normalized is not None:
        from .readout.native import execute_native_normalized

        return partial(
            _execute_native_states, execute_native_normalized, recipe.native_normalized
        )
    if recipe.native_diagonal is not None:
        from .diagonal.diagonal_native import execute_native_diagonal

        return partial(
            _execute_native_states, execute_native_diagonal, recipe.native_diagonal
        )
    return partial(_execute_materialized_states, recipe=recipe)


def _execute_native_states(
    execute,
    recipe,
    algorithm,
    *,
    values,
    initial_states,
    token_count,
    sequence_length,
    heads,
    return_final_state,
    **_,
):
    return execute(
        recipe,
        values,
        initial_states,
        sequence_count=token_count // sequence_length,
        sequence_length=sequence_length,
        heads=heads,
        return_final_state=return_final_state,
    )


def _execute_materialized_states(
    algorithm,
    *,
    values,
    initial_states,
    segment,
    role_sizes,
    heads,
    token_count,
    device,
    return_final_state,
    sequence_length,
    recipe,
):
    from ...stateful_operator import (
        Add,
        Abs,
        Cast,
        Clamp,
        Contract,
        Divide,
        Exp,
        Identity,
        Log,
        Multiply,
        Negate,
        Outer,
        Reciprocal,
        Sigmoid,
        Softmax,
        Sqrt,
        StateTerm,
        _evaluate,
    )
    from .reference import _matrix_view
    from .diagonal.diagonal_kernels import prepare_state, read_state

    specs = recipe.specs
    transitions = recipe.transitions
    identity_read = recipe.identity_read
    prepared: dict[str, Any] = {}
    final_states: dict[str, torch.Tensor] = {}
    term_values: dict[int, torch.Tensor] = {}
    input_values: dict[int, torch.Tensor] = {}
    broadcast_values: dict[int, torch.Tensor] = {}
    product_scores: dict[Any, torch.Tensor] = {}

    def vector(value: torch.Tensor, size: int) -> torch.Tensor:
        if value.numel() == 1:
            return value.reshape(1, 1, 1).expand(token_count, heads, size)
        return value.reshape(token_count, heads, -1).expand(token_count, heads, size)

    def input_expression(expression: Any) -> torch.Tensor:
        slot = recipe.expression_slots[id(expression)]
        if slot not in input_values:
            result = (
                _evaluate(expression, values, segment=segment, squeeze_features=True)
                .float()
                .contiguous()
            )
            input_values[slot] = result.unsqueeze(-1) if result.ndim == 2 else result
        return input_values[slot]

    def row_expression(expression: Any, size: int, lazy: bool) -> Any:
        if lazy and isinstance(expression, Outer):
            return (
                input_expression(expression.left).reshape(token_count, heads, -1),
                input_expression(expression.right).reshape(token_count, heads, -1),
            )
        return vector(input_expression(expression), size)

    def broadcast_value(size: int) -> torch.Tensor:
        value = broadcast_values.get(size)
        if value is None:
            value = torch.ones(
                token_count, heads, size, device=device, dtype=torch.float32
            )
            broadcast_values[size] = value
        return value

    def prepare(name: str):
        if name in prepared:
            return prepared[name]
        state = specs[name]
        transition = transitions[name]
        rows = role_sizes[state.feature_roles[0]]
        columns = 1 if state.rank == 1 else role_sizes[state.feature_roles[1]]
        identity = all(isinstance(node, Identity) for node in transition.propagations)
        if identity:
            row_gate, column_gate = None, None
        else:
            row_gate, column_gate = _axis_factors(
                transition.propagations,
                view=_matrix_view(state),
                evaluate=lambda expr, _: input_expression(expr),
                values=values,
                token_count=token_count,
                heads=heads,
                left_size=rows,
                right_size=columns,
                device=device,
            )
        key = None
        value = None
        scalar = None
        for factor in transition.injections[0].factors:
            if not factor.roles:
                scalar = input_expression(factor.expression)
            elif factor.roles[0] == state.feature_roles[0]:
                key = row_expression(
                    factor.expression, rows, identity and name not in identity_read
                )
            else:
                value = vector(input_expression(factor.expression), columns)
        if value is None:
            value = broadcast_value(columns)
        if scalar is not None:
            value = value * vector(scalar, 1)
        initial = initial_states[name]
        if state.rank == 1:
            if name in identity_read:
                initial = initial.unsqueeze(-2)
                value, key = key, value
                row_gate, column_gate = None, row_gate
            else:
                initial = initial.unsqueeze(-1)
        carry, final, factors = prepare_state(
            rows_gate=row_gate,
            columns_gate=column_gate,
            key=key,
            value=value,
            initial=initial,
            sequence_length=sequence_length,
            return_final_state=return_final_state,
            identity=identity,
        )
        if state.rank == 1:
            final = final.squeeze(-2 if name in identity_read else -1)
        final_states[name] = final
        prepared[name] = carry, factors, identity
        return prepared[name]

    expression_values: dict[int, torch.Tensor] = {}
    fused_outputs: dict[str, torch.Tensor] = {}
    if recipe.bonus_readout is not None:
        from .diagonal.diagonal_block import diagonal_block_bonus_readout

        (
            readout,
            state_term,
            query_input,
            query_scale,
            bonus_expression,
            key_expression,
        ) = recipe.bonus_readout
        prepared_state = prepare(state_term.state)
        query = values[query_input.name].reshape(token_count, heads, -1).contiguous()
        bonus = (
            values[bonus_expression.name].reshape(token_count, heads, -1).contiguous()
        )
        current_key = (
            values[key_expression.name].reshape(token_count, heads, -1).contiguous()
        )
        output = torch.empty(
            token_count,
            heads,
            prepared_state[0].shape[-1],
            device=device,
            dtype=torch.float32,
        )
        diagonal_block_bonus_readout(
            prepared_state[0],
            prepared_state[1],
            query,
            bonus,
            current_key,
            output,
            sequence_length=sequence_length,
            query_scale=query_scale,
        )
        fused_outputs[readout.name] = output.to(readout.output_dtype)
    elif recipe.normalized_readout is not None:
        from .readout.readout_regions import fused_normalized_region

        readout, numerator_term, denominator_term, epsilon = recipe.normalized_readout
        first = prepare(numerator_term.state)
        second = prepare(denominator_term.state)
        query = vector(input_expression(numerator_term.operand), first[0].shape[2])
        fused = fused_normalized_region(
            first,
            second,
            query,
            sequence_length=sequence_length,
            prior_first=numerator_term.prior,
            prior_second=denominator_term.prior,
            output_dtype=readout.output_dtype,
            epsilon=epsilon,
        )
        if fused is not None:
            fused_outputs[readout.name] = fused

    def evaluate(expression: Any) -> torch.Tensor:
        slot = recipe.expression_slots[id(expression)]
        if slot not in expression_values:
            expression_values[slot] = evaluate_uncached(expression)
        return expression_values[slot]

    def evaluate_uncached(expression: Any) -> torch.Tensor:
        if id(expression) in recipe.input_only:
            return input_expression(expression)
        if isinstance(expression, StateTerm):
            slot = recipe.expression_slots[id(expression)]
            if slot not in term_values:
                carry, factors, identity = prepare(expression.state)
                size = carry.shape[2]
                if expression.role is None:
                    query = vector(evaluate(expression.operand), 1)
                elif (
                    identity
                    and isinstance(expression.operand, Outer)
                    and id(expression.operand) in recipe.input_only
                ):
                    query = row_expression(expression.operand, size, True)
                else:
                    query = vector(evaluate(expression.operand), size)
                term_values[slot] = read_state(
                    carry,
                    factors,
                    query,
                    sequence_length=sequence_length,
                    prior=expression.prior,
                    identity=identity,
                    product_scores=product_scores,
                )
            return term_values[slot]
        if isinstance(expression, (Add, Multiply, Divide, Contract, Outer)):
            left, right = evaluate(expression.left), evaluate(expression.right)
            if isinstance(expression, Add):
                return left + right
            if isinstance(expression, Multiply):
                return left * right
            if isinstance(expression, Divide):
                return left / right
            if isinstance(expression, Contract):
                return (left.float() * right.float()).sum(-1, keepdim=True)
            return (left.float().unsqueeze(-1) * right.float().unsqueeze(-2)).flatten(2)
        operand = evaluate(expression.operand).float()
        if isinstance(expression, Negate):
            return -operand
        if isinstance(expression, Cast):
            return evaluate(expression.operand).to(expression.dtype)
        if isinstance(expression, Abs):
            return evaluate(expression.operand).float().abs()
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
        raise AssertionError("eligibility admitted an unsupported readout")

    try:
        outputs = {
            readout.name: (
                fused_outputs[readout.name]
                if readout.name in fused_outputs
                else evaluate(expression).to(readout.output_dtype)
            )
            for readout, expression in recipe.readouts
        }
        if return_final_state:
            for state in algorithm.states:
                prepare(state.name)
            return outputs, final_states
        return outputs, None
    finally:
        # These mutually recursive closures capture the per-call tensor caches.
        # Break their cycle so scratch is released without waiting for cyclic GC.
        evaluate = evaluate_uncached = None
