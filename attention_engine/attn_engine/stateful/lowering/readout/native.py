"""Native-layout identity contractions with authored input arithmetic at load."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...runtime.bindings import DenseRawBindingRecipe, compile_dense_binding


@dataclass(frozen=True)
class NativeNormalizedRecipe:
    output: str
    output_dtype: torch.dtype
    memory: str
    normalizer: str
    query: DenseRawBindingRecipe
    key: DenseRawBindingRecipe
    denominator_key: DenseRawBindingRecipe
    value: DenseRawBindingRecipe
    query_scale: float | None
    epsilon: float
    prior_memory: bool
    prior_normalizer: bool


def compile_native_normalized(algorithm, normalized):
    """Prove native vectors, identity updates and a shared scalar-scaled query."""
    from ....stateful_operator import (
        Batch,
        Constant,
        FeatureRole,
        Identity,
        Input,
        Multiply,
        Sequence,
    )

    if normalized is None or len(algorithm.states) != 2:
        return None
    readout, numerator, denominator, epsilon = normalized
    specs = {spec.name: spec for spec in algorithm.inputs}
    states = {state.name: state for state in algorithm.states}
    memory, normalizer = states[numerator.state], states[denominator.state]
    query = numerator.operand
    scale = None
    if isinstance(query, Multiply):
        if isinstance(query.left, Constant) and isinstance(query.right, Input):
            constant, query = query.left, query.right
        elif isinstance(query.right, Constant) and isinstance(query.left, Input):
            constant, query = query.right, query.left
        else:
            return None
        scale = constant.value
    if not isinstance(query, Input):
        return None
    keys = []
    value = None
    for name in (memory.name, normalizer.name):
        transition = next(item for item in algorithm.transitions if item.state == name)
        if len(transition.injections) != 1:
            return None
        if any(not isinstance(node, Identity) for node in transition.propagations):
            return None
        key = None
        for factor in transition.injections[0].factors:
            if not isinstance(factor.expression, Input):
                return None
            if factor.roles == (memory.feature_roles[0],):
                key = specs[factor.expression.name]
            elif name == memory.name and factor.roles == (memory.feature_roles[1],):
                value = specs[factor.expression.name]
            else:
                return None
        if key is None:
            return None
        keys.append(key)
    if value is None:
        return None
    query = specs[query.name]
    for spec, feature in (
        (query, memory.feature_roles[0]),
        (keys[0], memory.feature_roles[0]),
        (keys[1], memory.feature_roles[0]),
        (value, memory.feature_roles[1]),
    ):
        features = tuple(role for role in spec.roles if isinstance(role, FeatureRole))
        if (
            features != (feature,)
            or Batch not in spec.roles
            or Sequence not in spec.roles
        ):
            return None
    return NativeNormalizedRecipe(
        readout.name,
        readout.output_dtype,
        memory.name,
        normalizer.name,
        compile_dense_binding(query, memory.feature_roles[0]),
        compile_dense_binding(keys[0], memory.feature_roles[0]),
        compile_dense_binding(keys[1], memory.feature_roles[0]),
        compile_dense_binding(value, memory.feature_roles[1]),
        scale,
        epsilon,
        numerator.prior,
        denominator.prior,
    )


def _prepare_normalized_pair(
    key,
    den_key,
    value,
    initial,
    den_initial,
    *,
    sequence_length,
    heads,
    return_final_state,
):
    """Own paired summaries/carries and release recursive scratch before readout."""
    from ..diagonal.identity import identity_summary_pair
    from ..diagonal.identity_scan import identity_hierarchical_carry_pair

    rows, columns = initial.shape[-2:]
    bh = initial.shape[0] * heads
    chunks = (sequence_length + 63) // 64
    if chunks == 1:
        # The only exclusive carry is the initial State; no scratch is consumed.
        if not return_final_state:
            return initial, den_initial, None, None
        # A single chunk adds its summary directly to the requested final.
        final, den_final = torch.empty_like(initial), torch.empty_like(den_initial)
        identity_summary_pair(
            key,
            den_key,
            value,
            final,
            den_final,
            sequence_length=sequence_length,
            heads=heads,
            initial=initial,
            den_initial=den_initial,
        )
        return initial, den_initial, final, den_final
    carry = torch.empty(
        (bh, chunks, rows, columns), device=initial.device, dtype=torch.float32
    )
    den_carry = torch.empty(
        (bh, chunks, rows, 1), device=den_initial.device, dtype=torch.float32
    )
    final = torch.empty_like(initial) if return_final_state else initial
    den_final = torch.empty_like(den_initial) if return_final_state else den_initial
    identity_summary_pair(
        key,
        den_key,
        value,
        carry,
        den_carry,
        sequence_length=sequence_length,
        heads=heads,
    )
    carry, den_carry = identity_hierarchical_carry_pair(
        carry,
        den_carry,
        initial,
        den_initial,
        final,
        den_final,
        write_final=return_final_state,
    )
    return carry, den_carry, final, den_final


def execute_native_normalized(
    recipe,
    inputs,
    initial_states,
    *,
    sequence_count,
    sequence_length,
    heads,
    return_final_state,
):
    from ..diagonal.identity import native_vector_view
    from .identity import native_identity_normalized_readout

    bindings = {}

    def bind(binding):
        if binding.name not in bindings:
            bindings[binding.name] = binding.bind(inputs[binding.name], heads)
        return bindings[binding.name]

    query, qs, qg = bind(recipe.query)
    key, ks, kg = bind(recipe.key)
    den_key, dks, dkg = bind(recipe.denominator_key)
    value, vs, vg = bind(recipe.value)
    initial_memory = initial_states[recipe.memory]
    initial_normalizer = initial_states[recipe.normalizer].unsqueeze(-1)
    rows, columns = initial_memory.shape[-2:]
    query = native_vector_view(
        query, sequence_count, heads, sequence_length, rows, qs, qg
    )
    key = native_vector_view(key, sequence_count, heads, sequence_length, rows, ks, kg)
    den_key = native_vector_view(
        den_key, sequence_count, heads, sequence_length, rows, dks, dkg
    )
    value = native_vector_view(
        value, sequence_count, heads, sequence_length, columns, vs, vg
    )

    memory, normalizer, final_memory, final_normalizer = _prepare_normalized_pair(
        key,
        den_key,
        value,
        initial_memory,
        initial_normalizer,
        sequence_length=sequence_length,
        heads=heads,
        return_final_state=return_final_state,
    )
    output = torch.empty(
        sequence_count * sequence_length,
        heads,
        columns,
        device=query.device,
        dtype=recipe.output_dtype,
    )
    native_identity_normalized_readout(
        memory,
        key,
        value,
        normalizer,
        den_key,
        query,
        output,
        sequence_length=sequence_length,
        heads=heads,
        prior_memory=recipe.prior_memory,
        prior_normalizer=recipe.prior_normalizer,
        shared_key=recipe.key.name == recipe.denominator_key.name,
        epsilon=recipe.epsilon,
        query_scale=recipe.query_scale,
    )
    finals = (
        {recipe.memory: final_memory, recipe.normalizer: final_normalizer.squeeze(-1)}
        if return_final_state
        else None
    )
    return {recipe.output: output}, finals
