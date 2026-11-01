"""Based product-feature recurrence expressed with the Stateful Operator DSL.

Based's kernel polynomial ``1 + (q.k) + 0.5 (q.k)^2`` needs a second-order product
Feature as a State axis, so the Program keeps six States and normalizes with
``1e-6`` epsilon.
"""

from __future__ import annotations

import torch

from attn_engine import (
    AlgorithmIR,
    Batch,
    Constant,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    Outer,
    ProductFactor,
    ProductFeatureRole,
    ProductInjection,
    Readout,
    Sequence,
    StatefulOperator,
    StateSpec,
    StateTerm,
    StateTransition,
    TensorInput,
)

KEY = FeatureRole("key_dim")
VALUE = FeatureRole("value_dim")
QUERY_HEADS = HeadRole("query_heads")
KEY_HEADS = HeadRole("key_heads")
VALUE_HEADS = HeadRole("value_heads")
STATE_HEADS = HeadRole("state_heads")


def based_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """FLA ``fused_chunk_based``: second-order Taylor feature map with normalizer.

    The recurrence keeps six States: the zero, first, and second order numerator
    terms ``h0 [V]``, ``h1 [K, V]``, ``h2 [K*K, V]`` and their denominator
    counterparts ``z0 [1]``, ``z1 [K]``, ``z2 [K*K]``. The readout evaluates
    ``1 + (q.k) + 0.5 (q.k)^2`` through those States and divides by the matching
    denominator with the library's ``1e-6`` epsilon.
    """
    scale = dim**-0.5
    scaled_query = Input("query") * scale
    second_order = ProductFeatureRole(KEY, KEY)
    unit = FeatureRole("unit_dim")
    query_pair = Outer(scaled_query, scaled_query, second_order)
    key_pair = Outer(Input("key"), Input("key"), second_order)
    zero_order_injection = ProductInjection((ProductFactor(Constant(1.0), (unit,)),))
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
        ),
        states=(
            StateSpec("numerator_0", (Batch, STATE_HEADS, VALUE)),
            StateSpec("numerator_1", (Batch, STATE_HEADS, KEY, VALUE)),
            StateSpec("numerator_2", (Batch, STATE_HEADS, second_order, VALUE)),
            StateSpec("denominator_0", (Batch, STATE_HEADS, unit)),
            StateSpec("denominator_1", (Batch, STATE_HEADS, KEY)),
            StateSpec("denominator_2", (Batch, STATE_HEADS, second_order)),
        ),
        transition=(
            StateTransition(
                "numerator_0",
                injections=(
                    ProductInjection((ProductFactor(Input("value"), (VALUE,)),)),
                ),
            ),
            StateTransition(
                "numerator_1",
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(Input("key"), (KEY,)),
                            ProductFactor(Input("value"), (VALUE,)),
                        )
                    ),
                ),
            ),
            StateTransition(
                "numerator_2",
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(key_pair, (second_order,)),
                            ProductFactor(Input("value"), (VALUE,)),
                        )
                    ),
                ),
            ),
            StateTransition("denominator_0", injections=(zero_order_injection,)),
            StateTransition(
                "denominator_1",
                injections=(ProductInjection((ProductFactor(Input("key"), (KEY,)),)),),
            ),
            StateTransition(
                "denominator_2",
                injections=(
                    ProductInjection((ProductFactor(key_pair, (second_order,)),)),
                ),
            ),
        ),
        readouts=(
            Readout(
                "output",
                (
                    StateTerm("numerator_0", Constant(1.0), None)
                    + StateTerm("numerator_1", scaled_query, KEY)
                    + StateTerm("numerator_2", query_pair, second_order) * 0.5
                )
                / (
                    StateTerm("denominator_0", Constant(1.0), None)
                    + StateTerm("denominator_1", scaled_query, KEY)
                    + StateTerm("denominator_2", query_pair, second_order) * 0.5
                    + Constant(1e-6)
                ),
                dtype,
            ),
        ),
        head_mapping=HeadMapping(
            {
                QUERY_HEADS: STATE_HEADS,
                KEY_HEADS: STATE_HEADS,
                VALUE_HEADS: STATE_HEADS,
                STATE_HEADS: STATE_HEADS,
            }
        ),
    )
    return StatefulOperator(algorithm)
