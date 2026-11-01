"""ReBased product-feature recurrence expressed with the Stateful Operator DSL.

ReBased keeps only the squared feature map, so one product-Feature numerator
State and one matching denominator State suffice.
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


def rebased_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    eps: float = 1e-5,
) -> StatefulOperator:
    """FLA ``parallel_rebased``: squared attention with a normalizer.

    ``attn = (scale*q . k) ** 2`` keeps only the second-order feature map, so the
    Program declares a ``[K*K, V]`` numerator State and a ``[K*K]`` denominator
    State and divides by the library's configurable epsilon.
    """
    scale = dim**-0.5
    scaled_query = Input("query") * scale
    second_order = ProductFeatureRole(KEY, KEY)
    query_pair = Outer(scaled_query, scaled_query, second_order)
    key_pair = Outer(Input("key"), Input("key"), second_order)
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
        ),
        states=(
            StateSpec("numerator", (Batch, STATE_HEADS, second_order, VALUE)),
            StateSpec("denominator", (Batch, STATE_HEADS, second_order)),
        ),
        transition=(
            StateTransition(
                "numerator",
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(key_pair, (second_order,)),
                            ProductFactor(Input("value"), (VALUE,)),
                        )
                    ),
                ),
            ),
            StateTransition(
                "denominator",
                injections=(
                    ProductInjection((ProductFactor(key_pair, (second_order,)),)),
                ),
            ),
        ),
        readouts=(
            Readout(
                "output",
                StateTerm("numerator", query_pair, second_order)
                / (StateTerm("denominator", query_pair, second_order) + Constant(eps)),
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
