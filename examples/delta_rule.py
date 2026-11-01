"""Delta Rule with an error-correcting rank-one write.

Inputs use head-first [B, H, T, ...] layout. Beta is a supplied scalar write
strength, and keys arrive already normalized if the caller needs normalization.
"""

from __future__ import annotations

import torch

from attn_engine import (
    AlgorithmIR,
    Batch,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    ProductFactor,
    ProductInjection,
    RankOnePropagation,
    Sequence,
    StateContraction,
    StateSpec,
    StateTransition,
    StatefulOperator,
    TensorInput,
)

KEY = FeatureRole("key_dim")
VALUE = FeatureRole("value_dim")
QUERY_HEADS = HeadRole("query_heads")
KEY_HEADS = HeadRole("key_heads")
VALUE_HEADS = HeadRole("value_heads")
STATE_HEADS = HeadRole("state_heads")


def delta_rule_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """S <- S + beta k (v - k^T S)^T; o = q^T S / sqrt(K).

    ``kv_heads`` is retained for the linear_attention compatibility entry point.
    Bind head extents at compile time; each source head count must divide the
    State head count.
    """
    kv_heads = heads if kv_heads is None else kv_heads
    if kv_heads <= 0 or heads <= 0 or heads % kv_heads:
        raise ValueError("heads must be a positive multiple of kv_heads")
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
            TensorInput("beta", (Batch, STATE_HEADS, Sequence), torch.float32),
        ),
        states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            propagations=(
                RankOnePropagation(KEY, Input("key"), Input("key"), -Input("beta")),
            ),
            injections=(
                ProductInjection(
                    (
                        ProductFactor(Input("key"), (KEY,)),
                        ProductFactor(Input("value") * Input("beta"), (VALUE,)),
                    )
                ),
            ),
        ),
        readouts=(
            StateContraction(
                "output", "memory", Input("query") * (dim**-0.5), KEY, dtype
            ),
        ),
        head_mapping=HeadMapping(
            {
                QUERY_HEADS: STATE_HEADS,
                KEY_HEADS: STATE_HEADS,
                VALUE_HEADS: STATE_HEADS,
            }
        ),
    )
    return StatefulOperator(algorithm)
