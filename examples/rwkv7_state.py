"""RWKV-7 DPLR State algebra expressed with the Stateful Operator DSL.

``chunk_dplr_delta_rule`` advances ``S <- diag(exp(w)) S - beta k (a^T S) + beta k v^T``;
RWKV-7's channel mixing and token shift belong to its layer, so both arrive as
inputs."""

from __future__ import annotations

import torch

from attn_engine import (
    AlgorithmIR,
    AxisScale,
    Batch,
    Exp,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    ProductFactor,
    ProductInjection,
    RankOnePropagation,
    Sequence,
    StateContraction,
    StatefulOperator,
    StateSpec,
    StateTransition,
    TensorInput,
)

KEY = FeatureRole("key_dim")
VALUE = FeatureRole("value_dim")
QUERY_HEADS = HeadRole("query_heads")
KEY_HEADS = HeadRole("key_heads")
VALUE_HEADS = HeadRole("value_heads")
STATE_HEADS = HeadRole("state_heads")


def rwkv7_state(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """RWKV-7 / DPLR state algebra from ``chunk_dplr_delta_rule``.

    The transition is ``S <- diag(exp(w)) S - beta k (a^T S) + beta k v^T``.
    RWKV-7's channel mixing and token shift belong to its layer rather than to
    this recurrence, so both are supplied as inputs.
    """
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
            TensorInput("removal", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("decay", (Batch, STATE_HEADS, Sequence, KEY), torch.float32),
            TensorInput("beta", (Batch, STATE_HEADS, Sequence), torch.float32),
        ),
        states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            propagations=(
                AxisScale(Exp(Input("decay")), (KEY,)),
                RankOnePropagation(KEY, Input("key"), Input("removal"), -Input("beta")),
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
        readouts=(StateContraction("output", "memory", Input("query"), KEY, dtype),),
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
