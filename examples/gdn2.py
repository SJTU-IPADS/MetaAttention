"""Gated DeltaNet 2 with independent channel-wise erase and write gates.

Inputs use head-first [B, H, T, ...] layout. Gate is a natural-log KEY-channel
decay; erase and write are already activated KEY- and VALUE-channel gates.
"""

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
    Readout,
    StateTerm,
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


def gdn2(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """S <- (I - k (erase * k)^T) diag(exp(gate)) S + k (write * v)^T.

    Readout is q^T S / sqrt(K). Scalar erase=write=beta recovers KDA.
    """
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
            TensorInput("gate", (Batch, STATE_HEADS, Sequence, KEY), torch.float32),
            TensorInput("erase", (Batch, STATE_HEADS, Sequence, KEY), torch.float32),
            TensorInput("write", (Batch, STATE_HEADS, Sequence, VALUE), torch.float32),
        ),
        states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            propagations=(
                AxisScale(Exp(Input("gate")), (KEY,)),
                RankOnePropagation(
                    KEY, Input("key"), Input("erase") * Input("key"), -1
                ),
            ),
            injections=(
                ProductInjection(
                    (
                        ProductFactor(Input("key"), (KEY,)),
                        ProductFactor(Input("value") * Input("write"), (VALUE,)),
                    )
                ),
            ),
        ),
        readouts=(
            Readout(
                "output", StateTerm("memory", Input("query") * (dim**-0.5), KEY), dtype
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
