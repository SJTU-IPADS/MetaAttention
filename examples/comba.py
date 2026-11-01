"""ComBa with separate prediction and write keys.

Inputs use head-first [B, H, T, ...] layout. Gate is scalar natural-log decay,
and prediction_key supplies the projection used to read the write residual.
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


def comba(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """S' = exp(gate) S; S <- S' + beta k (v - p^T S')^T.

    Readout is q^T S / sqrt(K), with p supplied as prediction_key.
    """
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
            TensorInput("prediction_key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("gate", (Batch, STATE_HEADS, Sequence), torch.float32),
            TensorInput("beta", (Batch, STATE_HEADS, Sequence), torch.float32),
        ),
        states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            propagations=(
                AxisScale(Exp(Input("gate"))),
                RankOnePropagation(
                    KEY, Input("key"), Input("prediction_key"), -Input("beta")
                ),
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
