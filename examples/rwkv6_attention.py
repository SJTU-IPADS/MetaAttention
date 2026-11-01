"""RWKV6 recurrence expressed with the Stateful Operator DSL.

The recurrence decays the State per Key Feature, keeps a bonus skip for the
current token, and reads the State from before the current transition.
"""

from __future__ import annotations

import torch

from attn_engine import (
    AlgorithmIR,
    AxisScale,
    Batch,
    Constant,
    Contract,
    Exp,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    ProductFactor,
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


def rwkv6_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Build the RWKV6 decay, bonus-skip, prior-State recurrence.

    The readout uses a prior-State term and a Feature-axis contraction of
    ``scale * q * u * k``; ``w`` is the log decay.
    """
    scaled_query = Input("query") * (dim**-0.5)
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
            TensorInput("decay", (Batch, STATE_HEADS, Sequence, KEY), torch.float32),
            TensorInput("bonus", (Batch, STATE_HEADS, Sequence, KEY), dtype),
        ),
        states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            propagations=(AxisScale(Exp(Input("decay")), (KEY,)),),
            injections=(
                ProductInjection(
                    (
                        ProductFactor(Input("key"), (KEY,)),
                        ProductFactor(Input("value"), (VALUE,)),
                    )
                ),
            ),
        ),
        readouts=(
            Readout(
                "output",
                StateTerm("memory", scaled_query, KEY, prior=True)
                + Contract(
                    scaled_query * Input("bonus") * Input("key"),
                    Constant(1.0),
                    KEY,
                )
                * Input("value"),
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
