"""Gated Slot Attention expressed with the Stateful Operator DSL.

Two gated States are joined by a softmax over the slot axis. ABC's extra beta
Injection is out of scope.
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
    Readout,
    Sequence,
    Softmax,
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


def gated_slot_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_slot: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """FLA ``chunk_gsa``: two gated States joined by a slot softmax.

    ``hk [K, M]`` accumulates ``k (x) s`` and ``hv [M, V]`` accumulates ``s (x) v``,
    each decayed by the supplied per-slot gate. The readout contracts ``hk``,
    softmaxes the slot axis, and contracts ``hv`` with the result, which is how
    the library composes its two States.
    """
    slot = FeatureRole("slot_dim")
    slot_decay = Exp(Input("gate"))
    scaled_query = Input("query") * (dim**-0.5)
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("slot", (Batch, KEY_HEADS, Sequence, slot), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
            TensorInput("gate", (Batch, STATE_HEADS, Sequence, slot), dtype),
        ),
        states=(
            StateSpec("key_slot", (Batch, STATE_HEADS, KEY, slot)),
            StateSpec("slot_value", (Batch, STATE_HEADS, slot, VALUE)),
        ),
        transition=(
            StateTransition(
                "key_slot",
                propagations=(AxisScale(slot_decay, (slot,)),),
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(Input("key"), (KEY,)),
                            ProductFactor(Input("slot"), (slot,)),
                        )
                    ),
                ),
            ),
            StateTransition(
                "slot_value",
                propagations=(AxisScale(slot_decay, (slot,)),),
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(Input("slot"), (slot,)),
                            ProductFactor(Input("value"), (VALUE,)),
                        )
                    ),
                ),
            ),
        ),
        readouts=(
            Readout(
                "output",
                StateTerm(
                    "slot_value",
                    Softmax(StateTerm("key_slot", scaled_query, KEY), slot),
                    slot,
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
