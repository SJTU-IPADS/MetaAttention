"""HGRN expressed with the Stateful Operator DSL.

HGRN is a per-Feature vector State whose readout is the State itself, with the
per-Feature log decay supplied as an input.
"""

from __future__ import annotations

import torch

from attn_engine import (
    AlgorithmIR,
    AxisScale,
    Batch,
    Constant,
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

VALUE = FeatureRole("value_dim")
VALUE_HEADS = HeadRole("value_heads")
STATE_HEADS = HeadRole("state_heads")


def hgrn(
    batch: int,
    heads: int,
    seqlen: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Build the per-Feature vector-State recurrence.

    HGRN has no head axis, so callers pass ``heads=1`` and supply the headless
    ``[B, T, D]`` tensors as ``[B, 1, T, D]``; ``gate`` is the per-Feature log
    decay.
    """
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
            TensorInput("gate", (Batch, STATE_HEADS, Sequence, VALUE), torch.float32),
        ),
        states=(StateSpec("hidden", (Batch, STATE_HEADS, VALUE)),),
        transition=StateTransition(
            "hidden",
            propagations=(AxisScale(Exp(Input("gate")), (VALUE,)),),
            injections=(ProductInjection((ProductFactor(Input("value"), (VALUE,)),)),),
        ),
        readouts=(Readout("output", StateTerm("hidden", Constant(1.0), None), dtype),),
        head_mapping=HeadMapping({VALUE_HEADS: STATE_HEADS, STATE_HEADS: STATE_HEADS}),
    )
    return StatefulOperator(algorithm)
