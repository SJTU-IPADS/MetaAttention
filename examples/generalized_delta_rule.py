"""IPLR and DPLR recurrences from FLA's generalized_delta_rule references.

With state layout [K, V], both use a KEY-axis low-rank correction
beta (alpha^T S). DPLR adds a diagonal decay of the same prior State.
All inputs use head-first [B, H, T, ...] layout.
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


def _operator(dim: int, dtype: torch.dtype, *, diagonal: bool) -> StatefulOperator:
    inputs = [
        TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
        TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
        TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
        TensorInput("alpha", (Batch, KEY_HEADS, Sequence, KEY), dtype),
        TensorInput("beta", (Batch, KEY_HEADS, Sequence, KEY), dtype),
    ]
    if diagonal:
        inputs.append(
            TensorInput("gate", (Batch, STATE_HEADS, Sequence, KEY), torch.float32)
        )
        # FLA applies both terms to the prior State: (D + beta alpha^T) S.
        # (I + beta (D^-1 alpha)^T) D S has exactly the same action.
        propagations = (
            AxisScale(Exp(Input("gate")), (KEY,)),
            RankOnePropagation(
                KEY, Input("beta"), Input("alpha") * Exp(-Input("gate"))
            ),
        )
    else:
        propagations = (RankOnePropagation(KEY, Input("beta"), Input("alpha")),)
    return StatefulOperator(
        AlgorithmIR(
            inputs=inputs,
            states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
            transition=StateTransition(
                "memory",
                propagations=propagations,
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
                    StateTerm("memory", Input("query") * (dim**-0.5), KEY),
                    dtype,
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
    )


def iplr_delta_rule(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """S <- S + beta (alpha^T S) + k v^T; o = q^T S / sqrt(K).

    Alpha and beta are independent KEY vectors, not scalar learning rates.
    """
    return _operator(dim, dtype, diagonal=False)


def dplr_delta_rule(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """S <- diag(exp(gate)) S + beta (alpha^T S) + k v^T.

    Readout is q^T S / sqrt(K). Supply finite log gates for which both exp(gate)
    and exp(-gate) are representable in FP32; the factorization uses the inverse
    diagonal to keep the low-rank correction on the prior State.
    """
    return _operator(dim, dtype, diagonal=True)
