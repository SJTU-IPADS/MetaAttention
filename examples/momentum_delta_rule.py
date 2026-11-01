"""Momentum Delta Rule's coupled affine matrix-State update.

Follows FLA recurrent_momentum_delta_rule_ref's sign convention:
M_t = exp(log_mu) M_prior + eta k (p^T S_prior - v)^T
S_t = exp(log_alpha) S_prior - beta M_t
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
    StateInjection,
    StateReference,
    StateSpec,
    StateTerm,
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


def momentum_delta_rule(
    batch,
    heads,
    seqlen,
    dim,
    dim_value,
    *,
    dtype=torch.bfloat16,
    use_p_times_alpha=False,
):
    """Head-first inputs; scalar FP32 log gates, beta and eta; FP32 States.

    ``prediction_key`` is p. Set ``use_p_times_alpha=True`` to predict from the
    decayed memory, matching that option in FLA's public recurrent operator.
    Public State order is (memory, momentum); FLA's stacked cache is (S, M).
    Query scale is 1/sqrt(dim), as in FLA's default reference convention.
    """
    alpha = Exp(Input("log_alpha"))
    prediction = Input("prediction_key")
    if use_p_times_alpha:
        prediction = alpha * prediction
    return StatefulOperator(
        AlgorithmIR(
            inputs=(
                TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
                TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
                TensorInput("prediction_key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
                TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
                *(
                    TensorInput(name, (Batch, STATE_HEADS, Sequence), torch.float32)
                    for name in ("log_alpha", "log_mu", "beta", "eta")
                ),
            ),
            states=(
                StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),
                StateSpec("momentum", (Batch, STATE_HEADS, KEY, VALUE)),
            ),
            # Author order need not be topological. The current reference below
            # requires momentum to complete before memory can use its value.
            transition=(
                StateTransition(
                    "memory",
                    steps=(
                        AxisScale(alpha),
                        StateInjection(StateReference("momentum"), -Input("beta")),
                    ),
                ),
                StateTransition(
                    "momentum",
                    steps=(
                        AxisScale(Exp(Input("log_mu"))),
                        StateInjection(
                            StateReference("memory", prior=True),
                            Input("eta"),
                            role=KEY,
                            left=Input("key"),
                            right=prediction,
                        ),
                        ProductInjection(
                            (
                                ProductFactor(Input("key"), (KEY,)),
                                ProductFactor(-Input("eta") * Input("value"), (VALUE,)),
                            )
                        ),
                    ),
                ),
            ),
            readouts=(
                Readout(
                    "output",
                    StateTerm("memory", Input("query") * dim**-0.5, KEY),
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
