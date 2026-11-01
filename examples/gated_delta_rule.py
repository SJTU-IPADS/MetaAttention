from __future__ import annotations

import torch

from attn_engine import (
    Batch,
    FeatureRole,
    HeadRole,
    Sequence,
    AlgorithmIR,
    AxisScale,
    Exp,
    HeadMapping,
    Input,
    ProductFactor,
    ProductInjection,
    RankOnePropagation,
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


def gated_delta_rule(
    B: int,
    H: int,
    S: int,
    D: int,
    DV: int,
    *,
    HQ: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Build the GDN acceptance profile from generic Stateful Operator nodes."""
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
            TensorInput("gate", (Batch, STATE_HEADS, Sequence), torch.float32),
            TensorInput("beta", (Batch, STATE_HEADS, Sequence), torch.float32),
        ),
        states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            propagations=(
                AxisScale(Exp(Input("gate"))),
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
                "output", "memory", Input("query") * (D**-0.5), KEY, dtype
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


def compositional_pressure_profile(
    B: int = 2,
    H: int = 4,
    S: int = 129,
    D: int = 64,
    DV: int = 128,
    *,
    HQ: int = 2,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Model-independent composition stress profile for generic affine lowering."""
    source_heads = HeadRole("source_heads")
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("auxiliary_query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("left", (Batch, source_heads, Sequence, KEY), dtype),
            TensorInput("right", (Batch, source_heads, Sequence, KEY), dtype),
            TensorInput(
                "left_gate", (Batch, STATE_HEADS, Sequence, KEY), torch.float32
            ),
            TensorInput(
                "right_gate", (Batch, STATE_HEADS, Sequence, VALUE), torch.float32
            ),
            TensorInput("coefficient", (Batch, STATE_HEADS, Sequence), torch.float32),
            TensorInput("value", (Batch, source_heads, Sequence, VALUE), dtype),
        ),
        states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            propagations=(
                AxisScale(Exp(Input("left_gate")), (KEY,)),
                RankOnePropagation(
                    KEY, Input("left"), Input("right"), Input("coefficient")
                ),
                AxisScale(Exp(Input("right_gate")), (VALUE,)),
            ),
            injections=(
                ProductInjection(
                    (
                        ProductFactor(Input("left"), (KEY,)),
                        ProductFactor(Input("value"), (VALUE,)),
                    )
                ),
                ProductInjection(
                    (
                        ProductFactor(Input("right"), (KEY,)),
                        ProductFactor(Input("value") * Input("coefficient"), (VALUE,)),
                    )
                ),
            ),
        ),
        readouts=(
            StateContraction("output", "memory", Input("query"), KEY, dtype),
            StateContraction(
                "auxiliary", "memory", Input("auxiliary_query"), KEY, dtype
            ),
        ),
        head_mapping=HeadMapping({QUERY_HEADS: STATE_HEADS, source_heads: STATE_HEADS}),
    )
    return StatefulOperator(algorithm)


if __name__ == "__main__":
    device = torch.device("cuda")
    batch, query_heads, state_heads, length, dim = 1, 1, 2, 64, 128
    operator = gated_delta_rule(batch, state_heads, length, dim, dim, HQ=query_heads)
    query = torch.randn(
        batch, query_heads, length, dim, device=device, dtype=torch.bfloat16
    )
    key = torch.randn_like(query)
    value = torch.randn(
        batch, state_heads, length, dim, device=device, dtype=torch.bfloat16
    )
    gate = -torch.rand(batch, state_heads, length, device=device)
    beta = torch.rand(batch, state_heads, length, device=device)
    from attn_engine import Target

    compiled = operator.compile(
        role_sizes={
            QUERY_HEADS: query_heads,
            KEY_HEADS: query_heads,
            VALUE_HEADS: state_heads,
            STATE_HEADS: state_heads,
            KEY: dim,
            VALUE: dim,
        },
        target=Target(),
        mode="dense",
    )
    result = compiled(query=query, key=key, value=value, gate=gate, beta=beta)
    print(f"output={tuple(result.outputs['output'].shape)}")
