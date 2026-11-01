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


def gated_retention(B, H, S, D, DV, dtype=torch.bfloat16):
    """GLA acceptance profile expressed only with generic IR nodes."""
    algorithm = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
            TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
            TensorInput("gate", (Batch, STATE_HEADS, Sequence), torch.float32),
        ),
        states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            propagations=(AxisScale(Exp(Input("gate"))),),
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
            StateContraction(
                "output", "memory", Input("query") * (D**-0.5), KEY, dtype
            ),
        ),
        head_mapping=HeadMapping(
            {QUERY_HEADS: STATE_HEADS, KEY_HEADS: STATE_HEADS, VALUE_HEADS: STATE_HEADS}
        ),
    )
    return StatefulOperator(algorithm)
