"""ABC with input-only affine accumulation and normalized slot readout.

Each slot stores weighted key/value sums and a scalar weight sum. Normalization
happens at readout, so no State-dependent gate is required. This implements the
vanilla ABC formula in FLA's naive_cumsum_abc, using raw accumulators as its
continuation State rather than FLA's normalized key/value State tuple.
"""

from __future__ import annotations

import torch

from attn_engine import (
    AlgorithmIR,
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
    StateSpec,
    StateTerm,
    StateTransition,
    StatefulOperator,
    TensorInput,
)

KEY = FeatureRole("key_dim")
VALUE = FeatureRole("value_dim")
SLOT = FeatureRole("slot_dim")
QUERY_HEADS = HeadRole("query_heads")
KEY_HEADS = HeadRole("key_heads")
VALUE_HEADS = HeadRole("value_heads")
STATE_HEADS = HeadRole("state_heads")


def abc_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_slot: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Accumulate exp(slot_logits) weights, normalize per slot, then softmax.

    Inputs use head-first [B, H, T, ...] layout. At token t, key_slot is the
    sum of k_i exp(s_i)^T, slot_value is the sum of exp(s_i) v_i^T, and
    normalizer is the sum of exp(s_i), for i <= t. The output mixes normalized
    slot values using softmax(q^T key_slot / (sqrt(K) * normalizer)).

    Supply logits whose exponentials and cumulative sums remain finite and
    positive in FP32; this raw-accumulator example does not implement FLA's
    logcumsumexp stabilization. Continuation carries all three raw sums.
    """
    weight = Exp(Input("slot_logits"))
    normalizer = StateTerm("normalizer", 1, None)
    slot_scores = StateTerm("key_slot", Input("query") * (dim**-0.5), KEY) / normalizer
    return StatefulOperator(
        AlgorithmIR(
            inputs=(
                TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
                TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
                TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
                TensorInput(
                    "slot_logits", (Batch, KEY_HEADS, Sequence, SLOT), torch.float32
                ),
            ),
            states=(
                StateSpec("key_slot", (Batch, STATE_HEADS, KEY, SLOT)),
                StateSpec("slot_value", (Batch, STATE_HEADS, SLOT, VALUE)),
                StateSpec("normalizer", (Batch, STATE_HEADS, SLOT)),
            ),
            transition=(
                StateTransition(
                    "key_slot",
                    injections=(
                        ProductInjection(
                            (
                                ProductFactor(Input("key"), (KEY,)),
                                ProductFactor(weight, (SLOT,)),
                            )
                        ),
                    ),
                ),
                StateTransition(
                    "slot_value",
                    injections=(
                        ProductInjection(
                            (
                                ProductFactor(weight, (SLOT,)),
                                ProductFactor(Input("value"), (VALUE,)),
                            )
                        ),
                    ),
                ),
                StateTransition(
                    "normalizer",
                    injections=(ProductInjection((ProductFactor(weight, (SLOT,)),)),),
                ),
            ),
            readouts=(
                Readout(
                    "output",
                    StateTerm(
                        "slot_value", Softmax(slot_scores, SLOT) / normalizer, SLOT
                    ),
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
