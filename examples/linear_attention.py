"""Linear recurrent attention families expressed with the Stateful Operator DSL.

All factories use the head-first ``[B, H, T, ...]`` runtime layout expected by
``StatefulOperator``. Their mathematical state is a ``[K, V]`` matrix:

``H_t = P_t(H_{t-1}) + k_t v_t^T`` and ``o_t = q_t H_t / sqrt(K)``.

The module also contains scalar-gated, feature-gated, lightning, forgetting,
delta-rule, and explicit recurrent variants. Normalized linear attention uses
the separate two-State Program in :mod:`examples.normalized_linear_attention`.
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


def _operator(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    kv_heads: int,
    dtype: torch.dtype,
    gate: str | None,
) -> StatefulOperator:
    inputs = [
        TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
        TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
        TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
    ]
    if gate == "scalar":
        inputs.append(
            TensorInput("gate", (Batch, VALUE_HEADS, Sequence), torch.float32)
        )
    elif gate == "feature":
        inputs.append(
            TensorInput("gate", (Batch, VALUE_HEADS, Sequence, KEY), torch.float32)
        )
    elif gate == "dual":
        inputs.append(
            TensorInput("key_gate", (Batch, VALUE_HEADS, Sequence, KEY), torch.float32)
        )
        inputs.append(
            TensorInput(
                "value_gate", (Batch, VALUE_HEADS, Sequence, VALUE), torch.float32
            )
        )
    elif gate is not None:
        raise ValueError(f"unknown gate kind: {gate!r}")

    if gate == "dual":
        propagations = (
            AxisScale(Exp(Input("key_gate")), (KEY,)),
            AxisScale(Exp(Input("value_gate")), (VALUE,)),
        )
    else:
        propagation_roles = (KEY,) if gate == "feature" else ()
        propagations = (
            (AxisScale(Exp(Input("gate")), propagation_roles),)
            if gate in {"scalar", "feature"}
            else ()
        )
    algorithm = AlgorithmIR(
        inputs=tuple(inputs),
        states=(StateSpec("memory", (Batch, VALUE_HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            propagations=propagations,
            injections=(
                ProductInjection(
                    (
                        ProductFactor(Input("key") + 0.1, (KEY,)),
                        ProductFactor(Input("value"), (VALUE,)),
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
                QUERY_HEADS: VALUE_HEADS,
                KEY_HEADS: VALUE_HEADS,
                VALUE_HEADS: VALUE_HEADS,
            }
        ),
    )
    return StatefulOperator(algorithm)


def _validate_kv_heads(heads: int, kv_heads: int) -> None:
    if kv_heads <= 0 or heads % kv_heads:
        raise ValueError("heads must be a positive multiple of kv_heads")


def linear_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Build the ungated matrix-State recurrence."""
    kv_heads = heads if kv_heads is None else kv_heads
    _validate_kv_heads(heads, kv_heads)
    return _operator(
        batch, heads, seqlen, dim, dim_value, kv_heads=kv_heads, dtype=dtype, gate=None
    )


def simple_gla_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Build the scalar-gated matrix-State recurrence."""
    kv_heads = heads if kv_heads is None else kv_heads
    _validate_kv_heads(heads, kv_heads)
    return _operator(
        batch,
        heads,
        seqlen,
        dim,
        dim_value,
        kv_heads=kv_heads,
        dtype=dtype,
        gate="scalar",
    )


def gla_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Build the feature-gated matrix-State recurrence."""
    kv_heads = heads if kv_heads is None else kv_heads
    _validate_kv_heads(heads, kv_heads)
    return _operator(
        batch,
        heads,
        seqlen,
        dim,
        dim_value,
        kv_heads=kv_heads,
        dtype=dtype,
        gate="feature",
    )


def dual_gate_gla_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Build a matrix-State recurrence with a gate on each State axis.

    One feature-wise log gate decays the key axis and one decays the value axis.
    """
    kv_heads = heads if kv_heads is None else kv_heads
    _validate_kv_heads(heads, kv_heads)
    return _operator(
        batch,
        heads,
        seqlen,
        dim,
        dim_value,
        kv_heads=kv_heads,
        dtype=dtype,
        gate="dual",
    )


def lightning_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Build the Lightning Attention scalar-gate recurrence.

    The caller supplies the scalar log gate generated from ``layer_idx`` and
    ``num_layers``.
    """
    return simple_gla_attention(
        batch, heads, seqlen, dim, dim_value, kv_heads=kv_heads, dtype=dtype
    )


def forgetting_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Build the forgetting-attention scalar log-decay recurrence."""
    return simple_gla_attention(
        batch, heads, seqlen, dim, dim_value, kv_heads=kv_heads, dtype=dtype
    )


def delta_rule_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Build Delta Rule: S <- S + beta k (v - k^T S)^T.

    This compatibility entry point delegates to the self-contained Delta Rule
    example. Invocation requires a scalar FP32 ``beta`` input per head/token.
    """
    from examples.delta_rule import delta_rule_attention as build_delta_rule

    return build_delta_rule(
        batch, heads, seqlen, dim, dim_value, kv_heads=kv_heads, dtype=dtype
    )


def recurrent_linear_attention(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> StatefulOperator:
    """Build the explicit recurrent form of un-gated linear attention."""
    return linear_attention(
        batch, heads, seqlen, dim, dim_value, kv_heads=kv_heads, dtype=dtype
    )
