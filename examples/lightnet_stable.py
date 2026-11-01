"""Stable LightNet as a LogAddExp scan followed by a diagonal affine scan.

This is the FLA ``ops.lightnet`` contract, not the surrounding neural layer:
queries/values are already projected (and queries already activated, if desired).
There is no key-centering operation: raw logits enter a FP32 log-normalizer.
The layer's SiLU query activation, short convolutions and output norm/gate are
outside this attention operator. Default query scale is K**-0.5.

z_t = logaddexp(z_{t-1}, logits_t), z_0 = -inf
k_t = exp(logits_t-z_t)
g_t = z_{t-1}-z_t, or 0 for an empty prior
M_t = exp(g_t)[:,None] * M_{t-1} + k_t[:,None] * v_t[None,:]
o_t = (scale*q_t) @ M_t

FLA's zero first gate preserves a supplied nonzero memory when z_0=-inf;
finite z_0 instead decays that memory by exp(z_0-z_1). Continuations must
carry both States. States use z:[B,H,K] and memory:[B,H,K,V], unlike FLA's
extra singleton time dimension on z. Outputs retain the engine's B,H,T,V
layout (packed T,H,V). The gate stage exposes FP32 trajectories deliberately,
avoiding BF16 rounding before the matrix recurrence.
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
    TensorInput,
)
from attn_engine.stateful.logaddexp import LogAddExpScanIR
from attn_engine.stateful.stages import (
    ScanStage,
    ScanStageIR,
    StageBinding,
    StageOutput,
    StagedStatefulOperator,
)

KEY = FeatureRole("key_dim")
VALUE = FeatureRole("value_dim")
HEADS = HeadRole("heads")


def lightnet_stable(
    batch: int,
    heads: int,
    seqlen: int,
    dim: int,
    dim_value: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    scale: float | None = None,
) -> StagedStatefulOperator:
    """Build the staged attention; runtime Batch/Sequence remain dynamic."""
    scale = dim**-0.5 if scale is None else scale
    normalizer = LogAddExpScanIR(
        TensorInput("logits", (Batch, HEADS, Sequence, KEY), dtype),
        StateSpec("z", (Batch, HEADS, KEY)),
    )
    memory = AlgorithmIR(
        inputs=(
            TensorInput("query", (Batch, HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, HEADS, Sequence, VALUE), dtype),
            TensorInput("normalized_key", (Batch, HEADS, Sequence, KEY), torch.float32),
            TensorInput("log_gate", (Batch, HEADS, Sequence, KEY), torch.float32),
        ),
        states=(StateSpec("memory", (Batch, HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            propagations=(AxisScale(Exp(Input("log_gate")), (KEY,)),),
            injections=(
                ProductInjection(
                    (
                        ProductFactor(Input("normalized_key"), (KEY,)),
                        ProductFactor(Input("value"), (VALUE,)),
                    )
                ),
            ),
        ),
        readouts=(
            StateContraction("output", "memory", Input("query") * scale, KEY, dtype),
        ),
        head_mapping=HeadMapping({HEADS: HEADS}),
    )
    return StagedStatefulOperator(
        ScanStageIR(
            (
                ScanStage("normalizer", normalizer),
                ScanStage(
                    "memory_scan",
                    memory,
                    (
                        StageBinding("normalized_key", "normalizer", "normalized_key"),
                        StageBinding("log_gate", "normalizer", "log_gate"),
                    ),
                ),
            ),
            (StageOutput("output", "memory_scan", "output"),),
        )
    )
