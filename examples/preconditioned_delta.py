"""Preconditioned KDA/GDN through two separately affine scan stages."""

from __future__ import annotations

import math
import torch

from attn_engine import (
    AlgorithmIR,
    AxisScale,
    Batch,
    Contract,
    Exp,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    ProductFactor,
    ProductInjection,
    RankOnePropagation,
    Readout,
    ScanStage,
    ScanStageIR,
    Sequence,
    StageBinding,
    StageOutput,
    StateSpec,
    StateTerm,
    StateTransition,
    StagedStatefulOperator,
    TensorInput,
)

KEY = FeatureRole("key_dim")
VALUE = FeatureRole("value_dim")
KEY_HEADS = HeadRole("key_heads")
VALUE_HEADS = HeadRole("value_heads")


def preconditioned_delta(*, dtype=torch.bfloat16, kind="kda", x=1.5, eps=1e-6):
    """Head-first Inputs; public States (atk, memory).

    KDA uses per-key Feature log_decay and original q/k. GDN uses scalar
    log_decay and L2-normalizes q/k with FLA's 1e-12 norm floor. ATK is held
    at KEY_HEADS; grouped VALUE_HEADS share the same upstream trajectory.
    Center is an explicit FP32 per-token/head input (default FLA center: -0.2).
    This factory exposes the mathematical operator, not a model layer.
    """
    if kind not in ("kda", "gdn"):
        raise ValueError("kind must be kda or gdn")
    if not math.isfinite(x) or x <= 0 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("x and eps must be finite positive numbers")
    key = Input("key").to(torch.float32)
    query = Input("query").to(torch.float32)
    if kind == "gdn":
        key = key / Contract(key, key, KEY).clamp(1e-24, 3.402823e38).sqrt()
        query = query / Contract(query, query, KEY).clamp(1e-24, 3.402823e38).sqrt()
    kspec = TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype)
    atk = StateTerm("atk", 1, None)
    r = (atk + eps).log() - Input("center")
    write_key = key * Exp(-math.log(x) * r / (1 + r.abs()))
    auxiliary = AlgorithmIR(
        inputs=(
            kspec,
            TensorInput("log_atk_decay", (Batch, KEY_HEADS, Sequence), torch.float32),
            TensorInput("beta_atk", (Batch, KEY_HEADS, Sequence), torch.float32),
            TensorInput("center", (Batch, KEY_HEADS, Sequence), torch.float32),
        ),
        states=(StateSpec("atk", (Batch, KEY_HEADS, KEY)),),
        transition=StateTransition(
            "atk",
            propagations=(AxisScale(Exp(Input("log_atk_decay"))),),
            injections=(
                ProductInjection(
                    (ProductFactor(Input("beta_atk") * key * key, (KEY,)),)
                ),
            ),
        ),
        readouts=(Readout("write_key", write_key, torch.float32),),
        head_mapping=HeadMapping(),
    )
    # Query scaling remains a compile-time role specialization supplied by the
    # author. Use an explicit external scale so no feature extent is hidden.
    q = query * Input("scale")
    beta = Input("beta")
    write = Input("write_key")
    main = AlgorithmIR(
        inputs=(
            kspec,
            TensorInput("query", (Batch, KEY_HEADS, Sequence, KEY), dtype),
            TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
            TensorInput("write_key", (Batch, KEY_HEADS, Sequence, KEY), torch.float32),
            TensorInput("beta", (Batch, VALUE_HEADS, Sequence), torch.float32),
            TensorInput("scale", (Batch, KEY_HEADS, Sequence), torch.float32),
            TensorInput(
                "log_decay",
                (Batch, VALUE_HEADS, Sequence, KEY)
                if kind == "kda"
                else (Batch, VALUE_HEADS, Sequence),
                torch.float32,
            ),
        ),
        states=(StateSpec("memory", (Batch, VALUE_HEADS, KEY, VALUE)),),
        transition=StateTransition(
            "memory",
            steps=(
                AxisScale(Exp(Input("log_decay")), (KEY,) if kind == "kda" else ()),
                RankOnePropagation(KEY, write, key, -beta),
                ProductInjection(
                    (
                        ProductFactor(write, (KEY,)),
                        ProductFactor(
                            beta * Input("value").to(torch.float32), (VALUE,)
                        ),
                    )
                ),
            ),
        ),
        readouts=(Readout("output", StateTerm("memory", q, KEY), dtype),),
        head_mapping=HeadMapping({KEY_HEADS: VALUE_HEADS}),
    )
    return StagedStatefulOperator(
        ScanStageIR(
            (
                ScanStage("preconditioner", auxiliary),
                ScanStage(
                    "memory",
                    main,
                    (StageBinding("write_key", "preconditioner", "write_key"),),
                ),
            ),
            (StageOutput("output", "memory", "output"),),
        )
    )
