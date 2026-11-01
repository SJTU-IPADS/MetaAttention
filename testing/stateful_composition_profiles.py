from __future__ import annotations

from dataclasses import dataclass

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
    StateContraction,
    StateSpec,
    StateTransition,
    StatefulOperator,
    TensorInput,
)


@dataclass(frozen=True)
class CompositionProfile:
    name: str
    operator: StatefulOperator
    role_sizes: dict[HeadRole | FeatureRole, int]


def make_profile(kind: str, *, renamed: bool = False) -> CompositionProfile:
    if kind not in {"axis", "mixed", "injections", "rank-one"}:
        raise ValueError(f"unknown composition profile {kind!r}")
    tag = "renamed" if renamed else kind.replace("-", "_")
    row = FeatureRole(f"{tag}_row")
    column = FeatureRole(f"{tag}_column")
    query_heads = HeadRole(f"{tag}_query_heads")
    source_heads = HeadRole(f"{tag}_source_heads")
    state_heads = HeadRole(f"{tag}_state_heads")
    names = {
        "query": "probe" if renamed else "query",
        "auxiliary": "probe_aux" if renamed else "auxiliary_query",
        "left": "factor_l" if renamed else "left",
        "right": "factor_r" if renamed else "right",
        "value": "payload" if renamed else "value",
        "row_gate": "scale_l" if renamed else "row_gate",
        "column_gate": "scale_r" if renamed else "column_gate",
        "coefficient": "weight" if renamed else "coefficient",
    }
    inputs = [
        TensorInput(
            names["query"], (Batch, query_heads, Sequence, row), torch.bfloat16
        ),
    ]
    if kind != "axis":
        inputs.append(
            TensorInput(
                names["auxiliary"], (Batch, query_heads, Sequence, row), torch.bfloat16
            )
        )
    inputs.extend(
        [
            TensorInput(
                names["left"], (Batch, source_heads, Sequence, row), torch.bfloat16
            ),
            TensorInput(
                names["value"], (Batch, source_heads, Sequence, column), torch.bfloat16
            ),
        ]
    )
    if kind in {"axis", "mixed"}:
        inputs.extend(
            [
                TensorInput(
                    names["row_gate"],
                    (Batch, state_heads, Sequence, row),
                    torch.float32,
                ),
                TensorInput(
                    names["column_gate"],
                    (Batch, state_heads, Sequence, column),
                    torch.float32,
                ),
            ]
        )
    if kind in {"mixed", "injections", "rank-one"}:
        inputs.extend(
            [
                TensorInput(
                    names["right"], (Batch, source_heads, Sequence, row), torch.bfloat16
                ),
                TensorInput(
                    names["coefficient"], (Batch, state_heads, Sequence), torch.float32
                ),
            ]
        )
    state = StateSpec("state", (Batch, state_heads, row, column))
    injections = [
        ProductInjection(
            (
                ProductFactor(Input(names["left"]), (row,)),
                ProductFactor(Input(names["value"]), (column,)),
            )
        )
    ]
    propagations = []
    if kind == "axis":
        propagations = [
            AxisScale(Exp(Input(names["row_gate"])), (row,)),
            AxisScale(Exp(Input(names["column_gate"])), (column,)),
        ]
    elif kind == "mixed":
        propagations = [
            AxisScale(Exp(Input(names["row_gate"])), (row,)),
            RankOnePropagation(
                row,
                Input(names["left"]),
                Input(names["right"]),
                Input(names["coefficient"]),
            ),
            AxisScale(Exp(Input(names["column_gate"])), (column,)),
        ]
        injections.append(
            ProductInjection(
                (
                    ProductFactor(Input(names["right"]), (row,)),
                    ProductFactor(
                        Input(names["value"]) * Input(names["coefficient"]), (column,)
                    ),
                )
            )
        )
    elif kind == "injections":
        injections.append(
            ProductInjection(
                (
                    ProductFactor(Input(names["right"]), (row,)),
                    ProductFactor(
                        Input(names["value"]) * Input(names["coefficient"]), (column,)
                    ),
                )
            )
        )
    elif kind == "rank-one":
        propagations = [
            RankOnePropagation(
                row,
                Input(names["left"]),
                Input(names["right"]),
                Input(names["coefficient"]),
            )
        ]
    readouts = [
        StateContraction("output", "state", Input(names["query"]), row, torch.bfloat16)
    ]
    if kind != "axis":
        readouts.append(
            StateContraction(
                "auxiliary", "state", Input(names["auxiliary"]), row, torch.bfloat16
            )
        )
    algorithm = AlgorithmIR(
        inputs=tuple(inputs),
        states=(state,),
        transition=StateTransition(
            "state", propagations=tuple(propagations), injections=tuple(injections)
        ),
        readouts=tuple(readouts),
        head_mapping=HeadMapping({query_heads: state_heads, source_heads: state_heads}),
    )
    return CompositionProfile(
        kind,
        StatefulOperator(algorithm),
        {query_heads: 2, source_heads: 2, state_heads: 4, row: 64, column: 128},
    )


def make_inputs(
    profile: CompositionProfile, *, batch: int, length: int, device: torch.device
):
    generator = torch.Generator(device=device).manual_seed(1701 + length)
    values = {}
    for spec in profile.operator.algorithm.inputs:
        shape = tuple(
            batch
            if role == Batch
            else length
            if role == Sequence
            else profile.role_sizes[role]
            for role in spec.roles
        )
        if spec.dtype == torch.float32:
            tensor = 0.01 * torch.randn(
                shape, device=device, dtype=spec.dtype, generator=generator
            )
        else:
            tensor = 0.05 * torch.randn(
                shape, device=device, dtype=spec.dtype, generator=generator
            )
        values[spec.name] = tensor
    return values
