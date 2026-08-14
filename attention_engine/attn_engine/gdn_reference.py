# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# The mathematical behavior follows Flash Linear Attention's Gated Delta Rule
# reference implementation, distributed under the MIT License.

from __future__ import annotations

import torch


DEFAULT_SCALE = 128**-0.5


def gated_delta_rule_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate head-first Gated Delta Rule using an FP64 recurrent oracle.

    This deliberately simple implementation is CPU-safe and differentiable. It
    defines the public operator semantics; production H20 calls use generated
    TileLang kernels instead.
    """
    if scale is None:
        scale = DEFAULT_SCALE
    batch, query_heads, length, key_dim = query.shape
    value_heads = value.shape[1]
    groups = value_heads // query_heads

    q = query.to(torch.float64)
    k = key.to(torch.float64)
    v = value.to(torch.float64)
    g = gate.to(torch.float64)
    b = beta.to(torch.float64)
    if groups != 1:
        q = q.repeat_interleave(groups, dim=1)
        k = k.repeat_interleave(groups, dim=1)

    if initial_state is None:
        state = torch.zeros(
            batch,
            value_heads,
            key_dim,
            value.shape[-1],
            dtype=torch.float64,
            device=query.device,
        )
    else:
        state = initial_state.to(torch.float64)

    outputs = []
    for index in range(length):
        state = state * torch.exp(g[:, :, index, None, None])
        prediction = torch.einsum("bhk,bhkv->bhv", k[:, :, index], state)
        residual = (v[:, :, index] - prediction) * b[:, :, index, None]
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, :, index], residual)
        outputs.append(
            torch.einsum("bhk,bhkv->bhv", q[:, :, index] * scale, state)
        )

    output = torch.stack(outputs, dim=2).to(query.dtype)
    final_state = state.to(torch.float32) if output_final_state else None
    return output, final_state
