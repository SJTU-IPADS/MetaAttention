# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under the MIT License; see _flash_qla_hopper/LICENSE.
# Adapted from QwenLM/FlashQLA commit c18a4860ea9cb937f1075d606b4823d6ae34e880.

from __future__ import annotations

import torch

CHUNK_SIZE = 64


def _backend():
    from . import _flash_qla_hopper

    return _flash_qla_hopper


def _token_first(tensor: torch.Tensor) -> torch.Tensor:
    shape = (tensor.shape[0], tensor.shape[2], tensor.shape[1], *tensor.shape[3:])
    result = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    result.copy_(tensor.permute(0, 2, 1, *range(3, tensor.ndim)))
    return result


def _head_first(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        return tensor.permute(0, 2, 1, 3).contiguous()
    return tensor.permute(0, 2, 1).contiguous()


def _forward_token_first(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    backend = _backend()
    cumulative_gate = backend.chunk_local_cumsum(gate, chunk_size=CHUNK_SIZE)
    kkt = backend.kkt_solve(key, beta, chunk_size=CHUNK_SIZE)
    output, _, final_state = backend.fused_gdr_fwd(
        q=query,
        k=key,
        v=value,
        a=kkt,
        g=cumulative_gate,
        b=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        output_h=False,
        output_o=True,
        cu_seqlens=None,
        cp_seq_map=None,
        raw_cu_seqlens=None,
        state_v_first=False,
    )
    return output, final_state, cumulative_gate, kkt


class _FlashQLAGatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        has_initial_state: bool,
        output_final_state: bool,
    ):
        query_tf = _token_first(query)
        key_tf = _token_first(key)
        value_tf = _token_first(value)
        gate_tf = _token_first(gate)
        beta_tf = _token_first(beta)
        output_tf, final_state, cumulative_gate, kkt = _forward_token_first(
            query_tf,
            key_tf,
            value_tf,
            gate_tf,
            beta_tf,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        ctx.save_for_backward(query_tf, key_tf, value_tf, cumulative_gate, beta_tf, kkt, initial_state)
        ctx.scale = scale
        ctx.has_initial_state = has_initial_state
        return _head_first(output_tf), final_state

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor | None, final_state_grad: torch.Tensor | None):
        query, key, value, cumulative_gate, beta, kkt, initial_state = ctx.saved_tensors
        backend = _backend()
        if output_grad is None:
            output_grad_tf = torch.zeros_like(value)
        else:
            output_grad_tf = _token_first(output_grad)
        if final_state_grad is None:
            final_state_grad = torch.zeros_like(initial_state)

        chunk_states, _, _ = backend.fused_gdr_h(
            k=key,
            v=value,
            a=kkt,
            g=cumulative_gate,
            b=beta,
            initial_state=initial_state,
            output_final_state=False,
            output_h=True,
            cu_seqlens=None,
            num_warmup_chunks=None,
            state_v_first=False,
        )
        query_grad, key_grad, value_grad, gate_grad, beta_grad, initial_state_grad = backend.fused_gdr_bwd(
            q=query,
            k=key,
            v=value,
            a=kkt,
            g=cumulative_gate,
            b=beta,
            do=output_grad_tf,
            dht=final_state_grad.contiguous(),
            h=chunk_states,
            scale=ctx.scale,
            cu_seqlens=None,
            state_v_first=False,
        )
        query_heads = query.shape[2]
        value_heads = value.shape[2]
        if query_heads < value_heads:
            query_grad = backend.group_reduce_vector(query_grad, query_heads)
            key_grad = backend.group_reduce_vector(key_grad, query_heads)
        gate_grad = backend.chunk_local_cumsum(gate_grad, chunk_size=CHUNK_SIZE, reverse=True)
        if not ctx.has_initial_state:
            initial_state_grad = None
        return (
            _head_first(query_grad),
            _head_first(key_grad),
            _head_first(value_grad),
            _head_first(gate_grad),
            _head_first(beta_grad),
            None,
            initial_state_grad,
            None,
            None,
        )


def gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    has_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.zeros(
            query.shape[0],
            value.shape[1],
            query.shape[-1],
            value.shape[-1],
            dtype=torch.float32,
            device=query.device,
        )
    output, final_state = _FlashQLAGatedDeltaRule.apply(
        query,
        key,
        value,
        gate,
        beta,
        scale,
        initial_state,
        has_initial_state,
        output_final_state,
    )
    return output, final_state if output_final_state else None
