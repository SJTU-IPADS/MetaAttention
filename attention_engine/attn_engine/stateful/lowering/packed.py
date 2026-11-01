"""Compact device-segmented checkpoint pipeline shared with dense recurrence.

Offsets are never read on the host. Chunk capacity is ceil(totalT/16)+S,
not S*ceil(totalT/16); empty sequences have no chunk and preserve their State.
"""

from __future__ import annotations

from math import prod

import torch

from .lowrank.checkpoint import CHUNK_SIZE, metadata


def packed_chunk_grid(
    token_count: int, sequence_count: int, chunk_size: int = CHUNK_SIZE
) -> int:
    if token_count < 0 or sequence_count < 0 or chunk_size <= 0:
        raise ValueError("invalid packed grid dimensions")
    return (
        (token_count + chunk_size - 1) // chunk_size + sequence_count
        if token_count
        else 0
    )


def packed_workspace_bound(
    token_count: int,
    sequence_count: int,
    heads: int,
    rows: int,
    columns: int,
    *,
    chunk_size: int = CHUNK_SIZE,
    operations: int = 0,
    injections: int = 0,
    injection_scalars: int = 0,
    readouts: int = 0,
    left_width: int | None = None,
    offset_bytes: int = 8,
) -> dict[str, int]:
    """Exact kernel allocation envelope (elements are FP32 unless byte-labelled).

    Excludes differentiable input packing and per-output dtype conversions.
    Reverse workspace excludes saved entries, result gradients and cotangents;
    those are listed separately. All partials are live together with the reverse
    summaries, and reduction programs visit column blocks in ascending order.
    """
    if min(token_count, sequence_count, heads, rows, columns) < 0 or chunk_size <= 0:
        raise ValueError("invalid workspace dimensions")
    chunks = packed_chunk_grid(token_count, sequence_count, chunk_size)
    state = sequence_count * heads * rows * columns
    entries = chunks * heads * rows * columns
    transform = chunks * heads * rows * rows
    decay = chunks * heads * columns
    width = rows if left_width is None else left_width
    total = token_count * heads
    factors = total * (
        operations * (width + rows + 1)
        + injections * (rows + columns)
        + injection_scalars
        + readouts * rows
    )
    partials = (
        ((columns + 15) // 16)
        * total
        * (
            operations * (width + rows + 1)
            + injections * rows
            + injection_scalars
            + readouts * rows
        )
    )
    outputs = readouts * total * columns
    return {
        "grid_slots": chunks,
        "chunk_slots": chunks,
        "state_elements": state,
        "carry_elements": entries,
        "transform_elements": transform,
        "decay_elements": decay,
        "forward_workspace_elements": transform + decay + 2 * entries,
        "reverse_workspace_elements": transform + decay + 2 * entries + partials,
        "saved_checkpoint_elements": entries,
        "metadata_bytes": 24 * chunks + (offset_bytes + 4) * (sequence_count + 1),
        "packed_factor_elements": factors,
        "partial_gradient_elements": partials,
        "result_gradient_elements": factors + state,
        "output_elements": outputs,
        "final_elements": state,
        "absent_cotangent_elements": outputs + state,
    }


def _seq(value):
    if value is None:
        return ()
    return (value,) if isinstance(value, torch.Tensor) else tuple(value)


def _token_count(values):
    counts = [value.shape[0] for value in values if value is not None and value.ndim]
    # A broadcast [1,...] factor must not turn an all-empty binding into T=1.
    return (
        max(counts, default=0)
        if 0 not in counts
        else max((n for n in counts if n != 1), default=0)
    )


def _stack(parts, tokens, heads, width, device):
    """Use differentiable broadcast/padding; autograd reduces grouped bindings."""
    values = []
    for value in parts:
        if value is None:
            current = torch.zeros((1, 1, 1), device=device)
        elif value.ndim == 0:
            current = value.reshape(1, 1, 1)
        else:
            current = value.reshape(
                value.shape[0],
                value.shape[1] if value.ndim > 1 else 1,
                prod(value.shape[2:]) if value.ndim > 1 else 1,
            )
        if current.shape[0] not in (1, tokens):
            raise ValueError("binding token count mismatch")
        current = current.expand(tokens, -1, -1)
        groups = current.shape[1]
        if heads % groups:
            raise ValueError("binding heads must divide State heads")
        if groups != heads:
            current = current.repeat_interleave(heads // groups, dim=1)
        if current.shape[2] not in (1, width):
            raise ValueError("binding feature width mismatch")
        current = current.expand(-1, -1, width)
        values.append(current)
    if not values:
        return torch.empty(
            (0, tokens, heads, width), device=device, dtype=torch.float32
        )
    return torch.stack(values).float().contiguous()


class _PackedRecurrence(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        codes,
        left,
        right,
        coefficient,
        ik,
        iv,
        query,
        initial,
        offsets,
        dtypes,
        injection_scalars,
        injection_orders,
        readout_axis,
    ):
        from .lowrank.checkpoint import forward

        segmentation = metadata(
            query.shape[1], initial.shape[0], initial.device, offsets
        )
        output, final, checkpoints = forward(
            operation_codes=codes,
            left_factors=left,
            right_factors=right,
            coefficients=coefficient,
            injection_left=ik,
            injection_right=iv,
            readout=query,
            initial_state=initial,
            output_dtype=torch.float32,
            sequence_offsets=offsets,
            chunk_metadata=segmentation,
            injection_scalars=injection_scalars,
            injection_orders=injection_orders,
            readout_axis=readout_axis,
        )
        ctx.readout_axis = readout_axis
        ctx.codes = codes
        ctx.injection_orders = injection_orders
        ctx.output_shape = output.shape[1:]
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(
            left,
            right,
            coefficient,
            ik,
            iv,
            query,
            initial,
            checkpoints,
            injection_scalars,
            *segmentation,
        )
        return (
            *tuple(output[index].to(dtype) for index, dtype in enumerate(dtypes)),
            final,
        )

    @staticmethod
    def backward(ctx, *cotangents):
        from .lowrank.checkpoint import backward

        *outputs, final = cotangents
        (
            left,
            right,
            coefficient,
            ik,
            iv,
            query,
            initial,
            checkpoints,
            injection_scalars,
            offsets,
            prefix,
            chunks,
        ) = ctx.saved_tensors
        dy = (
            None
            if all(value is None for value in outputs)
            else torch.stack(
                [
                    torch.zeros(
                        ctx.output_shape, device=query.device, dtype=torch.float32
                    )
                    if value is None
                    else value.float()
                    for value in outputs
                ]
            ).contiguous()
        )
        gradients = backward(
            operation_codes=ctx.codes,
            left_factors=left,
            right_factors=right,
            coefficients=coefficient,
            injection_left=ik,
            injection_right=iv,
            readout=query,
            initial_state=initial,
            checkpoints=checkpoints,
            grad_output=dy,
            grad_final=final,
            sequence_offsets=offsets,
            chunk_metadata=(offsets, prefix, chunks),
            injection_scalars=injection_scalars,
            injection_orders=ctx.injection_orders,
            readout_axis=ctx.readout_axis,
        )
        return (None, *gradients[:7], None, None, gradients[7], None, None)


def execute_lowrank_packed(
    *,
    operation_codes,
    left_factors,
    right_factors,
    coefficients,
    injection_left,
    injection_right,
    readouts,
    initial_state,
    sequence_offsets,
    output_dtypes=None,
    return_final_state=False,
    injection_scalars=None,
    injection_orders=None,
    readout_axis=0,
):
    """Return per-readout [T,H,V] outputs and optional final, with all VJPs."""
    if initial_state.device.type != "cuda":
        raise ValueError("packed chunk pipeline requires CUDA")
    if sequence_offsets.ndim != 1 or sequence_offsets.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("sequence_offsets must be rank-1 int32/int64")
    if (
        initial_state.ndim != 4
        or sequence_offsets.shape[0] != initial_state.shape[0] + 1
    ):
        raise ValueError("invalid offsets or initial State shape")
    if sequence_offsets.device != initial_state.device:
        raise ValueError("offsets and initial State must share a device")
    reads, il, ir = _seq(readouts), _seq(injection_left), _seq(injection_right)
    codes = tuple(operation_codes)
    if not (len(codes) == len(left_factors) == len(right_factors) == len(coefficients)):
        raise ValueError("one factor slot is required per authored operation")
    if len(il) != len(ir):
        raise ValueError("injection factor counts differ")
    values = (*reads, *il, *ir, *left_factors, *right_factors, *coefficients)
    tokens = _token_count(values)
    heads, rows, columns = initial_state.shape[1:]
    device = initial_state.device
    width = max(rows, columns) if "column-scale" in codes else rows
    left_parts = []
    for code, value in zip(codes, left_factors, strict=True):
        feature = columns if code == "column-scale" else rows
        current = _stack((value,), tokens, heads, feature, device)[0]
        if feature < width:
            current = torch.nn.functional.pad(current, (0, width - feature))
        left_parts.append(current)
    left = (
        torch.stack(left_parts).unsqueeze(-1)
        if left_parts
        else torch.empty(
            (0, tokens, heads, width, 1), device=device, dtype=torch.float32
        )
    )
    right = _stack(right_factors, tokens, heads, rows, device).unsqueeze(-1)
    coeff = _stack(coefficients, tokens, heads, 1, device).unsqueeze(-1)
    keys = _stack(il, tokens, heads, rows, device).unsqueeze(-1)
    values = _stack(ir, tokens, heads, columns, device).unsqueeze(-2)
    queries = _stack(
        reads, tokens, heads, columns if readout_axis else rows, device
    ).unsqueeze(-1)
    dtypes = (
        tuple(value.dtype for value in reads)
        if output_dtypes is None
        else tuple(output_dtypes)
    )
    if len(dtypes) != len(reads):
        raise ValueError("one output dtype is required per readout")
    result = _PackedRecurrence.apply(
        codes,
        left,
        right,
        coeff,
        keys,
        values,
        queries,
        initial_state.float().contiguous(),
        sequence_offsets,
        dtypes,
        _stack(injection_scalars or [], tokens, heads, 1, device).unsqueeze(-1),
        injection_orders,
        readout_axis,
    )
    return tuple(result[:-1]), result[-1] if return_final_state else None


def execute_diagonal_packed(
    *,
    rows_gate=None,
    columns_gate=None,
    row_gates=None,
    column_gates=None,
    key=None,
    value=None,
    injection_left=None,
    injection_right=None,
    readouts=(),
    initial_state,
    sequence_offsets,
    output_dtypes=None,
    return_final_state=False,
):
    """Ordered row/column gates share the general compact checkpoint kernels."""
    row_ops = _seq(rows_gate if rows_gate is not None else row_gates)
    column_ops = _seq(columns_gate if columns_gate is not None else column_gates)
    il = injection_left if injection_left is not None else key
    ir = injection_right if injection_right is not None else value
    operations = ("row-scale",) * len(row_ops) + ("column-scale",) * len(column_ops)
    left = (*row_ops, *column_ops)
    tokens = _token_count((*left, *_seq(readouts), *_seq(il), *_seq(ir)))
    heads, rows = initial_state.shape[1:3]
    right = tuple(
        initial_state.new_zeros(()).expand(tokens, heads, rows, 1) for _ in left
    )
    coefficient = tuple(
        initial_state.new_zeros(()).expand(tokens, heads, 1, 1) for _ in left
    )
    return execute_lowrank_packed(
        operation_codes=operations,
        left_factors=left,
        right_factors=right,
        coefficients=coefficient,
        injection_left=il,
        injection_right=ir,
        readouts=readouts,
        initial_state=initial_state,
        sequence_offsets=sequence_offsets,
        output_dtypes=output_dtypes,
        return_final_state=return_final_state,
    )


__all__ = [
    "CHUNK_SIZE",
    "packed_chunk_grid",
    "packed_workspace_bound",
    "execute_diagonal_packed",
    "execute_lowrank_packed",
]
