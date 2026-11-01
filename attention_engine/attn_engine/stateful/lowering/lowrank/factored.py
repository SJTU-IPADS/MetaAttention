"""Ordered low-rank recurrence lowering with FP32 checkpointed gradients.

Inference uses register-State recurrence or block compact WY for scalar/rank-one
plans and affine summaries otherwise. Training retains checkpointed inverse-free VJP.
"""

from __future__ import annotations

import torch


def _stack(parts: list[torch.Tensor], shape: tuple[int, ...], device) -> torch.Tensor:
    if not parts:
        return torch.empty((0, *shape), device=device, dtype=torch.float32)
    return (
        torch.stack(
            [
                part.reshape((1,) * len(shape)).expand(shape)
                if part.numel() == 1
                else part.reshape(shape)
                for part in parts
            ],
            dim=0,
        )
        .float()
        .contiguous()
    )


def _stack_left(parts, codes, tokens, heads, rows, columns):
    width = max(rows, columns) if "column-scale" in codes else rows
    values = []
    for part, code in zip(parts, codes, strict=True):
        size = columns if code == "column-scale" else rows
        value = part.reshape(tokens, heads, size, 1)
        if size < width:
            value = torch.nn.functional.pad(value, (0, 0, 0, width - size))
        values.append(value)
    return _stack(values, (tokens, heads, width, 1), parts[0].device) if parts else None


def factored_recurrence_chunked(
    *,
    operation_codes: tuple[str, ...],
    left_factors: list[torch.Tensor | None],
    right_factors: list[torch.Tensor | None],
    coefficients: list[torch.Tensor | None],
    injection_left: list[torch.Tensor],
    injection_right: list[torch.Tensor],
    readouts: list[torch.Tensor],
    initial_state: torch.Tensor,
    output_dtypes: tuple[torch.dtype, ...],
    symmetric_rank_one: bool = False,
    injection_scalars: list[torch.Tensor] | None = None,
    injection_orders: tuple[tuple[int, ...], ...] | None = None,
    readout_axis: int = 0,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Execute scalar/rank-one schedules or ordered affine chunk replay."""
    from .lowrank_kernels import execute

    return execute(
        operation_codes=operation_codes,
        left_factors=left_factors,
        right_factors=right_factors,
        coefficients=coefficients,
        injection_left=injection_left,
        injection_right=injection_right,
        readouts=readouts,
        initial_state=initial_state,
        output_dtypes=output_dtypes,
        symmetric_rank_one=symmetric_rank_one,
        injection_scalars=injection_scalars,
        injection_orders=injection_orders,
        readout_axis=readout_axis,
    )


def factored_recurrence_backward(
    *,
    operation_codes: tuple[str, ...],
    left_factors: torch.Tensor,
    right_factors: torch.Tensor,
    coefficients: torch.Tensor,
    injection_left: torch.Tensor,
    injection_right: torch.Tensor,
    readout: torch.Tensor,
    recorded: torch.Tensor,
    initial_state: torch.Tensor,
    grad_output: torch.Tensor | None,
    grad_final: torch.Tensor | None,
    threads: int = 128,
    injection_scalars: torch.Tensor | None = None,
    injection_orders: tuple[tuple[int, ...], ...] | None = None,
    readout_axis: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Return factor/State gradients, with a final slot for supplied scalar factors.

    ``recorded`` holds FP32 checkpoints.

    Factors use [B*T,H,...], output cotangents [B,H,T,V], and recorded
    chunk-entry States [B,H,ceil(T/16),R,V]. Multiple readouts add a leading
    readout axis to query factors and output cotangents, not to States.
    Static authored codes select private kernels without a device opcode.
    """
    from .checkpoint import backward

    return backward(
        operation_codes=operation_codes,
        left_factors=left_factors,
        right_factors=right_factors,
        coefficients=coefficients,
        injection_left=injection_left,
        injection_right=injection_right,
        readout=readout,
        checkpoints=recorded,
        initial_state=initial_state,
        grad_output=grad_output,
        grad_final=grad_final,
        injection_scalars=injection_scalars,
        injection_orders=injection_orders,
        readout_axis=readout_axis,
    )


class _TileLangFactoredRecurrence(torch.autograd.Function):
    """Autograd transport for checkpointed factored recurrence lowering."""

    @staticmethod
    def forward(
        ctx,
        operation_codes,
        left,
        right,
        coefficient,
        injection_left,
        injection_right,
        readout,
        initial_state,
        injection_scalars,
        injection_orders,
        readout_axis,
        output_dtypes,
        backward,
        config,
    ):
        from .checkpoint import forward

        output, final, recorded = forward(
            operation_codes=operation_codes,
            left_factors=left,
            right_factors=right,
            coefficients=coefficient,
            injection_left=injection_left,
            injection_right=injection_right,
            readout=readout,
            initial_state=initial_state,
            output_dtype=torch.float32,
            record=backward is not None,
            injection_scalars=injection_scalars,
            injection_orders=injection_orders,
            readout_axis=readout_axis,
        )
        ctx.readout_axis = readout_axis
        ctx.injection_orders = injection_orders
        ctx.backward_binding = backward
        ctx.operation_codes = operation_codes
        ctx.output_dtypes = output_dtypes
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(
            left,
            right,
            coefficient,
            injection_left,
            injection_right,
            readout,
            recorded,
            initial_state,
            injection_scalars,
            final,
        )
        return (
            *tuple(output[i].to(dtype) for i, dtype in enumerate(output_dtypes)),
            final,
        )

    @staticmethod
    def backward(ctx, *cotangents):
        *output_grads, grad_final = cotangents
        if all(gradient is None for gradient in output_grads):
            grad_output = None
        else:
            final_shape = ctx.saved_tensors[-1].shape
            query = ctx.saved_tensors[5]
            shape = (
                final_shape[0],
                final_shape[1],
                query.shape[1] // final_shape[0],
                final_shape[2] if ctx.readout_axis else final_shape[3],
            )
            grad_output = torch.stack(
                [
                    gradient.float()
                    if gradient is not None
                    else torch.zeros(shape, device=query.device)
                    for gradient in output_grads
                ]
            ).contiguous()
        (
            left,
            right,
            coefficient,
            injection_left,
            injection_right,
            readout,
            recorded,
            initial_state,
            injection_scalars,
            final,
        ) = ctx.saved_tensors
        if ctx.backward_binding is None:
            raise RuntimeError("factored recurrence has no compiler-selected backward")
        context = {
            "factored.operation_codes": ctx.operation_codes,
            "factored.left": left,
            "factored.right": right,
            "factored.coefficient": coefficient,
            "factored.injection_left": injection_left,
            "factored.injection_right": injection_right,
            "factored.injection_scalars": injection_scalars,
            "factored.injection_orders": ctx.injection_orders,
            "factored.readout_axis": ctx.readout_axis,
            "factored.readout": readout,
            "factored.recorded": recorded,
            "factored.initial_state": initial_state,
            "factored.final_state": final,
        }
        gradients = ctx.backward_binding(context, (grad_output, grad_final))
        if len(gradients) == 7 and not injection_scalars.numel():
            # Existing internal bindings may omit the empty scalar-gradient slot.
            gradients = (*gradients, None)
        return (None, *gradients, None, None, None, None, None)


def factored_recurrence_dense(
    *,
    operation_codes: tuple[str, ...],
    left_factors: list[torch.Tensor],
    right_factors: list[torch.Tensor],
    coefficients: list[torch.Tensor],
    injection_left: list[torch.Tensor],
    injection_right: list[torch.Tensor],
    readouts: list[torch.Tensor],
    initial_state: torch.Tensor,
    output_dtypes: tuple[torch.dtype, ...],
    backward=None,
    config: tuple[tuple[str, int], ...] = (),
    threads: int = 128,
    injection_scalars: list[torch.Tensor] | None = None,
    injection_orders: tuple[tuple[int, ...], ...] | None = None,
    readout_axis: int = 0,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Run authored operations and readouts, saving only FP32 chunk entries."""
    if not readouts or len(readouts) != len(output_dtypes):
        raise ValueError("one output dtype is required per factored readout")
    source = readouts[0]
    tokens, heads = source.shape[:2]
    rows, columns = initial_state.shape[2:]
    shape = (tokens, heads, rows, 1)
    scalar_shape = (tokens, heads, 1, 1)
    columns_shape = (tokens, heads, 1, columns)
    device = initial_state.device
    left = _stack_left(left_factors, operation_codes, tokens, heads, rows, columns)
    if left is None:
        left = _stack([], shape, device)
    query = _stack(
        readouts, (tokens, heads, columns if readout_axis else rows, 1), device
    )
    result = _TileLangFactoredRecurrence.apply(
        tuple(operation_codes),
        left,
        _stack(right_factors, shape, device),
        _stack(coefficients, scalar_shape, device),
        _stack(injection_left, shape, device),
        _stack(injection_right, columns_shape, device),
        query,
        initial_state.contiguous(),
        _stack(injection_scalars or [], scalar_shape, device),
        injection_orders,
        readout_axis,
        tuple(output_dtypes),
        backward,
        tuple(config),
    )
    return tuple(result[:-1]), result[-1]
