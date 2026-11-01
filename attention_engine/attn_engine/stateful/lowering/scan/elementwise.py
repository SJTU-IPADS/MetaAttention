from __future__ import annotations

from functools import lru_cache

import torch


def tilelang_elementwise_prefix(
    scale: torch.Tensor,
    bias: torch.Tensor,
    segment: torch.Tensor,
    initial_state: torch.Tensor,
    backward=None,
    config: tuple[tuple[str, int], ...] = (),
    second: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compose affine summaries through a compiler-selected VJP binding."""
    if segment.shape != (bias.shape[0],):
        raise ValueError("invalid segmented affine prefix shapes")
    if scale.ndim != 4 or scale.shape[:2] != bias.shape[:2]:
        raise ValueError("invalid affine scale shape")
    rows, columns = bias.shape[2], bias.shape[3]
    if second is None and scale.shape[2:] not in (
        (1, 1),
        (rows, 1),
        (1, columns),
        (rows, columns),
    ):
        # a per-axis factor, a per-token scalar, or a full elementwise map
        raise ValueError(f"invalid affine scale shape {tuple(scale.shape)}")
    if second is not None and (
        scale.shape[:2] != bias.shape[:2]
        or second.shape[:2] != bias.shape[:2]
        or scale.shape[2:] != (rows, 1)
        or second.shape[2:] != (1, columns)
    ):
        # one factor per State axis composes the elementwise map per token
        raise ValueError("invalid paired affine factor shapes")
    if bias.numel() == 0:
        return bias
    if isinstance(config, int):
        config = (("block_tokens", config),)
    if backward is None:
        from ..backward import default_backward

        backward = default_backward("elementwise", config)
    return _TileLangElementwisePrefix.apply(
        scale, second, bias, segment, initial_state, backward, tuple(config)
    )


@lru_cache(maxsize=None)
def _compile_elementwise_prefix(
    token_count: int,
    heads: int,
    scale_rows: int,
    scale_columns: int,
    rows: int,
    columns: int,
    dtype: torch.dtype,
    device_index: int,
    block_tokens: int = 128,
    reverse: bool = False,
    paired: bool = False,
):
    """Compose a per-element affine prefix in one streaming kernel.

    Every thread owns one State element and walks the tokens of its segment
    accumulating the affine pair ``(A_t, B_t)`` with ``S_t = A_t S_0 + B_t``.
    A thread-local accumulation removes the cross-thread scan and its barriers,
    which dominated the previous schedule, and it composes per-axis diagonal
    factors exactly like the scalar ones. ``reverse`` walks the tokens from the
    end so a reverse pass needs no flipped copies. With ``paired`` the per-element
    factor is the product of one factor per State axis, which the kernel forms
    while it walks, so neither a ``[tokens, heads, rows, columns]`` factor nor a
    second reduction pass is needed.
    """
    import tilelang
    import tilelang.language as T

    block_tokens = int(block_tokens)
    if block_tokens <= 0 or block_tokens & (block_tokens - 1):
        raise ValueError("block_tokens must be a positive power of two")
    if paired and (scale_rows, scale_columns) != (rows, 1):
        # the row factor holds one entry per row; the column factor is validated
        # against the State columns by the caller
        raise ValueError("a paired prefix takes one factor per State axis")
    matrix_size = rows * columns
    matrix_blocks = tilelang.cdiv(matrix_size, block_tokens)
    write_every_element = scale_rows != 1 and scale_columns != 1

    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    def build_kernel():
        @T.prim_func
        def kernel(
            scale: T.Tensor(
                (token_count, heads, scale_rows, scale_columns), dtype=dtype
            ),
            second: T.Tensor((token_count, heads, 1, columns), dtype=dtype),
            bias: T.Tensor((token_count, heads, rows, columns), dtype=dtype),
            segment: T.Tensor((token_count,), dtype="int32"),
            output_scale: T.Tensor(
                (token_count, heads, scale_rows, scale_columns), dtype=dtype
            ),
            output_second: T.Tensor((token_count, heads, 1, columns), dtype=dtype),
            output_bias: T.Tensor((token_count, heads, rows, columns), dtype=dtype),
        ):
            if False:
                # tilelang resolves annotations through the kernel's free
                # variables, so names used only in a signature must be referenced.
                _ = (dtype, rows, paired)
            with T.Kernel(heads, matrix_blocks, threads=block_tokens) as (
                head,
                matrix_block,
            ):
                thread = T.get_thread_binding()
                index = matrix_block * block_tokens + thread
                accumulated_scale = T.alloc_local((1,), dtype="float32")
                accumulated_second = T.alloc_local((1,), dtype="float32")
                accumulated_bias = T.alloc_local((1,), dtype="float32")
                row_term = T.alloc_local((1,), dtype="float32")
                column_term = T.alloc_local((1,), dtype="float32")
                element_factor = T.alloc_local((1,), dtype="float32")
                current_segment = T.alloc_local((1,), dtype="int32")
                if index < matrix_size:
                    row = index // columns
                    column = index % columns
                    scale_row = 0 if scale_rows == 1 else row
                    scale_column = 0 if scale_columns == 1 else column
                    write_scale = T.alloc_local((1,), dtype="bool")
                    if paired:
                        # the row factor holds one entry per row and the column
                        # factor one per column; the elementwise product of the
                        # two is composed where it is applied
                        write_scale[0] = column == 0
                    elif write_every_element:
                        write_scale[0] = True
                    elif scale_rows == 1 and scale_columns == 1:
                        write_scale[0] = index == 0
                    elif scale_columns == 1:
                        # a first-feature factor holds one entry per row
                        write_scale[0] = column == 0
                    else:
                        # a second-feature factor holds one entry per column
                        write_scale[0] = row == 0
                    accumulated_scale[0] = 1.0
                    accumulated_second[0] = 1.0
                    accumulated_bias[0] = 0.0
                    current_segment[0] = -1
                    for step in T.Serial(token_count):
                        token = token_count - 1 - step if reverse else step
                        if segment[token] != current_segment[0]:
                            accumulated_scale[0] = 1.0
                            accumulated_second[0] = 1.0
                            accumulated_bias[0] = 0.0
                            current_segment[0] = segment[token]
                        row_term[0] = scale[token, head, scale_row, scale_column]
                        if paired:
                            column_term[0] = second[token, head, 0, column]
                            accumulated_second[0] = (
                                accumulated_second[0] * column_term[0]
                            )
                            element_factor[0] = row_term[0] * column_term[0]
                        else:
                            element_factor[0] = row_term[0]
                        accumulated_bias[0] = (
                            element_factor[0] * accumulated_bias[0]
                            + bias[token, head, row, column]
                        )
                        accumulated_scale[0] = row_term[0] * accumulated_scale[0]
                        output_bias[token, head, row, column] = accumulated_bias[0]
                        if write_scale[0]:
                            output_scale[token, head, scale_row, scale_column] = (
                                accumulated_scale[0]
                            )
                        if paired and row == 0:
                            output_second[token, head, 0, column] = accumulated_second[
                                0
                            ]

        return kernel

    return build_kernel()


def _tilelang_segmented_elementwise_scan(
    scale: torch.Tensor,
    bias: torch.Tensor,
    segment: torch.Tensor,
    *,
    inplace: bool = False,
    block_tokens: int = 128,
    reverse: bool = False,
    second: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Compose a per-element affine prefix, optionally from one factor per axis."""
    rows, columns = bias.shape[2], bias.shape[3]
    if second is None:
        output_scale = torch.empty_like(scale)
        output_second = None
    else:
        # one factor per State axis, composed per element inside the kernel
        output_scale = torch.empty(
            scale.shape[0],
            scale.shape[1],
            rows,
            1,
            device=scale.device,
            dtype=scale.dtype,
        )
        output_second = torch.empty(
            second.shape[0],
            second.shape[1],
            1,
            columns,
            device=second.device,
            dtype=second.dtype,
        )
    output_bias = bias if inplace else torch.empty_like(bias)
    if bias.numel() == 0:
        return output_scale, output_second, output_bias
    # an unpaired call still passes correctly shaped operands the kernel never
    # reads or writes, so one compiled signature serves both modes
    second_operand = (
        second
        if second is not None
        else torch.empty(
            scale.shape[0],
            scale.shape[1],
            1,
            columns,
            device=scale.device,
            dtype=scale.dtype,
        )
    )
    output_second_operand = (
        output_second
        if output_second is not None
        else torch.empty(
            scale.shape[0],
            scale.shape[1],
            1,
            columns,
            device=scale.device,
            dtype=scale.dtype,
        )
    )
    device_index = scale.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    kernel = _compile_elementwise_prefix(
        scale.shape[0],
        scale.shape[1],
        scale.shape[2],
        scale.shape[3],
        rows,
        columns,
        scale.dtype,
        device_index,
        block_tokens,
        reverse,
        second is not None,
    )
    kernel(
        scale,
        second_operand,
        bias,
        segment,
        output_scale,
        output_second_operand,
        output_bias,
    )
    return output_scale, output_second, output_bias


class _TileLangElementwisePrefix(torch.autograd.Function):
    """Autograd transport for a compiler-selected elementwise VJP binding."""

    @staticmethod
    def forward(ctx, scale, second, bias, segment, initial_state, backward, config):
        config = tuple(config)
        block_tokens = dict(config).get("block_tokens", 128)
        operands = [scale, bias, initial_state]
        if second is not None:
            operands.append(second)
        inference = not any(tensor.requires_grad for tensor in operands)
        output_scale, output_second, output_bias = _tilelang_segmented_elementwise_scan(
            scale,
            bias,
            segment,
            inplace=inference,
            block_tokens=block_tokens,
            second=second,
        )
        initial = initial_state[segment.long()]
        if inference and output_second is not None:
            output_bias.add_(output_scale * (output_second * initial))
            states = output_bias
        elif inference:
            states = output_bias.addcmul_(output_scale, initial)
        elif output_second is not None:
            states = output_scale * (output_second * initial) + output_bias
        else:
            states = output_scale * initial + output_bias
        ctx.backward_binding = backward
        ctx.save_for_backward(
            scale,
            second,
            segment,
            output_scale,
            output_second,
            output_bias,
            initial_state,
        )
        return states

    @staticmethod
    def backward(ctx, grad_states):
        (
            scale,
            second,
            segment,
            output_scale,
            output_second,
            output_bias,
            initial_state,
        ) = ctx.saved_tensors
        # the second factor slot is published as None when the Program scales one
        # axis, so the declared elementwise context is complete either way
        context = {
            "transition.scale": scale,
            "transition.second": second,
            "transition.output_second": output_second,
            "transition.segment": segment,
            "transition.output_scale": output_scale,
            "transition.output_bias": output_bias,
            "transition.initial_state": initial_state,
        }
        (
            scale_gradient,
            second_gradient,
            bias_gradient,
            segment_gradient,
            state_gradient,
        ) = ctx.backward_binding(context, grad_states)
        return (
            scale_gradient,
            second_gradient,
            bias_gradient,
            segment_gradient,
            state_gradient,
            None,
            None,
        )


def _sum_to_shape(value: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    for axis, (actual, expected) in enumerate(zip(value.shape, shape, strict=True)):
        if expected == 1 and actual != 1:
            value = value.sum(dim=axis, keepdim=True)
    return value
