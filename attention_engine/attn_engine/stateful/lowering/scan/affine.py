from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def _compile_dense_matmul_step(
    token_count: int,
    heads: int,
    rows: int,
    inner: int,
    columns: int,
    validity_offset: int,
    operand_offset: int,
    left_from_previous: bool,
    passthrough: int,
    addend: bool,
    device_index: int,
    threads: int = 128,
):
    import tilelang
    import tilelang.language as T

    from ..diagonal.identity import fp32_gemm_macro

    # Affine maps can be I plus a small correction. Single-TF32 GEMM loses
    # that correction on every composition, including the reverse scan.
    gemm = fp32_gemm_macro(T)
    block_rows = 32
    block_columns = 32
    block_inner = 32

    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    def build_kernel():
        @T.prim_func
        def kernel(
            left: T.Tensor((token_count, heads, rows, inner), dtype="float32"),
            right: T.Tensor((token_count, heads, inner, columns), dtype="float32"),
            segment: T.Tensor((token_count,), dtype="int32"),
            extra: T.Tensor((token_count, heads, rows, columns), dtype="float32"),
            output: T.Tensor((token_count, heads, rows, columns), dtype="float32"),
        ):
            with T.Kernel(
                token_count * heads,
                T.ceildiv(rows, block_rows),
                T.ceildiv(columns, block_columns),
                threads=threads,
            ) as (token_head, row_block, column_block):
                token = token_head // heads
                head = token_head % heads
                row_start = row_block * block_rows
                column_start = column_block * block_columns
                valid = (token >= validity_offset) & (
                    segment[token] == segment[token - validity_offset]
                )
                accumulator = T.alloc_fragment(
                    (block_rows, block_columns), dtype="float32"
                )
                left_shared = T.alloc_shared((block_rows, block_inner), dtype="float32")
                right_shared = T.alloc_shared(
                    (block_inner, block_columns), dtype="float32"
                )
                T.clear(accumulator)
                if valid:
                    left_token = token - operand_offset if left_from_previous else token
                    right_token = (
                        token if left_from_previous else token - operand_offset
                    )
                    for inner_block in T.Pipelined(
                        T.ceildiv(inner, block_inner), num_stages=1
                    ):
                        inner_start = inner_block * block_inner
                        T.copy(
                            left[
                                left_token,
                                head,
                                row_start : row_start + block_rows,
                                inner_start : inner_start + block_inner,
                            ],
                            left_shared,
                        )
                        T.copy(
                            right[
                                right_token,
                                head,
                                inner_start : inner_start + block_inner,
                                column_start : column_start + block_columns,
                            ],
                            right_shared,
                        )
                        gemm(left_shared, right_shared, accumulator)
                    if addend:
                        for row, column in T.Parallel(block_rows, block_columns):
                            accumulator[row, column] += extra[
                                token,
                                head,
                                row_start + row,
                                column_start + column,
                            ]
                    T.copy(
                        accumulator,
                        output[
                            token,
                            head,
                            row_start : row_start + block_rows,
                            column_start : column_start + block_columns,
                        ],
                    )
                else:
                    if passthrough == 1:
                        T.copy(
                            left[
                                token,
                                head,
                                row_start : row_start + block_rows,
                                column_start : column_start + block_columns,
                            ],
                            output[
                                token,
                                head,
                                row_start : row_start + block_rows,
                                column_start : column_start + block_columns,
                            ],
                        )
                    elif passthrough == 2:
                        T.copy(
                            right[
                                token,
                                head,
                                row_start : row_start + block_rows,
                                column_start : column_start + block_columns,
                            ],
                            output[
                                token,
                                head,
                                row_start : row_start + block_rows,
                                column_start : column_start + block_columns,
                            ],
                        )
                    else:
                        T.copy(
                            extra[
                                token,
                                head,
                                row_start : row_start + block_rows,
                                column_start : column_start + block_columns,
                            ],
                            output[
                                token,
                                head,
                                row_start : row_start + block_rows,
                                column_start : column_start + block_columns,
                            ],
                        )

        return kernel

    return build_kernel()


def _tilelang_dense_affine_scan(
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor,
    segment: torch.Tensor,
    *,
    threads: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_count, heads, rows, _ = left.shape
    columns = right.shape[-1]
    device_index = left.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    offset = 1
    while offset < token_count:
        next_left = torch.empty_like(left)
        next_right = torch.empty_like(right)
        temporary = torch.empty_like(bias)
        next_bias = torch.empty_like(bias)
        empty_bias = torch.empty_like(bias)
        empty_left = torch.empty_like(left)
        empty_right = torch.empty_like(right)
        _compile_dense_matmul_step(
            token_count,
            heads,
            rows,
            rows,
            rows,
            offset,
            offset,
            False,
            1,
            False,
            device_index,
            threads,
        )(left, left, segment, empty_left, next_left)
        _compile_dense_matmul_step(
            token_count,
            heads,
            columns,
            columns,
            columns,
            offset,
            offset,
            True,
            2,
            False,
            device_index,
            threads,
        )(right, right, segment, empty_right, next_right)
        _compile_dense_matmul_step(
            token_count,
            heads,
            rows,
            rows,
            columns,
            offset,
            offset,
            False,
            0,
            False,
            device_index,
            threads,
        )(left, bias, segment, empty_bias, temporary)
        _compile_dense_matmul_step(
            token_count,
            heads,
            rows,
            columns,
            columns,
            offset,
            0,
            False,
            0,
            True,
            device_index,
            threads,
        )(temporary, right, segment, bias, next_bias)
        left, right, bias = next_left, next_right, next_bias
        offset *= 2
    return left, right, bias


def tilelang_dense_affine_prefix(
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor,
    segment: torch.Tensor,
    initial_state: torch.Tensor,
    backward=None,
    config: tuple[tuple[str, int], ...] = (),
) -> torch.Tensor:
    """Compose dense summaries through a compiler-selected VJP binding."""
    if (
        left.ndim != 4
        or right.ndim != 4
        or bias.ndim != 4
        or left.shape[:2] != bias.shape[:2]
        or right.shape[:2] != bias.shape[:2]
        or segment.shape != (bias.shape[0],)
    ):
        raise ValueError("invalid dense affine prefix shapes")
    if isinstance(config, int):
        config = (("threads", config),)
    if backward is None:
        from ..backward import default_backward

        backward = default_backward("dense", config)
    return _TileLangDenseAffinePrefix.apply(
        left, right, bias, segment, initial_state, backward, tuple(config)
    )


class _TileLangDenseAffinePrefix(torch.autograd.Function):
    """Autograd transport for a compiler-selected dense VJP binding."""

    @staticmethod
    def forward(ctx, left, right, bias, segment, initial_state, backward, config):
        config = tuple(config)
        prefix_left, prefix_right, prefix_bias = _tilelang_dense_affine_scan(
            left, right, bias, segment, threads=dict(config).get("threads", 128)
        )
        states = (
            prefix_left @ initial_state[segment.long()] @ prefix_right + prefix_bias
        )
        ctx.backward_binding = backward
        ctx.save_for_backward(left, right, segment, initial_state, states)
        return states

    @staticmethod
    def backward(ctx, grad_states):
        left, right, segment, initial_state, states = ctx.saved_tensors
        context = {
            "transition.left": left,
            "transition.right": right,
            "transition.segment": segment,
            "transition.initial_state": initial_state,
            "transition.states": states,
        }
        return (*ctx.backward_binding(context, grad_states), None, None)
