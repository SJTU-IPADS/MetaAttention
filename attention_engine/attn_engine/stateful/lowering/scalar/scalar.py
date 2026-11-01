from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch


@lru_cache(maxsize=None)
def _compile_scalar_factorized_dense_raw(
    sequence_count: int,
    sequence_length: int,
    heads: int,
    rows: int,
    columns: int,
    propagation_heads: int,
    left_heads: int,
    right_heads: int,
    readout_heads: int,
    propagation_dtype: torch.dtype,
    left_dtype: torch.dtype,
    right_dtype: torch.dtype,
    readout_dtype: torch.dtype,
    output_dtype: torch.dtype,
    propagation_operation: str,
    left_operation: str,
    right_operation: str,
    readout_operation: str,
    propagation_is_log_scale: bool,
    left_is_log_scale: bool,
    right_is_log_scale: bool,
    readout_is_log_scale: bool,
    propagation_scale: float,
    left_scale: float,
    right_scale: float,
    readout_scale: float,
    propagation_offset: float,
    left_offset: float,
    right_offset: float,
    readout_offset: float,
    return_final_state: bool,
    threads: int,
    identity_propagation: bool,
    scale_steps: tuple[tuple[float, ...], ...] = ((), (), (), ()),
    expression_nodes: tuple[tuple, ...] = ((), (), (), ()),
):
    import tilelang
    import tilelang.language as T

    chunk_tokens = 64
    block_columns = 64
    chunk_count = tilelang.cdiv(sequence_length, chunk_tokens)
    propagation_shape = (
        (sequence_count, heads, rows, columns)
        if identity_propagation
        else (sequence_count, propagation_heads, sequence_length)
    )

    from .scalar_kernels import _transform

    propagation_transform = _transform(
        T,
        (
            propagation_operation,
            propagation_scale,
            False,
            propagation_offset,
            scale_steps[0],
            expression_nodes[0],
        ),
    )
    left_transform = _transform(
        T,
        (
            left_operation,
            left_scale,
            left_is_log_scale,
            left_offset,
            scale_steps[1],
            expression_nodes[1],
        ),
    )
    right_transform = _transform(
        T,
        (
            right_operation,
            right_scale,
            right_is_log_scale,
            right_offset,
            scale_steps[2],
            expression_nodes[2],
        ),
    )
    readout_transform = _transform(
        T,
        (
            readout_operation,
            readout_scale,
            readout_is_log_scale,
            readout_offset,
            scale_steps[3],
            expression_nodes[3],
        ),
    )
    preserve_scales = any(scale_steps)

    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: not preserve_scales},
        compile_flags=["-fmad=false"] if preserve_scales else None,
    )
    def build_kernel():
        @T.prim_func
        def kernel(
            propagation: T.Tensor(propagation_shape, dtype=propagation_dtype),
            left: T.Tensor(
                (sequence_count, left_heads, sequence_length, rows), dtype=left_dtype
            ),
            right: T.Tensor(
                (sequence_count, right_heads, sequence_length, columns),
                dtype=right_dtype,
            ),
            readout: T.Tensor(
                (sequence_count, readout_heads, sequence_length, rows),
                dtype=readout_dtype,
            ),
            initial: T.Tensor((sequence_count, heads, rows, columns), dtype="float32"),
            output: T.Tensor(
                (sequence_count, heads, sequence_length, columns), dtype=output_dtype
            ),
            final: T.Tensor((sequence_count, heads, rows, columns), dtype="float32"),
        ):
            if False:
                _ = (
                    propagation_dtype,
                    left_dtype,
                    right_dtype,
                    readout_dtype,
                    output_dtype,
                )
                _ = propagation_shape
            with T.Kernel(
                sequence_count * heads,
                T.ceildiv(columns, block_columns),
                threads=threads,
            ) as (sequence_head, column_block):
                sequence = sequence_head // heads
                head = sequence_head % heads
                propagation_head = head // (heads // propagation_heads)
                left_head = head // (heads // left_heads)
                right_head = head // (heads // right_heads)
                readout_head = head // (heads // readout_heads)
                column_start = column_block * block_columns
                state = T.alloc_shared((rows, block_columns), dtype="float32")
                readout_shared = T.alloc_shared((chunk_tokens, rows), dtype="float32")
                left_shared = T.alloc_shared((chunk_tokens, rows), dtype="float32")
                right_shared = T.alloc_shared(
                    (chunk_tokens, block_columns), dtype="float32"
                )
                weighted_left = T.alloc_shared((chunk_tokens, rows), dtype="float32")
                scores_shared = T.alloc_shared(
                    (chunk_tokens, chunk_tokens), dtype="float32"
                )
                gates = T.alloc_shared((chunk_tokens,), dtype="float32")
                prefix = T.alloc_shared((chunk_tokens,), dtype="float32")
                suffix = T.alloc_shared((chunk_tokens,), dtype="float32")
                running_weight = T.alloc_fragment((chunk_tokens,), dtype="float32")
                scores = T.alloc_fragment((chunk_tokens, chunk_tokens), dtype="float32")
                output_fragment = T.alloc_fragment(
                    (chunk_tokens, block_columns), dtype="float32"
                )
                update_fragment = T.alloc_fragment(
                    (rows, block_columns), dtype="float32"
                )

                for row, column in T.Parallel(rows, block_columns):
                    if column_start + column < columns:
                        state[row, column] = initial[
                            sequence, head, row, column_start + column
                        ]
                    else:
                        state[row, column] = 0.0
                T.sync_threads()

                for chunk in T.Serial(chunk_count):
                    chunk_start = chunk * chunk_tokens
                    for token, row in T.Parallel(chunk_tokens, rows):
                        logical_token = chunk_start + token
                        if logical_token < sequence_length:
                            readout_value = T.cast(
                                readout[sequence, readout_head, logical_token, row],
                                "float32",
                            )
                            left_value = T.cast(
                                left[sequence, left_head, logical_token, row], "float32"
                            )
                            readout_value = readout_transform(readout_value)
                            left_value = left_transform(left_value)
                            readout_shared[token, row] = readout_value
                            left_shared[token, row] = left_value
                        else:
                            readout_shared[token, row] = 0.0
                            left_shared[token, row] = 0.0
                    for token, column in T.Parallel(chunk_tokens, block_columns):
                        logical_token = chunk_start + token
                        if (
                            logical_token < sequence_length
                            and column_start + column < columns
                        ):
                            right_value = T.cast(
                                right[
                                    sequence,
                                    right_head,
                                    logical_token,
                                    column_start + column,
                                ],
                                "float32",
                            )
                            right_value = right_transform(right_value)
                            right_shared[token, column] = right_value
                        else:
                            right_shared[token, column] = 0.0
                    if not identity_propagation:
                        if propagation_is_log_scale:
                            if T.get_thread_binding() == 0:
                                running_log = T.alloc_local((1,), dtype="float32")
                                running_log[0] = 0.0
                                for token in T.Serial(chunk_tokens):
                                    logical_token = chunk_start + token
                                    if logical_token < sequence_length:
                                        propagation_value = T.cast(
                                            propagation[
                                                sequence,
                                                propagation_head,
                                                logical_token,
                                            ],
                                            "float32",
                                        )
                                        running_log[0] += propagation_transform(
                                            propagation_value
                                        )
                                    prefix[token] = running_log[0]
                        else:
                            for token in T.Parallel(chunk_tokens):
                                logical_token = chunk_start + token
                                if logical_token >= sequence_length:
                                    gates[token] = 1.0
                                else:
                                    propagation_value = T.cast(
                                        propagation[
                                            sequence, propagation_head, logical_token
                                        ],
                                        "float32",
                                    )
                                    gates[token] = propagation_transform(
                                        propagation_value
                                    )
                            T.sync_threads()
                            if T.get_thread_binding() == 0:
                                running = T.alloc_local((1,), dtype="float32")
                                running[0] = 1.0
                                for token in T.Serial(chunk_tokens):
                                    running[0] *= gates[token]
                                    prefix[token] = running[0]
                                running[0] = 1.0
                                for reverse_token in T.Serial(chunk_tokens):
                                    token = chunk_tokens - reverse_token - 1
                                    suffix[token] = running[0]
                                    running[0] *= gates[token]
                            T.fill(running_weight, 1.0)
                            for reverse_source in T.Serial(chunk_tokens):
                                source = chunk_tokens - reverse_source - 1
                                for token in T.Parallel(chunk_tokens):
                                    scores_shared[token, source] = T.if_then_else(
                                        source <= token, running_weight[token], 0.0
                                    )
                                    if source <= token:
                                        running_weight[token] *= gates[source]
                    T.sync_threads()

                    T.clear(output_fragment)
                    T.gemm(
                        readout_shared,
                        state,
                        output_fragment,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    if not identity_propagation:
                        for token, column in T.Parallel(chunk_tokens, block_columns):
                            if propagation_is_log_scale:
                                output_fragment[token, column] *= T.exp(prefix[token])
                            else:
                                output_fragment[token, column] *= prefix[token]

                    T.clear(scores)
                    T.gemm(
                        readout_shared,
                        left_shared,
                        scores,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for token, source in T.Parallel(chunk_tokens, chunk_tokens):
                        if identity_propagation:
                            scores[token, source] = T.if_then_else(
                                source <= token, scores[token, source], 0.0
                            )
                        elif propagation_is_log_scale:
                            scores[token, source] = T.if_then_else(
                                source <= token,
                                scores[token, source]
                                * T.exp(prefix[token] - prefix[source]),
                                0.0,
                            )
                        else:
                            scores[token, source] *= scores_shared[token, source]
                    T.copy(scores, scores_shared)
                    T.gemm(
                        scores_shared,
                        right_shared,
                        output_fragment,
                        clear_accum=False,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for token, column in T.Parallel(chunk_tokens, block_columns):
                        logical_token = chunk_start + token
                        if (
                            logical_token < sequence_length
                            and column_start + column < columns
                        ):
                            output[
                                sequence,
                                head,
                                logical_token,
                                column_start + column,
                            ] = output_fragment[token, column]

                    for token, row in T.Parallel(chunk_tokens, rows):
                        if identity_propagation:
                            weighted_left[token, row] = left_shared[token, row]
                        elif propagation_is_log_scale:
                            weighted_left[token, row] = left_shared[token, row] * T.exp(
                                prefix[chunk_tokens - 1] - prefix[token]
                            )
                        else:
                            weighted_left[token, row] = (
                                left_shared[token, row] * suffix[token]
                            )
                    T.clear(update_fragment)
                    T.gemm(
                        weighted_left,
                        right_shared,
                        update_fragment,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for row, column in T.Parallel(rows, block_columns):
                        if identity_propagation:
                            state[row, column] += update_fragment[row, column]
                        elif propagation_is_log_scale:
                            state[row, column] = (
                                T.exp(prefix[chunk_tokens - 1]) * state[row, column]
                                + update_fragment[row, column]
                            )
                        else:
                            state[row, column] = (
                                prefix[chunk_tokens - 1] * state[row, column]
                                + update_fragment[row, column]
                            )
                    T.sync_threads()
                if return_final_state:
                    for row, column in T.Parallel(rows, block_columns):
                        if column_start + column < columns:
                            final[sequence, head, row, column_start + column] = state[
                                row, column
                            ]

        return kernel

    return build_kernel()


def tilelang_scalar_factorized_dense_raw(
    raw_inputs: dict[str, torch.Tensor],
    initial_state: torch.Tensor,
    recipe: Any,
    *,
    sequence_count: int,
    sequence_length: int,
    output_dtype: torch.dtype,
    return_final_state: bool,
    threads: int = 128,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    identity_propagation = recipe.propagation[0].name == "__identity_propagation__"

    def bind(binding):
        # All schedules erase identity propagation loads at compile time. Reuse
        # a live FP32 pointer rather than launching a dense tensor of ones.
        if binding[0].name == "__identity_propagation__":
            return initial_state
        return raw_inputs[binding[0].name]

    bindings = tuple(
        bind(item)
        for item in (
            recipe.propagation,
            recipe.left_factor,
            recipe.right_factor,
            recipe.readout,
        )
    )
    propagation, left, right, readout = bindings
    _, heads, rows, columns = initial_state.shape
    output = torch.empty(
        sequence_count,
        heads,
        sequence_length,
        columns,
        device=initial_state.device,
        dtype=output_dtype,
    )
    final = torch.empty_like(initial_state) if return_final_state else initial_state
    expressions = (
        recipe.propagation[1],
        recipe.left_factor[1],
        recipe.right_factor[1],
        recipe.readout[1],
    )
    # A chunk grid exposes sequence parallelism when B*H cannot fill the GPU.
    # Fixed-size padded State tiles also handle row tails that the resident
    # GEMM schedule cannot represent. Decide per call, including continuation.
    if (sequence_length >= 128 and sequence_count * heads <= 32) or rows % 32:
        from .scalar_kernels import scalar_chunked_dense_raw

        scalar_chunked_dense_raw(
            propagation,
            left,
            right,
            readout,
            initial_state,
            output,
            final,
            expressions,
            sequence_length=sequence_length,
            identity_propagation=identity_propagation,
            return_final_state=return_final_state,
            threads=threads,
        )
        return output, final if return_final_state else None
    kernel = _compile_scalar_factorized_dense_raw(
        sequence_count,
        sequence_length,
        heads,
        rows,
        columns,
        heads if identity_propagation else propagation.shape[1],
        left.shape[1],
        right.shape[1],
        readout.shape[1],
        propagation.dtype,
        left.dtype,
        right.dtype,
        readout.dtype,
        output_dtype,
        expressions[0].operation,
        expressions[1].operation,
        expressions[2].operation,
        expressions[3].operation,
        expressions[0].is_log_scale,
        expressions[1].is_log_scale,
        expressions[2].is_log_scale,
        expressions[3].is_log_scale,
        expressions[0].scale,
        expressions[1].scale,
        expressions[2].scale,
        expressions[3].scale,
        getattr(expressions[0], "offset", 0.0),
        getattr(expressions[1], "offset", 0.0),
        getattr(expressions[2], "offset", 0.0),
        getattr(expressions[3], "offset", 0.0),
        return_final_state,
        threads,
        identity_propagation,
        tuple(getattr(expression, "scale_steps", ()) for expression in expressions),
        tuple(
            getattr(expression, "expression_nodes", ()) for expression in expressions
        ),
    )
    kernel(propagation, left, right, readout, initial_state, output, final)
    return output, final if return_final_state else None


@lru_cache(maxsize=None)
def _compile_scalar_factorized_forward(
    sequence_count: int,
    max_sequence_length: int,
    token_count: int,
    heads: int,
    rows: int,
    columns: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    device_index: int,
    return_final_state: bool,
    threads: int,
    left_offset: float,
    right_offset: float,
    readout_offset: float,
):
    import tilelang
    import tilelang.language as T

    chunk_tokens = 32
    block_columns = 32
    chunk_count = tilelang.cdiv(max_sequence_length, chunk_tokens)

    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def build_kernel():
        @T.prim_func
        def kernel(
            scale: T.Tensor((heads, token_count), dtype="float32"),
            left: T.Tensor((heads, token_count, rows), dtype=input_dtype),
            right: T.Tensor((heads, token_count, columns), dtype=input_dtype),
            readout: T.Tensor((heads, token_count, rows), dtype=input_dtype),
            offsets: T.Tensor((sequence_count + 1,), dtype="int32"),
            initial: T.Tensor((sequence_count, heads, rows, columns), dtype="float32"),
            output: T.Tensor((token_count, heads, columns), dtype=output_dtype),
            final: T.Tensor((sequence_count, heads, rows, columns), dtype="float32"),
        ):
            if False:
                # tilelang resolves annotations through the kernel's free
                # variables, so names used only in a signature must be referenced.
                _ = (token_count, input_dtype, output_dtype)
            with T.Kernel(
                sequence_count * heads,
                T.ceildiv(columns, block_columns),
                threads=threads,
            ) as (sequence_head, column_block):
                sequence = sequence_head // heads
                head = sequence_head % heads
                column_start = column_block * block_columns
                sequence_start = offsets[sequence]
                sequence_length = offsets[sequence + 1] - sequence_start
                state = T.alloc_shared((rows, block_columns), dtype="float32")
                query_shared = T.alloc_shared((chunk_tokens, rows), dtype="float32")
                readout_float = T.alloc_shared((chunk_tokens, rows), dtype="float32")
                key_shared = T.alloc_shared((chunk_tokens, rows), dtype="float32")
                value_shared = T.alloc_shared(
                    (chunk_tokens, block_columns), dtype="float32"
                )
                weighted_key_shared = T.alloc_shared(
                    (chunk_tokens, rows), dtype="float32"
                )
                scores_shared = T.alloc_shared(
                    (chunk_tokens, chunk_tokens), dtype="float32"
                )
                log_prefix = T.alloc_shared((chunk_tokens,), dtype="float32")
                scores = T.alloc_fragment((chunk_tokens, chunk_tokens), dtype="float32")
                output_fragment = T.alloc_fragment(
                    (chunk_tokens, block_columns), dtype="float32"
                )
                update_fragment = T.alloc_fragment(
                    (rows, block_columns), dtype="float32"
                )

                for row, column in T.Parallel(rows, block_columns):
                    state[row, column] = initial[
                        sequence, head, row, column_start + column
                    ]
                T.sync_threads()

                for chunk in T.Serial(chunk_count):
                    chunk_start = chunk * chunk_tokens
                    for token, row in T.Parallel(chunk_tokens, rows):
                        logical_token = chunk_start + token
                        physical_token = sequence_start + logical_token
                        if logical_token < sequence_length:
                            query_shared[token, row] = (
                                readout[head, physical_token, row] + readout_offset
                            )
                            readout_float[token, row] = (
                                readout[head, physical_token, row] + readout_offset
                            )
                            key_shared[token, row] = (
                                left[head, physical_token, row] + left_offset
                            )
                        else:
                            query_shared[token, row] = 0.0
                            key_shared[token, row] = 0.0
                            readout_float[token, row] = 0.0
                    for token, column in T.Parallel(chunk_tokens, block_columns):
                        logical_token = chunk_start + token
                        physical_token = sequence_start + logical_token
                        if (
                            logical_token < sequence_length
                            and column_start + column < columns
                        ):
                            value_shared[token, column] = (
                                right[head, physical_token, column_start + column]
                                + right_offset
                            )
                        else:
                            value_shared[token, column] = 0.0
                    if T.get_thread_binding() == 0:
                        running_log = T.alloc_local((1,), dtype="float32")
                        running_log[0] = 0.0
                        for token in T.Serial(chunk_tokens):
                            logical_token = chunk_start + token
                            if logical_token < sequence_length:
                                physical_token = sequence_start + logical_token
                                running_log[0] += T.log(scale[head, physical_token])
                            log_prefix[token] = running_log[0]
                    T.sync_threads()

                    T.clear(output_fragment)
                    T.gemm(
                        readout_float,
                        state,
                        output_fragment,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for token, column in T.Parallel(chunk_tokens, block_columns):
                        output_fragment[token, column] *= T.exp(log_prefix[token])

                    T.clear(scores)
                    T.gemm(
                        query_shared,
                        key_shared,
                        scores,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for token, source in T.Parallel(chunk_tokens, chunk_tokens):
                        scores[token, source] = T.if_then_else(
                            source <= token,
                            scores[token, source]
                            * T.exp(log_prefix[token] - log_prefix[source]),
                            0.0,
                        )
                    T.copy(scores, scores_shared)
                    T.gemm(
                        scores_shared,
                        value_shared,
                        output_fragment,
                        clear_accum=False,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for token, column in T.Parallel(chunk_tokens, block_columns):
                        logical_token = chunk_start + token
                        if (
                            logical_token < sequence_length
                            and column_start + column < columns
                        ):
                            output[
                                sequence_start + logical_token,
                                head,
                                column_start + column,
                            ] = output_fragment[token, column]

                    end_log = log_prefix[
                        T.min(chunk_tokens - 1, sequence_length - chunk_start - 1)
                    ]
                    for token, row in T.Parallel(chunk_tokens, rows):
                        weighted_key_shared[token, row] = key_shared[
                            token, row
                        ] * T.exp(end_log - log_prefix[token])
                    T.clear(update_fragment)
                    T.gemm(
                        weighted_key_shared,
                        value_shared,
                        update_fragment,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for row, column in T.Parallel(rows, block_columns):
                        state[row, column] = (
                            T.exp(end_log) * state[row, column]
                            + update_fragment[row, column]
                        )
                    T.sync_threads()
                if return_final_state:
                    for row, column in T.Parallel(rows, block_columns):
                        if column_start + column < columns:
                            final[sequence, head, row, column_start + column] = state[
                                row, column
                            ]

        return kernel

    return build_kernel()


def tilelang_scalar_factorized_forward(
    scale: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    readout: torch.Tensor,
    initial_state: torch.Tensor,
    offsets: torch.Tensor,
    *,
    sequence_count: int,
    max_sequence_length: int,
    output_dtype: torch.dtype,
    return_final_state: bool,
    threads: int = 128,
    left_offset: float = 0.0,
    right_offset: float = 0.0,
    readout_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    heads, rows, columns = initial_state.shape[1:]
    token_count = scale.shape[0]
    output = torch.empty(
        token_count, heads, columns, device=left.device, dtype=output_dtype
    )
    final = torch.empty_like(initial_state) if return_final_state else initial_state
    device_index = left.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    kernel = _compile_scalar_factorized_forward(
        sequence_count,
        max_sequence_length,
        token_count,
        heads,
        rows,
        columns,
        left.dtype,
        output_dtype,
        device_index,
        return_final_state,
        threads,
        left_offset,
        right_offset,
        readout_offset,
    )
    kernel(
        scale.reshape(token_count, heads).transpose(0, 1).contiguous(),
        left.reshape(token_count, heads, rows).transpose(0, 1).contiguous(),
        right.reshape(token_count, heads, columns).transpose(0, 1).contiguous(),
        readout.reshape(token_count, heads, rows).transpose(0, 1).contiguous(),
        offsets,
        initial_state,
        output,
        final,
    )
    return output, final if return_final_state else None


@lru_cache(maxsize=None)
def _dense_offsets(
    sequence_count: int, sequence_length: int, device_index: int
) -> torch.Tensor:
    return torch.arange(
        0,
        (sequence_count + 1) * sequence_length,
        sequence_length,
        device=torch.device("cuda", device_index),
        dtype=torch.int32,
    )


def tilelang_scalar_factorized_dense(
    scale: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    readout: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    sequence_count: int,
    sequence_length: int,
    output_dtype: torch.dtype,
    return_final_state: bool,
    threads: int = 128,
    left_offset: float = 0.0,
    right_offset: float = 0.0,
    readout_offset: float = 0.0,
):
    device_index = scale.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    offsets = _dense_offsets(sequence_count, sequence_length, device_index)
    output, final = tilelang_scalar_factorized_forward(
        scale,
        left,
        right,
        readout,
        initial_state,
        offsets,
        sequence_count=sequence_count,
        max_sequence_length=sequence_length,
        output_dtype=output_dtype,
        return_final_state=return_final_state,
        threads=threads,
        left_offset=left_offset,
        right_offset=right_offset,
        readout_offset=readout_offset,
    )
    output = output.reshape(
        sequence_count, sequence_length, output.shape[1], output.shape[2]
    ).permute(0, 2, 1, 3)
    return output, final
