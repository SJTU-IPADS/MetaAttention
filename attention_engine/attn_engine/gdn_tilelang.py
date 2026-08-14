# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# Gated Delta Rule mathematics and observable behavior follow Flash Linear
# Attention, distributed under the MIT License. This H20 TileLang schedule is
# an independent, correctness-first implementation.

from functools import lru_cache

import torch
import tilelang as tl
import tilelang.language as T


HEAD_DIM = 128
CHUNK_SIZE = 64
VALUE_TILE = 8
VALUE_TILES = HEAD_DIM // VALUE_TILE
THREADS = 128
_PASS_CONFIGS = {"tl.disable_warp_specialized": True}


@lru_cache(maxsize=None)
def _compile_gate_cumsum(batch: int, value_heads: int, length: int):
    @T.prim_func
    def gate_cumsum_kernel(
        gate: T.Tensor((batch, value_heads, length), "float32"),
        cumulative_gate: T.Tensor((batch, value_heads, length), "float32"),
    ):
        with T.Kernel(batch * value_heads, threads=THREADS) as batch_head:
            running = T.alloc_fragment((1,), "float32")
            batch_index = batch_head // value_heads
            value_head = batch_head % value_heads
            for chunk in T.serial(length // CHUNK_SIZE):
                running[0] = 0.0
                for offset in T.serial(CHUNK_SIZE):
                    token = chunk * CHUNK_SIZE + offset
                    running[0] = running[0] + gate[batch_index, value_head, token]
                    cumulative_gate[batch_index, value_head, token] = running[0]

    return tl.compile(
        gate_cumsum_kernel,
        out_idx=[1],
        pass_configs=_PASS_CONFIGS,
        execution_backend="cython",
    )


@lru_cache(maxsize=None)
def _compile_kkt(batch: int, query_heads: int, value_heads: int, length: int):
    groups = value_heads // query_heads

    @T.prim_func
    def kkt_kernel(
        key: T.Tensor((batch, query_heads, length, HEAD_DIM), "bfloat16"),
        beta: T.Tensor((batch, value_heads, length), "float32"),
        cumulative_gate: T.Tensor((batch, value_heads, length), "float32"),
        key_inverse: T.Tensor((batch, value_heads, length, CHUNK_SIZE), "float32"),
        value_inverse: T.Tensor((batch, value_heads, length, CHUNK_SIZE), "float32"),
    ):
        with T.Kernel(length // CHUNK_SIZE, batch * value_heads, threads=THREADS) as (chunk, batch_head):
            key_matrix = T.alloc_fragment((CHUNK_SIZE, CHUNK_SIZE), "float32")
            value_matrix = T.alloc_fragment((CHUNK_SIZE, CHUNK_SIZE), "float32")
            key_row = T.alloc_fragment((CHUNK_SIZE,), "float32")
            value_row = T.alloc_fragment((CHUNK_SIZE,), "float32")
            batch_index = batch_head // value_heads
            value_head = batch_head % value_heads
            query_head = value_head // groups

            for row, column in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                key_matrix[row, column] = 0.0
                value_matrix[row, column] = 0.0
            for row in T.serial(CHUNK_SIZE):
                for column in T.serial(row):
                    dot = T.alloc_fragment((1,), "float32")
                    dot[0] = 0.0
                    for dim in T.serial(HEAD_DIM):
                        dot[0] = dot[0] + T.cast(
                            key[batch_index, query_head, chunk * CHUNK_SIZE + row, dim],
                            "float32",
                        ) * T.cast(
                            key[batch_index, query_head, chunk * CHUNK_SIZE + column, dim],
                            "float32",
                        )
                    base = -beta[batch_index, value_head, chunk * CHUNK_SIZE + row] * dot[0]
                    key_matrix[row, column] = base
                    value_matrix[row, column] = base

            for row in T.serial(CHUNK_SIZE):
                for column in T.Parallel(CHUNK_SIZE):
                    key_row[column] = key_matrix[row, column]
                    value_row[column] = value_matrix[row, column]
                for column in T.serial(row):
                    key_value = T.alloc_fragment((1,), "float32")
                    value_value = T.alloc_fragment((1,), "float32")
                    key_value[0] = key_row[column]
                    value_value[0] = value_row[column]
                    for inner in T.serial(column + 1, row):
                        key_value[0] = key_value[0] + key_row[inner] * key_matrix[inner, column]
                        value_value[0] = value_value[0] + value_row[inner] * value_matrix[inner, column]
                    key_matrix[row, column] = key_value[0]
                    value_matrix[row, column] = value_value[0]
                key_matrix[row, row] = 1.0
                value_matrix[row, row] = 1.0

            for row, column in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                key_inverse[
                    batch_index, value_head, chunk * CHUNK_SIZE + row, column
                ] = key_matrix[row, column]
                value_inverse[
                    batch_index, value_head, chunk * CHUNK_SIZE + row, column
                ] = value_matrix[row, column]

    return tl.compile(
        kkt_kernel,
        out_idx=[3, 4],
        pass_configs=_PASS_CONFIGS,
        execution_backend="cython",
    )


@lru_cache(maxsize=None)
def _compile_wy(batch: int, query_heads: int, value_heads: int, length: int):
    groups = value_heads // query_heads

    @T.prim_func
    def wy_kernel(
        key: T.Tensor((batch, query_heads, length, HEAD_DIM), "bfloat16"),
        value: T.Tensor((batch, value_heads, length, HEAD_DIM), "bfloat16"),
        beta: T.Tensor((batch, value_heads, length), "float32"),
        cumulative_gate: T.Tensor((batch, value_heads, length), "float32"),
        key_inverse: T.Tensor((batch, value_heads, length, CHUNK_SIZE), "float32"),
        value_inverse: T.Tensor((batch, value_heads, length, CHUNK_SIZE), "float32"),
        corrected_key: T.Tensor((batch, value_heads, length, HEAD_DIM), "float32"),
        corrected_value: T.Tensor((batch, value_heads, length, HEAD_DIM), "float32"),
    ):
        with T.Kernel(VALUE_TILES, batch * value_heads, threads=THREADS) as (tile, batch_head):
            batch_index = batch_head // value_heads
            value_head = batch_head % value_heads
            query_head = value_head // groups
            for token in T.serial(length):
                chunk_start = token // CHUNK_SIZE * CHUNK_SIZE
                local_row = token % CHUNK_SIZE
                for column in T.serial(VALUE_TILE):
                    key_result = T.alloc_fragment((1,), "float32")
                    value_result = T.alloc_fragment((1,), "float32")
                    key_result[0] = 0.0
                    value_result[0] = 0.0
                    for local_source in T.serial(local_row + 1):
                        source = chunk_start + local_source
                        key_result[0] = key_result[0] + T.cast(
                            key_inverse[batch_index, value_head, token, local_source], "float32"
                        ) * beta[batch_index, value_head, source] * T.cast(
                            key[batch_index, query_head, source, tile * VALUE_TILE + column], "float32"
                        )
                        value_result[0] = value_result[0] + T.cast(
                            value_inverse[batch_index, value_head, token, local_source], "float32"
                        ) * beta[batch_index, value_head, source] * T.exp(
                            cumulative_gate[batch_index, value_head, token]
                            - cumulative_gate[batch_index, value_head, source]
                        ) * T.cast(value[batch_index, value_head, source, tile * VALUE_TILE + column], "float32")
                    corrected_key[batch_index, value_head, token, tile * VALUE_TILE + column] = key_result[0]
                    corrected_value[batch_index, value_head, token, tile * VALUE_TILE + column] = value_result[0]

    return tl.compile(
        wy_kernel,
        out_idx=[6, 7],
        pass_configs=_PASS_CONFIGS,
        execution_backend="cython",
    )


@lru_cache(maxsize=None)
def _compile_state_output(
    batch: int,
    query_heads: int,
    value_heads: int,
    length: int,
    scale: float,
    store_final_state: bool,
):
    groups = value_heads // query_heads
    final_shape = (batch, value_heads, HEAD_DIM, HEAD_DIM) if store_final_state else (1,)

    @T.prim_func
    def state_output_kernel(
        query: T.Tensor((batch, query_heads, length, HEAD_DIM), "bfloat16"),
        key: T.Tensor((batch, query_heads, length, HEAD_DIM), "bfloat16"),
        cumulative_gate: T.Tensor((batch, value_heads, length), "float32"),
        corrected_key: T.Tensor((batch, value_heads, length, HEAD_DIM), "float32"),
        corrected_value: T.Tensor((batch, value_heads, length, HEAD_DIM), "float32"),
        initial_state: T.Tensor((batch, value_heads, HEAD_DIM, HEAD_DIM), "float32"),
        output: T.Tensor((batch, value_heads, length, HEAD_DIM), "bfloat16"),
        final_state: T.Tensor(final_shape, "float32"),
    ):
        with T.Kernel(VALUE_TILES, batch * value_heads, threads=THREADS) as (tile, batch_head):
            state = T.alloc_fragment((HEAD_DIM, VALUE_TILE), "float32")
            new_value = T.alloc_fragment((CHUNK_SIZE, VALUE_TILE), "float32")
            batch_index = batch_head // value_heads
            value_head = batch_head % value_heads
            query_head = value_head // groups
            for state_dim, state_column in T.Parallel(HEAD_DIM, VALUE_TILE):
                state[state_dim, state_column] = initial_state[
                    batch_index, value_head, state_dim, tile * VALUE_TILE + state_column
                ]

            for chunk in T.serial(length // CHUNK_SIZE):
                chunk_start = chunk * CHUNK_SIZE
                for row in T.serial(CHUNK_SIZE):
                    token = chunk_start + row
                    row_gate = T.exp(cumulative_gate[batch_index, value_head, token])
                    for column in T.serial(VALUE_TILE):
                        correction = T.alloc_fragment((1,), "float32")
                        correction[0] = 0.0
                        for dim in T.serial(HEAD_DIM):
                            correction[0] = correction[0] + corrected_key[
                                batch_index, value_head, token, dim
                            ] * row_gate * state[dim, column]
                        new_value[row, column] = corrected_value[
                            batch_index, value_head, token, tile * VALUE_TILE + column
                        ] - correction[0]

                for row in T.serial(CHUNK_SIZE):
                    token = chunk_start + row
                    row_gate = cumulative_gate[batch_index, value_head, token]
                    for column in T.serial(VALUE_TILE):
                        result = T.alloc_fragment((1,), "float32")
                        result[0] = 0.0
                        for dim in T.serial(HEAD_DIM):
                            result[0] = result[0] + T.cast(
                                query[batch_index, query_head, token, dim], "float32"
                            ) * T.exp(row_gate) * state[dim, column]
                        for source_row in T.serial(row + 1):
                            source = chunk_start + source_row
                            score = T.alloc_fragment((1,), "float32")
                            score[0] = 0.0
                            for dim in T.serial(HEAD_DIM):
                                score[0] = score[0] + T.cast(
                                    query[batch_index, query_head, token, dim], "float32"
                                ) * T.cast(
                                    key[batch_index, query_head, source, dim], "float32"
                                )
                            result[0] = result[0] + score[0] * T.exp(
                                row_gate
                                - cumulative_gate[batch_index, value_head, source]
                            ) * new_value[source_row, column]
                        output[
                            batch_index, value_head, token, tile * VALUE_TILE + column
                        ] = T.cast(result[0] * scale, "bfloat16")

                chunk_gate = cumulative_gate[
                    batch_index, value_head, chunk_start + CHUNK_SIZE - 1
                ]
                for dim, column in T.Parallel(HEAD_DIM, VALUE_TILE):
                    state[dim, column] = state[dim, column] * T.exp(chunk_gate)
                for source_row in T.serial(CHUNK_SIZE):
                    source = chunk_start + source_row
                    source_decay = T.exp(
                        chunk_gate - cumulative_gate[batch_index, value_head, source]
                    )
                    for dim in T.serial(HEAD_DIM):
                        key_value = T.cast(
                            key[batch_index, query_head, source, dim], "float32"
                        ) * source_decay
                        for column in T.Parallel(VALUE_TILE):
                            state[dim, column] = (
                                state[dim, column] + key_value * new_value[source_row, column]
                            )

            if store_final_state:
                for state_dim, state_column in T.Parallel(HEAD_DIM, VALUE_TILE):
                    final_state[
                        batch_index, value_head, state_dim, tile * VALUE_TILE + state_column
                    ] = state[state_dim, state_column]

    return tl.compile(
        state_output_kernel,
        out_idx=[6, 7],
        pass_configs=_PASS_CONFIGS,
        execution_backend="cython",
    )


@lru_cache(maxsize=None)
def _compile_chunk_states(batch: int, query_heads: int, value_heads: int, length: int):
    groups = value_heads // query_heads
    chunk_count = length // CHUNK_SIZE

    @T.prim_func
    def chunk_state_kernel(
        key: T.Tensor((batch, query_heads, length, HEAD_DIM), "bfloat16"),
        value: T.Tensor((batch, value_heads, length, HEAD_DIM), "bfloat16"),
        gate: T.Tensor((batch, value_heads, length), "float32"),
        beta: T.Tensor((batch, value_heads, length), "float32"),
        initial_state: T.Tensor((batch, value_heads, HEAD_DIM, HEAD_DIM), "float32"),
        chunk_states: T.Tensor((batch, value_heads, chunk_count, HEAD_DIM, HEAD_DIM), "float32"),
    ):
        with T.Kernel(VALUE_TILES, batch * value_heads, threads=THREADS) as (tile, batch_head):
            state = T.alloc_fragment((HEAD_DIM, VALUE_TILE), "float32")
            prediction = T.alloc_fragment((VALUE_TILE,), "float32")
            residual = T.alloc_fragment((VALUE_TILE,), "float32")
            batch_index = batch_head // value_heads
            value_head = batch_head % value_heads
            query_head = value_head // groups
            for state_dim, state_column in T.Parallel(HEAD_DIM, VALUE_TILE):
                state[state_dim, state_column] = initial_state[batch_index, value_head, state_dim, tile * VALUE_TILE + state_column]
            for chunk in T.serial(chunk_count):
                for state_dim, state_column in T.Parallel(HEAD_DIM, VALUE_TILE):
                    chunk_states[batch_index, value_head, chunk, state_dim, tile * VALUE_TILE + state_column] = state[state_dim, state_column]
                for offset in T.serial(CHUNK_SIZE):
                    token = chunk * CHUNK_SIZE + offset
                    decay = T.exp(gate[batch_index, value_head, token])
                    for state_dim, state_column in T.Parallel(HEAD_DIM, VALUE_TILE):
                        state[state_dim, state_column] = state[state_dim, state_column] * decay
                    for output_column in T.Parallel(VALUE_TILE):
                        prediction[output_column] = 0.0
                    for state_dim in T.serial(HEAD_DIM):
                        key_value = T.cast(key[batch_index, query_head, token, state_dim], "float32")
                        for output_column in T.Parallel(VALUE_TILE):
                            prediction[output_column] = prediction[output_column] + key_value * state[state_dim, output_column]
                    for output_column in T.Parallel(VALUE_TILE):
                        residual[output_column] = beta[batch_index, value_head, token] * (T.cast(value[batch_index, value_head, token, tile * VALUE_TILE + output_column], "float32") - prediction[output_column])
                    for state_dim in T.serial(HEAD_DIM):
                        key_value = T.cast(key[batch_index, query_head, token, state_dim], "float32")
                        for state_column in T.Parallel(VALUE_TILE):
                            state[state_dim, state_column] = state[state_dim, state_column] + key_value * residual[state_column]

    return tl.compile(chunk_state_kernel, out_idx=[5], pass_configs=_PASS_CONFIGS, execution_backend="cython")


@lru_cache(maxsize=None)
def _compile_backward(batch: int, query_heads: int, value_heads: int, length: int, scale: float):
    groups = value_heads // query_heads

    @T.prim_func
    def backward_kernel(
        query: T.Tensor((batch, query_heads, length, HEAD_DIM), "bfloat16"),
        key: T.Tensor((batch, query_heads, length, HEAD_DIM), "bfloat16"),
        value: T.Tensor((batch, value_heads, length, HEAD_DIM), "bfloat16"),
        gate: T.Tensor((batch, value_heads, length), "float32"),
        beta: T.Tensor((batch, value_heads, length), "float32"),
        initial_state: T.Tensor((batch, value_heads, HEAD_DIM, HEAD_DIM), "float32"),
        chunk_states: T.Tensor((batch, value_heads, length // CHUNK_SIZE, HEAD_DIM, HEAD_DIM), "float32"),
        output_grad: T.Tensor((batch, value_heads, length, HEAD_DIM), "bfloat16"),
        final_state_grad: T.Tensor((batch, value_heads, HEAD_DIM, HEAD_DIM), "float32"),
        query_grad_parts: T.Tensor((batch, value_heads, length, HEAD_DIM, VALUE_TILES), "float32"),
        key_grad_parts: T.Tensor((batch, value_heads, length, HEAD_DIM, VALUE_TILES), "float32"),
        value_grad: T.Tensor((batch, value_heads, length, HEAD_DIM), "float32"),
        gate_grad_parts: T.Tensor((batch, value_heads, length, VALUE_TILES), "float32"),
        beta_grad_parts: T.Tensor((batch, value_heads, length, VALUE_TILES), "float32"),
        initial_state_grad: T.Tensor((batch, value_heads, HEAD_DIM, HEAD_DIM), "float32"),
    ):
        with T.Kernel(VALUE_TILES, batch * value_heads, threads=THREADS) as (tile, batch_head):
            state = T.alloc_fragment((HEAD_DIM, VALUE_TILE), "float32")
            state_grad = T.alloc_fragment((HEAD_DIM, VALUE_TILE), "float32")
            residual = T.alloc_fragment((VALUE_TILE,), "float32")
            residual_grad = T.alloc_fragment((VALUE_TILE,), "float32")
            prediction = T.alloc_fragment((VALUE_TILE,), "float32")
            prediction_grad = T.alloc_fragment((VALUE_TILE,), "float32")
            batch_index = batch_head // value_heads
            value_head = batch_head % value_heads
            query_head = value_head // groups

            for grad_dim, grad_column in T.Parallel(HEAD_DIM, VALUE_TILE):
                state_grad[grad_dim, grad_column] = final_state_grad[
                    batch_index, value_head, grad_dim, tile * VALUE_TILE + grad_column
                ]

            for reverse_chunk in T.serial(length // CHUNK_SIZE):
                chunk = length // CHUNK_SIZE - reverse_chunk - 1
                chunk_start = chunk * CHUNK_SIZE
                for reverse_offset in T.serial(CHUNK_SIZE):
                    offset = CHUNK_SIZE - reverse_offset - 1
                    token = chunk_start + offset
                    decay = T.exp(gate[batch_index, value_head, token])
                    for state_dim, state_column in T.Parallel(HEAD_DIM, VALUE_TILE):
                        state[state_dim, state_column] = chunk_states[
                            batch_index,
                            value_head,
                            chunk,
                            state_dim,
                            tile * VALUE_TILE + state_column,
                        ]
                    for replay in T.serial(CHUNK_SIZE):
                        if replay < offset:
                            replay_token = chunk_start + replay
                            replay_decay = T.exp(gate[batch_index, value_head, replay_token])
                            for replay_dim, replay_column in T.Parallel(HEAD_DIM, VALUE_TILE):
                                state[replay_dim, replay_column] = (
                                    state[replay_dim, replay_column] * replay_decay
                                )
                            for prediction_column in T.Parallel(VALUE_TILE):
                                prediction[prediction_column] = 0.0
                            for replay_dim in T.serial(HEAD_DIM):
                                key_value = T.cast(
                                    key[batch_index, query_head, replay_token, replay_dim],
                                    "float32",
                                )
                                for prediction_column in T.Parallel(VALUE_TILE):
                                    prediction[prediction_column] = (
                                        prediction[prediction_column]
                                        + key_value * state[replay_dim, prediction_column]
                                    )
                            for residual_column in T.Parallel(VALUE_TILE):
                                residual[residual_column] = beta[
                                    batch_index, value_head, replay_token
                                ] * (
                                    T.cast(
                                        value[
                                            batch_index,
                                            value_head,
                                            replay_token,
                                            tile * VALUE_TILE + residual_column,
                                        ],
                                        "float32",
                                    )
                                    - prediction[residual_column]
                                )
                            for replay_dim in T.serial(HEAD_DIM):
                                key_value = T.cast(
                                    key[batch_index, query_head, replay_token, replay_dim],
                                    "float32",
                                )
                                for residual_column in T.Parallel(VALUE_TILE):
                                    state[replay_dim, residual_column] = (
                                        state[replay_dim, residual_column]
                                        + key_value * residual[residual_column]
                                    )
                    for dim, column in T.Parallel(HEAD_DIM, VALUE_TILE):
                        state[dim, column] = state[dim, column] * decay
                    for column in T.Parallel(VALUE_TILE):
                        prediction[column] = 0.0
                    for dim in T.serial(HEAD_DIM):
                        key_value = T.cast(
                            key[batch_index, query_head, token, dim], "float32"
                        )
                        for column in T.Parallel(VALUE_TILE):
                            prediction[column] = (
                                prediction[column] + key_value * state[dim, column]
                            )
                    for column in T.Parallel(VALUE_TILE):
                        residual[column] = beta[batch_index, value_head, token] * (
                            T.cast(
                                value[
                                    batch_index,
                                    value_head,
                                    token,
                                    tile * VALUE_TILE + column,
                                ],
                                "float32",
                            )
                            - prediction[column]
                        )
                    for dim, column in T.Parallel(HEAD_DIM, VALUE_TILE):
                        state[dim, column] = state[dim, column] + T.cast(
                            key[batch_index, query_head, token, dim], "float32"
                        ) * residual[column]

                    for dim in T.serial(HEAD_DIM):
                        query_part = T.alloc_fragment((1,), "float32")
                        query_part[0] = 0.0
                        for column in T.serial(VALUE_TILE):
                            output_gradient = T.cast(
                                output_grad[
                                    batch_index,
                                    value_head,
                                    token,
                                    tile * VALUE_TILE + column,
                                ],
                                "float32",
                            )
                            query_part[0] = (
                                query_part[0]
                                + state[dim, column] * output_gradient * scale
                            )
                            state_grad[dim, column] = (
                                state_grad[dim, column]
                                + T.cast(
                                    query[batch_index, query_head, token, dim], "float32"
                                )
                                * scale
                                * output_gradient
                            )
                        query_grad_parts[
                            batch_index, value_head, token, dim, tile
                        ] = query_part[0]

                    for column in T.Parallel(VALUE_TILE):
                        residual_grad[column] = 0.0
                    for dim in T.serial(HEAD_DIM):
                        key_value = T.cast(
                            key[batch_index, query_head, token, dim], "float32"
                        )
                        for column in T.Parallel(VALUE_TILE):
                            residual_grad[column] = (
                                residual_grad[column] + key_value * state_grad[dim, column]
                            )
                    beta_part = T.alloc_fragment((1,), "float32")
                    beta_part[0] = 0.0
                    for column in T.serial(VALUE_TILE):
                        beta_part[0] = beta_part[0] + residual_grad[column] * (
                            T.cast(
                                value[
                                    batch_index,
                                    value_head,
                                    token,
                                    tile * VALUE_TILE + column,
                                ],
                                "float32",
                            )
                            - prediction[column]
                        )
                        value_grad[
                            batch_index,
                            value_head,
                            token,
                            tile * VALUE_TILE + column,
                        ] = residual_grad[column] * beta[batch_index, value_head, token]
                        prediction_grad[column] = -residual_grad[column] * beta[
                            batch_index, value_head, token
                        ]
                    beta_grad_parts[batch_index, value_head, token, tile] = beta_part[0]

                    gate_part = T.alloc_fragment((1,), "float32")
                    gate_part[0] = 0.0
                    for dim in T.serial(HEAD_DIM):
                        key_part = T.alloc_fragment((1,), "float32")
                        key_part[0] = 0.0
                        key_value = T.cast(
                            key[batch_index, query_head, token, dim], "float32"
                        )
                        for column in T.serial(VALUE_TILE):
                            pre_update_state = state[dim, column] - key_value * residual[column]
                            key_part[0] = (
                                key_part[0] + state_grad[dim, column] * residual[column]
                            )
                            key_part[0] = (
                                key_part[0] + pre_update_state * prediction_grad[column]
                            )
                            state_grad[dim, column] = (
                                state_grad[dim, column] + key_value * prediction_grad[column]
                            )
                            gate_part[0] = (
                                gate_part[0] + state_grad[dim, column] * pre_update_state
                            )
                            state_grad[dim, column] = state_grad[dim, column] * decay
                        key_grad_parts[
                            batch_index, value_head, token, dim, tile
                        ] = key_part[0]
                    gate_grad_parts[
                        batch_index, value_head, token, tile
                    ] = gate_part[0]

            for dim, column in T.Parallel(HEAD_DIM, VALUE_TILE):
                initial_state_grad[
                    batch_index, value_head, dim, tile * VALUE_TILE + column
                ] = state_grad[dim, column]

    return tl.compile(
        backward_kernel,
        out_idx=[9, 10, 11, 12, 13, 14],
        pass_configs=_PASS_CONFIGS,
        execution_backend="cython",
    )


@lru_cache(maxsize=None)
def _compile_reduce_gradients(batch: int, query_heads: int, value_heads: int, length: int):
    groups = value_heads // query_heads

    @T.prim_func
    def reduce_kernel(
        query_grad_parts: T.Tensor((batch, value_heads, length, HEAD_DIM, VALUE_TILES), "float32"),
        key_grad_parts: T.Tensor((batch, value_heads, length, HEAD_DIM, VALUE_TILES), "float32"),
        value_grad_float: T.Tensor((batch, value_heads, length, HEAD_DIM), "float32"),
        gate_grad_parts: T.Tensor((batch, value_heads, length, VALUE_TILES), "float32"),
        beta_grad_parts: T.Tensor((batch, value_heads, length, VALUE_TILES), "float32"),
        query_grad: T.Tensor((batch, query_heads, length, HEAD_DIM), "bfloat16"),
        key_grad: T.Tensor((batch, query_heads, length, HEAD_DIM), "bfloat16"),
        value_grad: T.Tensor((batch, value_heads, length, HEAD_DIM), "bfloat16"),
        gate_grad: T.Tensor((batch, value_heads, length), "float32"),
        beta_grad: T.Tensor((batch, value_heads, length), "float32"),
    ):
        with T.Kernel(batch * query_heads, threads=THREADS) as batch_head:
            batch_index = batch_head // query_heads
            query_head = batch_head % query_heads
            for token in T.serial(length):
                for dim in T.serial(HEAD_DIM):
                    dq = T.alloc_fragment((1,), "float32")
                    dk = T.alloc_fragment((1,), "float32")
                    dq[0] = 0.0
                    dk[0] = 0.0
                    for group in T.serial(groups):
                        grouped_value_head = query_head * groups + group
                        for tile in T.serial(VALUE_TILES):
                            dq[0] = dq[0] + query_grad_parts[batch_index, grouped_value_head, token, dim, tile]
                            dk[0] = dk[0] + key_grad_parts[batch_index, grouped_value_head, token, dim, tile]
                    query_grad[batch_index, query_head, token, dim] = T.cast(dq[0], "bfloat16")
                    key_grad[batch_index, query_head, token, dim] = T.cast(dk[0], "bfloat16")

        with T.Kernel(batch * value_heads, threads=THREADS) as batch_head:
            batch_index = batch_head // value_heads
            reduction_value_head = batch_head % value_heads
            for token in T.serial(length):
                dg = T.alloc_fragment((1,), "float32")
                db = T.alloc_fragment((1,), "float32")
                dg[0] = 0.0
                db[0] = 0.0
                for tile in T.serial(VALUE_TILES):
                    dg[0] = dg[0] + gate_grad_parts[batch_index, reduction_value_head, token, tile]
                    db[0] = db[0] + beta_grad_parts[batch_index, reduction_value_head, token, tile]
                gate_grad[batch_index, reduction_value_head, token] = dg[0]
                beta_grad[batch_index, reduction_value_head, token] = db[0]
                for grad_dim in T.Parallel(HEAD_DIM):
                    value_grad[batch_index, reduction_value_head, token, grad_dim] = T.cast(
                        value_grad_float[batch_index, reduction_value_head, token, grad_dim], "bfloat16"
                    )

    return tl.compile(
        reduce_kernel,
        out_idx=[5, 6, 7, 8, 9],
        pass_configs=_PASS_CONFIGS,
        execution_backend="cython",
    )


class _GatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, query, key, value, gate, beta, scale, initial_state, has_initial_state, output_final_state
    ):
        batch, query_heads, length, _ = query.shape
        value_heads = value.shape[1]
        cumulative_gate = _compile_gate_cumsum(batch, value_heads, length)(gate)
        key_inverse, value_inverse = _compile_kkt(batch, query_heads, value_heads, length)(
            key, beta, cumulative_gate
        )
        corrected_key, corrected_value = _compile_wy(batch, query_heads, value_heads, length)(
            key, value, beta, cumulative_gate, key_inverse, value_inverse
        )
        output, final_state = _compile_state_output(
            batch, query_heads, value_heads, length, scale, output_final_state
        )(query, key, cumulative_gate, corrected_key, corrected_value, initial_state)
        ctx.save_for_backward(
            query, key, value, gate, beta, cumulative_gate,
            key_inverse, value_inverse, initial_state,
        )
        ctx.scale = scale
        ctx.has_initial_state = has_initial_state
        ctx.output_final_state = output_final_state
        return output, final_state

    @staticmethod
    def backward(ctx, output_grad, final_state_grad):
        query, key, value, gate, beta, _, _, _, initial_state = ctx.saved_tensors
        batch, query_heads, length, _ = query.shape
        value_heads = value.shape[1]
        if output_grad is None:
            output_grad = torch.zeros_like(value)
        if not ctx.output_final_state or final_state_grad is None:
            final_state_grad = torch.zeros_like(initial_state)

        chunk_states = _compile_chunk_states(
            batch, query_heads, value_heads, length
        )(key, value, gate, beta, initial_state)
        (
            query_grad_parts,
            key_grad_parts,
            value_grad_float,
            gate_grad_parts,
            beta_grad_parts,
            initial_state_grad,
        ) = _compile_backward(
            batch, query_heads, value_heads, length, ctx.scale
        )(
            query,
            key,
            value,
            gate,
            beta,
            initial_state,
            chunk_states,
            output_grad.contiguous(),
            final_state_grad.contiguous(),
        )
        query_grad, key_grad, value_grad, gate_grad, beta_grad = _compile_reduce_gradients(
            batch, query_heads, value_heads, length
        )(
            query_grad_parts,
            key_grad_parts,
            value_grad_float,
            gate_grad_parts,
            beta_grad_parts,
        )
        if not ctx.has_initial_state:
            initial_state_grad = None
        return (
            query_grad,
            key_grad,
            value_grad,
            gate_grad,
            beta_grad,
            None,
            initial_state_grad,
            None,
            None,
        )


def gated_delta_rule(
    query,
    key,
    value,
    gate,
    beta,
    *,
    scale,
    initial_state,
    output_final_state,
):
    has_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.zeros(
            query.shape[0],
            value.shape[1],
            HEAD_DIM,
            HEAD_DIM,
            dtype=torch.float32,
            device=query.device,
        )
    output, final_state = _GatedDeltaRule.apply(
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
