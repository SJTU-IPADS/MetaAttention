"""Tiled vector propagation for feature-gated rank-two recurrences.

This kernel is intentionally narrow: one first-Feature vector gate, one
rank-one injection, and a first-Feature readout.  Each block owns one
sequence/head and a tile of State columns.  A thread carries one State column
and advances the whole key vector in registers, so no token-by-State
propagation summary is materialized.
"""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def _compile_vector_propagation(
    sequence_count: int,
    sequence_length: int,
    heads: int,
    rows: int,
    columns: int,
    output_dtype: torch.dtype,
    thread_columns: int,
    device_index: int,
):
    import tilelang
    import tilelang.language as T

    block_columns = min(columns, thread_columns)
    column_blocks = tilelang.cdiv(columns, block_columns)

    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def build_kernel():
        @T.prim_func
        def vector_propagation_kernel(
            log_gate: T.Tensor(
                (sequence_count * sequence_length, heads, rows), "float32"
            ),
            key: T.Tensor((sequence_count * sequence_length, heads, rows), "float32"),
            value: T.Tensor(
                (sequence_count * sequence_length, heads, columns), "float32"
            ),
            query: T.Tensor((sequence_count * sequence_length, heads, rows), "float32"),
            initial: T.Tensor((sequence_count, heads, rows, columns), "float32"),
            output: T.Tensor(
                (sequence_count, heads, sequence_length, columns), dtype=output_dtype
            ),
            final: T.Tensor((sequence_count, heads, rows, columns), "float32"),
        ):
            if False:
                _ = (
                    sequence_count,
                    sequence_length,
                    heads,
                    rows,
                    columns,
                    column_blocks,
                    output_dtype,
                )
            with T.Kernel(
                sequence_count * heads, column_blocks, threads=block_columns
            ) as (sequence_head, column_block):
                sequence = sequence_head // heads
                head = sequence_head % heads
                column = column_block * block_columns + T.get_thread_binding()
                state = T.alloc_local((rows,), dtype="float32")
                accumulated = T.alloc_local((1,), dtype="float32")
                gate_shared = T.alloc_shared((rows,), dtype="float32")
                if column < columns:
                    for row in T.serial(rows):
                        state[row] = initial[sequence, head, row, column]
                for token in T.serial(sequence_length):
                    flat = sequence * sequence_length + token
                    for row in T.Parallel(rows):
                        gate_shared[row] = T.exp(log_gate[flat, head, row])
                    T.sync_threads()
                    if column < columns:
                        for row in T.serial(rows):
                            state[row] = (
                                gate_shared[row] * state[row]
                                + key[flat, head, row] * value[flat, head, column]
                            )
                        accumulated[0] = 0.0
                        for row in T.serial(rows):
                            accumulated[0] = (
                                accumulated[0] + query[flat, head, row] * state[row]
                            )
                        output[sequence, head, token, column] = accumulated[0]
                    T.sync_threads()
                if column < columns:
                    for row in T.serial(rows):
                        final[sequence, head, row, column] = state[row]

        return vector_propagation_kernel

    return build_kernel()


def vector_propagation_dense(
    *,
    log_gate: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query: torch.Tensor,
    initial_state: torch.Tensor,
    output_dtype: torch.dtype,
    thread_columns: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute feature-gated recurrence with ``exp(log_gate)`` in the kernel."""
    if initial_state.ndim != 4:
        raise ValueError("vector propagation requires a rank-two State")
    if not all(
        tensor.device.type == "cuda"
        for tensor in (log_gate, key, value, query, initial_state)
    ):
        raise ValueError("vector propagation requires CUDA tensors")
    sequence_count, heads, rows, columns = initial_state.shape
    if any(tensor.ndim != 3 for tensor in (log_gate, key, value, query)):
        raise ValueError(
            "vector propagation inputs must be token-major rank-three tensors"
        )
    token_count = log_gate.shape[0]
    if any(tensor.shape[0] != token_count for tensor in (key, value, query)):
        raise ValueError("vector propagation inputs must have equal token counts")
    if token_count % sequence_count:
        raise ValueError("dense vector propagation requires uniform sequence lengths")
    sequence_length = token_count // sequence_count
    log_gate = log_gate.reshape(token_count, heads, rows).float().contiguous()
    key = key.reshape(token_count, heads, rows).float().contiguous()
    value = value.reshape(token_count, heads, columns).float().contiguous()
    query = query.reshape(token_count, heads, rows).float().contiguous()
    device_index = initial_state.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    kernel = _compile_vector_propagation(
        sequence_count,
        sequence_length,
        heads,
        rows,
        columns,
        output_dtype,
        thread_columns,
        device_index,
    )
    output = torch.empty(
        sequence_count,
        heads,
        sequence_length,
        columns,
        device=initial_state.device,
        dtype=output_dtype,
    )
    final = torch.empty_like(initial_state, dtype=torch.float32)
    kernel(
        log_gate, key, value, query, initial_state.float().contiguous(), output, final
    )
    return output, final
