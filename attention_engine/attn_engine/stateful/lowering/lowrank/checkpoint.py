"""FP32 compact chunk checkpoints and inverse-free, deterministic local VJPs.

Dense and packed recurrences share device-segmented TileLang kernels. Affine
summaries carry chunk-entry States; only local replay has a 16-token State axis.
All segmentation, recurrence and gradient reductions remain on the device.
"""

from functools import lru_cache

import torch

from ..diagonal.identity import fp32_gemm_macro

CHUNK_SIZE = 16
_CODES = {"scale": 0, "row-scale": 1, "rank-one": 2, "column-scale": 3}


def _ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def _dtype(tensor):
    return str(tensor.dtype).removeprefix("torch.")


def _address(index, shape, strides):
    """Construct a logical row-major address in a strided source tensor."""
    address = 0
    for extent, stride in reversed(tuple(zip(shape, strides))):
        address += index % extent * stride
        index = index // extent
    return address


def _operation_code(T, codes, op):
    code = 0
    for index, value in enumerate(codes):
        code = T.if_then_else(op == index, value, code)
    return code


def _scalar_derivative(
    T, IK, IV, IS, orders, total, rows, columns, scalar_index, flat, r, c
):
    result = 0.0
    for index, order in enumerate(orders):
        for factor in order:
            if factor >= 0:
                value = _write_product(
                    T,
                    IK,
                    IV,
                    IS,
                    orders,
                    total,
                    rows,
                    columns,
                    index,
                    flat,
                    r,
                    c,
                    factor,
                )
                result = T.if_then_else(scalar_index == factor, value, result)
    return result


def _write_product(
    T, IK, IV, IS, orders, total, rows, columns, injection, flat, r, c, skip=-3
):
    """Emit the authored FP32 product, or its zero-safe factor derivative."""
    result = 0.0
    for index, order in enumerate(orders):
        product = None
        for factor in order:
            if factor == skip:
                continue
            if factor == -1:
                value = T.if_then_else(
                    r < rows,
                    T.cast(IK[(index * total + flat) * rows + r], "float32"),
                    0.0,
                )
            elif factor == -2:
                value = T.if_then_else(
                    c < columns,
                    T.cast(IV[(index * total + flat) * columns + c], "float32"),
                    0.0,
                )
            else:
                value = T.cast(IS[factor * total + flat], "float32")
            product = value if product is None else product * value
        result = T.if_then_else(injection == index, product, result)
    return result


@lru_cache(maxsize=None)
def _compile_copy(shape, strides, source_dtype, result_dtype, device_index):
    import tilelang
    import tilelang.language as T

    size = 1
    for extent in shape:
        size *= extent
    span = 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build():
        @T.prim_func
        def kernel(
            Source: T.Tensor((span,), source_dtype),
            Result: T.Tensor((size,), result_dtype),
        ):
            with T.Kernel(_ceildiv(size, 256), threads=128) as block:
                for lane in T.Parallel(256):
                    index = block * 256 + lane
                    if index < size:
                        Result[index] = T.cast(
                            Source[_address(index, shape, strides)], result_dtype
                        )

        return kernel

    return build()


def _copy(source, result):
    """Launch a strided copy/cast, including the zero-capacity State path."""
    if not source.numel():
        return
    shape, strides = tuple(source.shape), tuple(source.stride())
    span = 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))
    storage = torch.as_strided(source, (span,), (1,))
    _compile_copy(shape, strides, _dtype(source), _dtype(result), source.device.index)(
        storage,
        result.view(-1),
    )


def _flat(tensor):
    if tensor.is_contiguous():
        return tensor.view(-1)
    result = torch.empty(tensor.shape, device=tensor.device, dtype=tensor.dtype)
    _copy(tensor, result)
    return result.view(-1)


@lru_cache(maxsize=None)
def _compile_prefix(
    sequences, offsets_dtype, dense_length, device_index, chunk_size=CHUNK_SIZE
):
    import tilelang
    import tilelang.language as T

    block = 1 << max(0, (max(1, sequences) - 1).bit_length())

    @tilelang.jit
    def build():
        @T.prim_func
        def kernel(
            Offsets: T.Tensor((sequences + 1,), offsets_dtype),
            Prefix: T.Tensor((sequences + 1,), "int32"),
        ):
            with T.Kernel(1, threads=128):
                counts = T.alloc_shared((block,), "int32")
                if dense_length >= 0:
                    for index in T.Parallel(sequences + 1):
                        Offsets[index] = index * dense_length
                    T.sync_threads()
                for index in T.Parallel(block):
                    counts[index] = T.if_then_else(
                        index < sequences,
                        T.cast(
                            (Offsets[index + 1] - Offsets[index] + chunk_size - 1)
                            // chunk_size,
                            "int32",
                        ),
                        0,
                    )
                T.cumsum(counts, dim=0)
                for index in T.Parallel(sequences + 1):
                    Prefix[index] = T.if_then_else(index == 0, 0, counts[index - 1])

        return kernel

    return build()


@lru_cache(maxsize=None)
def _compile_metadata(
    sequences, capacity, offsets_dtype, device_index, chunk_size=CHUNK_SIZE
):
    import tilelang
    import tilelang.language as T

    steps = (sequences + 1).bit_length()

    @tilelang.jit
    def build():
        @T.prim_func
        def kernel(
            Offsets: T.Tensor((sequences + 1,), offsets_dtype),
            Prefix: T.Tensor((sequences + 1,), "int32"),
            Metadata: T.Tensor((3 * capacity,), "int64"),
        ):
            with T.Kernel(_ceildiv(capacity, 128), threads=128) as block:
                for lane in T.Parallel(128):
                    chunk = block * 128 + lane
                    if chunk < capacity:
                        low = T.alloc_var("int32", init=0)
                        high = T.alloc_var("int32", init=sequences)
                        for iteration in T.serial(steps):
                            middle = (low + high) // 2
                            boundary = T.if_then_else(
                                middle <= sequences, Prefix[middle], Prefix[sequences]
                            )
                            if boundary <= chunk:
                                low = middle + 1
                            else:
                                high = middle
                        sequence = T.if_then_else(chunk < Prefix[sequences], low - 1, 0)
                        Metadata[chunk] = T.if_then_else(
                            chunk < Prefix[sequences], sequence, -1
                        )
                        Metadata[capacity + chunk] = (
                            Offsets[sequence] + (chunk - Prefix[sequence]) * chunk_size
                        )
                        Metadata[2 * capacity + chunk] = Offsets[sequence + 1]

        return kernel

    return build()


def metadata(
    token_count, sequence_count, device, sequence_offsets=None, *, chunk_size=CHUNK_SIZE
):
    """Build exact chunk segmentation without observing device offsets on CPU."""
    if sequence_offsets is None:
        if sequence_count == 0 or token_count % sequence_count:
            if token_count:
                raise ValueError("dense token count must divide the State batch")
            length = 0
        else:
            length = token_count // sequence_count
        offsets = torch.empty(sequence_count + 1, device=device, dtype=torch.int64)
        capacity = sequence_count * _ceildiv(length, chunk_size)
    else:
        offsets = _flat(sequence_offsets)
        length = -1
        capacity = (
            _ceildiv(token_count, chunk_size) + sequence_count if token_count else 0
        )
    prefix = torch.empty(sequence_count + 1, device=device, dtype=torch.int32)
    chunks = torch.empty((3, capacity), device=device, dtype=torch.int64)
    _compile_prefix(
        sequence_count, _dtype(offsets), length, offsets.device.index, chunk_size
    )(offsets, prefix)
    if capacity:
        _compile_metadata(
            sequence_count, capacity, _dtype(offsets), offsets.device.index, chunk_size
        )(
            offsets,
            prefix,
            chunks.view(-1),
        )
    return offsets, prefix, chunks


def _apply_macro(T, rows, columns, width, total, br, bc):
    @T.macro
    def apply(State, Left, Right, Coeff, flat, op, code, column_base, transpose):
        if code == 0:
            for r, c in T.Parallel(br, bc):
                State[r, c] = State[r, c] * T.cast(Coeff[op * total + flat], "float32")
        elif code == 1:
            for r, c in T.Parallel(br, bc):
                gate = T.if_then_else(
                    r < rows,
                    T.cast(Left[(op * total + flat) * width + r], "float32"),
                    1.0,
                )
                State[r, c] = State[r, c] * gate
        elif code == 3:
            for r, c in T.Parallel(br, bc):
                column = column_base + c
                gate = T.if_then_else(
                    (column >= 0) & (column < columns),
                    T.cast(Left[(op * total + flat) * width + column], "float32"),
                    1.0,
                )
                State[r, c] = State[r, c] * gate
        else:
            products = T.alloc_fragment((br, bc), "float32")
            projection = T.alloc_fragment((bc,), "float32")
            for r, c in T.Parallel(br, bc):
                factor = T.if_then_else(
                    r < rows,
                    T.cast(Left[(op * total + flat) * width + r], "float32")
                    if transpose
                    else T.cast(Right[(op * total + flat) * rows + r], "float32"),
                    0.0,
                )
                products[r, c] = State[r, c] * factor
            T.reduce_sum(products, projection, dim=0)
            for r, c in T.Parallel(br, bc):
                factor = T.if_then_else(
                    r < rows,
                    T.cast(Right[(op * total + flat) * rows + r], "float32")
                    if transpose
                    else T.cast(Left[(op * total + flat) * width + r], "float32"),
                    0.0,
                )
                State[r, c] = (
                    State[r, c]
                    + T.cast(Coeff[op * total + flat], "float32")
                    * factor
                    * projection[c]
                )

    return apply


@lru_cache(maxsize=None)
def _compile_summary(
    sizes,
    dtypes,
    *,
    heads,
    rows,
    columns,
    total,
    width,
    capacity,
    codes,
    injections,
    injection_orders,
    readouts,
    length,
    packed,
    reverse,
    device_index,
    readout_axis=0,
    chunk_size=CHUNK_SIZE,
):
    import tilelang
    import tilelang.language as T

    br, bc = max(16, 1 << (rows - 1).bit_length()), 32
    linear_blocks = _ceildiv(rows, bc)
    apply = _apply_macro(T, rows, columns, width, total, br, bc)

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build():
        @T.prim_func
        def kernel(
            Left: T.Tensor((sizes[0],), dtypes[0]),
            Right: T.Tensor((sizes[1],), dtypes[1]),
            Coeff: T.Tensor((sizes[2],), dtypes[2]),
            IK: T.Tensor((sizes[3],), dtypes[3]),
            IV: T.Tensor((sizes[4],), dtypes[4]),
            Q: T.Tensor((sizes[5],), dtypes[5]),
            IS: T.Tensor((sizes[6],), dtypes[6]),
            DY: T.Tensor((sizes[7],), dtypes[7]),
            Metadata: T.Tensor((3 * capacity,), "int64"),
            Offsets: T.Tensor((sizes[8],), dtypes[8]),
            Transform: T.Tensor((capacity * heads * rows * rows,), "float32"),
            Decay: T.Tensor((capacity * heads * columns,), "float32"),
            Bias: T.Tensor((capacity * heads * rows * columns,), "float32"),
        ):
            with T.Kernel(
                capacity, heads, linear_blocks + _ceildiv(columns, bc), threads=128
            ) as (chunk, head, block):
                sequence = T.cast(Metadata[chunk], "int32")
                if sequence >= 0:
                    start = T.cast(Metadata[capacity + chunk], "int32")
                    end = T.cast(Metadata[2 * capacity + chunk], "int32")
                    matrix = T.alloc_fragment((br, bc), "float32")
                    decay = T.alloc_fragment((bc,), "float32")
                    value_base = (block - linear_blocks) * bc
                    for r, c in T.Parallel(br, bc):
                        matrix[r, c] = T.if_then_else(
                            (block < linear_blocks) & (r == block * bc + c), 1.0, 0.0
                        )
                    T.fill(decay, 1.0)
                    for step in T.serial(chunk_size):
                        token = (
                            start + chunk_size - 1 - step if reverse else start + step
                        )
                        if token < end:
                            flat = token * heads + head
                            if reverse:
                                output_flat = (
                                    token * heads + head
                                    if packed
                                    else (sequence * heads + head) * length
                                    + token
                                    - T.cast(Offsets[sequence], "int32")
                                )
                                for readout in T.unroll(readouts):
                                    for r, c in T.Parallel(br, bc):
                                        query = T.if_then_else(
                                            (value_base + c < columns)
                                            & (block >= linear_blocks)
                                            if readout_axis
                                            else r < rows,
                                            T.cast(
                                                Q[
                                                    (readout * total + flat) * columns
                                                    + value_base
                                                    + c
                                                ]
                                                if readout_axis
                                                else Q[
                                                    (readout * total + flat) * rows + r
                                                ],
                                                "float32",
                                            ),
                                            0.0,
                                        )
                                        dy = T.if_then_else(
                                            (block >= linear_blocks)
                                            & (
                                                (r < rows)
                                                if readout_axis
                                                else (value_base + c < columns)
                                            ),
                                            T.cast(
                                                DY[
                                                    (readout * total + output_flat)
                                                    * (
                                                        rows
                                                        if readout_axis
                                                        else columns
                                                    )
                                                    + (
                                                        r
                                                        if readout_axis
                                                        else value_base + c
                                                    )
                                                ],
                                                "float32",
                                            ),
                                            0.0,
                                        )
                                        matrix[r, c] = matrix[r, c] + query * dy
                            for index in T.unroll(len(codes)):
                                op = len(codes) - 1 - index if reverse else index
                                code = _operation_code(T, codes, op)
                                if code == 3:
                                    apply(
                                        matrix,
                                        Left,
                                        Right,
                                        Coeff,
                                        flat,
                                        op,
                                        code,
                                        value_base,
                                        reverse,
                                    )
                                    for c in T.Parallel(bc):
                                        gate = T.if_then_else(
                                            (block >= linear_blocks)
                                            & (value_base + c < columns),
                                            T.cast(
                                                Left[
                                                    (op * total + flat) * width
                                                    + value_base
                                                    + c
                                                ],
                                                "float32",
                                            ),
                                            1.0,
                                        )
                                        decay[c] = decay[c] * gate
                                else:
                                    apply(
                                        matrix,
                                        Left,
                                        Right,
                                        Coeff,
                                        flat,
                                        op,
                                        code,
                                        value_base,
                                        reverse,
                                    )
                            if not reverse:
                                for injection in T.unroll(injections):
                                    for r, c in T.Parallel(br, bc):
                                        if block >= linear_blocks:
                                            matrix[r, c] = matrix[
                                                r, c
                                            ] + _write_product(
                                                T,
                                                IK,
                                                IV,
                                                IS,
                                                injection_orders,
                                                total,
                                                rows,
                                                columns,
                                                injection,
                                                flat,
                                                r,
                                                value_base + c,
                                            )
                    slot = chunk * heads + head
                    for r, c in T.Parallel(br, bc):
                        if r < rows:
                            if block < linear_blocks:
                                if block * bc + c < rows:
                                    Transform[
                                        (slot * rows + r) * rows + block * bc + c
                                    ] = matrix[r, c]
                            else:
                                if value_base + c < columns:
                                    Bias[
                                        (slot * rows + r) * columns + value_base + c
                                    ] = matrix[r, c]
                    for c in T.Parallel(bc):
                        if (block >= linear_blocks) & (value_base + c < columns):
                            Decay[slot * columns + value_base + c] = decay[c]

        return kernel

    return build()


@lru_cache(maxsize=None)
def _compile_carry(
    sequences,
    heads,
    rows,
    columns,
    capacity,
    initial_dtype,
    final_dtype,
    reverse,
    device_index,
):
    import tilelang
    import tilelang.language as T

    br, bc = max(32, 1 << (rows - 1).bit_length()), 32
    bk = min(br, 64)

    # Bounded K tiles keep shared storage linear in rows. Exact 64-row tiles
    # consume the staged fragment directly; other shapes keep the proven staging.
    gemm = fp32_gemm_macro(T)

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build():
        @T.prim_func
        def kernel(
            Transform: T.Tensor((capacity * heads * rows * rows,), "float32"),
            Decay: T.Tensor((capacity * heads * columns,), "float32"),
            Bias: T.Tensor((capacity * heads * rows * columns,), "float32"),
            Initial: T.Tensor((sequences * heads * rows * columns,), initial_dtype),
            Prefix: T.Tensor((sequences + 1,), "int32"),
            Checkpoints: T.Tensor((capacity * heads * rows * columns,), "float32"),
            Final: T.Tensor((sequences * heads * rows * columns,), final_dtype),
        ):
            with T.Kernel(sequences * heads, _ceildiv(columns, bc), threads=128) as (
                sh,
                block,
            ):
                sequence, head = sh // heads, sh % heads
                first, last = Prefix[sequence], Prefix[sequence + 1]
                state = T.alloc_fragment((br, bc), "float32")
                previous = T.alloc_shared((br, bc), "float32")
                next_state = T.alloc_fragment((br, bc), "float32")
                transform_tile = T.alloc_shared((br, bk), "float32")
                for r, c in T.Parallel(br, bc):
                    state[r, c] = T.if_then_else(
                        (r < rows) & (block * bc + c < columns),
                        T.cast(
                            Initial[(sh * rows + r) * columns + block * bc + c],
                            "float32",
                        ),
                        0.0,
                    )
                for index in T.serial(last - first):
                    chunk = last - 1 - index if reverse else first + index
                    slot = chunk * heads + head
                    for r, c in T.Parallel(br, bc):
                        if (r < rows) & (block * bc + c < columns):
                            Checkpoints[
                                (slot * rows + r) * columns + block * bc + c
                            ] = state[r, c]
                    if reverse or index + 1 < last - first:
                        T.copy(state, previous)
                        T.clear(next_state)
                        if rows == 64:
                            for r, k in T.Parallel(br, bk):
                                transform_tile[r, k] = Transform[
                                    (slot * rows + r) * rows + k
                                ]
                            gemm(transform_tile, previous, next_state)
                        else:
                            state_tile = T.alloc_shared((bk, bc), "float32")
                            for kb in T.serial(_ceildiv(rows, bk)):
                                for k, c in T.Parallel(bk, bc):
                                    state_tile[k, c] = T.if_then_else(
                                        kb * bk + k < rows,
                                        previous[kb * bk + k, c],
                                        0.0,
                                    )
                                for r, k in T.Parallel(br, bk):
                                    transform_tile[r, k] = T.if_then_else(
                                        (r < rows) & (kb * bk + k < rows),
                                        Transform[
                                            (slot * rows + r) * rows + kb * bk + k
                                        ],
                                        0.0,
                                    )
                                gemm(transform_tile, state_tile, next_state)
                        for r, c in T.Parallel(br, bc):
                            decay = T.if_then_else(
                                block * bc + c < columns,
                                Decay[slot * columns + block * bc + c],
                                1.0,
                            )
                            bias = T.if_then_else(
                                (r < rows) & (block * bc + c < columns),
                                Bias[(slot * rows + r) * columns + block * bc + c],
                                0.0,
                            )
                            state[r, c] = next_state[r, c] * decay + bias
                if reverse or first == last:
                    for r, c in T.Parallel(br, bc):
                        if (r < rows) & (block * bc + c < columns):
                            Final[(sh * rows + r) * columns + block * bc + c] = T.cast(
                                state[r, c], final_dtype
                            )

        return kernel

    return build()


@lru_cache(maxsize=None)
def _compile_replay(
    sizes,
    dtypes,
    *,
    sequences,
    heads,
    rows,
    columns,
    total,
    width,
    capacity,
    codes,
    injections,
    injection_orders,
    readouts,
    length,
    packed,
    output_dtype,
    final_dtype,
    device_index,
    readout_axis=0,
    chunk_size=CHUNK_SIZE,
):
    import tilelang
    import tilelang.language as T

    br = max(16, 1 << (rows - 1).bit_length())
    # Weighted multi-update inference reuses each row reduction across more
    # value columns. Bound the fragment at 128 rows so register pressure stays
    # comparable to the opposite-axis replay already supported here.
    wide_replay = (
        chunk_size == 64
        and codes.count(2) > 1
        and any(factor >= 0 for order in injection_orders for factor in order)
        and rows <= 128
        and columns >= 64
    )
    bc = (
        max(16, 1 << (columns - 1).bit_length())
        if readout_axis
        else 64
        if wide_replay
        else 16
    )
    apply = _apply_macro(T, rows, columns, width, total, br, bc)

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build():
        @T.prim_func
        def kernel(
            Left: T.Tensor((sizes[0],), dtypes[0]),
            Right: T.Tensor((sizes[1],), dtypes[1]),
            Coeff: T.Tensor((sizes[2],), dtypes[2]),
            IK: T.Tensor((sizes[3],), dtypes[3]),
            IV: T.Tensor((sizes[4],), dtypes[4]),
            Q: T.Tensor((sizes[5],), dtypes[5]),
            IS: T.Tensor((sizes[6],), dtypes[6]),
            Metadata: T.Tensor((3 * capacity,), "int64"),
            Offsets: T.Tensor((sizes[7],), dtypes[7]),
            Checkpoints: T.Tensor((capacity * heads * rows * columns,), "float32"),
            Output: T.Tensor(
                (readouts * total * (rows if readout_axis else columns),), output_dtype
            ),
            Final: T.Tensor((sequences * heads * rows * columns,), final_dtype),
        ):
            with T.Kernel(capacity, heads, _ceildiv(columns, bc), threads=128) as (
                chunk,
                head,
                block,
            ):
                sequence = T.cast(Metadata[chunk], "int32")
                if sequence >= 0:
                    start = T.cast(Metadata[capacity + chunk], "int32")
                    end = T.cast(Metadata[2 * capacity + chunk], "int32")
                    state = T.alloc_fragment((br, bc), "float32")
                    products = T.alloc_fragment((br, bc), "float32")
                    output = T.alloc_fragment((br if readout_axis else bc,), "float32")
                    slot = chunk * heads + head
                    for r, c in T.Parallel(br, bc):
                        state[r, c] = T.if_then_else(
                            (r < rows) & (block * bc + c < columns),
                            Checkpoints[(slot * rows + r) * columns + block * bc + c],
                            0.0,
                        )
                    for step in T.serial(chunk_size):
                        token = start + step
                        if token < end:
                            flat = token * heads + head
                            for op in T.unroll(len(codes)):
                                apply(
                                    state,
                                    Left,
                                    Right,
                                    Coeff,
                                    flat,
                                    op,
                                    _operation_code(T, codes, op),
                                    block * bc,
                                    False,
                                )
                            for injection in T.unroll(injections):
                                for r, c in T.Parallel(br, bc):
                                    state[r, c] = state[r, c] + _write_product(
                                        T,
                                        IK,
                                        IV,
                                        IS,
                                        injection_orders,
                                        total,
                                        rows,
                                        columns,
                                        injection,
                                        flat,
                                        r,
                                        block * bc + c,
                                    )
                            output_flat = (
                                token * heads + head
                                if packed
                                else (sequence * heads + head) * length
                                + token
                                - T.cast(Offsets[sequence], "int32")
                            )
                            for readout in T.unroll(readouts):
                                for r, c in T.Parallel(br, bc):
                                    query = T.if_then_else(
                                        c < columns if readout_axis else r < rows,
                                        T.cast(
                                            Q[(readout * total + flat) * columns + c]
                                            if readout_axis
                                            else Q[(readout * total + flat) * rows + r],
                                            "float32",
                                        ),
                                        0.0,
                                    )
                                    products[r, c] = state[r, c] * query
                                if readout_axis:
                                    T.reduce_sum(products, output, dim=1)
                                    for r in T.Parallel(br):
                                        if r < rows:
                                            Output[
                                                (readout * total + output_flat) * rows
                                                + r
                                            ] = T.cast(output[r], output_dtype)
                                else:
                                    T.reduce_sum(products, output, dim=0)
                                    for c in T.Parallel(bc):
                                        if block * bc + c < columns:
                                            Output[
                                                (readout * total + output_flat)
                                                * columns
                                                + block * bc
                                                + c
                                            ] = T.cast(output[c], output_dtype)
                    if start + chunk_size >= end:
                        for r, c in T.Parallel(br, bc):
                            if (r < rows) & (block * bc + c < columns):
                                Final[
                                    ((sequence * heads + head) * rows + r) * columns
                                    + block * bc
                                    + c
                                ] = T.cast(state[r, c], final_dtype)

        return kernel

    return build()


@lru_cache(maxsize=None)
def _compile_local_vjp(
    sizes,
    dtypes,
    *,
    heads,
    rows,
    columns,
    total,
    width,
    capacity,
    codes,
    injections,
    injection_orders,
    readouts,
    length,
    packed,
    device_index,
    readout_axis=0,
):
    import tilelang
    import tilelang.language as T

    br, bc = max(16, 1 << (rows - 1).bit_length()), 16
    blocks, operations = _ceildiv(columns, bc), len(codes)
    apply = _apply_macro(T, rows, columns, width, total, br, bc)

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build():
        @T.prim_func
        def kernel(
            Left: T.Tensor((sizes[0],), dtypes[0]),
            Right: T.Tensor((sizes[1],), dtypes[1]),
            Coeff: T.Tensor((sizes[2],), dtypes[2]),
            IK: T.Tensor((sizes[3],), dtypes[3]),
            IV: T.Tensor((sizes[4],), dtypes[4]),
            Q: T.Tensor((sizes[5],), dtypes[5]),
            IS: T.Tensor((sizes[6],), dtypes[6]),
            Metadata: T.Tensor((3 * capacity,), "int64"),
            Offsets: T.Tensor((sizes[7],), dtypes[7]),
            Checkpoints: T.Tensor((capacity * heads * rows * columns,), "float32"),
            Carry: T.Tensor((capacity * heads * rows * columns,), "float32"),
            DY: T.Tensor(
                (readouts * total * (rows if readout_axis else columns),), dtypes[8]
            ),
            DL: T.Tensor((blocks * sizes[0],), "float32"),
            DR: T.Tensor((blocks * sizes[1],), "float32"),
            DC: T.Tensor((blocks * sizes[2],), "float32"),
            DIK: T.Tensor((blocks * sizes[3],), "float32"),
            DIV: T.Tensor((sizes[4],), "float32"),
            DQ: T.Tensor((blocks * sizes[5],), "float32"),
            DIS: T.Tensor((blocks * sizes[6],), "float32"),
        ):
            with T.Kernel(capacity, heads, blocks, threads=128) as (chunk, head, block):
                sequence = T.cast(Metadata[chunk], "int32")
                if sequence >= 0:
                    start = T.cast(Metadata[capacity + chunk], "int32")
                    end = T.cast(Metadata[2 * capacity + chunk], "int32")
                    state = T.alloc_fragment((br, bc), "float32")
                    previous = T.alloc_fragment((br, bc), "float32")
                    adjoint = T.alloc_fragment((br, bc), "float32")
                    history = T.alloc_shared((CHUNK_SIZE, br, bc), "float32")
                    products = T.alloc_fragment((br, bc), "float32")
                    row_sum = T.alloc_fragment((br,), "float32")
                    column_sum = T.alloc_fragment((bc,), "float32")
                    projection = T.alloc_fragment((bc,), "float32")
                    adjoint_projection = T.alloc_fragment((bc,), "float32")
                    scalar = T.alloc_fragment((1,), "float32")
                    scalar_products = T.alloc_fragment((1, bc), "float32")
                    slot = chunk * heads + head
                    for r, c in T.Parallel(br, bc):
                        state[r, c] = T.if_then_else(
                            (r < rows) & (block * bc + c < columns),
                            Checkpoints[(slot * rows + r) * columns + block * bc + c],
                            0.0,
                        )
                    for step in T.serial(CHUNK_SIZE):
                        for r, c in T.Parallel(br, bc):
                            history[step, r, c] = state[r, c]
                        token = start + step
                        if token < end:
                            flat = token * heads + head
                            for op in T.unroll(operations):
                                apply(
                                    state,
                                    Left,
                                    Right,
                                    Coeff,
                                    flat,
                                    op,
                                    _operation_code(T, codes, op),
                                    block * bc,
                                    False,
                                )
                            for injection in T.unroll(injections):
                                for r, c in T.Parallel(br, bc):
                                    state[r, c] = state[r, c] + _write_product(
                                        T,
                                        IK,
                                        IV,
                                        IS,
                                        injection_orders,
                                        total,
                                        rows,
                                        columns,
                                        injection,
                                        flat,
                                        r,
                                        block * bc + c,
                                    )
                    for r, c in T.Parallel(br, bc):
                        adjoint[r, c] = T.if_then_else(
                            (r < rows) & (block * bc + c < columns),
                            Carry[(slot * rows + r) * columns + block * bc + c],
                            0.0,
                        )
                    for reverse_step in T.serial(CHUNK_SIZE):
                        step = CHUNK_SIZE - 1 - reverse_step
                        token = start + step
                        if token < end:
                            flat = token * heads + head
                            for r, c in T.Parallel(br, bc):
                                previous[r, c] = history[step, r, c]
                                state[r, c] = previous[r, c]
                            for op in T.unroll(operations):
                                apply(
                                    state,
                                    Left,
                                    Right,
                                    Coeff,
                                    flat,
                                    op,
                                    _operation_code(T, codes, op),
                                    block * bc,
                                    False,
                                )
                            for injection in T.unroll(injections):
                                for r, c in T.Parallel(br, bc):
                                    state[r, c] = state[r, c] + _write_product(
                                        T,
                                        IK,
                                        IV,
                                        IS,
                                        injection_orders,
                                        total,
                                        rows,
                                        columns,
                                        injection,
                                        flat,
                                        r,
                                        block * bc + c,
                                    )
                            output_flat = (
                                token * heads + head
                                if packed
                                else (sequence * heads + head) * length
                                + token
                                - T.cast(Offsets[sequence], "int32")
                            )
                            for readout in T.unroll(readouts):
                                for r, c in T.Parallel(br, bc):
                                    dy = T.if_then_else(
                                        r < rows
                                        if readout_axis
                                        else block * bc + c < columns,
                                        T.cast(
                                            DY[
                                                (readout * total + output_flat)
                                                * (rows if readout_axis else columns)
                                                + (
                                                    r
                                                    if readout_axis
                                                    else block * bc + c
                                                )
                                            ],
                                            "float32",
                                        ),
                                        0.0,
                                    )
                                    products[r, c] = state[r, c] * dy
                                if readout_axis:
                                    T.reduce_sum(products, column_sum, dim=0)
                                    for c in T.Parallel(bc):
                                        if block * bc + c < columns:
                                            DQ[
                                                (
                                                    (block * readouts + readout) * total
                                                    + flat
                                                )
                                                * columns
                                                + block * bc
                                                + c
                                            ] = column_sum[c]
                                else:
                                    T.reduce_sum(products, row_sum, dim=1)
                                    for r in T.Parallel(br):
                                        if r < rows:
                                            DQ[
                                                (
                                                    (block * readouts + readout) * total
                                                    + flat
                                                )
                                                * rows
                                                + r
                                            ] = row_sum[r]
                                for r, c in T.Parallel(br, bc):
                                    dy = T.if_then_else(
                                        r < rows
                                        if readout_axis
                                        else block * bc + c < columns,
                                        T.cast(
                                            DY[
                                                (readout * total + output_flat)
                                                * (rows if readout_axis else columns)
                                                + (
                                                    r
                                                    if readout_axis
                                                    else block * bc + c
                                                )
                                            ],
                                            "float32",
                                        ),
                                        0.0,
                                    )
                                    query = T.if_then_else(
                                        block * bc + c < columns
                                        if readout_axis
                                        else r < rows,
                                        T.cast(
                                            Q[
                                                (readout * total + flat) * columns
                                                + block * bc
                                                + c
                                            ]
                                            if readout_axis
                                            else Q[(readout * total + flat) * rows + r],
                                            "float32",
                                        ),
                                        0.0,
                                    )
                                    adjoint[r, c] = adjoint[r, c] + query * dy
                            for injection in T.unroll(injections):
                                for r, c in T.Parallel(br, bc):
                                    products[r, c] = adjoint[r, c] * _write_product(
                                        T,
                                        IK,
                                        IV,
                                        IS,
                                        injection_orders,
                                        total,
                                        rows,
                                        columns,
                                        injection,
                                        flat,
                                        r,
                                        block * bc + c,
                                        -1,
                                    )
                                T.reduce_sum(products, row_sum, dim=1)
                                for r in T.Parallel(br):
                                    if r < rows:
                                        DIK[
                                            (
                                                (block * injections + injection) * total
                                                + flat
                                            )
                                            * rows
                                            + r
                                        ] = row_sum[r]
                                for r, c in T.Parallel(br, bc):
                                    products[r, c] = adjoint[r, c] * _write_product(
                                        T,
                                        IK,
                                        IV,
                                        IS,
                                        injection_orders,
                                        total,
                                        rows,
                                        columns,
                                        injection,
                                        flat,
                                        r,
                                        block * bc + c,
                                        -2,
                                    )
                                T.reduce_sum(products, column_sum, dim=0)
                                for c in T.Parallel(bc):
                                    if block * bc + c < columns:
                                        DIV[
                                            (injection * total + flat) * columns
                                            + block * bc
                                            + c
                                        ] = column_sum[c]
                            for scalar_index in T.unroll(sizes[6] // max(1, total)):
                                # Each scalar slot belongs to exactly one injection.
                                for r, c in T.Parallel(br, bc):
                                    products[r, c] = adjoint[r, c] * _scalar_derivative(
                                        T,
                                        IK,
                                        IV,
                                        IS,
                                        injection_orders,
                                        total,
                                        rows,
                                        columns,
                                        scalar_index,
                                        flat,
                                        r,
                                        block * bc + c,
                                    )
                                T.reduce_sum(products, column_sum, dim=0)
                                for c in T.Parallel(bc):
                                    scalar_products[0, c] = column_sum[c]
                                T.reduce_sum(scalar_products, scalar, dim=1)
                                DIS[
                                    (block * (sizes[6] // total) + scalar_index) * total
                                    + flat
                                ] = scalar[0]
                            for reverse_op in T.unroll(operations):
                                op = operations - 1 - reverse_op
                                code = _operation_code(T, codes, op)
                                T.copy(previous, state)
                                for earlier in T.unroll(operations):
                                    if earlier < op:
                                        apply(
                                            state,
                                            Left,
                                            Right,
                                            Coeff,
                                            flat,
                                            earlier,
                                            _operation_code(T, codes, earlier),
                                            block * bc,
                                            False,
                                        )
                                if code == 0:
                                    for r, c in T.Parallel(br, bc):
                                        products[r, c] = adjoint[r, c] * state[r, c]
                                    T.reduce_sum(products, column_sum, dim=0)
                                    for c in T.Parallel(bc):
                                        scalar_products[0, c] = column_sum[c]
                                    T.reduce_sum(scalar_products, scalar, dim=1)
                                    DC[(block * operations + op) * total + flat] = (
                                        scalar[0]
                                    )
                                    for r, c in T.Parallel(br, bc):
                                        adjoint[r, c] = adjoint[r, c] * T.cast(
                                            Coeff[op * total + flat], "float32"
                                        )
                                elif code == 1:
                                    for r, c in T.Parallel(br, bc):
                                        products[r, c] = adjoint[r, c] * state[r, c]
                                    T.reduce_sum(products, row_sum, dim=1)
                                    for r in T.Parallel(br):
                                        if r < rows:
                                            DL[
                                                (
                                                    (block * operations + op) * total
                                                    + flat
                                                )
                                                * width
                                                + r
                                            ] = row_sum[r]
                                    apply(
                                        adjoint,
                                        Left,
                                        Right,
                                        Coeff,
                                        flat,
                                        op,
                                        code,
                                        block * bc,
                                        True,
                                    )
                                elif code == 3:
                                    for r, c in T.Parallel(br, bc):
                                        products[r, c] = adjoint[r, c] * state[r, c]
                                    T.reduce_sum(products, column_sum, dim=0)
                                    for c in T.Parallel(bc):
                                        if block * bc + c < columns:
                                            DL[
                                                (
                                                    (block * operations + op) * total
                                                    + flat
                                                )
                                                * width
                                                + block * bc
                                                + c
                                            ] = column_sum[c]
                                    apply(
                                        adjoint,
                                        Left,
                                        Right,
                                        Coeff,
                                        flat,
                                        op,
                                        code,
                                        block * bc,
                                        True,
                                    )
                                else:
                                    for r, c in T.Parallel(br, bc):
                                        right = T.if_then_else(
                                            r < rows,
                                            T.cast(
                                                Right[(op * total + flat) * rows + r],
                                                "float32",
                                            ),
                                            0.0,
                                        )
                                        products[r, c] = state[r, c] * right
                                    T.reduce_sum(products, projection, dim=0)
                                    for r, c in T.Parallel(br, bc):
                                        left = T.if_then_else(
                                            r < rows,
                                            T.cast(
                                                Left[(op * total + flat) * width + r],
                                                "float32",
                                            ),
                                            0.0,
                                        )
                                        products[r, c] = adjoint[r, c] * left
                                    T.reduce_sum(products, adjoint_projection, dim=0)
                                    for r, c in T.Parallel(br, bc):
                                        products[r, c] = adjoint[r, c] * projection[c]
                                    T.reduce_sum(products, row_sum, dim=1)
                                    for r in T.Parallel(br):
                                        if r < rows:
                                            DL[
                                                (
                                                    (block * operations + op) * total
                                                    + flat
                                                )
                                                * width
                                                + r
                                            ] = (
                                                T.cast(
                                                    Coeff[op * total + flat], "float32"
                                                )
                                                * row_sum[r]
                                            )
                                    for r, c in T.Parallel(br, bc):
                                        products[r, c] = (
                                            state[r, c] * adjoint_projection[c]
                                        )
                                    T.reduce_sum(products, row_sum, dim=1)
                                    for r in T.Parallel(br):
                                        if r < rows:
                                            DR[
                                                (
                                                    (block * operations + op) * total
                                                    + flat
                                                )
                                                * rows
                                                + r
                                            ] = (
                                                T.cast(
                                                    Coeff[op * total + flat], "float32"
                                                )
                                                * row_sum[r]
                                            )
                                    for c in T.Parallel(bc):
                                        scalar_products[0, c] = (
                                            adjoint_projection[c] * projection[c]
                                        )
                                    T.reduce_sum(scalar_products, scalar, dim=1)
                                    DC[(block * operations + op) * total + flat] = (
                                        scalar[0]
                                    )
                                    for r, c in T.Parallel(br, bc):
                                        right = T.if_then_else(
                                            r < rows,
                                            T.cast(
                                                Right[(op * total + flat) * rows + r],
                                                "float32",
                                            ),
                                            0.0,
                                        )
                                        adjoint[r, c] = (
                                            adjoint[r, c]
                                            + T.cast(
                                                Coeff[op * total + flat], "float32"
                                            )
                                            * right
                                            * adjoint_projection[c]
                                        )

        return kernel

    return build()


@lru_cache(maxsize=None)
def _compile_reduce(size, blocks, device_index):
    import tilelang
    import tilelang.language as T

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build():
        @T.prim_func
        def kernel(
            Partial: T.Tensor((blocks * size,), "float32"),
            Result: T.Tensor((size,), "float32"),
        ):
            with T.Kernel(_ceildiv(size, 256), threads=128) as tile:
                value = T.alloc_fragment((256,), "float32")
                T.clear(value)
                for block in T.serial(blocks):
                    for lane in T.Parallel(256):
                        index = tile * 256 + lane
                        value[lane] = value[lane] + T.if_then_else(
                            index < size, Partial[block * size + index], 0.0
                        )
                for lane in T.Parallel(256):
                    index = tile * 256 + lane
                    if index < size:
                        Result[index] = value[lane]

        return kernel

    return build()


def _layout(readout, initial_state, sequence_offsets, readout_axis=0):
    sequences, heads, rows, columns = initial_state.shape
    tokens = readout.shape[-4]
    packed = sequence_offsets is not None
    length = 0 if packed or not sequences else tokens // sequences
    count = readout.shape[0] if readout.ndim == 5 else 1
    output_width = rows if readout_axis else columns
    shape = (
        (tokens, heads, output_width)
        if packed
        else (sequences, heads, length, output_width)
    )
    if readout.ndim == 5:
        shape = (count, *shape)
    return sequences, heads, rows, columns, tokens, length, count, packed, shape


def _parameters(
    factors,
    *,
    heads,
    rows,
    columns,
    tokens,
    capacity,
    operation_codes,
    injections,
    readouts,
    length,
    packed,
    device,
    injection_orders=None,
    readout_axis=0,
):
    return dict(
        heads=heads,
        rows=rows,
        columns=columns,
        total=tokens * heads,
        width=factors[0].shape[-2],
        capacity=capacity,
        codes=tuple(_CODES[code] for code in operation_codes),
        injections=injections,
        injection_orders=(
            tuple(injection_orders)
            if injection_orders is not None
            else ((-1, -2),) * injections
        ),
        readouts=readouts,
        readout_axis=readout_axis,
        length=length,
        packed=packed,
        device_index=device.index,
    )


def forward(
    *,
    operation_codes,
    left_factors,
    right_factors,
    coefficients,
    injection_left,
    injection_right,
    readout,
    initial_state,
    output_dtype,
    record=True,
    sequence_offsets=None,
    chunk_metadata=None,
    injection_scalars=None,
    injection_orders=None,
    readout_axis=0,
):
    """Return outputs, final State and FP32 entries; segmentation stays on device."""
    sequences, heads, rows, columns, tokens, length, count, packed, shape = _layout(
        readout, initial_state, sequence_offsets, readout_axis
    )
    # Long inference amortizes carry with 64-token chunks. Short dense sequences
    # retain Triton's 16-token replay granularity to expose enough independent
    # CTAs. Recorded/packed/supplied metadata keep the backward checkpoint ABI.
    chunk_size = (
        64
        if not record
        and chunk_metadata is None
        and sequence_offsets is None
        and length > 128
        else CHUNK_SIZE
    )
    offsets, prefix, chunks = (
        metadata(
            tokens,
            sequences,
            initial_state.device,
            sequence_offsets,
            chunk_size=chunk_size,
        )
        if chunk_metadata is None
        else chunk_metadata
    )
    capacity, device = chunks.shape[1], initial_state.device
    if injection_scalars is None:
        injection_scalars = torch.empty((0, tokens, heads, 1, 1), device=device)
    checkpoints = torch.empty(
        (capacity, heads, rows, columns), device=device, dtype=torch.float32
    )
    output = torch.empty(shape, device=device, dtype=output_dtype)
    final = torch.empty(initial_state.shape, device=device, dtype=initial_state.dtype)
    if not capacity:
        _copy(initial_state, final)
    else:
        factors = (
            left_factors,
            right_factors,
            coefficients,
            injection_left,
            injection_right,
            readout,
            injection_scalars,
        )
        flat_factors = tuple(_flat(value) for value in factors)
        offsets_flat, prefix_flat, metadata_flat = (
            _flat(offsets),
            _flat(prefix),
            _flat(chunks),
        )
        parameters = _parameters(
            factors,
            heads=heads,
            rows=rows,
            columns=columns,
            tokens=tokens,
            capacity=capacity,
            operation_codes=operation_codes,
            injections=injection_left.shape[0],
            readouts=count,
            length=length,
            packed=packed,
            device=device,
            injection_orders=injection_orders,
            readout_axis=readout_axis,
        )
        parameters["chunk_size"] = chunk_size
        transform = torch.empty(
            (capacity, heads, rows, rows), device=device, dtype=torch.float32
        )
        decay = torch.empty(
            (capacity, heads, columns), device=device, dtype=torch.float32
        )
        bias = torch.empty_like(checkpoints)
        summary_inputs = (*flat_factors, output.view(-1), offsets_flat)
        _compile_summary(
            tuple(value.numel() for value in summary_inputs),
            tuple(_dtype(value) for value in summary_inputs),
            **parameters,
            reverse=False,
        )(
            *flat_factors,
            output.view(-1),
            metadata_flat,
            offsets_flat,
            transform.view(-1),
            decay.view(-1),
            bias.view(-1),
        )
        _compile_carry(
            sequences,
            heads,
            rows,
            columns,
            capacity,
            _dtype(initial_state),
            _dtype(final),
            False,
            device.index,
        )(
            transform.view(-1),
            decay.view(-1),
            bias.view(-1),
            _flat(initial_state),
            prefix_flat,
            checkpoints.view(-1),
            final.view(-1),
        )
        replay_inputs = (*flat_factors, offsets_flat)
        _compile_replay(
            tuple(value.numel() for value in replay_inputs),
            tuple(_dtype(value) for value in replay_inputs),
            **parameters,
            sequences=sequences,
            output_dtype=_dtype(output),
            final_dtype=_dtype(final),
        )(
            *flat_factors,
            metadata_flat,
            offsets_flat,
            checkpoints.view(-1),
            output.view(-1),
            final.view(-1),
        )
    if not packed:
        checkpoints = checkpoints.reshape(
            sequences, _ceildiv(length, chunk_size), heads, rows, columns
        ).permute(0, 2, 1, 3, 4)
    if not record:
        checkpoints = checkpoints[:0]
    return output, final, checkpoints


def backward(
    *,
    operation_codes,
    left_factors,
    right_factors,
    coefficients,
    injection_left,
    injection_right,
    readout,
    checkpoints,
    initial_state,
    grad_output,
    grad_final,
    sequence_offsets=None,
    chunk_metadata=None,
    injection_scalars=None,
    injection_orders=None,
    readout_axis=0,
):
    """Reverse affine carry followed by parallel, inverse-free chunk-local VJPs."""
    sequences, heads, rows, columns, tokens, length, count, packed, shape = _layout(
        readout, initial_state, sequence_offsets, readout_axis
    )
    offsets, prefix, chunks = (
        metadata(tokens, sequences, initial_state.device, sequence_offsets)
        if chunk_metadata is None
        else chunk_metadata
    )
    capacity, device = chunks.shape[1], initial_state.device
    has_scalars = injection_scalars is not None
    if injection_scalars is None:
        injection_scalars = torch.empty((0, tokens, heads, 1, 1), device=device)
    if not packed:
        checkpoints = checkpoints.permute(0, 2, 1, 3, 4)
    factors = (
        left_factors,
        right_factors,
        coefficients,
        injection_left,
        injection_right,
        readout,
        injection_scalars,
    )
    gradients = [
        torch.zeros(value.shape, device=device, dtype=torch.float32)
        for value in factors
    ]
    di = torch.empty(initial_state.shape, device=device, dtype=torch.float32)
    df = (
        torch.zeros(initial_state.shape, device=device, dtype=torch.float32)
        if grad_final is None
        else grad_final
    )
    if not capacity:
        _copy(df, di)
        return (
            (*gradients[:6], di, gradients[6]) if has_scalars else (*gradients[:6], di)
        )
    dy = (
        torch.zeros(shape, device=device, dtype=torch.float32)
        if grad_output is None
        else grad_output
    )
    flat_factors = tuple(_flat(value) for value in factors)
    offsets_flat, prefix_flat, metadata_flat = (
        _flat(offsets),
        _flat(prefix),
        _flat(chunks),
    )
    dy_flat = _flat(dy)
    parameters = _parameters(
        factors,
        heads=heads,
        rows=rows,
        columns=columns,
        tokens=tokens,
        capacity=capacity,
        operation_codes=operation_codes,
        injections=injection_left.shape[0],
        readouts=count,
        length=length,
        packed=packed,
        device=device,
        injection_orders=injection_orders,
        readout_axis=readout_axis,
    )
    blocks = _ceildiv(columns, 16)
    transform = torch.empty(
        (capacity, heads, rows, rows), device=device, dtype=torch.float32
    )
    decay = torch.empty((capacity, heads, columns), device=device, dtype=torch.float32)
    source = torch.empty(
        (capacity, heads, rows, columns), device=device, dtype=torch.float32
    )
    carry = torch.empty_like(source)
    summary_inputs = (*flat_factors, dy_flat, offsets_flat)
    _compile_summary(
        tuple(value.numel() for value in summary_inputs),
        tuple(_dtype(value) for value in summary_inputs),
        **parameters,
        reverse=True,
    )(
        *flat_factors,
        dy_flat,
        metadata_flat,
        offsets_flat,
        transform.view(-1),
        decay.view(-1),
        source.view(-1),
    )
    _compile_carry(
        sequences,
        heads,
        rows,
        columns,
        capacity,
        _dtype(df),
        _dtype(di),
        True,
        device.index,
    )(
        transform.view(-1),
        decay.view(-1),
        source.view(-1),
        _flat(df),
        prefix_flat,
        carry.view(-1),
        di.view(-1),
    )
    partial_indices = (0, 1, 2, 3, 5, 6)
    partials = {
        index: torch.zeros(
            (blocks, factors[index].numel()), device=device, dtype=torch.float32
        )
        for index in partial_indices
    }
    local_inputs = (*flat_factors, offsets_flat, dy_flat)
    _compile_local_vjp(
        tuple(value.numel() for value in local_inputs),
        tuple(_dtype(value) for value in local_inputs),
        **parameters,
    )(
        *flat_factors,
        metadata_flat,
        offsets_flat,
        _flat(checkpoints),
        carry.view(-1),
        dy_flat,
        partials[0].view(-1),
        partials[1].view(-1),
        partials[2].view(-1),
        partials[3].view(-1),
        gradients[4].view(-1),
        partials[5].view(-1),
        partials[6].view(-1),
    )
    for index in partial_indices:
        size = factors[index].numel()
        if size:
            _compile_reduce(size, blocks, device.index)(
                partials[index].view(-1), gradients[index].view(-1)
            )
    return (*gradients[:6], di, gradients[6]) if has_scalars else (*gradients[:6], di)
