"""Raw-input scalar affine summaries, exclusive carries and chunk readout.

Summary and readout CTAs run independently over 64-token chunks. Only the
FP32 State carry traverses chunks in order. Products never divide by a gate or
a prefix, so signed gates, resets and underflow preserve later injections.
"""

from functools import lru_cache
from typing import Any

import torch


CHUNK_TOKENS = 64
STATE_BLOCK = 64
CARRY_BLOCK = 256


def _metadata(tensor: torch.Tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype).removeprefix("torch."),
    )


def _transform(T, expression):
    operation, scale, is_log = expression[:3]
    offset = expression[3] if len(expression) > 3 else 0.0
    scale_steps = expression[4] if len(expression) > 4 else ()
    nodes = expression[5] if len(expression) > 5 else ()

    def transform(value):
        if nodes:
            values = []
            for node in nodes:
                if node[0] == "input":
                    result = value
                elif node[0] == "constant":
                    result = float(torch.tensor(node[1], dtype=node[2]).item())
                elif node[0] == "binary":
                    left, right = values[node[1]], values[node[2]]
                    result = left + right if node[4] == "+" else left * right
                    if node[3] == torch.bfloat16:
                        result = T.cast(T.cast(result, "bfloat16"), "float32")
                elif node[0] == "unary":
                    operand = values[node[1]]
                    result = (
                        -operand
                        if node[2] == "neg"
                        else T.cast(
                            T.call_extern("float64", "exp", T.cast(operand, "float64")),
                            "float32",
                        )
                    )
                else:
                    raise ValueError(f"unsupported scalar expression node: {node[0]}")
                values.append(result)
            return values[-1]

        # The fast exp intrinsic can differ by one ULP near one. Repeated
        # weak decay amplifies that error in long FP32 State recurrences.
        if operation == "exp":
            value = T.cast(
                T.call_extern("float64", "exp", T.cast(value, "float64")), "float32"
            )
        if scale_steps:
            for factor in scale_steps:
                # Keep each authored FP32 rounding point through TIR/CUDA
                # optimization; an ordinary Mul can be reassociated.
                value = T.ieee_mul(value, factor)
        else:
            value = value * scale
        if is_log:
            value = T.cast(
                T.call_extern("float64", "exp", T.cast(value, "float64")), "float32"
            )
        return value + offset

    return transform


def _gate_macro(T, heads, length, identity, expression):
    transform = _transform(T, expression)

    @T.macro
    def load_gates(Propagation, Gates, sequence, head, chunk):
        for token in T.Parallel(CHUNK_TOKENS):
            logical = chunk * CHUNK_TOKENS + token
            if identity:
                Gates[token] = 1.0
            elif logical < length:
                Gates[token] = transform(
                    T.cast(
                        Propagation[
                            sequence, head // (heads // Propagation.shape[1]), logical
                        ],
                        "float32",
                    )
                )
            else:
                # Padding must not decay a partial chunk's terminal State.
                Gates[token] = 1.0
        T.sync_threads()

    return load_gates


def _product_scan_macro(T, reverse):
    """Division-free scan; alternate storage without a copy-back at each level."""

    @T.macro
    def scan(Values):
        scratch = T.alloc_shared((CHUNK_TOKENS,), "float32")
        # Six levels finish in Values. Each level reads only the other buffer,
        # preserving the original multiplication tree and cross-warp ordering.
        for pair in T.unroll(3):
            distance = 1 << (2 * pair)
            for token in T.Parallel(CHUNK_TOKENS):
                if reverse:
                    scratch[token] = Values[token] * T.if_then_else(
                        token + distance < CHUNK_TOKENS, Values[token + distance], 1.0
                    )
                else:
                    scratch[token] = Values[token] * T.if_then_else(
                        token >= distance, Values[token - distance], 1.0
                    )
            T.sync_threads()
            for token in T.Parallel(CHUNK_TOKENS):
                if reverse:
                    Values[token] = scratch[token] * T.if_then_else(
                        token + 2 * distance < CHUNK_TOKENS,
                        scratch[token + 2 * distance],
                        1.0,
                    )
                else:
                    Values[token] = scratch[token] * T.if_then_else(
                        token >= 2 * distance, scratch[token - 2 * distance], 1.0
                    )
            T.sync_threads()

    return scan


@lru_cache(maxsize=None)
def _compile_summary(
    propagation_meta,
    left_meta,
    right_meta,
    carry_meta,
    scale_meta,
    expressions,
    *,
    heads,
    length,
    identity,
    threads,
    device_index,
):
    import tilelang
    import tilelang.language as T

    from ..diagonal.identity import fp32_gemm_macro

    batch, _, chunks, rows, columns = carry_meta[0]
    load_gates = _gate_macro(T, heads, length, identity, expressions[0])
    scan_suffix = _product_scan_macro(T, True)
    left_transform = _transform(T, expressions[1])
    right_transform = _transform(T, expressions[2])
    gemm = fp32_gemm_macro(T)

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build_kernel():
        @T.prim_func
        def kernel(
            Propagation: T.StridedTensor(*propagation_meta),
            Left: T.StridedTensor(*left_meta),
            Right: T.StridedTensor(*right_meta),
            Carry: T.StridedTensor(*carry_meta),
            Scale: T.StridedTensor(*scale_meta),
        ):
            with T.Kernel(
                batch * heads,
                chunks,
                T.ceildiv(rows, STATE_BLOCK) * T.ceildiv(columns, STATE_BLOCK),
                threads=threads,
            ) as (sh, chunk, tile):
                sequence, head = sh // heads, sh % heads
                rb = tile // T.ceildiv(columns, STATE_BLOCK)
                cb = tile % T.ceildiv(columns, STATE_BLOCK)
                key = T.alloc_shared((STATE_BLOCK, CHUNK_TOKENS), "float32")
                value = T.alloc_shared((CHUNK_TOKENS, STATE_BLOCK), "float32")
                gates = T.alloc_shared((CHUNK_TOKENS,), "float32")
                suffix = T.alloc_shared((CHUNK_TOKENS,), "float32")
                result = T.alloc_fragment((STATE_BLOCK, STATE_BLOCK), "float32")
                if not identity:
                    load_gates(Propagation, gates, sequence, head, chunk)
                    # The injection at t is multiplied by gates t+1..end.
                    for token in T.Parallel(CHUNK_TOKENS):
                        suffix[token] = T.if_then_else(
                            token + 1 < CHUNK_TOKENS, gates[token + 1], 1.0
                        )
                    T.sync_threads()
                    scan_suffix(suffix)
                    if tile == 0 and T.get_thread_binding() == 0:
                        Scale[sh, chunk] = gates[0] * suffix[0]
                for row, token in T.Parallel(STATE_BLOCK, CHUNK_TOKENS):
                    logical = chunk * CHUNK_TOKENS + token
                    if rb * STATE_BLOCK + row < rows and logical < length:
                        key[row, token] = left_transform(
                            T.cast(
                                Left[
                                    sequence,
                                    head // (heads // Left.shape[1]),
                                    logical,
                                    rb * STATE_BLOCK + row,
                                ],
                                "float32",
                            )
                        )
                        if not identity:
                            key[row, token] *= suffix[token]
                    else:
                        key[row, token] = 0.0
                for token, column in T.Parallel(CHUNK_TOKENS, STATE_BLOCK):
                    logical = chunk * CHUNK_TOKENS + token
                    if cb * STATE_BLOCK + column < columns and logical < length:
                        value[token, column] = right_transform(
                            T.cast(
                                Right[
                                    sequence,
                                    head // (heads // Right.shape[1]),
                                    logical,
                                    cb * STATE_BLOCK + column,
                                ],
                                "float32",
                            )
                        )
                    else:
                        value[token, column] = 0.0
                T.clear(result)
                gemm(key, value, result)
                for row, column in T.Parallel(STATE_BLOCK, STATE_BLOCK):
                    if (
                        rb * STATE_BLOCK + row < rows
                        and cb * STATE_BLOCK + column < columns
                    ):
                        Carry[
                            sequence,
                            head,
                            chunk,
                            rb * STATE_BLOCK + row,
                            cb * STATE_BLOCK + column,
                        ] = result[row, column]

        return kernel

    return build_kernel()


@lru_cache(maxsize=None)
def _compile_carry(
    scale_meta,
    carry_meta,
    initial_meta,
    final_meta,
    *,
    identity,
    write_final,
    carry_block,
    device_index,
):
    import tilelang
    import tilelang.language as T

    batch, heads, chunks, rows, columns = carry_meta[0]

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build_kernel():
        @T.prim_func
        def kernel(
            Scale: T.StridedTensor(*scale_meta),
            Carry: T.StridedTensor(*carry_meta),
            Initial: T.StridedTensor(*initial_meta),
            Final: T.StridedTensor(*final_meta),
        ):
            with T.Kernel(
                batch * heads, T.ceildiv(rows * columns, carry_block), threads=128
            ) as (sh, block):
                sequence, head = sh // heads, sh % heads
                state = T.alloc_fragment((carry_block,), "float32")
                bias = T.alloc_fragment((carry_block,), "float32")
                for lane in T.Parallel(carry_block):
                    element = block * carry_block + lane
                    if element < rows * columns:
                        state[lane] = Initial[
                            sequence, head, element // columns, element % columns
                        ]
                    else:
                        state[lane] = 0.0
                for chunk in T.serial(chunks):
                    for lane in T.Parallel(carry_block):
                        element = block * carry_block + lane
                        if element < rows * columns:
                            bias[lane] = Carry[
                                sequence,
                                head,
                                chunk,
                                element // columns,
                                element % columns,
                            ]
                            # Consume the summary before replacing it with the
                            # exclusive carry. No second matrix workspace needed.
                            Carry[
                                sequence,
                                head,
                                chunk,
                                element // columns,
                                element % columns,
                            ] = state[lane]
                            if identity:
                                state[lane] += bias[lane]
                            else:
                                state[lane] = (
                                    Scale[sh, chunk] * state[lane] + bias[lane]
                                )
                if write_final:
                    for lane in T.Parallel(carry_block):
                        element = block * carry_block + lane
                        if element < rows * columns:
                            Final[
                                sequence, head, element // columns, element % columns
                            ] = state[lane]

        return kernel

    return build_kernel()


@lru_cache(maxsize=None)
def _compile_readout(
    propagation_meta,
    left_meta,
    right_meta,
    readout_meta,
    carry_meta,
    output_meta,
    expressions,
    *,
    heads,
    length,
    identity,
    threads,
    device_index,
):
    import tilelang
    import tilelang.language as T

    from ..diagonal.identity import fp32_gemm_macro

    batch, _, chunks, rows, columns = carry_meta[0]
    load_gates = _gate_macro(T, heads, length, identity, expressions[0])
    scan_prefix = _product_scan_macro(T, False)
    left_transform = _transform(T, expressions[1])
    right_transform = _transform(T, expressions[2])
    readout_transform = _transform(T, expressions[3])
    gemm = fp32_gemm_macro(T)

    @T.macro
    def prepared_left_gemm(AH, AL, B, C):
        """FP32 GEMM with a pre-split left operand.

        The readout query feeds both query-key and query-State products.  The
        hi/lo split is invariant across those products, so retain it once in
        shared memory while each right operand still receives the standard
        three-term TF32 decomposition.
        """
        bh = T.alloc_shared(B.shape, T.tfloat32)
        bl = T.alloc_shared(B.shape, T.tfloat32)
        for k, j in T.Parallel(B.shape[0], B.shape[1]):
            bh[k, j] = T.cast(B[k, j], T.tfloat32)
            bl[k, j] = T.cast(B[k, j] - T.cast(bh[k, j], "float32"), T.tfloat32)
        T.gemm(AL, bh, C)
        T.gemm(AH, bl, C)
        T.gemm(AH, bh, C)

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build_kernel():
        @T.prim_func
        def kernel(
            Propagation: T.StridedTensor(*propagation_meta),
            Left: T.StridedTensor(*left_meta),
            Right: T.StridedTensor(*right_meta),
            Readout: T.StridedTensor(*readout_meta),
            Carry: T.StridedTensor(*carry_meta),
            Output: T.StridedTensor(*output_meta),
        ):
            with T.Kernel(
                batch * heads, chunks, T.ceildiv(columns, STATE_BLOCK), threads=threads
            ) as (sh, chunk, cb):
                sequence, head = sh // heads, sh % heads
                query = T.alloc_shared((CHUNK_TOKENS, STATE_BLOCK), "float32")
                query_hi = T.alloc_shared((CHUNK_TOKENS, STATE_BLOCK), T.tfloat32)
                query_lo = T.alloc_shared((CHUNK_TOKENS, STATE_BLOCK), T.tfloat32)
                key = T.alloc_shared((STATE_BLOCK, CHUNK_TOKENS), "float32")
                state = T.alloc_shared((STATE_BLOCK, STATE_BLOCK), "float32")
                value = T.alloc_shared((CHUNK_TOKENS, STATE_BLOCK), "float32")
                weights = T.alloc_shared((CHUNK_TOKENS, CHUNK_TOKENS), "float32")
                weight_scratch = T.alloc_shared((CHUNK_TOKENS, CHUNK_TOKENS), "float32")
                gates = T.alloc_shared((CHUNK_TOKENS,), "float32")
                prefix = T.alloc_shared((CHUNK_TOKENS,), "float32")
                scores = T.alloc_fragment((CHUNK_TOKENS, CHUNK_TOKENS), "float32")
                output = T.alloc_fragment((CHUNK_TOKENS, STATE_BLOCK), "float32")
                if not identity:
                    load_gates(Propagation, gates, sequence, head, chunk)
                    T.copy(gates, prefix)
                    T.sync_threads()
                    scan_prefix(prefix)
                    # Reverse inclusive products over sources give the product
                    # gates[source+1..token], without prefix ratios/log subtraction.
                    for token, source in T.Parallel(CHUNK_TOKENS, CHUNK_TOKENS):
                        weights[token, source] = T.if_then_else(
                            source < token, gates[source + 1], 1.0
                        )
                    T.sync_threads()
                    for pair in T.unroll(3):
                        distance = 1 << (2 * pair)
                        for token, source in T.Parallel(CHUNK_TOKENS, CHUNK_TOKENS):
                            weight_scratch[token, source] = weights[
                                token, source
                            ] * T.if_then_else(
                                source + distance < CHUNK_TOKENS,
                                weights[token, source + distance],
                                1.0,
                            )
                        T.sync_threads()
                        for token, source in T.Parallel(CHUNK_TOKENS, CHUNK_TOKENS):
                            weights[token, source] = weight_scratch[
                                token, source
                            ] * T.if_then_else(
                                source + 2 * distance < CHUNK_TOKENS,
                                weight_scratch[token, source + 2 * distance],
                                1.0,
                            )
                        T.sync_threads()
                T.clear(scores)
                T.clear(output)
                for rb in T.serial(T.ceildiv(rows, STATE_BLOCK)):
                    for token, row in T.Parallel(CHUNK_TOKENS, STATE_BLOCK):
                        logical = chunk * CHUNK_TOKENS + token
                        query_value = T.alloc_var("float32", init=0.0)
                        if logical < length and rb * STATE_BLOCK + row < rows:
                            query_value = readout_transform(
                                T.cast(
                                    Readout[
                                        sequence,
                                        head // (heads // Readout.shape[1]),
                                        logical,
                                        rb * STATE_BLOCK + row,
                                    ],
                                    "float32",
                                )
                            )
                            key[row, token] = left_transform(
                                T.cast(
                                    Left[
                                        sequence,
                                        head // (heads // Left.shape[1]),
                                        logical,
                                        rb * STATE_BLOCK + row,
                                    ],
                                    "float32",
                                )
                            )
                        else:
                            key[row, token] = 0.0
                        if identity:
                            query[token, row] = query_value
                        else:
                            query_hi[token, row] = T.cast(query_value, T.tfloat32)
                            query_lo[token, row] = T.cast(
                                query_value - T.cast(query_hi[token, row], "float32"),
                                T.tfloat32,
                            )
                    for row, column in T.Parallel(STATE_BLOCK, STATE_BLOCK):
                        if (
                            rb * STATE_BLOCK + row < rows
                            and cb * STATE_BLOCK + column < columns
                        ):
                            state[row, column] = Carry[
                                sequence,
                                head,
                                chunk,
                                rb * STATE_BLOCK + row,
                                cb * STATE_BLOCK + column,
                            ]
                        else:
                            state[row, column] = 0.0
                    # Identity retains its measured staging schedule. Gated
                    # readouts split directly from the loaded FP32 lane value.
                    if identity:
                        for token, row in T.Parallel(CHUNK_TOKENS, STATE_BLOCK):
                            query_hi[token, row] = T.cast(query[token, row], T.tfloat32)
                            query_lo[token, row] = T.cast(
                                query[token, row]
                                - T.cast(query_hi[token, row], "float32"),
                                T.tfloat32,
                            )
                    T.sync_threads()
                    prepared_left_gemm(query_hi, query_lo, key, scores)
                    prepared_left_gemm(query_hi, query_lo, state, output)
                for token, column in T.Parallel(CHUNK_TOKENS, STATE_BLOCK):
                    if not identity:
                        output[token, column] *= prefix[token]
                for token, source in T.Parallel(CHUNK_TOKENS, CHUNK_TOKENS):
                    if source <= token:
                        if identity:
                            weights[token, source] = scores[token, source]
                        else:
                            weights[token, source] *= scores[token, source]
                    else:
                        weights[token, source] = 0.0
                for token, column in T.Parallel(CHUNK_TOKENS, STATE_BLOCK):
                    logical = chunk * CHUNK_TOKENS + token
                    if logical < length and cb * STATE_BLOCK + column < columns:
                        value[token, column] = right_transform(
                            T.cast(
                                Right[
                                    sequence,
                                    head // (heads // Right.shape[1]),
                                    logical,
                                    cb * STATE_BLOCK + column,
                                ],
                                "float32",
                            )
                        )
                    else:
                        value[token, column] = 0.0
                gemm(weights, value, output)
                for token, column in T.Parallel(CHUNK_TOKENS, STATE_BLOCK):
                    logical = chunk * CHUNK_TOKENS + token
                    if logical < length and cb * STATE_BLOCK + column < columns:
                        Output[sequence, head, logical, cb * STATE_BLOCK + column] = (
                            output[token, column]
                        )

        return kernel

    return build_kernel()


def scalar_chunked_dense_raw(
    propagation: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    readout: torch.Tensor,
    initial: torch.Tensor,
    output: torch.Tensor,
    final: torch.Tensor,
    expressions: tuple[Any, Any, Any, Any],
    *,
    sequence_length: int,
    identity_propagation: bool,
    return_final_state: bool,
    threads: int,
) -> None:
    """Launch summary → exclusive carry → readout using the established raw ABI."""
    sequences, heads, rows, columns = initial.shape
    chunks = (sequence_length + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    carry = torch.empty(
        sequences,
        heads,
        chunks,
        rows,
        columns,
        device=initial.device,
        dtype=torch.float32,
    )
    # Identity kernels erase Scale accesses; retain a live pointer, not dummy storage.
    scale = (
        initial
        if identity_propagation
        else torch.empty(
            sequences * heads, chunks, device=initial.device, dtype=torch.float32
        )
    )
    expression_meta = tuple(
        (
            expression.operation,
            expression.scale,
            expression.is_log_scale,
            getattr(expression, "offset", 0.0),
            getattr(expression, "scale_steps", ()),
            getattr(expression, "expression_nodes", ()),
        )
        for expression in expressions
    )
    device_index = initial.device.index
    if initial.is_cuda and device_index is None:
        device_index = torch.cuda.current_device()
    common = dict(
        heads=heads,
        length=sequence_length,
        identity=identity_propagation,
        threads=threads,
        device_index=device_index,
    )
    _compile_summary(
        _metadata(propagation),
        _metadata(left),
        _metadata(right),
        _metadata(carry),
        _metadata(scale),
        expression_meta,
        **common,
    )(propagation, left, right, carry, scale)
    # Regular substantial dense states use 512-element carry blocks, halving
    # carry CTAs without changing the original tail schedule.
    carry_elements = rows * columns
    carry_block = (
        512 if carry_elements >= 4096 and carry_elements % 512 == 0 else CARRY_BLOCK
    )
    _compile_carry(
        _metadata(scale),
        _metadata(carry),
        _metadata(initial),
        _metadata(final),
        identity=identity_propagation,
        write_final=return_final_state,
        carry_block=carry_block,
        device_index=device_index,
    )(scale, carry, initial, final)
    _compile_readout(
        _metadata(propagation),
        _metadata(left),
        _metadata(right),
        _metadata(readout),
        _metadata(carry),
        _metadata(output),
        expression_meta,
        **common,
    )(propagation, left, right, readout, carry, output)
