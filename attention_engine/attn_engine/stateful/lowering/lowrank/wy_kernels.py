"""Ordered scalar/rank-one inference: register recurrence and block compact WY."""

from functools import lru_cache

import torch

from ..diagonal.identity import fp32_gemm_macro

_BLOCK = 32
_SHORT = 128


def eligible(operation_codes, injection_count, rows, columns):
    events = operation_codes.count("rank-one") + injection_count
    return (
        "rank-one" in operation_codes
        and all(x in ("scale", "rank-one") for x in operation_codes)
        and 1 <= events <= 4
        and 0 < rows <= 128
        and 0 < columns <= 128
    )


def chunk_tokens(event_count):
    return 16


def scratch_bytes_per_chunk(
    operation_codes, injection_count, readout_count, rows, columns
):
    """Live global WY scratch for one chunk/head (factor/output packing excluded)."""
    events = operation_codes.count("rank-one") + injection_count
    be = max(32, 1 << (16 * events - 1).bit_length())
    br = max(32, 1 << (rows - 1).bit_length())
    return 4 * (
        2 * be * br + be * columns + br * columns + readout_count * 16 * be + 16
    )


def _meta(x):
    return tuple(x.shape), tuple(x.stride()), str(x.dtype).removeprefix("torch.")


def _select(T, values, index):
    result = 0
    for i, value in enumerate(values):
        result = T.if_then_else(index == i, value, result)
    return result


def _gemm_macro(T):
    """Bound K staging; retain the existing triple-TF32 factor contract."""
    fp32 = fp32_gemm_macro(T)

    @T.macro
    def gemm(A, B, C):
        a = T.alloc_shared((A.shape[0], 32), "float32")
        b = T.alloc_shared((32, B.shape[1]), "float32")
        for kb in T.serial((A.shape[1] + 31) // 32):
            for i, k in T.Parallel(A.shape[0], 32):
                a[i, k] = T.if_then_else(
                    kb * 32 + k < A.shape[1], A[i, kb * 32 + k], 0.0
                )
            for k, j in T.Parallel(32, B.shape[1]):
                b[k, j] = T.if_then_else(
                    kb * 32 + k < B.shape[0], B[kb * 32 + k, j], 0.0
                )
            fp32(a, b, C)

    return gemm


@lru_cache(maxsize=None)
def _compile_direct(
    metas,
    *,
    batch,
    heads,
    rows,
    columns,
    length,
    codes,
    injections,
    readouts,
    symmetric_rank_one,
    device_index,
):
    import tilelang
    import tilelang.language as T

    br = max(16, 1 << (rows - 1).bit_length())
    ranks = tuple(i for i, code in enumerate(codes) if code == 2)
    rank_slot = tuple(
        ranks.index(i) if code == 2 else 0 for i, code in enumerate(codes)
    )

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build():
        @T.prim_func
        def kernel(
            L: T.StridedTensor(*metas[0]),
            R: T.StridedTensor(*metas[1]),
            C: T.StridedTensor(*metas[2]),
            IL: T.StridedTensor(*metas[3]),
            IR: T.StridedTensor(*metas[4]),
            Q: T.StridedTensor(*metas[5]),
            Initial: T.StridedTensor(*metas[6]),
            Output: T.StridedTensor(*metas[7]),
            Final: T.StridedTensor(*metas[8]),
        ):
            with T.Kernel(
                batch * heads, (columns + _BLOCK - 1) // _BLOCK, threads=128
            ) as (sh, cb):
                sequence, head = sh // heads, sh % heads
                state = T.alloc_fragment((br, _BLOCK), "float32")
                products = T.alloc_fragment((br, _BLOCK), "float32")
                projection = T.alloc_fragment((_BLOCK,), "float32")
                for r, c in T.Parallel(br, _BLOCK):
                    state[r, c] = T.if_then_else(
                        (r < rows) & (cb * _BLOCK + c < columns),
                        T.cast(Initial[sequence, head, r, cb * _BLOCK + c], "float32"),
                        0.0,
                    )
                for t in T.serial(length):
                    token = sequence * length + t
                    for op in T.unroll(len(codes)):
                        if _select(T, codes, op) == 0:
                            for r, c in T.Parallel(br, _BLOCK):
                                state[r, c] = state[r, c] * C[op, token, head]
                        else:
                            for r, c in T.Parallel(br, _BLOCK):
                                factor = T.if_then_else(
                                    r < rows,
                                    L[_select(T, rank_slot, op), token, head, r]
                                    if symmetric_rank_one
                                    else R[_select(T, rank_slot, op), token, head, r],
                                    0.0,
                                )
                                products[r, c] = state[r, c] * factor
                            T.reduce_sum(products, projection, dim=0)
                            for r, c in T.Parallel(br, _BLOCK):
                                factor = T.if_then_else(
                                    r < rows,
                                    L[_select(T, rank_slot, op), token, head, r],
                                    0.0,
                                )
                                state[r, c] = (
                                    state[r, c]
                                    + C[op, token, head] * factor * projection[c]
                                )
                    for j in T.unroll(injections):
                        for r, c in T.Parallel(br, _BLOCK):
                            key = T.if_then_else(r < rows, IL[j, token, head, r], 0.0)
                            value = T.if_then_else(
                                cb * _BLOCK + c < columns,
                                IR[j, token, head, cb * _BLOCK + c],
                                0.0,
                            )
                            state[r, c] = state[r, c] + key * value
                    for q in T.unroll(readouts):
                        for r, c in T.Parallel(br, _BLOCK):
                            products[r, c] = state[r, c] * T.if_then_else(
                                r < rows, Q[q, token, head, r], 0.0
                            )
                        T.reduce_sum(products, projection, dim=0)
                        for c in T.Parallel(_BLOCK):
                            if cb * _BLOCK + c < columns:
                                Output[q, token, head, cb * _BLOCK + c] = projection[c]
                for r, c in T.Parallel(br, _BLOCK):
                    if (r < rows) & (cb * _BLOCK + c < columns):
                        Final[sequence, head, r, cb * _BLOCK + c] = state[r, c]

        return kernel

    return build()


@lru_cache(maxsize=None)
def _compile_solve(
    metas,
    *,
    batch,
    heads,
    rows,
    columns,
    length,
    codes,
    injections,
    readouts,
    symmetric_rank_one,
    device_index,
):
    import tilelang
    import tilelang.language as T

    chunk = 16
    ranks = tuple(i for i, code in enumerate(codes) if code == 2)
    scales = tuple(i for i, code in enumerate(codes) if code == 0)
    events = len(ranks) + injections
    ne = chunk * events
    be = max(32, 1 << (ne - 1).bit_length())
    br = max(32, 1 << (rows - 1).bit_length())
    chunks = (length + chunk - 1) // chunk
    gemm = _gemm_macro(T)

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build():
        @T.prim_func
        def kernel(
            L: T.StridedTensor(*metas[0]),
            R: T.StridedTensor(*metas[1]),
            C: T.StridedTensor(*metas[2]),
            IL: T.StridedTensor(*metas[3]),
            IR: T.StridedTensor(*metas[4]),
            Q: T.StridedTensor(*metas[5]),
            D: T.Tensor((batch, heads, chunks, be, br), "float32"),
            F: T.Tensor((batch, heads, chunks, be, columns), "float32"),
            U: T.Tensor((batch, heads, chunks, br, be), "float32"),
            W: T.Tensor((readouts, batch, heads, chunks, chunk, be), "float32"),
            P: T.Tensor((batch, heads, chunks, chunk), "float32"),
        ):
            with T.Kernel(batch * heads, chunks, threads=128) as (sh, ci):
                sequence, head = sh // heads, sh % heads
                u = T.alloc_shared((br, be), "float32")
                right = T.alloc_shared((be, br), "float32")
                seed = T.alloc_shared((be, br), "float32")
                interaction = T.alloc_fragment((be, be), "float32")
                lower = T.alloc_shared((be, be), "float32")
                inverse = T.alloc_shared((be, be), "float32")
                inverse_row = T.alloc_fragment((be, be), "float32")
                reduced = T.alloc_fragment((be,), "float32")
                gates = T.alloc_shared((chunk,), "float32")
                prefix = T.alloc_shared((chunk,), "float32")
                intervals = T.alloc_shared((chunk, chunk), "float32")
                for t in T.Parallel(chunk):
                    token = sequence * length + ci * chunk + t
                    gates[t] = 1.0
                    for s in T.unroll(len(scales)):
                        gates[t] = gates[t] * T.if_then_else(
                            ci * chunk + t < length,
                            C[_select(T, scales, s), token, head],
                            1.0,
                        )
                T.sync_threads()
                prefix[0] = gates[0]
                for t in T.serial(1, chunk):
                    prefix[t] = prefix[t - 1] * gates[t]
                for p in T.Parallel(chunk):
                    intervals[p, p] = 1.0
                    for t in T.serial(chunk):
                        if t > p:
                            intervals[p, t] = intervals[p, t - 1] * gates[t]
                T.sync_threads()
                for e, r in T.Parallel(be, br):
                    t, kind = e // events, e % events
                    token = sequence * length + ci * chunk + t
                    valid = (e < ne) & (ci * chunk + t < length) & (r < rows)
                    u[r, e] = 0.0
                    right[e, r] = 0.0
                    for k in T.unroll(len(ranks)):
                        if kind == k:
                            u[r, e] = T.if_then_else(
                                valid,
                                C[_select(T, ranks, k), token, head]
                                * L[k, token, head, r],
                                0.0,
                            )
                            right[e, r] = T.if_then_else(
                                valid,
                                L[k, token, head, r]
                                if symmetric_rank_one
                                else R[k, token, head, r],
                                0.0,
                            )
                    for j in T.unroll(injections):
                        if kind == len(ranks) + j:
                            u[r, e] = T.if_then_else(valid, IL[j, token, head, r], 0.0)
                    seed[e, r] = right[e, r] * T.if_then_else(e < ne, prefix[t], 0.0)
                T.clear(interaction)
                gemm(right, u, interaction)
                for e, p in T.Parallel(be, be):
                    lower[e, p] = T.if_then_else(
                        (p < e) & (e < ne),
                        interaction[e, p] * intervals[p // events, e // events],
                        0.0,
                    )
                    inverse[e, p] = T.if_then_else(e == p, 1.0, 0.0)
                T.sync_threads()
                # (I-L)^-1 by ordered forward substitution. Each row's K
                # reduction and all RHS columns are parallel, never scalar dots.
                for e in T.serial(ne):
                    for p, k in T.Parallel(be, be):
                        inverse_row[k, p] = T.if_then_else(
                            k < e, lower[e, k] * inverse[k, p], 0.0
                        )
                    T.reduce_sum(inverse_row, reduced, dim=0)
                    for p in T.Parallel(be):
                        inverse[e, p] = reduced[p] + T.if_then_else(e == p, 1.0, 0.0)
                    T.sync_threads()
                solved = T.alloc_fragment((be, br), "float32")
                T.clear(solved)
                gemm(inverse, seed, solved)
                for e, r in T.Parallel(be, br):
                    D[sequence, head, ci, e, r] = solved[e, r]
                    U[sequence, head, ci, r, e] = u[r, e] * T.if_then_else(
                        e < ne, intervals[e // events, chunk - 1], 0.0
                    )
                values = T.alloc_shared((be, _BLOCK), "float32")
                bias = T.alloc_fragment((be, _BLOCK), "float32")
                for cb in T.serial((columns + _BLOCK - 1) // _BLOCK):
                    for e, c in T.Parallel(be, _BLOCK):
                        t, kind = e // events, e % events
                        token = sequence * length + ci * chunk + t
                        values[e, c] = 0.0
                        for j in T.unroll(injections):
                            if kind == len(ranks) + j:
                                values[e, c] = T.if_then_else(
                                    (e < ne)
                                    & (ci * chunk + t < length)
                                    & (cb * _BLOCK + c < columns),
                                    IR[j, token, head, cb * _BLOCK + c],
                                    0.0,
                                )
                    T.clear(bias)
                    gemm(inverse, values, bias)
                    for e, c in T.Parallel(be, _BLOCK):
                        if cb * _BLOCK + c < columns:
                            F[sequence, head, ci, e, cb * _BLOCK + c] = bias[e, c]
                queries = T.alloc_shared((chunk, br), "float32")
                weights = T.alloc_fragment((chunk, be), "float32")
                for q in T.unroll(readouts):
                    for t, r in T.Parallel(chunk, br):
                        queries[t, r] = T.if_then_else(
                            (ci * chunk + t < length) & (r < rows),
                            Q[q, sequence * length + ci * chunk + t, head, r],
                            0.0,
                        )
                    T.clear(weights)
                    gemm(queries, u, weights)
                    for t, e in T.Parallel(chunk, be):
                        W[q, sequence, head, ci, t, e] = T.if_then_else(
                            (e < ne) & (e // events <= t),
                            weights[t, e] * intervals[e // events, t],
                            0.0,
                        )
                for t in T.Parallel(chunk):
                    P[sequence, head, ci, t] = prefix[t]

        return kernel

    return build()


@lru_cache(maxsize=None)
def _compile_carry(
    initial_meta,
    final_meta,
    *,
    batch,
    heads,
    rows,
    columns,
    length,
    events,
    device_index,
):
    import tilelang
    import tilelang.language as T

    chunk, chunks = 16, (length + 15) // 16
    be = max(32, 1 << (chunk * events - 1).bit_length())
    br = max(32, 1 << (rows - 1).bit_length())
    gemm = _gemm_macro(T)

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build():
        @T.prim_func
        def kernel(
            D: T.Tensor((batch, heads, chunks, be, br), "float32"),
            F: T.Tensor((batch, heads, chunks, be, columns), "float32"),
            U: T.Tensor((batch, heads, chunks, br, be), "float32"),
            P: T.Tensor((batch, heads, chunks, chunk), "float32"),
            Initial: T.StridedTensor(*initial_meta),
            Entry: T.Tensor((batch, heads, chunks, br, columns), "float32"),
            Final: T.StridedTensor(*final_meta),
        ):
            with T.Kernel(
                batch * heads, (columns + _BLOCK - 1) // _BLOCK, threads=128
            ) as (sh, cb):
                sequence, head = sh // heads, sh % heads
                state = T.alloc_shared((br, _BLOCK), "float32")
                d = T.alloc_shared((be, br), "float32")
                u = T.alloc_shared((br, be), "float32")
                response = T.alloc_fragment((be, _BLOCK), "float32")
                responses = T.alloc_shared((be, _BLOCK), "float32")
                next_state = T.alloc_fragment((br, _BLOCK), "float32")
                for r, c in T.Parallel(br, _BLOCK):
                    state[r, c] = T.if_then_else(
                        (r < rows) & (cb * _BLOCK + c < columns),
                        T.cast(Initial[sequence, head, r, cb * _BLOCK + c], "float32"),
                        0.0,
                    )
                for ci in T.serial(chunks):
                    for r, c in T.Parallel(br, _BLOCK):
                        if cb * _BLOCK + c < columns:
                            Entry[sequence, head, ci, r, cb * _BLOCK + c] = state[r, c]
                    for e, r in T.Parallel(be, br):
                        d[e, r] = D[sequence, head, ci, e, r]
                        u[r, e] = U[sequence, head, ci, r, e]
                    T.clear(response)
                    gemm(d, state, response)
                    for e, c in T.Parallel(be, _BLOCK):
                        value = T.if_then_else(
                            cb * _BLOCK + c < columns,
                            F[sequence, head, ci, e, cb * _BLOCK + c],
                            0.0,
                        )
                        responses[e, c] = response[e, c] + value
                        if cb * _BLOCK + c < columns:
                            F[sequence, head, ci, e, cb * _BLOCK + c] = responses[e, c]
                    T.clear(next_state)
                    gemm(u, responses, next_state)
                    for r, c in T.Parallel(br, _BLOCK):
                        state[r, c] = (
                            P[sequence, head, ci, chunk - 1] * state[r, c]
                            + next_state[r, c]
                        )
                    T.sync_threads()
                for r, c in T.Parallel(br, _BLOCK):
                    if (r < rows) & (cb * _BLOCK + c < columns):
                        Final[sequence, head, r, cb * _BLOCK + c] = state[r, c]

        return kernel

    return build()


@lru_cache(maxsize=None)
def _compile_readout(
    query_meta,
    output_meta,
    *,
    batch,
    heads,
    rows,
    columns,
    length,
    events,
    readouts,
    device_index,
):
    import tilelang
    import tilelang.language as T

    chunk, chunks = 16, (length + 15) // 16
    be = max(32, 1 << (chunk * events - 1).bit_length())
    br = max(32, 1 << (rows - 1).bit_length())
    gemm = _gemm_macro(T)

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build():
        @T.prim_func
        def kernel(
            Q: T.StridedTensor(*query_meta),
            W: T.Tensor((readouts, batch, heads, chunks, chunk, be), "float32"),
            P: T.Tensor((batch, heads, chunks, chunk), "float32"),
            Entry: T.Tensor((batch, heads, chunks, br, columns), "float32"),
            Response: T.Tensor((batch, heads, chunks, be, columns), "float32"),
            Output: T.StridedTensor(*output_meta),
        ):
            with T.Kernel(
                batch * heads, chunks, (columns + _BLOCK - 1) // _BLOCK, threads=128
            ) as (sh, ci, cb):
                sequence, head = sh // heads, sh % heads
                queries = T.alloc_shared((chunk, br), "float32")
                weights = T.alloc_shared((chunk, be), "float32")
                entry = T.alloc_shared((br, _BLOCK), "float32")
                responses = T.alloc_shared((be, _BLOCK), "float32")
                seed = T.alloc_fragment((chunk, _BLOCK), "float32")
                out = T.alloc_fragment((chunk, _BLOCK), "float32")
                for r, c in T.Parallel(br, _BLOCK):
                    entry[r, c] = T.if_then_else(
                        cb * _BLOCK + c < columns,
                        Entry[sequence, head, ci, r, cb * _BLOCK + c],
                        0.0,
                    )
                for e, c in T.Parallel(be, _BLOCK):
                    responses[e, c] = T.if_then_else(
                        cb * _BLOCK + c < columns,
                        Response[sequence, head, ci, e, cb * _BLOCK + c],
                        0.0,
                    )
                for q in T.unroll(readouts):
                    for t, r in T.Parallel(chunk, br):
                        queries[t, r] = T.if_then_else(
                            (ci * chunk + t < length) & (r < rows),
                            Q[q, sequence * length + ci * chunk + t, head, r],
                            0.0,
                        )
                    for t, e in T.Parallel(chunk, be):
                        weights[t, e] = W[q, sequence, head, ci, t, e]
                    T.clear(seed)
                    T.clear(out)
                    gemm(queries, entry, seed)
                    gemm(weights, responses, out)
                    for t, c in T.Parallel(chunk, _BLOCK):
                        if (ci * chunk + t < length) & (cb * _BLOCK + c < columns):
                            Output[
                                q,
                                sequence * length + ci * chunk + t,
                                head,
                                cb * _BLOCK + c,
                            ] = P[sequence, head, ci, t] * seed[t, c] + out[t, c]

        return kernel

    return build()


def execute(
    *,
    codes,
    left,
    right,
    coefficient,
    injection_left,
    injection_right,
    query,
    initial_state,
    length,
    output_dtypes,
    symmetric_rank_one=False,
):
    batch, heads, rows, columns = initial_state.shape
    device = initial_state.device

    def bind(x, size):
        x = x.float()
        if x.numel() == 1:
            return x.reshape(1, 1, 1).expand(batch * length, heads, size)
        return x.reshape(batch * length, heads, -1).expand(batch * length, heads, size)

    L = torch.stack([bind(x, rows) for x in left])
    R = L if symmetric_rank_one else torch.stack([bind(x, rows) for x in right])
    C = torch.stack([bind(x, 1).squeeze(-1) for x in coefficient])
    IL = (
        torch.stack([bind(x, rows) for x in injection_left])
        if injection_left
        else torch.empty((0, batch * length, heads, rows), device=device)
    )
    IR = (
        torch.stack([bind(x, columns) for x in injection_right])
        if injection_right
        else torch.empty((0, batch * length, heads, columns), device=device)
    )
    Q = torch.stack([bind(x, rows) for x in query])
    output = torch.empty(
        (len(query), batch * length, heads, columns), device=device, dtype=torch.float32
    )
    final = torch.empty(
        (batch, heads, rows, columns), device=device, dtype=torch.float32
    )
    tensors = (L, R, C, IL, IR, Q, initial_state, output, final)
    metas = tuple(_meta(x) for x in tensors)
    parameters = dict(
        batch=batch,
        heads=heads,
        rows=rows,
        columns=columns,
        length=length,
        codes=tuple(codes),
        injections=len(injection_left),
        readouts=len(query),
        symmetric_rank_one=symmetric_rank_one,
        device_index=device.index,
    )
    if length <= _SHORT:
        _compile_direct(metas, **parameters)(*tensors)
    else:
        del tensors
        events = len(left) + len(injection_left)
        chunks = (length + 15) // 16
        be = max(32, 1 << (16 * events - 1).bit_length())
        br = max(32, 1 << (rows - 1).bit_length())
        shape = (batch, heads, chunks)

        def empty(*tail):
            return torch.empty((*shape, *tail), device=device, dtype=torch.float32)

        d, u = empty(be, br), empty(br, be)
        # Carry reads each chunk's additive term before overwriting it with the
        # response consumed by readout; their global storage can be shared.
        f = empty(be, columns)
        entry, prefix = empty(br, columns), empty(16)
        weights = torch.empty(
            (len(query), *shape, 16, be), device=device, dtype=torch.float32
        )
        _compile_solve(metas, **parameters)(
            L, R, C, IL, IR, Q, d, f, u, weights, prefix
        )
        del L, R, C, IL, IR
        carry = dict(
            batch=batch,
            heads=heads,
            rows=rows,
            columns=columns,
            length=length,
            events=events,
            device_index=device.index,
        )
        _compile_carry(metas[6], metas[8], **carry)(
            d, f, u, prefix, initial_state, entry, final
        )
        del d, u
        _compile_readout(metas[5], metas[7], readouts=len(query), **carry)(
            Q, weights, prefix, entry, f, output
        )
    return tuple(
        output[i].reshape(batch, length, heads, columns).permute(0, 2, 1, 3).to(dtype)
        for i, dtype in enumerate(output_dtypes)
    ), final.to(initial_state.dtype)
