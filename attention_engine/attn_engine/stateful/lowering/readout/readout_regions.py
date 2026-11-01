"""TileLang chunk-local normalized, softmax, and shared State contractions."""

from functools import lru_cache
import hashlib
import linecache

import torch


CHUNK = 64


def _next_power_of_two(value):
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _metadata(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype).removeprefix("torch."),
    )


def _query_parts(query, tokens, heads, rows):
    if isinstance(query, tuple):
        q, q2 = (part.reshape(tokens, heads, -1) for part in query)
        if q.shape[-1] * q2.shape[-1] != rows:
            return None
        return q, q2, q.shape[-1], q2.shape[-1]
    q = query.reshape(tokens, heads, rows)
    return q, q, rows, 0


def readout_region_resources(first_shape, second_shape, *, softmax=True):
    """Conservative live FP32 element bound for the two-State region."""
    _, _, r1, d1 = first_shape
    _, _, r2, d2 = second_shape
    a, b, c, d = map(_next_power_of_two, (r1, d1, r2, d2))
    live = (
        a * b
        + c * d
        + max(a * b, c * d)
        + 4 * (a + b + c + d)
        + (3 * b if softmax else b)
    )
    return {
        "live_fp32_elements": live,
        "live_fp32_bytes": 4 * live,
        "eligible": max(a, b, c, d) <= 256 and live <= 16384,
    }


def _factor(T, key, key2, flat, head, r, kr):
    if kr:
        return T.cast(key[flat, head, r // kr], "float32") * T.cast(
            key2[flat, head, r % kr], "float32"
        )
    return T.cast(key[flat, head, r], "float32")


def _advance_macro(T, rows, columns, kr, unit_row, unit_column):
    @T.macro
    def advance(State, Row, Column, Key, Key2, Value, flat, head):
        for r, c in T.Parallel(rows, columns):
            rg = 1.0 if unit_row else T.cast(Row[flat, head, r], "float32")
            cg = 1.0 if unit_column else T.cast(Column[flat, head, c], "float32")
            key = _factor(T, Key, Key2, flat, head, r, kr)
            State[r, c] = (rg * cg) * State[r, c] + key * T.cast(
                Value[flat, head, c], "float32"
            )
        T.sync_threads()

    return advance


@lru_cache(maxsize=None)
def _compile_region(
    metas,
    length,
    heads,
    kr1,
    kr2,
    qr,
    unit1,
    unit2,
    prior1,
    prior2,
    softmax,
    epsilon,
    device_index,
):
    import tilelang
    import tilelang.language as T

    c1m, r1m, d1m, k1m, k12m, v1m, c2m, r2m, d2m, k2m, k22m, v2m, qm, q2m, om = metas
    bh, chunks, r1, d1 = c1m[0]
    _, _, r2, d2 = c2m[0]
    advance1 = _advance_macro(T, r1, d1, kr1, *unit1)
    advance2 = _advance_macro(T, r2, d2, kr2, *unit2)

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build():
        @T.prim_func
        def kernel(
            C1: T.StridedTensor(*c1m),
            R1: T.StridedTensor(*r1m),
            D1: T.StridedTensor(*d1m),
            K1: T.StridedTensor(*k1m),
            K12: T.StridedTensor(*k12m),
            V1: T.StridedTensor(*v1m),
            C2: T.StridedTensor(*c2m),
            R2: T.StridedTensor(*r2m),
            D2: T.StridedTensor(*d2m),
            K2: T.StridedTensor(*k2m),
            K22: T.StridedTensor(*k22m),
            V2: T.StridedTensor(*v2m),
            Q: T.StridedTensor(*qm),
            Q2: T.StridedTensor(*q2m),
            Output: T.StridedTensor(*om),
        ):
            with T.Kernel(bh, chunks, threads=128) as (sh, chunk):
                seq, head = sh // heads, sh % heads
                s1 = T.alloc_shared((r1, d1), "float32")
                s2 = T.alloc_shared((r2, d2), "float32")
                first = T.alloc_shared((d1,), "float32")
                second = T.alloc_shared((d2,), "float32")
                normalizer = T.alloc_shared((2,), "float32")
                for r, c in T.Parallel(r1, d1):
                    s1[r, c] = T.cast(C1[sh, chunk, r, c], "float32")
                for r, c in T.Parallel(r2, d2):
                    s2[r, c] = T.cast(C2[sh, chunk, r, c], "float32")
                T.sync_threads()
                for step in T.serial(CHUNK):
                    t = chunk * CHUNK + step
                    if t < length:
                        flat = seq * length + t
                        if not prior1:
                            advance1(s1, R1, D1, K1, K12, V1, flat, head)
                        if not prior2:
                            advance2(s2, R2, D2, K2, K22, V2, flat, head)
                        for c in T.Parallel(d1):
                            total = T.alloc_local((1,), "float32")
                            total[0] = 0.0
                            for r in T.serial(r1):
                                q = _factor(T, Q, Q2, flat, head, r, qr)
                                total[0] += q * s1[r, c]
                            first[c] = total[0]
                        T.sync_threads()
                        if softmax:
                            if T.get_thread_binding() == 0:
                                maximum = T.alloc_local((1,), "float32")
                                denominator = T.alloc_local((1,), "float32")
                                maximum[0], denominator[0] = -3.402823466e38, 0.0
                                for slot in T.serial(d1):
                                    maximum[0] = T.max(maximum[0], first[slot])
                                for slot in T.serial(d1):
                                    denominator[0] += T.exp(first[slot] - maximum[0])
                                normalizer[0], normalizer[1] = (
                                    maximum[0],
                                    denominator[0],
                                )
                            T.sync_threads()
                            for slot in T.Parallel(d1):
                                first[slot] = (
                                    T.exp(first[slot] - normalizer[0]) / normalizer[1]
                                )
                            T.sync_threads()
                        for c in T.Parallel(d2):
                            total = T.alloc_local((1,), "float32")
                            total[0] = 0.0
                            for r in T.serial(r2):
                                if softmax:
                                    q = first[r]
                                else:
                                    q = _factor(T, Q, Q2, flat, head, r, qr)
                                total[0] += q * s2[r, c]
                            second[c] = total[0]
                        T.sync_threads()
                        if softmax:
                            for c in T.Parallel(d2):
                                Output[flat, head, c] = second[c]
                        else:
                            for c in T.Parallel(d1):
                                Output[flat, head, c] = first[c] / (second[0] + epsilon)
                        if prior1:
                            advance1(s1, R1, D1, K1, K12, V1, flat, head)
                        if prior2:
                            advance2(s2, R2, D2, K2, K22, V2, flat, head)
                        T.sync_threads()

        return kernel

    return build()


def _bound_factors(factors):
    row, column, key, key2, value, kl, kr = factors
    return (
        (
            key if row is None else row,
            value if column is None else column,
            key,
            key if not kr else key2,
            value,
        ),
        (row is None, column is None),
        kr,
    )


def _region(
    first,
    second,
    query,
    length,
    prior_first,
    prior_second,
    output_dtype,
    *,
    softmax,
    epsilon=0.0,
):
    c1, f1, _ = first
    c2, f2, _ = second
    if length <= 0 or c1.shape[:2] != c2.shape[:2]:
        return None
    bh, chunks, r1, d1 = c1.shape
    _, _, r2, d2 = c2.shape
    if d1 != r2 if softmax else r1 != r2 or d2 != 1:
        return None
    if not readout_region_resources(c1.shape, c2.shape, softmax=softmax)["eligible"]:
        return None
    tokens, heads = f1[2].shape[:2]
    if (
        f2[2].shape[:2] != (tokens, heads)
        or tokens % length
        or bh != tokens // length * heads
        or chunks != (length + 63) // 64
    ):
        return None
    parts = _query_parts(query, tokens, heads, r1)
    if parts is None:
        return None
    q, q2, _, qr = parts
    bound1, unit1, kr1 = _bound_factors(f1)
    bound2, unit2, kr2 = _bound_factors(f2)
    output = torch.empty(
        tokens, heads, d2 if softmax else d1, device=c1.device, dtype=output_dtype
    )
    args = (c1, *bound1, c2, *bound2, q, q2, output)
    _compile_region(
        tuple(_metadata(t) for t in args),
        length,
        heads,
        kr1,
        kr2,
        qr,
        unit1,
        unit2,
        prior_first,
        prior_second,
        softmax,
        epsilon,
        c1.device.index,
    )(*args)
    return output


def fused_readout_region(
    first,
    second,
    query,
    *,
    sequence_length,
    prior_first=False,
    prior_second=False,
    output_dtype=torch.float32,
):
    return _region(
        first,
        second,
        query,
        sequence_length,
        prior_first,
        prior_second,
        output_dtype,
        softmax=True,
    )


def fused_normalized_region(
    first,
    second,
    query,
    *,
    sequence_length,
    prior_first=False,
    prior_second=False,
    output_dtype=torch.float32,
    epsilon=0.0,
):
    return _region(
        first,
        second,
        query,
        sequence_length,
        prior_first,
        prior_second,
        output_dtype,
        softmax=False,
        epsilon=epsilon,
    )


@lru_cache(maxsize=None)
def _compile_shared(
    metas, count, qrs, length, heads, kr, unit, priors, epilogue, device_index
):
    import tilelang
    import tilelang.language as T

    bh, chunks, rows, columns = metas[0][0]
    names = ["Carry", "Row", "Column", "Key", "Key2", "Value"]
    names += [name for i in range(count) for name in (f"Q{i}", f"Q2_{i}")]
    outputs = 1 if epilogue else count
    names += [f"O{i}" for i in range(outputs)]
    annotations = ", ".join(
        f"{name}: T.StridedTensor(*metas[{i}])" for i, name in enumerate(names)
    )
    body = []
    for i in range(count):
        body.append(
            f"                result{i} = T.alloc_shared(({columns},), 'float32')"
        )
    body.extend(
        [
            "                for step in T.serial(64):",
            "                    t = chunk * 64 + step",
            f"                    if t < {length}:",
            f"                        flat = seq * {length} + t",
        ]
    )
    for post in (False, True):
        if post:
            body.append(
                "                        advance(state, Row, Column, Key, Key2, Value, flat, head)"
            )
        for i in range(count):
            if (not priors[i]) != post:
                continue
            body.extend(
                [
                    f"                        for c in T.Parallel({columns}):",
                    "                            total = T.alloc_local((1,), 'float32')",
                    "                            total[0] = 0.0",
                    f"                            for r in T.serial({rows}):",
                    f"                                q = _factor(T, Q{i}, Q2_{i}, flat, head, r, {qrs[i]})",
                    "                                total[0] += q * state[r, c]",
                    f"                            result{i}[c] = total[0]",
                    "                        T.sync_threads()",
                ]
            )
    body.append(f"                        for c in T.Parallel({columns}):")
    if epilogue:
        symbol = {"add": "+", "multiply": "*", "divide": "/"}[epilogue]
        body.append(
            f"                            O0[flat, head, c] = result0[c] {symbol} result1[c]"
        )
    else:
        body.extend(
            f"                            O{i}[flat, head, c] = result{i}[c]"
            for i in range(count)
        )
    body.append("                        T.sync_threads()")
    source = (
        "@tilelang.jit(compile_flags=['-fmad=false'])\n"
        "def build():\n"
        "    @T.prim_func\n"
        f"    def kernel({annotations}):\n"
        f"        with T.Kernel({bh}, {chunks}, threads=128) as (sh, chunk):\n"
        f"                seq, head = sh // {heads}, sh % {heads}\n"
        f"                state = T.alloc_shared(({rows}, {columns}), 'float32')\n"
        f"                for r, c in T.Parallel({rows}, {columns}):\n"
        "                    state[r, c] = T.cast(Carry[sh, chunk, r, c], 'float32')\n"
        "                T.sync_threads()\n" + "\n".join(body) + "\n    return kernel\n"
    )
    filename = (
        f"<stateful-shared-readout-{hashlib.sha256(source.encode()).hexdigest()}>"
    )
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {
        "tilelang": tilelang,
        "T": T,
        "metas": metas,
        "_factor": _factor,
        "advance": _advance_macro(T, rows, columns, kr, *unit),
    }
    exec(compile(source, filename, "exec"), namespace)
    return namespace["build"]()


def fused_shared_readouts(
    prepared,
    queries,
    *,
    sequence_length,
    priors=None,
    output_dtypes=None,
    epilogue=None,
):
    """Replay once for 2--4 independent contractions or one binary epilogue."""
    carry, factors, _ = prepared
    bh, chunks, rows, columns = carry.shape
    count = len(queries)
    if (
        count < 2
        or count > 4
        or epilogue not in (None, "add", "multiply", "divide")
        or (epilogue and count != 2)
    ):
        return None
    br, bc = _next_power_of_two(rows), _next_power_of_two(columns)
    live = 2 * br * bc + 4 * (br + bc) + count * (br + 2 * bc)
    if max(br, bc) > 256 or live > 16384:
        return None
    tokens, heads = factors[2].shape[:2]
    if (
        sequence_length <= 0
        or tokens % sequence_length
        or bh != tokens // sequence_length * heads
        or chunks != (sequence_length + 63) // 64
    ):
        return None
    priors = (False,) * count if priors is None else tuple(priors)
    output_count = 1 if epilogue else count
    output_dtypes = (
        (torch.float32,) * output_count
        if output_dtypes is None
        else tuple(output_dtypes)
    )
    if len(priors) != count or len(output_dtypes) != output_count:
        return None
    parts = tuple(_query_parts(query, tokens, heads, rows) for query in queries)
    if any(part is None for part in parts):
        return None
    bound, unit, kr = _bound_factors(factors)
    outputs = tuple(
        torch.empty(tokens, heads, columns, device=carry.device, dtype=dtype)
        for dtype in output_dtypes
    )
    args = (carry, *bound, *(t for part in parts for t in part[:2]), *outputs)
    _compile_shared(
        tuple(_metadata(t) for t in args),
        count,
        tuple(part[3] for part in parts),
        sequence_length,
        heads,
        kr,
        unit,
        priors,
        epilogue,
        carry.device.index,
    )(*args)
    return outputs
