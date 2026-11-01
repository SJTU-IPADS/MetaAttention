"""Raw-loader ProductFeature summaries and value-tiled normalized readouts.

Only chunk carries retain the expanded Feature axis. Local scores use the
existing two-base-GEMM factorization; no quadratic sequence workspace is built.
"""

import hashlib
import linecache
from functools import lru_cache

from ..diagonal.identity import fp32_gemm_macro
from ..diagonal.native_tile_kernels import RawLoaderContext

_SUMMARY_KERNELS = {}
_READOUT_KERNELS = {}


def _key(recipe, context, expressions, tensors, length, heads):
    return (
        recipe,
        context.specialization_key(expressions),
        length,
        heads,
        tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in tensors),
    )


def _annotation(name, tensor):
    dtype = str(tensor.dtype).removeprefix("torch.")
    return f"{name}: T.StridedTensor({tuple(tensor.shape)}, {tuple(tensor.stride())}, {dtype!r})"


@lru_cache(maxsize=None)
def _compile(body, annotations, device):
    import tilelang
    import tilelang.language as T

    text = (
        "@tilelang.jit(compile_flags=['-fmad=false'])\n"
        "def build():\n"
        "    @T.prim_func\n"
        f"    def kernel({annotations}):\n{body}\n"
        "    return kernel\n"
    )
    filename = f"<native-product-{hashlib.sha256(text.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(text), None, text.splitlines(True), filename)
    namespace = {"tilelang": tilelang, "T": T, "gemm": fp32_gemm_macro(T)}
    exec(compile(text, filename, "exec"), namespace)
    return namespace["build"]()


def _factor(context, expressions, right, token, feature):
    if not expressions:
        return "1.0"
    if len(expressions) == 1:
        return context.emit_value(expressions[0], token, "head", feature)
    left = context.emit_value(expressions[0], token, "head", f"({feature}) // {right}")
    rhs = context.emit_value(expressions[1], token, "head", f"({feature}) % {right}")
    return f"({left} * {rhs})"


def launch_product_summary(recipe, inputs, summaries, *, sequence_length, heads):
    expressions = tuple(e for p in recipe.pairs for e in (*p.key, p.value))
    context = RawLoaderContext(expressions, inputs, heads, sequence_length)
    chunks = (sequence_length + 63) // 64
    bh = summaries[0][0].shape[0]
    arguments = tuple(t for pair in summaries for t in pair)
    cache_key = _key(recipe, context, expressions, arguments, sequence_length, heads)
    kernel = _SUMMARY_KERNELS.get(cache_key)
    if kernel is not None:
        kernel(*context.pointers, *arguments)
        return
    annotations = list(context.annotations)
    for i, (num, den) in enumerate(summaries):
        annotations.extend((_annotation(f"N{i}", num), _annotation(f"D{i}", den)))
    sections, offset = [], 0
    for i, pair in enumerate(recipe.pairs):
        for denominator in (False, True):
            columns = 1 if denominator else recipe.columns
            tiles = ((pair.rows + 31) // 32) * ((columns + 31) // 32)
            key = _factor(context, pair.key, pair.right, "flat", "row")
            value = (
                "1.0"
                if denominator
                else context.emit_value(pair.value, "flat", "head", "column")
            )
            name = f"D{i}" if denominator else f"N{i}"
            tile_columns = (columns + 31) // 32
            tile_row = f"((tile - {offset}) // {tile_columns})"
            tile_column = f"((tile - {offset}) % {tile_columns})"
            sections.append(f"""            if (tile >= {offset}) & (tile < {offset + tiles}):
                for r, t in T.Parallel(32, 64):
                    row = {tile_row} * 32 + r
                    token = chunk * 64 + t
                    flat = sequence * {sequence_length} + token
                    k[r, t] = T.if_then_else((row < {pair.rows}) & (token < {sequence_length}), {key}, 0.0)
                for t, c in T.Parallel(64, 32):
                    column = {tile_column} * 32 + c
                    token = chunk * 64 + t
                    flat = sequence * {sequence_length} + token
                    v[t, c] = T.if_then_else((column < {columns}) & (token < {sequence_length}), {value}, 0.0)
                T.clear(result)
                gemm(k, v, result)
                for r, c in T.Parallel(32, 32):
                    if ({tile_row} * 32 + r < {pair.rows}) & ({tile_column} * 32 + c < {columns}):
                        {name}[sh, chunk, {tile_row} * 32 + r, {tile_column} * 32 + c] = result[r, c]""")
            offset += tiles
    body = f"""        with T.Kernel({bh}, {chunks}, {offset}, threads=128) as (sh, chunk, tile):
            sequence = sh // {heads}
            head = sh % {heads}
            k = T.alloc_shared((32, 64), 'float32')
            v = T.alloc_shared((64, 32), 'float32')
            result = T.alloc_fragment((32, 32), 'float32')
""" + "\n".join(sections)
    kernel = _compile(body, ", ".join(annotations), arguments[0].device)
    _SUMMARY_KERNELS[cache_key] = kernel
    kernel(*context.pointers, *arguments)


def launch_product_normalized(
    recipe, inputs, carries, output, *, sequence_length, heads
):
    from ....stateful_operator import Add, Constant, Multiply, StateTerm

    expressions = tuple(e for p in recipe.pairs for e in (*p.key, *p.query, p.value))
    context = RawLoaderContext(expressions, inputs, heads, sequence_length)
    bh, chunks = carries[0][0].shape[:2]
    arguments = tuple(t for pair in carries for t in pair)
    cache_key = _key(
        recipe, context, expressions, (*arguments, output), sequence_length, heads
    )
    kernel = _READOUT_KERNELS.get(cache_key)
    if kernel is not None:
        kernel(*context.pointers, *arguments, output)
        return
    annotations = list(context.annotations)
    for i, (num, den) in enumerate(carries):
        annotations.extend((_annotation(f"N{i}", num), _annotation(f"D{i}", den)))
    annotations.append(_annotation("Output", output))
    declarations, sections, term_names = [], [], {}
    for i, (pair, terms) in enumerate(zip(recipe.pairs, recipe.terms)):
        declarations.append(f"            p{i} = T.alloc_fragment((64, 32), 'float32')")
        declarations.append(
            f"            result{i} = T.alloc_shared((64, 32), 'float32')"
        )
        term_names[terms[0]] = f"result{i}[t, c]"
        term_names[terms[1]] = f"result{i}[t, 16]"
        query = _factor(context, pair.query, pair.right, "flat", "row")
        sections.append(f"""            T.clear(p{i})
            for rb in T.serial(T.ceildiv({pair.rows}, 32)):
                for t, r in T.Parallel(64, 32):
                    token = chunk * 64 + t
                    flat = sequence * {sequence_length} + token
                    row = rb * 32 + r
                    q[t, r] = T.if_then_else((token < {sequence_length}) & (row < {pair.rows}), {query}, 0.0)
                for r, c in T.Parallel(32, 32):
                    row = rb * 32 + r
                    column = column_block * 16 + c
                    state[r, c] = 0.0
                    if row < {pair.rows}:
                        if (c < 16) & (column < {recipe.columns}):
                            state[r, c] = N{i}[sh, chunk, row, column]
                        if c == 16:
                            state[r, c] = D{i}[sh, chunk, row, 0]
                gemm(q, state, p{i})
            T.clear(score)
            T.clear(other)""")
        for factor_index, (key, query_part) in enumerate(zip(pair.key, pair.query)):
            dimension = (
                pair.rows
                if len(pair.key) == 1
                else (pair.rows // pair.right if factor_index == 0 else pair.right)
            )
            qexpr = context.emit_value(query_part, "flat", "head", "feature")
            kexpr = context.emit_value(key, "flat", "head", "feature")
            accumulator = "score" if factor_index == 0 else "other"
            sections.append(f"""            for fb in T.serial(T.ceildiv({dimension}, 32)):
                for t, r in T.Parallel(64, 32):
                    token = chunk * 64 + t
                    flat = sequence * {sequence_length} + token
                    feature = fb * 32 + r
                    q[t, r] = T.if_then_else((token < {sequence_length}) & (feature < {dimension}), {qexpr}, 0.0)
                for r, t in T.Parallel(32, 64):
                    token = chunk * 64 + t
                    flat = sequence * {sequence_length} + token
                    feature = fb * 32 + r
                    k[r, t] = T.if_then_else((token < {sequence_length}) & (feature < {dimension}), {kexpr}, 0.0)
                gemm(q, k, {accumulator})""")
        local_score = (
            "1.0"
            if not pair.key
            else "score[t, s]"
            if len(pair.key) == 1
            else "score[t, s] * other[t, s]"
        )
        causal = "s < t" if pair.prior else "s <= t"
        value = context.emit_value(pair.value, "flat", "head", "column")
        sections.append(f"""            for t, s in T.Parallel(64, 64):
                score_shared[t, s] = T.if_then_else((chunk * 64 + t < {sequence_length}) & (chunk * 64 + s < {sequence_length}) & ({causal}), {local_score}, 0.0)
            for t, c in T.Parallel(64, 32):
                token = chunk * 64 + t
                flat = sequence * {sequence_length} + token
                column = column_block * 16 + c
                v[t, c] = 0.0
                if token < {sequence_length}:
                    if (c < 16) & (column < {recipe.columns}):
                        v[t, c] = {value}
                    if c == 16:
                        v[t, c] = 1.0
            gemm(score_shared, v, p{i})
            T.copy(p{i}, result{i})""")

    def emit(node):
        if isinstance(node, StateTerm):
            return term_names[node]
        if isinstance(node, Constant):
            return f"T.cast({node.value!r}, 'float32')"
        symbol = (
            "+" if isinstance(node, Add) else "*" if isinstance(node, Multiply) else "/"
        )
        return f"({emit(node.left)} {symbol} {emit(node.right)})"

    body = (
        f"""        with T.Kernel({bh}, {chunks}, T.ceildiv({recipe.columns}, 16), threads=128) as (sh, chunk, column_block):
            sequence = sh // {heads}
            head = sh % {heads}
            q = T.alloc_shared((64, 32), 'float32')
            k = T.alloc_shared((32, 64), 'float32')
            state = T.alloc_shared((32, 32), 'float32')
            v = T.alloc_shared((64, 32), 'float32')
            score_shared = T.alloc_shared((64, 64), 'float32')
            score = T.alloc_fragment((64, 64), 'float32')
            other = T.alloc_fragment((64, 64), 'float32')
"""
        + "\n".join(declarations + sections)
        + f"""
            for t, c in T.Parallel(64, 16):
                token = chunk * 64 + t
                column = column_block * 16 + c
                if (token < {sequence_length}) & (column < {recipe.columns}):
                    Output[sequence * {sequence_length} + token, head, column] = {emit(recipe.expression)}"""
    )
    kernel = _compile(body, ", ".join(annotations), output.device)
    _READOUT_KERNELS[cache_key] = kernel
    kernel(*context.pointers, *arguments, output)
