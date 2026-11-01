"""Native TileLang kernels for diagonal state summaries and replay.

This module is deliberately independent from :mod:`diagonal_native`.  It lowers
``DenseLoader`` records directly to raw storage loads; unlike the legacy native
path it never materializes an authored expression with Python/Torch indexing.
All arithmetic is authored as FP32, with an explicit BF16 round trip at every
DenseLoader BF16 node.

Replay retains one carry tile per independent chunk in row/column fragments.
State transitions remain token-ordered; only the Feature-axis contraction is
distributed across the CTA instead of serialized inside each column lane.
"""

import hashlib
import linecache
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import torch


_CHUNK = 64
_TILE = 32
_SUMMARY_KERNELS = {}
_REPLAY_KERNELS = {}


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _storage_size(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().nbytes() // tensor.element_size())


def _tensor_metadata(tensor: torch.Tensor) -> tuple:
    """Specialization metadata only; never retain a tensor or storage pointer."""
    return (
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        int(tensor.storage_offset()),
        _storage_size(tensor),
    )


def _loader_metadata(expressions, sources, maps, loaders) -> tuple:
    # Keep semantic use order, including repeated expressions and absent gates.
    recipes = tuple(
        None
        if expression is None
        else (loaders[id(expression)].cache_key, tuple(maps[id(expression)].items()))
        for expression in expressions
    )
    bindings = tuple(
        (_tensor_metadata(source.tensor), source.strides, source.group, source.offset)
        for source in sources
    )
    return recipes, bindings


def _raw_storage_view(tensor: torch.Tensor) -> torch.Tensor:
    """Make a one-dimensional view rooted at storage base.

    The generated offset includes ``storage_offset``.  Keeping this explicit is
    important for sliced authored inputs and avoids silently assuming contiguous
    or zero-offset sources.
    """
    return tensor.as_strided((_storage_size(tensor),), (1,), storage_offset=0)


def _expressions(items: Iterable[Any]) -> tuple[Any, ...]:
    result = []
    seen = set()

    def visit(item):
        if item is None:
            return
        if isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
            return
        key = id(item)
        if key not in seen:
            seen.add(key)
            result.append(item)

    visit(tuple(items))
    return tuple(result)


@dataclass(frozen=True)
class _RawSource:
    tensor: torch.Tensor
    strides: tuple[int, int, int, int]
    group: int
    offset: int


def _bind(expressions: Iterable[Any], inputs: dict[str, torch.Tensor], heads: int):
    """Bind every expression independently (feature axes may differ by use)."""
    source_specs: list[_RawSource] = []
    bound_sources = {}
    source_maps: dict[int, dict[str, int]] = {}
    loaders: dict[int, Any] = {}
    for expression in _expressions(expressions):
        if not hasattr(expression, "loader") or not hasattr(expression, "bindings"):
            raise TypeError("native diagonal kernels require DenseExpression values")
        loader = expression.loader
        source_maps[id(expression)] = {}
        loaders[id(expression)] = loader
        for binding in expression.bindings:
            # Reuse identical axis recipes within this call only. The same
            # input can still have distinct row/column feature bindings.
            slot = bound_sources.get(binding)
            if slot is None:
                tensor, strides, group = binding.bind(inputs[binding.name], heads)
                slot = len(source_specs)
                source_specs.append(
                    _RawSource(
                        tensor, strides, int(group), int(tensor.storage_offset())
                    )
                )
                bound_sources[binding] = slot
            source_maps[id(expression)][binding.name] = slot
    return tuple(source_specs), source_maps, loaders


class RawLoaderContext:
    """Reusable raw-binding/code-generation context for native TileLang kernels."""

    def __init__(
        self,
        expressions: Iterable[Any],
        inputs: dict[str, torch.Tensor],
        heads: int,
        sequence_length: int,
    ):
        self.sequence_length = int(sequence_length)
        self.sources, self.maps, self.loaders = _bind(expressions, inputs, int(heads))
        self.pointers = tuple(
            _raw_storage_view(source.tensor) for source in self.sources
        )
        self._annotations = None

    @property
    def annotations(self):
        # Cache hits need current pointers/metadata, never generated source.
        if self._annotations is None:
            self._annotations = tuple(
                f"X{i}: T.Tensor(({_storage_size(source.tensor)},), {_dtype_name(source.tensor.dtype)!r})"
                for i, source in enumerate(self.sources)
            )
        return self._annotations

    def emit_value(
        self,
        expression: Any,
        token: str = "token",
        head: str = "head",
        feature: str = "feature",
        *,
        local_values=None,
    ):
        return emit_dense_loader(
            self.loaders[id(expression)],
            self.maps[id(expression)],
            self.sources,
            token=token,
            head=head,
            feature=feature,
            sequence_length=self.sequence_length,
            local_values=local_values,
        )

    def specialization_key(self, expressions):
        return _loader_metadata(expressions, self.sources, self.maps, self.loaders)


def _round_bf16(value: str) -> str:
    return f"T.cast(T.cast({value}, 'bfloat16'), 'float32')"


def _constant(value: Any, dtype: Any) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if dtype == torch.bfloat16:
        value = torch.tensor(value, dtype=torch.bfloat16).item()
        return _round_bf16(repr(value))
    return f"T.cast({repr(value)}, 'float32')"


def emit_dense_loader(
    loader: Any,
    source_map: dict[str, int],
    sources: tuple[_RawSource, ...],
    *,
    token: str = "token",
    head: str = "head",
    feature: str = "feature",
    sequence_length: int,
    local_values=None,
) -> str:
    """Return a TileLang expression evaluating one ``DenseLoader``.

    ``sources`` are raw storage pointers and ``source_map`` is expression-local,
    allowing one authored tensor to be used with distinct row/column feature
    bindings without losing its stride metadata.
    """
    values: list[str] = []
    for node in loader.nodes:
        op = node[0]
        if op == "local":
            values.append(local_values[node[1]])
        elif op == "input":
            slot = source_map[node[1]]
            source = sources[slot]
            s0, s1, s2, s3 = source.strides
            offset = (
                f"{source.offset} + ({token} // {sequence_length}) * {s0} + "
                f"({head} // {source.group}) * {s1} + ({token} % {sequence_length}) * {s2} + "
                f"{feature} * {s3}"
            )
            values.append(f"T.cast(X{slot}[{offset}], 'float32')")
        elif op == "constant":
            values.append(_constant(node[1], node[2]))
        elif op == "cast":
            value = values[node[1]]
            dtype = str(node[2]).removeprefix("torch.")
            values.append(f"T.cast(T.cast({value}, {dtype!r}), 'float32')")
        elif op == "binary":
            left, right, dtype, symbol = node[1], node[2], node[3], node[4]
            lhs, rhs = values[left], values[right]
            value = f"({lhs} {symbol} {rhs})"
            values.append(_round_bf16(value) if dtype == torch.bfloat16 else value)
        elif op == "unary":
            operand, kind = values[node[1]], node[2]
            if kind == "neg":
                value = f"(-{operand})"
            elif kind == "abs":
                value = f"T.abs({operand})"
            elif kind in ("exp", "log", "sqrt"):
                value = f"T.{kind}({operand})"
            elif kind == "sigmoid":
                value = f"(1.0 / (1.0 + T.exp(-{operand})))"
            elif kind == "reciprocal":
                value = f"(1.0 / {operand})"
            elif kind == "clamp":
                value = f"T.max(T.min({operand}, {repr(node[4])}), {repr(node[3])})"
            else:
                raise ValueError(f"unsupported DenseLoader unary node {kind!r}")
            values.append(value)
        elif op == "factor":
            values.append(f"({values[node[1]]} * {values[node[2]]})")
        else:
            raise ValueError(f"unsupported DenseLoader node {op!r}")
    return values[loader.root]


def _annotation_sources(sources: tuple[_RawSource, ...]) -> str:
    return ", ".join(
        f"X{i}: T.Tensor(({_storage_size(source.tensor)},), {_dtype_name(source.tensor.dtype)!r})"
        for i, source in enumerate(sources)
    )


@lru_cache(maxsize=None)
def _kernel_source(body: str, annotations: str, name: str, device: torch.device) -> Any:
    import tilelang
    import tilelang.language as T

    text = (
        "@tilelang.jit(compile_flags=['-fmad=false'])\n"
        "def build():\n"
        "    @T.prim_func\n"
        f"    def {name}({annotations}):\n"
        f"{body}\n"
        "    return " + name + "\n"
    )
    filename = f"<native-diagonal-{hashlib.sha256(text.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(text), None, text.splitlines(True), filename)
    namespace = {"tilelang": tilelang, "T": T}
    exec(compile(text, filename, "exec"), namespace)
    return namespace["build"]()


def _expr_ref(
    expression: Any,
    sources,
    maps,
    loaders,
    *,
    token: str,
    head: str,
    feature: str,
    length: int,
):
    if expression is None:
        return "1.0"
    return emit_dense_loader(
        loaders[id(expression)],
        maps[id(expression)],
        sources,
        token=token,
        head=head,
        feature=feature,
        sequence_length=length,
    )


def _summary_kernel(
    sources,
    maps,
    loaders,
    row_gate,
    column_gate,
    injections,
    *,
    batch,
    length,
    heads,
    rows,
    columns,
    scale,
    bias,
):

    rg = _expr_ref(
        row_gate,
        sources,
        maps,
        loaders,
        token="flat",
        head="head",
        feature="r",
        length=length,
    )
    cg = _expr_ref(
        column_gate,
        sources,
        maps,
        loaders,
        token="flat",
        head="head",
        feature="c",
        length=length,
    )
    terms = [
        " + ".join(
            f"({_expr_ref(key, sources, maps, loaders, token='flat', head='head', feature='r', length=length)} * "
            f"{_expr_ref(value, sources, maps, loaders, token='flat', head='head', feature='c', length=length)})"
            for key, value in injections
        )
        or "0.0"
    ][0]
    body = f"""        with T.Kernel({batch * heads * ((length + _CHUNK - 1) // _CHUNK)}, T.ceildiv({rows * columns}, {_TILE}), threads={_TILE}) as (shc, tile):
            tid = T.get_thread_binding()
            element = tile * {_TILE} + tid
            seq_head = shc // {((length + _CHUNK - 1) // _CHUNK)}
            chunk = shc % {((length + _CHUNK - 1) // _CHUNK)}
            seq = seq_head // {heads}
            head = seq_head % {heads}
            r = element // {columns}
            c = element % {columns}
            a = T.alloc_local((1,), 'float32')
            d = T.alloc_local((1,), 'float32')
            b = T.alloc_local((1,), 'float32')
            a[0] = 1.0
            d[0] = 1.0
            b[0] = 0.0
            for step in T.serial({_CHUNK}):
                t = chunk * {_CHUNK} + step
                flat = seq * {length} + t
                valid = (t < {length}) & (r < {rows}) & (c < {columns})
                if valid:
                    gr = {rg}
                    gc = {cg}
                    a[0] = a[0] * gr
                    d[0] = d[0] * gc
                    b[0] = b[0] * gr * gc + {terms}
            if element < {rows * columns}:
                Scale[seq_head, chunk, r, c] = a[0] * d[0]
                Bias[seq_head, chunk, r, c] = b[0]"""
    annotations = _annotation_sources(sources)
    if annotations:
        annotations += ", "
    annotations += f"Scale: T.StridedTensor({tuple(scale.shape)}, {tuple(scale.stride())}, 'float32'), "
    annotations += (
        f"Bias: T.StridedTensor({tuple(bias.shape)}, {tuple(bias.stride())}, 'float32')"
    )
    return _kernel_source(body, annotations, "summary", bias.device)


def launch_native_summary(
    row_gate: Any,
    column_gate: Any,
    injections: Iterable[tuple[Any, Any]],
    inputs: dict[str, torch.Tensor],
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    sequence_length: int,
    heads: int,
) -> None:
    """Evaluate gates and every injection directly into chunk summaries."""
    batch_heads, chunks, rows, columns = (int(x) for x in bias.shape)
    if batch_heads % heads:
        raise ValueError("summary batch/head extent is not divisible by heads")
    injections = tuple(injections)
    expressions = _expressions((row_gate, column_gate, injections))
    sources, maps, loaders = _bind(expressions, inputs, heads)
    key = (
        _loader_metadata(
            (
                row_gate,
                column_gate,
                *(expression for pair in injections for expression in pair),
            ),
            sources,
            maps,
            loaders,
        ),
        batch_heads // heads,
        heads,
        sequence_length,
        chunks,
        rows,
        columns,
        _CHUNK,
        _TILE,
        _tensor_metadata(scale),
        _tensor_metadata(bias),
    )
    kernel = _SUMMARY_KERNELS.get(key)
    if kernel is None:
        kernel = _summary_kernel(
            sources,
            maps,
            loaders,
            row_gate,
            column_gate,
            injections,
            batch=batch_heads // heads,
            length=sequence_length,
            heads=heads,
            rows=rows,
            columns=columns,
            scale=scale,
            bias=bias,
        )
        _SUMMARY_KERNELS[key] = kernel
    kernel(*(_raw_storage_view(source.tensor) for source in sources), scale, bias)


def launch_native_carry(
    scale, bias, initial, carry, final=None, *, identity=False, write_final=False
):
    """Run the existing native hierarchical affine carry without host indexing."""
    from .diagonal_scan import hierarchical_carry

    return hierarchical_carry(
        scale, bias, initial, carry, final, identity=identity, write_final=write_final
    )


def _replay_kernel(
    sources,
    maps,
    loaders,
    row_gate,
    column_gate,
    injections,
    query_expression,
    *,
    batch,
    length,
    heads,
    rows,
    columns,
    prior,
    carry,
    query,
    output,
    final,
):

    rg = _expr_ref(
        row_gate,
        sources,
        maps,
        loaders,
        token="flat",
        head="head",
        feature="r",
        length=length,
    )
    cg = _expr_ref(
        column_gate,
        sources,
        maps,
        loaders,
        token="flat",
        head="head",
        feature="c",
        length=length,
    )
    terms = (
        " + ".join(
            f"({_expr_ref(key, sources, maps, loaders, token='flat', head='head', feature='r', length=length)} * "
            f"{_expr_ref(value, sources, maps, loaders, token='flat', head='head', feature='c', length=length)})"
            for key, value in injections
        )
        or "0.0"
    )
    if query_expression is None:
        query_load = "T.cast(Query[flat, head, r], 'float32')"
    else:
        query_load = _expr_ref(
            query_expression,
            sources,
            maps,
            loaders,
            token="flat",
            head="head",
            feature="r",
            length=length,
        )
    chunks = (length + _CHUNK - 1) // _CHUNK
    padded = 1 << (rows - 1).bit_length()
    body = f"""        with T.Kernel({batch * heads * chunks}, T.ceildiv({columns}, {_TILE}), threads=128) as (shc, block):
            seq_head = shc // {chunks}
            chunk = shc % {chunks}
            seq = seq_head // {heads}
            head = seq_head % {heads}
            state = T.alloc_fragment(({padded}, {_TILE}), 'float32')
            products = T.alloc_fragment(({padded}, {_TILE}), 'float32')
            result = T.alloc_fragment(({_TILE},), 'float32')
            for r, lane in T.Parallel({padded}, {_TILE}):
                c = block * {_TILE} + lane
                state[r, lane] = T.if_then_else((r < {rows}) & (c < {columns}), T.cast(Carry[seq_head, chunk, r, c], 'float32'), 0.0)
            for step in T.serial({_CHUNK}):
                token = chunk * {_CHUNK} + step
                if token < {length}:
                    flat = seq * {length} + token
                    for r, lane in T.Parallel({padded}, {_TILE}):
                        c = block * {_TILE} + lane
                        products[r, lane] = 0.0
                        if (r < {rows}) & (c < {columns}):
                            if not {bool(prior)}:
                                state[r, lane] = state[r, lane] * ({rg}) * ({cg}) + ({terms})
                            products[r, lane] = ({query_load}) * state[r, lane]
                            if {bool(prior)}:
                                state[r, lane] = state[r, lane] * ({rg}) * ({cg}) + ({terms})
                    T.reduce_sum(products, result, dim=0)
                    for lane in T.Parallel({_TILE}):
                        c = block * {_TILE} + lane
                        if c < {columns}:
                            Output[flat, head, c] = result[lane]"""
    if final is not None:
        body += f"""
            for r, lane in T.Parallel({padded}, {_TILE}):
                c = block * {_TILE} + lane
                if (r < {rows}) & (c < {columns}):
                    Final[seq, head, r, c] = state[r, lane]"""
    annotations = _annotation_sources(sources)
    if annotations:
        annotations += ", "
    if query_expression is None:
        annotations += f"Query: T.StridedTensor({tuple(query.shape)}, {tuple(query.stride())}, {_dtype_name(query.dtype)!r}), "
    annotations += f"Carry: T.StridedTensor({tuple(carry.shape)}, {tuple(carry.stride())}, 'float32'), "
    annotations += f"Output: T.StridedTensor({tuple(output.shape)}, {tuple(output.stride())}, {_dtype_name(output.dtype)!r})"
    if final is not None:
        annotations += f", Final: T.StridedTensor({tuple(final.shape)}, {tuple(final.stride())}, 'float32')"
    return _kernel_source(body, annotations, "replay", carry.device)


def launch_native_replay(
    carry: torch.Tensor,
    row_gate: Any,
    column_gate: Any,
    injections: Iterable[tuple[Any, Any]],
    query: torch.Tensor | Any,
    inputs: dict[str, torch.Tensor],
    output: torch.Tensor,
    *,
    sequence_length: int,
    heads: int,
    prior: bool = False,
    final: torch.Tensor | None = None,
) -> None:
    """Replay a carry with raw DenseLoader factors and an explicit/query loader.

    ``query`` may be an arbitrary-stride ``(tokens, heads, rows)`` tensor (the
    nonfused cut) or a DenseExpression, in which case it is loaded with the same
    raw binding machinery as gates and injections.
    """
    batch_heads, chunks, rows, columns = (int(x) for x in carry.shape)
    if batch_heads % heads:
        raise ValueError("carry batch/head extent is not divisible by heads")
    if final is not None and (
        chunks != 1 or final.shape != (batch_heads // heads, heads, rows, columns)
    ):
        raise ValueError(
            "direct replay final requires one chunk and a matching rank-two State"
        )
    query_expression = query if hasattr(query, "loader") else None
    injections = tuple(injections)
    expressions = _expressions((row_gate, column_gate, injections, query_expression))
    sources, maps, loaders = _bind(expressions, inputs, heads)
    query_tensor = None if query_expression is not None else query
    key = (
        _loader_metadata(
            (
                row_gate,
                column_gate,
                *(expression for pair in injections for expression in pair),
                query_expression,
            ),
            sources,
            maps,
            loaders,
        ),
        batch_heads // heads,
        heads,
        sequence_length,
        chunks,
        rows,
        columns,
        _CHUNK,
        _TILE,
        bool(prior),
        _tensor_metadata(carry),
        _tensor_metadata(output),
        _tensor_metadata(query_tensor) if query_expression is None else None,
        _tensor_metadata(final) if final is not None else None,
    )
    kernel = _REPLAY_KERNELS.get(key)
    if kernel is None:
        kernel = _replay_kernel(
            sources,
            maps,
            loaders,
            row_gate,
            column_gate,
            injections,
            query_expression,
            batch=batch_heads // heads,
            length=sequence_length,
            heads=heads,
            rows=rows,
            columns=columns,
            prior=prior,
            carry=carry,
            query=query_tensor,
            output=output,
            final=final,
        )
        _REPLAY_KERNELS[key] = kernel
    args = [*(_raw_storage_view(source.tensor) for source in sources)]
    if query_expression is None:
        args.append(query_tensor)
    args.extend((carry, output))
    if final is not None:
        args.append(final)
    kernel(*args)


__all__ = [
    "emit_dense_loader",
    "RawLoaderContext",
    "launch_native_summary",
    "launch_native_carry",
    "launch_native_replay",
]
