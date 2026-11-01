"""TileLang lowering for pointwise Stateful expression DAGs.

Authored arithmetic rounds at each node. Raw storage views retain descriptor
strides, grouped heads, and offsets without materializing expanded inputs.
"""

from __future__ import annotations

import hashlib
import linecache
from typing import Any, Mapping

import torch

from ....stateful_operator import Batch, HeadRole, Sequence as SequenceRole
from ...runtime.bindings import _PACKED_TOKEN

_CACHE: dict[tuple[Any, ...], Any] = {}


def _descriptor(source: Any) -> bool:
    return hasattr(source, "source_roles") and hasattr(source, "source_strides")


def _tensor(source):
    return source.tensor if _descriptor(source) else source


def _logical_shape(source: Any) -> tuple[int, ...]:
    if not _descriptor(source):
        return tuple(source.shape)
    roles = tuple(source.source_roles)
    extents = tuple(
        int(source.source_shape[roles.index(role)]) if role in roles else 1
        for role in source.feature_roles
    )
    return (int(source.token_count), int(source.target_heads), *extents)


def _storage_view(source):
    tensor = _tensor(source)
    # The address formula includes the storage offset, so the pointer starts at
    # the storage base rather than at the first element of a sliced tensor.
    size = tensor.untyped_storage().nbytes() // tensor.element_size()
    return torch.as_strided(tensor, (size,), (1,), storage_offset=0)


def _source_key(source):
    tensor = _tensor(source)
    raw = (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.dtype,
        tensor.untyped_storage().nbytes() // tensor.element_size(),
    )
    if not _descriptor(source):
        return ("tensor", *raw)
    return (
        "descriptor",
        *raw,
        tuple(source.source_roles),
        tuple(source.feature_roles),
        tuple(source.source_shape),
        tuple(source.source_strides),
        source.storage_offset,
        source.target_heads,
        source.head_group_size,
        source.mode,
        source.sequence_length,
        source.token_count,
    )


def _coordinates(shape):
    remainder = "p"
    coordinates = []
    for extent in reversed(shape):
        coordinates.append(f"(({remainder}) % {max(1, extent)})")
        remainder = f"(({remainder}) // {max(1, extent)})"
    return tuple(reversed(coordinates))


def _address(source, shape):
    logical = _logical_shape(source)
    coords = _coordinates(shape)[len(shape) - len(logical) :]
    coords = tuple(
        "0" if extent == 1 else coordinate
        for extent, coordinate in zip(logical, coords, strict=True)
    )
    tensor = _tensor(source)
    if not _descriptor(source):
        return " + ".join(
            (
                str(tensor.storage_offset()),
                *(
                    f"({coordinate}) * {stride}"
                    for coordinate, stride in zip(coords, tensor.stride(), strict=True)
                ),
            )
        )
    token, head, *features = coords
    length = max(1, int(source.sequence_length or 1))
    feature_indices = dict(zip(source.feature_roles, features, strict=True))
    terms = [str(source.storage_offset)]
    for role, extent, stride in zip(
        source.source_roles, source.source_shape, source.source_strides, strict=True
    ):
        if extent == 1:
            index = "0"
        elif role is _PACKED_TOKEN:
            index = token
        elif role is Batch:
            index = f"({token}) // {length}"
        elif role is SequenceRole:
            index = f"({token}) % {length}"
        elif isinstance(role, HeadRole):
            index = f"({head}) // {source.head_group_size}"
        else:
            index = feature_indices.get(role, "0")
        terms.append(f"({index}) * {stride}")
    return " + ".join(terms)


def _node_types(compiled, leaves):
    types = []
    scalars = []
    for node in compiled._nodes:
        if node.kind in ("input", "state"):
            source = leaves[node.expression]
            dtype = torch.float32 if node.kind == "state" else _tensor(source).dtype
            scalar = not _logical_shape(source)
        elif node.kind == "constant":
            dtype, scalar = node.expression.dtype, True
        elif node.kind in ("add", "multiply", "divide"):
            left, right = node.children
            scalar = scalars[left] and scalars[right]
            dtype = (
                types[right]
                if scalars[left] and not scalars[right]
                else types[left]
                if scalars[right] and not scalars[left]
                else torch.promote_types(types[left], types[right])
            )
        elif node.kind == "cast":
            dtype, scalar = node.expression.dtype, scalars[node.children[0]]
        elif node.kind == "negate":
            dtype, scalar = types[node.children[0]], scalars[node.children[0]]
        else:
            dtype, scalar = torch.float32, scalars[node.children[0]]
        types.append(dtype)
        scalars.append(scalar)
    return tuple(types)


def _cast(value, dtype):
    name = str(dtype).removeprefix("torch.")
    return f"T.cast(T.cast({value}, {name!r}), 'float32')"


def _make_kernel(compiled, sources, shape, node_types, output_dtypes, block):
    import tilelang
    import tilelang.language as T

    total = 1
    for extent in shape:
        total *= extent
    annotations = []
    for index, source in enumerate(sources):
        tensor = _tensor(source)
        size = tensor.untyped_storage().nbytes() // tensor.element_size()
        annotations.append(
            f"X{index}: T.Tensor(({size},), {str(tensor.dtype).removeprefix('torch.')!r})"
        )
    annotations.extend(
        f"O{index}: T.Tensor(({total},), {str(dtype).removeprefix('torch.')!r})"
        for index, dtype in enumerate(output_dtypes)
    )
    lines = []
    leaf = 0
    for index, node in enumerate(compiled._nodes):
        kind = node.kind
        if kind in ("input", "state"):
            value = f"T.cast(X{leaf}[{_address(sources[leaf], shape)}], 'float32')"
            leaf += 1
        elif kind == "constant":
            value = repr(
                torch.tensor(node.expression.value, dtype=node.expression.dtype).item()
            )
        elif kind in ("add", "multiply", "divide"):
            left, right = (f"v{child}" for child in node.children)
            # Torch wrapped scalar operands convert to the vector computation
            # dtype before arithmetic, including BF16 scalar constants.
            operands = []
            for child, operand in zip(node.children, (left, right), strict=True):
                expression = compiled._nodes[child].expression
                if compiled._nodes[child].kind == "constant":
                    # TVM folds nested constant casts through BF16. Convert the
                    # authored scalar before embedding it in device source.
                    operand = repr(
                        torch.tensor(expression.value, dtype=expression.dtype)
                        .to(node_types[index])
                        .item()
                    )
                operands.append(_cast(operand, node_types[index]))
            left, right = operands
            symbol = {"add": "+", "multiply": "*", "divide": "/"}[kind]
            value = f"({left} {symbol} {right})"
        else:
            operand = f"v{node.children[0]}"
            if kind == "cast":
                value = operand
            elif kind == "abs":
                value = f"T.abs({operand})"
            elif kind == "negate":
                value = f"(-{operand})"
            elif kind in ("exp", "log", "sqrt"):
                value = f"T.{kind}({operand})"
            elif kind == "sigmoid":
                value = f"(1.0 / (1.0 + T.exp(-{operand})))"
            elif kind == "reciprocal":
                value = f"(1.0 / {operand})"
            elif kind == "clamp":
                value = f"T.max(T.min({operand}, {node.expression.maximum!r}), {node.expression.minimum!r})"
            else:
                raise RuntimeError(f"unsupported CUDA expression node {kind}")
        lines.append(
            f"                    v{index} = {_cast(value, node_types[index])}"
        )
    lines.extend(
        f"                    O{index}[p] = v{root}"
        for index, root in enumerate(compiled._roots)
    )
    text = (
        "@tilelang.jit(compile_flags=['-fmad=false'])\n"
        "def build():\n"
        "    @T.prim_func\n"
        f"    def fused({', '.join(annotations)}):\n"
        f"        with T.Kernel({(total + block - 1) // block}, threads={block}) as bx:\n"
        f"            for lane in T.Parallel({block}):\n"
        f"                p = bx * {block} + lane\n"
        f"                if p < {total}:\n" + "\n".join(lines) + "\n    return fused\n"
    )
    filename = f"<stateful-expression-{hashlib.sha256(text.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(text), None, text.splitlines(True), filename)
    namespace = {"tilelang": tilelang, "T": T}
    exec(compile(text, filename, "exec"), namespace)
    return namespace["build"]()


def launch_expression(
    compiled: Any,
    bindings: Mapping[str, Any],
    states: Mapping[Any, Any],
    output_dtype: torch.dtype | None = None,
) -> Any:
    """Execute one TileLang kernel, or return None for CPU-only evaluation.

    Reduction/sequence cuts belong to the enclosing Stateful lowering and must
    not silently route a CUDA graph to the reference evaluator.
    """
    if any(
        node.kind in ("outer", "contract", "shift", "softmax")
        for node in compiled._nodes
    ):
        raise RuntimeError(
            "CUDA expression DAG contains a materialization cut; use the lowering that owns the cut"
        )
    sources, leaves = [], {}
    for node in compiled._nodes:
        if node.kind == "input":
            source = bindings[node.expression.name]
        elif node.kind == "state":
            source = states.get(node.expression, states.get(node.expression.state))
            if source is None:
                raise KeyError(
                    f"missing materialized StateTerm binding for {node.expression.state!r}"
                )
        else:
            continue
        if not isinstance(source, torch.Tensor) and not _descriptor(source):
            raise TypeError("unsupported expression binding")
        sources.append(source)
        leaves[node.expression] = source
    if not sources or not any(_tensor(source).is_cuda for source in sources):
        return None
    device = _tensor(sources[0]).device
    if (
        not all(_tensor(source).device == device for source in sources)
        or device.type != "cuda"
    ):
        raise RuntimeError("CUDA expression bindings must share a device")
    shape = tuple(
        torch.broadcast_shapes(*(_logical_shape(source) for source in sources))
    )
    node_types = _node_types(compiled, leaves)
    output_dtypes = tuple(output_dtype or node_types[root] for root in compiled._roots)
    outputs = tuple(
        torch.empty(shape, device=device, dtype=dtype) for dtype in output_dtypes
    )
    if any(extent == 0 for extent in shape):
        return outputs[0] if len(outputs) == 1 else outputs
    key = (
        compiled._kernel_identity,
        tuple(_source_key(source) for source in sources),
        shape,
        node_types,
        output_dtypes,
        device.index,
    )
    kernel = _CACHE.get(key)
    if kernel is None:
        kernel = _make_kernel(compiled, sources, shape, node_types, output_dtypes, 128)
        _CACHE[key] = kernel
    kernel(
        *(_storage_view(source) for source in sources),
        *(output.view(-1) for output in outputs),
    )
    return outputs[0] if len(outputs) == 1 else outputs


__all__ = ["launch_expression"]
