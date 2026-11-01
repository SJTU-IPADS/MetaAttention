"""Dense diagonal state preparation and readout dispatch."""

from __future__ import annotations
import torch

CHUNK = 64


def _cdiv(x, y):
    return (x + y - 1) // y


def prepare_state(
    *,
    rows_gate,
    columns_gate,
    key,
    value,
    initial,
    sequence_length,
    return_final_state,
    identity=False,
):
    batch, heads, rows, columns = initial.shape
    initial = initial.contiguous()
    chunks = _cdiv(sequence_length, CHUNK)
    if isinstance(key, tuple):
        key, key2 = (part.float().contiguous() for part in key)
        kl, kr = key.shape[-1], key2.shape[-1]
    else:
        key = key.reshape(batch * sequence_length, heads, rows).float().contiguous()
        key2, kl, kr = key, rows, 0
    value = value.reshape(batch * sequence_length, heads, columns).float().contiguous()
    tokens = batch * sequence_length
    rows_gate = (
        None
        if rows_gate is None
        else rows_gate.reshape(tokens, heads, -1)
        .expand(tokens, heads, rows)
        .float()
        .contiguous()
    )
    columns_gate = (
        None
        if columns_gate is None
        else columns_gate.reshape(tokens, heads, -1)
        .expand(tokens, heads, columns)
        .float()
        .contiguous()
    )
    bias = torch.empty(
        batch * heads, chunks, rows, columns, device=initial.device, dtype=torch.float32
    )
    scale = None if identity else torch.empty_like(bias)
    carry = torch.empty_like(bias)
    final = torch.empty_like(initial) if return_final_state else initial
    if identity:
        from .identity import identity_summary

        def vector(t):
            return t.reshape(batch, sequence_length, heads, -1).permute(0, 2, 1, 3)

        identity_summary(
            vector(key),
            vector(key2),
            vector(value),
            bias,
            sequence_length=sequence_length,
            heads=heads,
            kr=kr,
        )
    else:
        from .diagonal_block import diagonal_block_summary

        diagonal_block_summary(
            rows_gate,
            columns_gate,
            key,
            key2,
            value,
            scale,
            bias,
            sequence_length=sequence_length,
            heads=heads,
            rows=rows,
            columns=columns,
            chunks=chunks,
            kl=kl,
            kr=kr,
        )
    from .diagonal_scan import hierarchical_carry

    hierarchical_carry(
        scale,
        bias,
        initial,
        carry,
        final,
        identity=identity,
        write_final=return_final_state,
    )
    return carry, final, (rows_gate, columns_gate, key, key2, value, kl, kr)


def read_state(
    carry,
    factors,
    query,
    *,
    sequence_length,
    prior=False,
    identity=False,
    product_scores=None,
):
    batch_heads, chunks, rows, columns = carry.shape
    rows_gate, columns_gate, key, key2, value, kl, kr = factors
    heads = key.shape[1]
    if isinstance(query, tuple):
        query, query2 = (part.float().contiguous() for part in query)
        ql, qr = query.shape[-1], query2.shape[-1]
    else:
        query = query.reshape(key.shape[0], heads, rows).float().contiguous()
        query2, ql, qr = query, rows, 0
    if identity and kr and qr and kl == ql and kr == qr:
        from ..readout.product_kernels import product_chunk_scores, read_product_state

        scores = None
        if product_scores is not None:
            key_cache = (
                key.data_ptr(),
                key2.data_ptr(),
                query.data_ptr(),
                query2.data_ptr(),
                kl,
                kr,
                sequence_length,
                prior,
            )
            scores = product_scores.get(key_cache)
            if scores is None:
                scores = product_chunk_scores(
                    (key, key2),
                    (query, query2),
                    sequence_length=sequence_length,
                    prior=prior,
                )
                product_scores[key_cache] = scores
        return read_product_state(
            carry,
            (key, key2),
            (query, query2),
            value,
            sequence_length=sequence_length,
            prior=prior,
            scores=scores,
        )
    output = torch.empty(
        key.shape[0], heads, columns, device=key.device, dtype=torch.float32
    )
    if identity:
        from .identity import identity_readout

        batch = batch_heads // heads

        def vector(t):
            return t.reshape(batch, sequence_length, heads, -1).permute(0, 2, 1, 3)

        identity_readout(
            vector(key),
            vector(key2),
            vector(value),
            vector(query),
            vector(query2),
            carry,
            output,
            sequence_length=sequence_length,
            heads=heads,
            kr=kr,
            qr=qr,
            prior=prior,
        )
    else:
        from .diagonal_block import diagonal_block_readout

        diagonal_block_readout(
            carry,
            factors,
            query,
            query2,
            output,
            sequence_length=sequence_length,
            prior=prior,
            ql=ql,
            qr=qr,
        )
    return output
