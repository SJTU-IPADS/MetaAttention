"""TileLang lowering for ordered low-rank affine recurrences.

Scalar-only eligible plans use short register-State recurrence or block compact
WY. Noncommuting plans retain affine summaries, exclusive chunk carries and
parallel replay with the historical token-major factor ABI.
"""

import torch


def execute(
    *,
    operation_codes,
    left_factors,
    right_factors,
    coefficients,
    injection_left,
    injection_right,
    readouts,
    initial_state,
    output_dtypes,
    symmetric_rank_one=False,
    injection_scalars=None,
    injection_orders=None,
    readout_axis=0,
):
    batch, heads, rows, columns = initial_state.shape
    tokens = readouts[0].shape[0]
    if tokens % batch:
        raise ValueError("low-rank token count must divide the State batch")
    length = tokens // batch
    device = initial_state.device
    codes = tuple(
        {"scale": 0, "row-scale": 1, "rank-one": 2, "column-scale": 3}[x]
        for x in operation_codes
    )

    def binding(part, size):
        if part.numel() == 1:
            return part.reshape(1, 1, 1).expand(tokens, heads, size)
        return part.reshape(tokens, heads, -1).expand(tokens, heads, size)

    from .wy_kernels import eligible, execute as execute_wy

    if (
        readout_axis == 0
        and not injection_scalars
        and eligible(tuple(operation_codes), len(injection_left), rows, columns)
    ):
        rank_ops = tuple(i for i, code in enumerate(codes) if code == 2)
        return execute_wy(
            codes=codes,
            left=tuple(left_factors[i] for i in rank_ops),
            right=tuple(right_factors[i] for i in rank_ops),
            coefficient=coefficients,
            injection_left=injection_left,
            injection_right=injection_right,
            query=readouts,
            initial_state=initial_state,
            length=length,
            output_dtypes=output_dtypes,
            symmetric_rank_one=symmetric_rank_one,
        )

    from .checkpoint import forward

    width = max(rows, columns) if 3 in codes else rows
    left = torch.empty(
        (len(codes), tokens, heads, width, 1), device=device, dtype=torch.float32
    )
    right = torch.empty(
        (len(codes), tokens, heads, rows, 1), device=device, dtype=torch.float32
    )
    coeff = torch.empty(
        (len(codes), tokens, heads, 1, 1), device=device, dtype=torch.float32
    )
    for i, code in enumerate(codes):
        if code in (1, 2, 3):
            size = columns if code == 3 else rows
            left[i, :, :, :size, 0].copy_(binding(left_factors[i], size))
        if code == 2:
            right[i, :, :, :, 0].copy_(binding(right_factors[i], rows))
        if code in (0, 2):
            coeff[i, :, :, 0, 0].copy_(binding(coefficients[i], 1).squeeze(-1))

    def stack(parts, size):
        if not parts:
            return torch.empty(
                (0, tokens, heads, size), device=device, dtype=torch.float32
            )
        return torch.stack([binding(x, size) for x in parts]).float().contiguous()

    il = stack(injection_left, rows).unsqueeze(-1)
    ir = stack(injection_right, columns).unsqueeze(-2)
    query = stack(readouts, columns if readout_axis else rows).unsqueeze(-1)
    output, final, _ = forward(
        operation_codes=operation_codes,
        left_factors=left,
        right_factors=right,
        coefficients=coeff,
        injection_left=il,
        injection_right=ir,
        readout=query,
        initial_state=initial_state,
        output_dtype=torch.float32,
        record=False,
        injection_scalars=stack(injection_scalars or [], 1).unsqueeze(-1),
        injection_orders=injection_orders,
        readout_axis=readout_axis,
    )
    return tuple(
        output[i].to(dtype) for i, dtype in enumerate(output_dtypes)
    ), final.to(initial_state.dtype)
