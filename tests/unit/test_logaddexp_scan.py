"""LogAddExp semantics, strict contracts, VJP and optional H20/FLA evidence."""

import os
import sys

import pytest
import torch

from attn_engine import (
    Batch,
    FeatureRole,
    HeadRole,
    Sequence,
    StateSpec,
    StateTuple,
    StatefulCompilationError,
    Target,
    TensorInput,
)
from attn_engine.stateful.logaddexp import (
    LogAddExpOperator,
    LogAddExpScanIR,
    logaddexp_combine,
)

pytestmark = pytest.mark.unit
H = HeadRole("heads")
K = FeatureRole("key")


@pytest.fixture(autouse=True)
def cpu_contract(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")


def compiled(mode="dense", backward=True, dtype=torch.float32, workspace=1 << 30):
    ir = LogAddExpScanIR(TensorInput("logits", (Batch, H, Sequence, K), dtype))
    return LogAddExpOperator(ir).compile(
        role_sizes={H: 2, K: 3},
        target=Target(workspace_bytes=workspace),
        mode=mode,
        backward=backward,
    )


def reference(x, prior):
    if x.shape[0] == 0:
        return x, x, prior
    z = x.logcumsumexp(0)
    if prior is not None:
        z = torch.logaddexp(z, prior)
        previous = torch.cat((prior[None], z[:-1]))
    else:
        previous = torch.cat((z[:1], z[:-1]))
    return (x - z).exp(), torch.nan_to_num(previous - z, neginf=0.0), z[-1]


@pytest.mark.parametrize(
    "mode,lengths",
    [
        ("dense", (17, 17)),
        ("packed", (0, 1, 7, 0, 19, 0)),
        ("packed", (0, 0)),
        ("dense", (0, 0)),
    ],
)
@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("loss_kind", ["all", "outputs", "final"])
def test_prefix_outputs_final_and_all_loss_gradients(mode, lengths, warm, loss_kind):
    torch.manual_seed(41)
    x = torch.randn(sum(lengths), 2, 3, requires_grad=True)
    p = torch.randn(len(lengths), 2, 3, requires_grad=True) if warm else None
    xr = x.detach().clone().requires_grad_()
    pr = p.detach().clone().requires_grad_() if warm else None
    offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    physical = (
        x
        if mode == "packed"
        else x.reshape(len(lengths), lengths[0], 2, 3).permute(0, 2, 1, 3).contiguous()
    )
    result = compiled(mode)(
        logits=physical,
        sequence_offsets=offsets if mode == "packed" else None,
        initial_state=StateTuple(("z",), (p,)) if warm else None,
        return_final_state=True,
    )
    triples = [
        reference(piece, pr[i] if warm else None)
        for i, piece in enumerate(xr.split(lengths))
    ]
    kr, gr = (torch.cat([t[j] for t in triples]) for j in (0, 1))
    zr = torch.stack(
        [t[2] if t[2] is not None else torch.full((2, 3), -torch.inf) for t in triples]
    )
    k, g = result.outputs
    if mode == "dense":
        k, g = (t.permute(0, 2, 1, 3).reshape_as(x) for t in (k, g))
    torch.testing.assert_close(k, kr)
    torch.testing.assert_close(g, gr)
    torch.testing.assert_close(result.final_state[0], zr)
    weights = [torch.randn_like(t) for t in (k, g, zr)]
    if loss_kind == "outputs":
        weights[2].zero_()
    elif loss_kind == "final":
        weights[0].zero_()
        weights[1].zero_()
    # Infinite empty identities are constants, not a finite scalar loss.
    actual_loss = (
        (k * weights[0]).sum()
        + (g * weights[1]).sum()
        + (
            torch.where(torch.isfinite(result.final_state[0]), result.final_state[0], 0)
            * weights[2]
        ).sum()
    )
    ref_loss = (
        (kr * weights[0]).sum()
        + (gr * weights[1]).sum()
        + (torch.where(torch.isfinite(zr), zr, 0) * weights[2]).sum()
    )
    actual_loss.backward()
    if ref_loss.requires_grad:
        ref_loss.backward()
        torch.testing.assert_close(x.grad, xr.grad, atol=3e-6, rtol=3e-5)
        if warm:
            torch.testing.assert_close(p.grad, pr.grad, atol=3e-6, rtol=3e-5)
    assert torch.isfinite(x.grad).all()


def test_identity_associativity_masked_and_continuation():
    a = torch.tensor([-torch.inf, 10000.0, -10000.0], requires_grad=True)
    identity = torch.full_like(a, -torch.inf, requires_grad=True)
    b, c = torch.randn(3), torch.randn(3)
    torch.testing.assert_close(logaddexp_combine(a, identity), a)
    torch.testing.assert_close(
        logaddexp_combine(logaddexp_combine(a, b), c),
        logaddexp_combine(a, logaddexp_combine(b, c)),
    )
    logaddexp_combine(a, identity).sum().backward()
    assert a.grad.isfinite().all() and identity.grad.isfinite().all()
    op = compiled()
    x = (torch.randn(2, 2, 9, 3) + 1000).requires_grad_()
    full = op(logits=x, return_final_state=True)
    first = op(logits=x[:, :, :4].contiguous(), return_final_state=True)
    last = op(
        logits=x[:, :, 4:].contiguous(),
        initial_state=first.final_state,
        return_final_state=True,
    )
    for whole, lhs, rhs in zip(full.outputs, first.outputs, last.outputs):
        torch.testing.assert_close(
            whole, torch.cat((lhs, rhs), 2), atol=1e-4, rtol=2e-4
        )
    torch.testing.assert_close(full.final_state[0], last.final_state[0])
    masked = torch.full((2, 2, 5, 3), -torch.inf, requires_grad=True)
    result = op(logits=masked, return_final_state=True)
    assert all(torch.equal(t, torch.zeros_like(t)) for t in result.outputs)
    sum(t.sum() for t in result.outputs).backward()
    assert torch.equal(masked.grad, torch.zeros_like(masked))


def test_contracts_metadata_and_workspace():
    op = compiled()
    x = torch.randn(2, 2, 5, 3)
    for values, code in [
        ({}, "RUNTIME_BINDING"),
        ({"logits": x, "extra": x}, "RUNTIME_BINDING"),
        ({"logits": x.bfloat16()}, "RUNTIME_DTYPE"),
        ({"logits": x.transpose(2, 3)}, "RUNTIME_LAYOUT"),
        ({"logits": torch.randn(2, 2, 5, 4)}, "RUNTIME_SHAPE"),
    ]:
        with pytest.raises(StatefulCompilationError) as error:
            op(**values)
        assert error.value.code == code
    with pytest.raises(StatefulCompilationError):
        compiled(backward=False)(logits=x.requires_grad_())
    with pytest.raises(StatefulCompilationError):
        compiled(workspace=1).workspace_bytes(10, sequence_count=2, sequence_length=5)
    with pytest.raises(StatefulCompilationError):
        op.workspace_bytes(-1)
    ir = op.operator.algorithm
    for roles in ({H: 2}, {H: True, K: 3}, {H: 2, K: 0}):
        with pytest.raises(StatefulCompilationError):
            LogAddExpOperator(ir).compile(
                role_sizes=roles, target=Target(), mode="dense"
            )
    with pytest.raises(StatefulCompilationError):
        LogAddExpScanIR(
            TensorInput("x", (Batch, H, Sequence, K), torch.float32),
            StateSpec("z", (Batch, H, FeatureRole("other"))),
        )


def test_staged_lightnet_reference_and_gradients():
    from examples.lightnet_stable import HEADS, KEY, VALUE, lightnet_stable

    op = lightnet_stable(2, 2, 7, 3, 4, dtype=torch.float32).compile(
        role_sizes={HEADS: 2, KEY: 3, VALUE: 4},
        target=Target(),
        mode="dense",
        backward=True,
    )
    torch.manual_seed(42)
    q, x, v = [
        torch.randn(*shape, requires_grad=True)
        for shape in ((2, 2, 7, 3), (2, 2, 7, 3), (2, 2, 7, 4))
    ]
    z0 = torch.randn(2, 2, 3, requires_grad=True)
    m0 = torch.randn(2, 2, 3, 4, requires_grad=True)
    result = op(
        query=q,
        logits=x,
        value=v,
        initial_state=StateTuple(("z", "memory"), (z0, m0)),
        return_final_state=True,
    )
    qr, xr, vr, zr, mr = [
        t.detach().clone().requires_grad_() for t in (q, x, v, z0, m0)
    ]
    z, m = zr, mr
    outputs = []
    for t in range(7):
        zp = z
        z = torch.logaddexp(z, xr[:, :, t])
        k, g = (xr[:, :, t] - z).exp(), zp - z
        m = g.exp()[..., None] * m + k[..., None] * vr[:, :, t, None, :]
        outputs.append(torch.einsum("bhk,bhkv->bhv", qr[:, :, t] * 3**-0.5, m))
    expected = torch.stack(outputs, 2)
    torch.testing.assert_close(result.outputs[0], expected, atol=2e-6, rtol=2e-5)
    weights = [torch.randn_like(t) for t in (expected, z, m)]
    sum(
        (a * w).sum() for a, w in zip((result.outputs[0], *result.final_state), weights)
    ).backward()
    sum((a * w).sum() for a, w in zip((expected, z, m), weights)).backward()
    for a, b in zip((q, x, v, z0, m0), (qr, xr, vr, zr, mr)):
        torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("LIGHTNET_FLA_CHECK") != "1",
    reason="opt-in H20 FLA comparison",
)
@pytest.mark.parametrize("warm", [False, True])
def test_h20_fla_gate_all_loss_vjp(warm, monkeypatch):
    monkeypatch.delenv("STATEFUL_OPERATOR_TEST_CPU")
    fla_path = os.environ.get("LIGHTNET_FLA_PATH")
    if fla_path:
        sys.path.insert(0, fla_path)
    from fla.ops.lightnet.gate import fused_lightnet_gate

    torch.manual_seed(17)
    x = torch.randn(1, 67, 2, 3, device="cuda", requires_grad=True)
    offsets = torch.tensor([0, 0, 1, 34, 34, 67, 67], device="cuda", dtype=torch.int32)
    prior = torch.randn(6, 2, 3, device="cuda", requires_grad=True) if warm else None
    xr = x.detach().clone().requires_grad_()
    pr = prior.detach().clone().requires_grad_() if warm else None
    result = compiled("packed")(
        logits=x.squeeze(0).contiguous(),
        sequence_offsets=offsets,
        initial_state=StateTuple(("z",), (prior,)) if warm else None,
        return_final_state=True,
    )
    k, g, z = fused_lightnet_gate(xr, pr[:, None] if warm else None, offsets)
    actual = (*result.outputs, result.final_state[0])
    expected = (k.squeeze(0), g.squeeze(0), z.squeeze(1))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
    weights = [torch.randn_like(t) for t in actual]
    sum(
        (torch.where(torch.isfinite(a), a, 0) * w).sum()
        for a, w in zip(actual, weights)
    ).backward()
    sum(
        (torch.where(torch.isfinite(a), a, 0) * w).sum()
        for a, w in zip(expected, weights)
    ).backward()
    torch.testing.assert_close(x.grad, xr.grad, atol=8e-6, rtol=5e-5)
    if warm:
        torch.testing.assert_close(prior.grad, pr.grad, atol=8e-6, rtol=5e-5)


@pytest.mark.parametrize("cold_prior", [False, True])
def test_staged_cold_memory_empty_and_continuation(cold_prior):
    from examples.lightnet_stable import HEADS, KEY, VALUE, lightnet_stable

    op = lightnet_stable(1, 2, 9, 3, 4, dtype=torch.float32).compile(
        role_sizes={HEADS: 2, KEY: 3, VALUE: 4},
        target=Target(),
        mode="dense",
        backward=True,
    )
    q, x, v = [
        torch.randn(*shape, requires_grad=True)
        for shape in ((1, 2, 9, 3), (1, 2, 9, 3), (1, 2, 9, 4))
    ]
    m0 = torch.randn(1, 2, 3, 4, requires_grad=True)
    z0 = (
        torch.full((1, 2, 3), -torch.inf, requires_grad=True)
        if cold_prior
        else torch.randn(1, 2, 3, requires_grad=True)
    )
    initial = StateTuple(("z", "memory"), (z0, m0))
    full = op(
        query=q, logits=x, value=v, initial_state=initial, return_final_state=True
    )
    one = op(
        query=q[:, :, :4].contiguous(),
        logits=x[:, :, :4].contiguous(),
        value=v[:, :, :4].contiguous(),
        initial_state=initial,
        return_final_state=True,
    )
    two = op(
        query=q[:, :, 4:].contiguous(),
        logits=x[:, :, 4:].contiguous(),
        value=v[:, :, 4:].contiguous(),
        initial_state=one.final_state,
        return_final_state=True,
    )
    torch.testing.assert_close(
        full.outputs[0], torch.cat((one.outputs[0], two.outputs[0]), 2)
    )
    for a, b in zip(full.final_state, two.final_state):
        torch.testing.assert_close(a, b)
    weights = [torch.randn_like(t) for t in (full.outputs[0], *full.final_state)]
    gf = torch.autograd.grad(
        sum(
            (a * w).sum() for a, w in zip((full.outputs[0], *full.final_state), weights)
        ),
        (q, x, v, z0, m0),
        retain_graph=True,
    )
    gc = torch.autograd.grad(
        sum(
            (a * w).sum()
            for a, w in zip(
                (torch.cat((one.outputs[0], two.outputs[0]), 2), *two.final_state),
                weights,
            )
        ),
        (q, x, v, z0, m0),
    )
    for a, b in zip(gf, gc):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
        assert torch.isfinite(a).all()
    if cold_prior:
        torch.testing.assert_close(
            full.outputs[0][:, :, 0],
            torch.einsum(
                "bhk,bhkv->bhv", q[:, :, 0] * 3**-0.5, m0 + v[:, :, 0, None, :]
            ),
        )
    packed_op = lightnet_stable(1, 2, 0, 3, 4, dtype=torch.float32).compile(
        role_sizes={HEADS: 2, KEY: 3, VALUE: 4},
        target=Target(),
        mode="packed",
        backward=True,
    )
    empty = packed_op(
        query=q.new_empty(0, 2, 3),
        logits=x.new_empty(0, 2, 3),
        value=v.new_empty(0, 2, 4),
        sequence_offsets=torch.tensor([0, 0], dtype=torch.int32),
        initial_state=initial,
        return_final_state=True,
    )
    assert empty.outputs[0].shape == (0, 2, 4)
    for a, b in zip(empty.final_state, initial):
        torch.testing.assert_close(a, b)


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("LIGHTNET_FLA_CHECK") != "1",
    reason="opt-in H20 FLA comparison",
)
@pytest.mark.parametrize("warm", [False, True])
def test_h20_fla_staged_lightnet(warm, monkeypatch):
    monkeypatch.delenv("STATEFUL_OPERATOR_TEST_CPU")
    fla_path = os.environ.get("LIGHTNET_FLA_PATH")
    if fla_path:
        sys.path.insert(0, fla_path)
    from fla.ops.lightnet import chunk_lightnet
    from examples.lightnet_stable import HEADS, KEY, VALUE, lightnet_stable

    # FLA's FP32 chunk kernel currently misaligns tail blocks (T=17);
    # use its supported full block for direct all-loss VJP comparison.
    op = lightnet_stable(1, 2, 64, 64, 64, dtype=torch.float32).compile(
        role_sizes={HEADS: 2, KEY: 64, VALUE: 64},
        target=Target(),
        mode="dense",
        backward=True,
    )
    torch.manual_seed(24)
    q, x, v = [
        torch.randn(1, 2, 64, 64, device="cuda", requires_grad=True) for _ in range(3)
    ]
    m0 = torch.randn(1, 2, 64, 64, device="cuda", requires_grad=True) if warm else None
    z0 = torch.randn(1, 2, 64, device="cuda", requires_grad=True) if warm else None
    result = op(
        query=q,
        logits=x,
        value=v,
        initial_state=StateTuple(("z", "memory"), (z0, m0)) if warm else None,
        return_final_state=True,
    )
    qr, xr, vr = [
        a.detach().permute(0, 2, 1, 3).contiguous().requires_grad_() for a in (q, x, v)
    ]
    mr = m0.detach().clone().requires_grad_() if warm else None
    zr = z0.detach().clone().requires_grad_() if warm else None
    output, (mt, zt) = chunk_lightnet(
        qr,
        xr,
        vr,
        initial_state=(mr, zr[:, None]) if warm else None,
        output_final_state=True,
    )
    actual = (result.outputs[0], *result.final_state)
    expected = (output.permute(0, 2, 1, 3), zt[:, 0], mt)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=8e-4, rtol=3e-3)
    weights = [torch.randn_like(t) for t in actual]
    sum((a * w).sum() for a, w in zip(actual, weights)).backward()
    sum((a * w).sum() for a, w in zip(expected, weights)).backward()
    for a, b in zip((q, x, v), (qr, xr, vr)):
        torch.testing.assert_close(
            a.grad, b.grad.permute(0, 2, 1, 3), atol=2e-3, rtol=5e-3
        )
    if warm:
        torch.testing.assert_close(z0.grad, zr.grad, atol=2e-3, rtol=5e-3)
        torch.testing.assert_close(m0.grad, mr.grad, atol=2e-3, rtol=5e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_staged_segmented_matches_independent_sequences(dtype):
    from examples.lightnet_stable import HEADS, KEY, VALUE, lightnet_stable

    builder = lightnet_stable(3, 2, 6, 3, 4, dtype=dtype)
    roles = {HEADS: 2, KEY: 3, VALUE: 4}
    packed = builder.compile(
        role_sizes=roles, target=Target(), mode="packed", backward=True
    )
    dense = builder.compile(
        role_sizes=roles, target=Target(), mode="dense", backward=True
    )
    lengths = (0, 1, 5, 0, 3)
    offsets = torch.tensor([0, 0, 1, 6, 6, 9], dtype=torch.int32)
    q = torch.randn(9, 2, 3, dtype=dtype, requires_grad=True)
    x = torch.randn(9, 2, 3, dtype=dtype, requires_grad=True)
    v = torch.randn(9, 2, 4, dtype=dtype, requires_grad=True)
    result = packed(
        query=q, logits=x, value=v, sequence_offsets=offsets, return_final_state=True
    )
    pieces, z, memory = [], [], []
    for i, length in enumerate(lengths):
        if length == 0:
            z.append(torch.full((2, 3), -torch.inf))
            memory.append(torch.zeros(2, 3, 4))
            continue
        start, end = offsets[i].item(), offsets[i + 1].item()
        values = [a[start:end].permute(1, 0, 2)[None].contiguous() for a in (q, x, v)]
        one = dense(
            query=values[0], logits=values[1], value=values[2], return_final_state=True
        )
        pieces.append(one.outputs[0][0].permute(1, 0, 2))
        z.append(one.final_state[0][0])
        memory.append(one.final_state[1][0])
    torch.testing.assert_close(result.outputs[0], torch.cat(pieces))
    torch.testing.assert_close(result.final_state[0], torch.stack(z))
    torch.testing.assert_close(result.final_state[1], torch.stack(memory))
    result.outputs[0].float().sum().backward()
    assert all(a.grad.isfinite().all() for a in (q, x, v))


def test_role_permutation_and_runtime_rejects_nonfinite_algebra():
    ir = LogAddExpScanIR(TensorInput("logits", (Sequence, K, Batch, H), torch.float32))
    op = LogAddExpOperator(ir).compile(
        role_sizes={H: 2, K: 3}, target=Target(), mode="dense", backward=True
    )
    x = torch.randn(2, 2, 7, 3)
    actual = op(logits=x.permute(2, 3, 0, 1).contiguous(), return_final_state=True)
    expected = compiled()(logits=x, return_final_state=True)
    for a, b in zip(actual.outputs, expected.outputs):
        torch.testing.assert_close(a, b)
    for bad in (torch.nan, torch.inf):
        values = x.clone()
        values[0, 0, 0, 0] = bad
        with pytest.raises(RuntimeError, match="finite values"):
            compiled()(logits=values)
