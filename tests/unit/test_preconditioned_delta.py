"""Independent recurrence oracle for stage-derived preconditioned coefficients."""

import pytest
import torch
import torch.nn.functional as F

from attn_engine import StateTuple, Target
from examples.preconditioned_delta import (
    KEY,
    VALUE,
    KEY_HEADS,
    VALUE_HEADS,
    preconditioned_delta,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")


def _inputs(kind, dtype, device="cpu", length=9):
    torch.manual_seed(31)
    b, hk, hv, k, v = 2, 2, 4, 3, 5
    vector = (b, hv, length, k) if kind == "kda" else (b, hv, length)
    values = {
        "query": torch.randn(b, hk, length, k, device=device, dtype=dtype) * 0.3,
        "key": torch.randn(b, hk, length, k, device=device, dtype=dtype) * 0.3,
        "value": torch.randn(b, hv, length, v, device=device, dtype=dtype) * 0.3,
        "log_atk_decay": -torch.rand(b, hk, length, device=device) * 0.2,
        "beta_atk": torch.rand(b, hk, length, device=device) * 0.4,
        "center": torch.full((b, hk, length), -0.2, device=device),
        "scale": torch.full((b, hk, length), k**-0.5, device=device),
        "beta": torch.rand(b, hv, length, device=device) * 0.3,
        "log_decay": -torch.rand(vector, device=device) * 0.1,
    }
    initial = (
        torch.rand(b, hk, k, device=device) * 0.1,
        torch.randn(b, hv, k, v, device=device) * 0.1,
    )
    return {
        name: tensor.contiguous().requires_grad_() for name, tensor in values.items()
    }, tuple(t.requires_grad_() for t in initial)


def _oracle(values, initial, kind):
    a, s = initial
    q, k, v = (values[name].float() for name in ("query", "key", "value"))
    if kind == "gdn":
        q, k = (F.normalize(t, dim=-1) for t in (q, k))
    outputs = []
    for t in range(k.shape[2]):
        kt = k[:, :, t]
        a = (
            values["log_atk_decay"][:, :, t].exp()[..., None] * a
            + values["beta_atk"][:, :, t, None] * kt.square()
        )
        r = (a + 1e-6).log() - values["center"][:, :, t, None]
        write = kt * torch.exp(-__import__("math").log(1.5) * r / (1 + r.abs()))
        kt, write, qt = (
            z.repeat_interleave(s.shape[1] // a.shape[1], dim=1)
            for z in (kt, write, q[:, :, t] * values["scale"][:, :, t, None])
        )
        decay = values["log_decay"][:, :, t].exp()
        s = s * (decay[..., None] if kind == "kda" else decay[..., None, None])
        error = v[:, :, t] - (kt[..., None] * s).sum(-2)
        s = s + write[..., None] * (values["beta"][:, :, t, None] * error)[..., None, :]
        outputs.append((qt[..., None] * s).sum(-2))
    return torch.stack(outputs, dim=2).to(values["value"].dtype), (a, s)


def _compile(kind, dtype, device="cpu", packed=False):
    return preconditioned_delta(dtype=dtype, kind=kind).compile(
        role_sizes={KEY_HEADS: 2, VALUE_HEADS: 4, KEY: 3, VALUE: 5},
        target=Target(device_type=device),
        mode="packed" if packed else "dense",
        backward=True,
    )


@pytest.mark.parametrize("kind", ["kda", "gdn"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("loss_kind", ["output", "final", "both"])
def test_output_finals_all_gradients(kind, dtype, loss_kind):
    values, initial = _inputs(kind, dtype)
    result = _compile(kind, dtype)(
        **values,
        initial_state=StateTuple(("atk", "memory"), initial),
        return_final_state=True,
    )
    output, finals = _oracle(values, initial, kind)
    tol = 3e-5 if dtype == torch.float32 else 0.004
    torch.testing.assert_close(result.outputs["output"], output, atol=tol, rtol=tol)
    for actual, wanted in zip(result.final_state, finals, strict=True):
        torch.testing.assert_close(actual, wanted, atol=3e-6, rtol=3e-5)
    loss = (
        result.outputs["output"].float().square().sum() if loss_kind != "final" else 0
    )
    expected_loss = output.float().square().sum() if loss_kind != "final" else 0
    if loss_kind != "output":
        loss = loss + sum(t.square().sum() for t in result.final_state)
        expected_loss = expected_loss + sum(t.square().sum() for t in finals)
    tensors = (*values.values(), *initial)
    actual = torch.autograd.grad(loss, tensors, allow_unused=True)
    expected = torch.autograd.grad(expected_loss, tensors, allow_unused=True)
    for grad, wanted in zip(actual, expected, strict=True):
        if wanted is None:
            assert grad is None
        else:
            torch.testing.assert_close(grad, wanted, atol=tol * 2, rtol=tol * 2)


@pytest.mark.parametrize("kind", ["kda", "gdn"])
def test_continuation_preserves_shared_atk_cache(kind):
    values, initial = _inputs(kind, torch.float32)
    compiled = _compile(kind, torch.float32)
    first = compiled(
        **{n: v[:, :, :4].contiguous() for n, v in values.items()},
        initial_state=StateTuple(("atk", "memory"), initial),
        return_final_state=True,
    )
    second = compiled(
        **{n: v[:, :, 4:].contiguous() for n, v in values.items()},
        initial_state=first.final_state,
        return_final_state=True,
    )
    full = compiled(
        **values,
        initial_state=StateTuple(("atk", "memory"), initial),
        return_final_state=True,
    )
    assert first.final_state["atk"].shape == (2, 2, 3)
    assert first.final_state["memory"].shape == (2, 4, 3, 5)
    torch.testing.assert_close(
        torch.cat((first.outputs["output"], second.outputs["output"]), dim=2),
        full.outputs["output"],
    )
    for actual, wanted in zip(second.final_state, full.final_state, strict=True):
        torch.testing.assert_close(actual, wanted)


@pytest.mark.parametrize("kind", ["kda", "gdn"])
@pytest.mark.parametrize("all_empty", [False, True])
def test_packed_states_and_gradients(kind, all_empty):
    values, initial = _inputs(kind, torch.float32)
    length = 0 if all_empty else 9
    values = {
        name: value[0, :, :length].movedim(0, 1).contiguous().detach().requires_grad_()
        for name, value in values.items()
    }
    offsets = [0, 0, length, length]
    initial = tuple(
        t[:1].expand(3, *t.shape[1:]).contiguous().detach().requires_grad_()
        for t in initial
    )
    result = _compile(kind, torch.float32, packed=True)(
        **values,
        initial_state=StateTuple(("atk", "memory"), initial),
        sequence_offsets=torch.tensor(offsets, dtype=torch.int32),
        return_final_state=True,
    )
    if not all_empty:
        dense = {n: v.movedim(0, 1).unsqueeze(0) for n, v in values.items()}
        output, finals = _oracle(dense, tuple(t[1:2] for t in initial), kind)
        torch.testing.assert_close(result.outputs["output"], output[0].movedim(0, 1))
        for actual, wanted in zip(result.final_state, finals, strict=True):
            torch.testing.assert_close(actual[1:2], wanted)
    for state, original in zip(result.final_state, initial, strict=True):
        torch.testing.assert_close(state[[0, 2]], original[[0, 2]])
    gradients = torch.autograd.grad(
        result.outputs["output"].sum() + sum(t.sum() for t in result.final_state),
        (*values.values(), *initial),
    )
    assert all(torch.isfinite(t).all() for t in gradients)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["kda", "gdn"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_h20_stage_outputs_and_gradients(kind, dtype, monkeypatch):
    monkeypatch.delenv("STATEFUL_OPERATOR_TEST_CPU", raising=False)
    values, initial = _inputs(kind, dtype, "cuda", length=65)
    result = _compile(kind, dtype, "cuda")(
        **values,
        initial_state=StateTuple(("atk", "memory"), initial),
        return_final_state=True,
    )
    output, finals = _oracle(values, initial, kind)
    tol = 0.004 if dtype == torch.bfloat16 else 4e-5
    torch.testing.assert_close(result.outputs["output"], output, atol=tol, rtol=tol)
    for actual, wanted in zip(result.final_state, finals, strict=True):
        torch.testing.assert_close(actual, wanted, atol=3e-5, rtol=3e-5)
    actual = torch.autograd.grad(
        result.outputs["output"].float().square().sum()
        + sum(t.square().sum() for t in result.final_state),
        (*values.values(), *initial),
    )
    expected = torch.autograd.grad(
        output.float().square().sum() + sum(t.square().sum() for t in finals),
        (*values.values(), *initial),
    )
    for grad, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(grad, wanted, atol=tol * 3, rtol=tol * 3)


def test_gdn_zero_query_key_matches_normalization_floor_gradients():
    values, initial = _inputs("gdn", torch.float32)
    values["query"] = torch.zeros_like(values["query"], requires_grad=True)
    values["key"] = torch.zeros_like(values["key"], requires_grad=True)
    result = _compile("gdn", torch.float32)(
        **values,
        initial_state=StateTuple(("atk", "memory"), initial),
        return_final_state=True,
    )
    output, finals = _oracle(values, initial, "gdn")
    actual = torch.autograd.grad(
        result.outputs[0].sum() + sum(t.sum() for t in result.final_state),
        (*values.values(), *initial),
    )
    expected = torch.autograd.grad(
        output.sum() + sum(t.sum() for t in finals), (*values.values(), *initial)
    )
    for lhs, rhs in zip(actual, expected, strict=True):
        assert lhs.isfinite().all()
        torch.testing.assert_close(lhs, rhs, atol=1e4, rtol=2e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["kda", "gdn"])
@pytest.mark.parametrize("all_empty", [False, True])
def test_h20_packed_checkpoint_gradient_equivalence(kind, all_empty, monkeypatch):
    monkeypatch.delenv("STATEFUL_OPERATOR_TEST_CPU", raising=False)
    values, initial = _inputs(kind, torch.float32, "cuda", length=7)
    length = 0 if all_empty else 7
    values = {
        name: tensor[0, :, :length].movedim(0, 1).contiguous().detach().requires_grad_()
        for name, tensor in values.items()
    }
    initial = tuple(
        tensor[:1].expand(3, *tensor.shape[1:]).contiguous().detach().requires_grad_()
        for tensor in initial
    )
    offsets = torch.tensor([0, 0, length, length], device="cuda", dtype=torch.int32)
    normal = _compile(kind, torch.float32, "cuda", packed=True)
    checked = preconditioned_delta(dtype=torch.float32, kind=kind).compile(
        role_sizes={KEY_HEADS: 2, VALUE_HEADS: 4, KEY: 3, VALUE: 5},
        target=Target(),
        mode="packed",
        backward=True,
        checkpoint=True,
    )
    results = [
        compiled(
            **values,
            initial_state=StateTuple(("atk", "memory"), initial),
            sequence_offsets=offsets,
            return_final_state=True,
        )
        for compiled in (normal, checked)
    ]
    for lhs, rhs in zip(
        (*results[0].outputs, *results[0].final_state),
        (*results[1].outputs, *results[1].final_state),
        strict=True,
    ):
        torch.testing.assert_close(lhs, rhs)
    gradients = [
        torch.autograd.grad(
            result.outputs[0].square().sum()
            + sum(t.square().sum() for t in result.final_state),
            (*values.values(), *initial),
        )
        for result in results
    ]
    for lhs, rhs in zip(*gradients, strict=True):
        assert torch.isfinite(lhs).all()
        torch.testing.assert_close(lhs, rhs)
    if not all_empty:
        dense = {
            name: tensor.movedim(0, 1).unsqueeze(0) for name, tensor in values.items()
        }
        output, finals = _oracle(dense, tuple(t[1:2] for t in initial), kind)
        torch.testing.assert_close(
            results[0].outputs[0], output[0].movedim(0, 1), atol=3e-5, rtol=3e-5
        )
        for actual, wanted in zip(results[0].final_state, finals, strict=True):
            torch.testing.assert_close(actual[1:2], wanted, atol=3e-5, rtol=3e-5)
