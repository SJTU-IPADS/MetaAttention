"""Abs/Cast preserve explicit precision through analysis and expression lowering."""

import pytest
import torch

from attn_engine import (
    AlgorithmIR,
    Batch,
    Constant,
    FeatureRole,
    HeadRole,
    HeadMapping,
    Input,
    ProductFactor,
    ProductInjection,
    Readout,
    Sequence,
    StateSpec,
    StateTerm,
    StateTransition,
    StatefulOperator,
    Target,
    TensorInput,
)
from attn_engine.stateful.lowering.expression.expression import compile_expression
from attn_engine.stateful.lowering.expression.dense_expression import (
    compile_dense_expression,
)
from attn_engine.stateful.analyse.algorithm import simplify_expression


def test_abs_cast_precision_and_cpu_vjp():
    x = torch.tensor([-1.003, 0.0, 0.125], dtype=torch.bfloat16, requires_grad=True)
    expr = (Input("x").to(torch.float32) + 0.001).abs().to(torch.bfloat16)
    actual = compile_expression(expr)({"x": x})
    expected = (x.float() + 0.001).abs().bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([-1.0, 1.0, 1.0], dtype=x.dtype))
    z = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    compile_expression(Input("z").abs())({"z": z}).sum().backward()
    torch.testing.assert_close(z.grad, torch.tensor([-1.0, 0.0, 1.0]))
    # BF16 arithmetic followed by Cast differs observably from Cast then arithmetic.
    a = compile_expression((Input("x") + 0.001).to(torch.float32))({"x": x})
    b = compile_expression(Input("x").to(torch.float32) + 0.001)({"x": x})
    assert not torch.equal(a, b)
    assert simplify_expression(expr) == expr
    literal = Constant(1.003).to(torch.bfloat16).to(torch.float32)
    assert simplify_expression(literal) == literal


def _operator():
    feature, heads = FeatureRole("feature"), HeadRole("heads")
    expr = Input("x").to(torch.float32).abs()
    op = StatefulOperator(
        AlgorithmIR(
            inputs=(
                TensorInput("x", (Batch, heads, Sequence, feature), torch.bfloat16),
            ),
            head_mapping=HeadMapping({heads: heads}),
            states=(StateSpec("memory", (Batch, heads, feature)),),
            transition=(
                StateTransition(
                    "memory",
                    injections=(ProductInjection((ProductFactor(expr, (feature,)),)),),
                ),
            ),
            readouts=(
                Readout(
                    "output", StateTerm("memory", 1, None).abs().to(torch.bfloat16)
                ),
            ),
        )
    )
    return op, feature, heads, expr


def test_abs_cast_analysis_dependency_and_native_loader():
    op, feature, heads, expr = _operator()
    compiled = op.compile(
        role_sizes={feature: 3, heads: 1}, target=Target(), mode="dense", backward=True
    )
    executable = op._plans[compiled.specialization]
    assert "x" in executable.backward_ir.input_gradients
    dense = compile_dense_expression(
        expr, {s.name: s for s in op.algorithm.inputs}, feature
    )
    assert dense is not None
    assert any(node[0] == "cast" for node in dense.loader.nodes)
    assert any(node[:3] == ("unary", 1, "abs") for node in dense.loader.nodes)


@pytest.mark.gpu
def test_abs_cast_cuda_pointwise_and_native_loader(gpu_device):
    from attn_engine.stateful.lowering.diagonal.native_tile_kernels import (
        RawLoaderContext,
    )
    from attn_engine.stateful.lowering.diagonal.native_tile_softmax import (
        _compile,
        _annotation,
    )

    op, feature, heads, expr = _operator()
    host = torch.tensor([-1.003, 0.0, 0.125], dtype=torch.bfloat16).reshape(1, 1, 1, 3)
    x = host.to(gpu_device)
    dag = {
        "precise": (Input("x").to(torch.float32) + 0.001).abs(),
        "rounded": (Input("x") + 0.001).to(torch.float32).abs(),
        "down": Input("x").abs().to(torch.bfloat16),
    }
    actual = compile_expression(dag)({"x": x})
    expected = compile_expression(dag)({"x": host})
    for name in dag:
        torch.testing.assert_close(actual[name].cpu(), expected[name], rtol=0, atol=0)
    native = compile_dense_expression(
        dag["precise"], {s.name: s for s in op.algorithm.inputs}, feature
    )
    context = RawLoaderContext((native,), {"x": x}, 1, 1)
    out = torch.empty((3,), device=gpu_device)
    body = [
        "head = 0",
        "token = 0",
        "for r in T.Parallel(3):",
        "    Output[r] = " + context.emit_value(native, "token", "head", "r"),
    ]
    _compile(
        list(context.annotations) + [_annotation("Output", out)],
        body,
        1,
        device=gpu_device,
    )(*context.pointers, out)
    torch.testing.assert_close(out.cpu(), expected["precise"].flatten(), rtol=0, atol=0)


@pytest.mark.gpu
def test_abs_cast_public_backward_preserves_node_derivatives(gpu_device):
    op, feature, heads, _ = _operator()
    compiled = op.compile(
        role_sizes={feature: 3, heads: 1}, target=Target(), mode="dense", backward=True
    )
    x = (
        torch.tensor(
            [-1.0, 0.0, 1.0, 0.5, -0.25, 0.0], dtype=torch.bfloat16, device=gpu_device
        )
        .reshape(1, 1, 2, 3)
        .requires_grad_()
    )
    oracle_input = x.detach().clone().requires_grad_()
    result = compiled(x=x, return_final_state=True)
    (
        result.outputs["output"].float().sum() + result.final_state["memory"].sum()
    ).backward()
    trajectory = oracle_input.float().abs().cumsum(2)
    (trajectory.abs().bfloat16().float().sum() + trajectory[:, :, -1].sum()).backward()
    torch.testing.assert_close(x.grad, oracle_input.grad, rtol=0, atol=0)
