import operator

import pytest
import torch
import torch.fx as fx

from core.utils import IndentedCode


pytestmark = pytest.mark.unit


def sliding_window_mask(b, h, q_idx, kv_idx):
    return torch.logical_and(q_idx >= kv_idx, q_idx < kv_idx + 128)


supported_ops = {
    torch.logical_and: "operator.and_",
}


def is_operator_func(func):
    return func in operator.__dict__.values()


def tl_codegen(node: fx.Node) -> str:
    if node.op == "call_function":
        if is_operator_func(node.target):
            return f"{node} = operator.{node.target.__name__}({', '.join([str(arg) for arg in node.args])})"
        if node.target in supported_ops:
            return f"{node} = {supported_ops[node.target]}({', '.join([str(arg) for arg in node.args])})"
        raise NotImplementedError(f"Operator {node.target} is not supported")
    if node.op in {"placeholder", "output"}:
        return ""
    raise NotImplementedError(f"Operator {node.op} is not supported")


def lower_graph(mask_graph: fx.GraphModule) -> IndentedCode:
    mask_code = IndentedCode()
    for node in mask_graph.graph.nodes:
        mask_code.add_line(tl_codegen(node))
    return mask_code


def test_sliding_window_mask_lowering_uses_operator_and():
    mask_code = lower_graph(fx.symbolic_trace(sliding_window_mask))
    assert "operator.and_" in str(mask_code)


def test_lowering_unsupported_operator_raises():
    def unsupported_mask(x):
        return torch.sin(x)

    with pytest.raises(NotImplementedError, match="is not supported"):
        lower_graph(fx.symbolic_trace(unsupported_mask))
