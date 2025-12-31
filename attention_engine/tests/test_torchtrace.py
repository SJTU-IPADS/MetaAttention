import torch
import torch.fx as fx
import operator
from core.transform.core import IndentedCode
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
        elif node.target in supported_ops:
            return f"{node} = {supported_ops[node.target]}({', '.join([str(arg) for arg in node.args])})"
        else:
            raise NotImplementedError(f"Operator {node.target} is not supported")
    elif node.op == "placeholder":
        return ""
    elif node.op == "output":
        return ""
    else:
        raise NotImplementedError(f"Operator {node.op} is not supported")
                
def lower_graph(mask_graph: fx.GraphModule)->IndentedCode:
    graph = mask_graph.graph
    mask_code = IndentedCode()
    for node in graph.nodes:
        print(node)
        print(node.op)
        print(node.args)
        print(tl_codegen(node))
        mask_code.add_line(tl_codegen(node))
    return mask_code

if __name__ == "__main__":
    mask_graph = fx.symbolic_trace(sliding_window_mask)
    print(mask_graph)
    print(type(mask_graph))
    mask_code = lower_graph(mask_graph)
    print(mask_code)
    
