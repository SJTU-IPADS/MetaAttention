# TODO: implement more op bwd
class Node:
    def __init__(self, type: str):
        self.type = type
        self.inputs = []
        self.grad = None

    def _backward(self, grad):
        raise NotImplementedError

    def backward(self, grad=None):
        if grad:
            self.grad = grad
        self._backward(self.grad)
        for node in self.inputs:
            node.backward()

    def __str__(self):
        code = f"{self.type}("
        for node in self.inputs:
            code += f"{node}, "
        code = code + ")"
        return code


class Var(Node):
    def __init__(self, name: str):
        super().__init__("Var")
        self.name = name

    def _backward(self, grad: Node):
        pass

    def __str__(self):
        return f"{self.type}(\"{self.name}\")"

    def print_grad(self):
        print(f"{self.name} grad: {self.grad}")


class Const(Node):
    def __init__(self, value: float):
        super().__init__("Const")
        self.value = value

    def _backward(self, grad: Node):
        pass

    def __str__(self):
        return f"{self.type}({self.value})"


class Add(Node):
    def __init__(self, left: Node, right: Node):
        super().__init__("Add")
        self.inputs = [left, right]

    def _backward(self, grad: Node):
        grad_0 = grad
        grad_1 = grad
        if self.inputs[0].grad:
            grad_0 = Add(grad_0, self.inputs[0].grad)
        if self.inputs[1].grad:
            grad_1 = Add(grad_1, self.inputs[1].grad)
        self.inputs[0].grad = grad_0
        self.inputs[1].grad = grad_1


class Mul(Node):
    def __init__(self, left: Node, right: Node):
        super().__init__("Mul")
        self.inputs = [left, right]

    def _backward(self, grad: Node):
        grad0 = Mul(grad, self.inputs[1])
        grad1 = Mul(grad, self.inputs[0])
        if self.inputs[0].grad:
            grad0 = Add(grad0, self.inputs[0].grad)
        if self.inputs[1].grad:
            grad1 = Add(grad1, self.inputs[1].grad)
        self.inputs[0].grad = grad0
        self.inputs[1].grad = grad1


class Neg(Node):
    def __init__(self, node: Node):
        super().__init__("Neg")
        self.inputs = [node]

    def _backward(self, grad: Node):
        grad = Neg(grad)
        if self.inputs[0].grad:
            grad = Add(grad, self.inputs[0].grad)
        self.inputs[0].grad = grad


class Sub(Node):
    def __init__(self, left: Node, right: Node):
        super().__init__("Sub")
        self.inputs = [left, right]

    def _backward(self, grad: Node):
        raise NotImplementedError


class Div(Node):
    def __init__(self, left: Node, right: Node):
        super().__init__("Div")
        self.inputs = [left, right]

    def _backward(self, grad: Node):
        grad0 = Div(grad, self.inputs[1])
        grad1 = Mul(grad0, Neg(self.inputs[0]))
        grad1 = Div(grad1, self.inputs[1])
        if self.inputs[0].grad:
            grad0 = Add(grad0, self.inputs[0].grad)
        if self.inputs[1].grad:
            grad1 = Add(grad1, self.inputs[1].grad)
        self.inputs[0].grad = grad0
        self.inputs[1].grad = grad1


class Exp(Node):
    def __init__(self, node: Node):
        super().__init__("Exp")
        self.inputs = [node]

    def _backward(self, grad: Node):
        raise NotImplementedError


class Exp2(Node):
    def __init__(self, node: Node):
        super().__init__("Exp2")
        self.inputs = [node]

    def _backward(self, grad: Node):
        raise NotImplementedError


class Log(Node):
    def __init__(self, node: Node):
        super().__init__("Log")
        self.inputs = [node]

    def _backward(self, grad: Node):
        raise NotImplementedError
    
class Log2(Node):
    def __init__(self, node: Node):
        super().__init__("Log2")
        self.inputs = [node]

    def _backward(self, grad: Node):
        raise NotImplementedError


class Tanh(Node):
    def __init__(self, node: Node):
        super().__init__("Tanh")
        self.inputs = [node]

    def _backward(self, grad: Node):
        raise NotImplementedError


class Abs(Node):
    def __init__(self, node: Node):
        super().__init__("Abs")
        self.inputs = [node]

    def _backward(self, grad: Node):
        raise NotImplementedError


class Max(Node):
    def __init__(self, left: Node, right: Node):
        super().__init__("Max")
        self.inputs = [left, right]

    def _backward(self, grad: Node):
        raise NotImplementedError
    
class MaxBwd(Node):
    def __init__(self, grad: Node, left: Node, right: Node):
        super().__init__("MaxBwd")
        self.inputs = [grad, left, right]
        
    def _backward(self, grad: Node):
        raise NotImplementedError

# reduce ops


class ReduceSum(Node):
    def __init__(self, node: Node):
        super().__init__("ReduceSum")
        self.inputs = [node]

    def _backward(self, grad: Node):
        raise NotImplementedError


class ReduceMax(Node):
    def __init__(self, node: Node):
        super().__init__("ReduceMax")
        self.inputs = [node]

    def _backward(self, grad: Node):
        raise NotImplementedError


class ReduceAbsSum(Node):
    def __init__(self, node: Node):
        super().__init__("ReduceAbsSum")
        self.inputs = [node]

    def _backward(self, grad: Node):
        raise NotImplementedError

class ColReduceMax(Node):
    def __init__(self, node: Node):
        super().__init__("ColReduceMax")
        self.inputs = [node]

    def _backward(self, grad: Node):
        raise NotImplementedError
    
class ColReduceSum(Node):
    def __init__(self, node: Node):
        super().__init__("ColReduceSum")
        self.inputs = [node]

    def _backward(self, grad: Node):
        raise NotImplementedError 
    
class ColReduceAbsSum(Node):
    def __init__(self, node: Node):
        super().__init__("ColReduceAbsSum")
        self.inputs = [node]

    def _backward(self, grad: Node):
        raise NotImplementedError



if __name__ == "__main__":
    a = Var("a")
    b = Var("b")
    c = Add(a, b)
    c = Mul(c, c)
    d = Mul(c, Var("c"))
    d.backward(Var("d"))
    # print(c.grad)
    # print(d.grad)
    a.print_grad()
    # print(b.grad)
