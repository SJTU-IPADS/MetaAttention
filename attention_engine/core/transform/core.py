import torch
from typing import Literal, Type
import functools
from .graph import *
from ..utils import IndentedCode
import functools
from copy import copy, deepcopy

from torch import Tensor
from typing import Optional, Union, Any

import sympy
import random

# TODO: support online_func extern_input_tensor


def plus_count(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.count += 1
        for arg in args:
            if isinstance(arg, SymbolScalar):
                arg.count += 1
            # else:
            #     print(arg)
        for key, arg in kwargs.items():
            if isinstance(arg, SymbolScalar):
                arg.count += 1
            # else:
            #     print(arg)
        return func(self, *args, **kwargs)
    return wrapper


class SymbolScalar:
    """
    symbolic scalar/array
    """
    def __init__(self, varname: str, value: Node, prev=[], shape_idx: list = ["block_M"],
                 require_grad: bool = True, dtype="float"):
        self.varname = varname
        self.code = value
        self.prev = prev
        self.shape_idx = [str(i) for i in shape_idx]
        self.require_grad = require_grad

        self.count = 0
        self.use_list = []
        # for reuse optimization
        self.allow_reuse = True
        # for lower
        self.lowered = False
        self.visit_count = 0

        self.grad = None  # SymbolicScalar
        
        self.dtype = dtype
    
    def set_allow_reuse(self, allow_reuse: bool):
        """
        set reuse optimization
        """
        self.allow_reuse = allow_reuse
        
    def __repr__(self):
        return f"SymbolScalar({self.varname}, {self.code}, {self.prev}, {self.shape_idx}, {self.require_grad}, {self.dtype})"

    @property
    def name(self):
        return self.varname
    
    @property
    def shape(self):
        # cannot use sympy.symbols() because of constant
        shapes = [sympy.simplify(sh_idx) for sh_idx in self.shape_idx]
        return shapes
    
    # @plus_count # TODO: plus count bug
    def op(self, code: Type[Node], others: list = [],
           shape_idx: list = None, varname_suffix: str = None):
        """
        op for symbolic scalar
        """
        for i, other in enumerate(others):
            # if other is python scalar
            if isinstance(other, (int, float)):
                others[i] = SymbolicConst(other)
            else:
                assert isinstance(other, SymbolScalar)
        if shape_idx is None:
            shape_idx = self.shape_idx
        output_varname = self.varname
        if varname_suffix is not None:
            output_varname = f"{output_varname}_{varname_suffix}"

        code = code(*[x.code for x in [self] + others])
        # TODO: now must be var+1 not 1+var
        output = self.__class__(
            f"{output_varname}_{self.count}",
            code,
            [self] + others,
            shape_idx)
        self.use_list.append(output)
        self.count += 1
        # print(self.count)
        # print(len(self.use_list))
        # if self.count != len(self.use_list):
        #     print(self,self.varname)
        for other in others:
            other.use_list.append(output)
            other.count += 1
            # print(other.count)
            # print(len(other.use_list))
            # if other.count != len(other.use_list):
            #     print(other,other.varname)
        return output

    def backward(self, grad=None):  # SymbolicScalar
        if grad:
            self.grad = grad
        self._backward(self.grad)
        for node in self.prev:
            node.backward()

    def _backward(self, grad):  # symblocscalar
        if self.code.type == "Var" or self.code.type == "Const":
            return
        if self.code.type == "Add":
            grad_0 = copy(grad)
            grad_1 = copy(grad)
            if self.prev[0].require_grad:
                if self.prev[0].grad:
                    grad_0 = grad_0 + self.prev[0].grad
                self.prev[0].grad = grad_0
            if self.prev[1].require_grad:
                if self.prev[1].grad:
                    grad_1 = grad_1 + self.prev[1].grad
                self.prev[1].grad = grad_1
        elif self.code.type == "Mul":
            if self.prev[0].require_grad:
                grad0 = grad * self.prev[1]
                if self.prev[0].grad:
                    grad0 = grad0 + self.prev[0].grad
                self.prev[0].grad = grad0
            if self.prev[1].require_grad:
                grad1 = grad * self.prev[0]
                if self.prev[1].grad:
                    grad1 = grad1 + self.prev[1].grad
                self.prev[1].grad = grad1

        elif self.code.type == "Div":
            if self.prev[1].require_grad or self.prev[0].require_grad:
                grad0 = grad / self.prev[1]
                if self.prev[0].require_grad:
                    if self.prev[0].grad:
                        grad0 = grad0 + self.prev[0].grad
                    self.prev[0].grad = grad0
                if self.prev[1].require_grad:
                    grad1 = - grad0 * self.prev[0] / self.prev[1]
                    if self.prev[1].grad:
                        grad1 = grad1 + self.prev[1].grad
                    self.prev[1].grad = grad1

        elif self.code.type == "Tanh":
            if self.prev[0].require_grad:
                # grad_t = self * grad
                # grad_t = self * grad_t
                # grad0 = grad - grad_t
                grad0 = grad - grad * self * self
                if self.prev[0].grad:
                    grad0 = grad0 + self.prev[0].grad
                self.prev[0].grad = grad0
        elif self.code.type == "Max":
            # TODO: max backward,implement is then else lower
            if self.prev[0].require_grad:
                grad0 = grad.maxbwd(self, self.prev[1])
                if self.prev[0].grad:
                    grad0 = grad0 + self.prev[0].grad
                self.prev[0].grad = grad0
            if self.prev[1].require_grad:
                grad1 = grad.maxbwd(self, self.prev[0])
                if self.prev[1].grad:
                    grad1 = grad1 + self.prev[1].grad
                self.prev[1].grad = grad1

        elif self.code.type == "Log":
            if self.prev[0].require_grad:
                grad0 = grad / self.prev[0]
                if self.prev[0].grad:
                    grad0 = grad0 + self.prev[0].grad
                self.prev[0].grad = grad0
        else:
            raise NotImplementedError(
                f"backward for {self.code.type} is not implemented")
        # change shape_idx
        for idx, node in enumerate(self.prev):
            if node.require_grad:
                self.prev[idx].grad.shape_idx = self.prev[idx].shape_idx

    def clear_usecount(self):
        self.count = 0
        self.use_list.clear()

    def clear_visit(self):
        self.visit_count = 0
        self.lowered = False

    def clear_codegen(self):
        self.clear_usecount()
        self.clear_visit()

    def __add__(self, other):
        return self.op(Add, [other])

    def __neg__(self):
        return self.op(Neg)

    def __sub__(self, other):
        return self.op(Sub, [other])

    def __mul__(self, other):
        return self.op(Mul, [other])

    def __truediv__(self, other):
        return self.op(Div, [other])

    def abs(self):
        return self.op(Abs)

    def tanh(self):
        return self.op(Tanh)

    def exp(self):
        return self.op(Exp)

    def exp2(self):
        return self.op(Exp2)

    def log(self):
        return self.op(Log)
    
    def log2(self):
        return self.op(Log2)

    def max(self, other):
        return self.op(Max, [other])
    
    def maxbwd(self, other1, other2):
        return self.op(MaxBwd, [other1, other2])


class SymbolicArray(SymbolScalar):
    """
    2D Array for OnlineFunc.online_fwd
    """

    def __init__(self, varname: str = "", code: Node = Var(" "),
                 prev=[], shape_idx: list = ["block_M", "block_N"]):
        super().__init__(varname, code, prev, shape_idx)

    def get_reduce(self, op: Literal["sum", "max", "abssum"]):
        """
        get row reduce result of array
        """
        shape_idx = self.shape_idx[:-1]

        if op == "sum":
            return self.op(
                ReduceSum, shape_idx=shape_idx, varname_suffix="sum")
        elif op == "max":
            return self.op(
                ReduceMax, shape_idx=shape_idx, varname_suffix="max")
        elif op == "abssum":
            return self.op(
                ReduceAbsSum, shape_idx=shape_idx, varname_suffix="abssum")
        else:
            raise NotImplementedError

class SymbolicColReduceArray(SymbolScalar):
    """
    2D Array for OnlineFunc.combine
    """
    def __init__(self, varname: str = "", code: Node = Var(" "),
                 prev=[], shape_idx: list = ["block_M", "block_N"]):
        super().__init__(varname, code, prev, shape_idx)
        
    def get_reduce(self, op: Literal["sum", "max", "abssum"]):
        """
        get col reduce result of array
        """
        shape_idx = self.shape_idx[:-2] + [self.shape_idx[-1]]

        if op == "sum":
            return self.op(
                ColReduceSum, shape_idx=shape_idx, varname_suffix="sum")
        elif op == "max":
            return self.op(
                ColReduceMax, shape_idx=shape_idx, varname_suffix="max")
        elif op == "abssum":
            return self.op(
                ColReduceAbsSum, shape_idx=shape_idx, varname_suffix="abssum")
        else:
            raise NotImplementedError
        
class SymbolicTensor(SymbolScalar):
    """
    Tensor for CustomIO
    """

    def __init__(self, varname: str, shape: tuple):
        # convert shape to shape_idx
        super().__init__(
            varname, Var(varname), shape_idx=[
                str(i) for i in shape])
        # self.shape = shape


class SymbolicConst(SymbolScalar):
    """
    Const for constant value
    """

    def __init__(self, value):
        # TODO: not float const # str(value)+(".f" if isinstance(value, int)
        # else "f")
        super().__init__(f"float({str(value)})", Const(value), prev=[], shape_idx=[],
                         require_grad=False)


class CustomIO:
    def __init__(self, input_tensors: dict[str, tuple] = {}):
        self.input_tensors: dict[str, SymbolicTensor] = {}
        for k, v in input_tensors.items():
            self.input_tensors[k] = SymbolicTensor(k, v)

    def __call__(self, tensor_name: str, tensor_shape: tuple):
        if tensor_name in self.input_tensors:
            raise ValueError(f"Tensor {tensor_name} already exists")
        # self.input_tensors[tensor_name].shape = tensor_shape
        self.input_tensors[tensor_name].shape_idx = [str(i) for i in tensor_shape]

        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)

                return result
            return wrapper
        return decorator


def create_mask(
    mod_fn,
    B: Optional[int],
    H: Optional[int],
    Q_LEN: int,
    KV_LEN: int,
    device: str = "cuda",
) -> torch.Tensor:
    r"""This function creates a mask tensor from a mod_fn function.

    Args:
        mod_fn (Union[_score_mod_signature, _mask_mod_signature]): Function to modify attention scores.
        B (int): Batch size.
        H (int): Number of query heads.
        Q_LEN (int): Sequence length of query.
        KV_LEN (int): Sequence length of key/value.
        device (str): Device to run the mask creation on.

    Returns:
        mask (Tensor): A mask tensor with shape (B, H, M, N).
    """
    if B is None:
        B = 1
    if H is None:
        H = 1
    b = torch.arange(0, B, device=device)
    h = torch.arange(0, H, device=device)
    m = torch.arange(0, Q_LEN, device=device)
    n = torch.arange(0, KV_LEN, device=device)

    # with TransformGetItemToIndex():
    #     # elif mod_type == _ModificationType.MASK_MOD:
    #     mask_mod = mod_fn
    #     mask_mod = _vmap_for_bhqkv(mask_mod, prefix=())
    #     mask = mask_mod(b, h, m, n)
    #     return mask
    
    dimensions = [
        (None, None, None, 0),
        (None, None, 0, None),
        (None, 0, None, None),
        (0, None, None, None),
    ]
    for dim in dimensions:
        mod_fn = torch.vmap(mod_fn, in_dims=dim, out_dims=0)
    mask = mod_fn(b, h, m, n)
    return mask

_DEFAULT_SPARSE_BLOCK_SIZE = 128

def _broadcast_to_dim(x, dim):
    while x.dim() < dim:
        x = x.unsqueeze(0)
    return x

def _round_up_to_multiple(x, multiple):
    return (x + multiple - 1) // multiple * multiple

def _convert_mask_to_block_mask(
    mask: torch.Tensor,
    Q_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
    KV_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
    separate_full_blocks: bool = False,
) -> tuple[Tensor, Optional[Tensor]]:
    assert mask.dtype == torch.bool
    mask = _broadcast_to_dim(mask, 4)

    def padding_needed_for_multiple(x, multiple):
        return _round_up_to_multiple(x, multiple) - x

    mask = torch.nn.functional.pad(
        mask,
        (
            0,
            padding_needed_for_multiple(mask.shape[-1], KV_BLOCK_SIZE),
            0,
            padding_needed_for_multiple(mask.shape[-2], Q_BLOCK_SIZE),
        ),
    )
    B, H, Q, KV = mask.shape
    assert Q % Q_BLOCK_SIZE == 0
    assert KV % KV_BLOCK_SIZE == 0
    mask = mask.view(
        B, H, Q // Q_BLOCK_SIZE, Q_BLOCK_SIZE, KV // KV_BLOCK_SIZE, KV_BLOCK_SIZE
    )  # [B, H, Q//Q_BLOCK_SIZE, Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE, KV_BLOCK_SIZE]
    mask = mask.permute(
        0, 1, 2, 4, 3, 5
    )  # [B, H, Q//Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE, Q_BLOCK_SIZE, KV_BLOCK_SIZE]
    mask_block_sum = mask.sum(
        dim=[-2, -1]
    )  # [B, H, Q//Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE]
    if separate_full_blocks:
        full_block_sum = Q_BLOCK_SIZE * KV_BLOCK_SIZE
        full_blocks = mask_block_sum == full_block_sum
        partial_blocks = (mask_block_sum > 0) & (mask_block_sum < full_block_sum)
        partial_blocks = partial_blocks.to(dtype=torch.int8)
        full_blocks = full_blocks.to(dtype=torch.int8)
        return partial_blocks, full_blocks
    else:
        partial_blocks = mask_block_sum > 0
        partial_blocks = partial_blocks.to(dtype=torch.int8)
        return partial_blocks, None

def is_causal_mask(mask_tensor, block_M, block_N):
    """
    mask_tensor: (B,H,seqlen//BLOCKM,seqlen//BLOCKN)
    
    return:
    True: mask_tensor is a causal mask
    False: mask_tensor is not a causal mask
    """
    # 获取张量的形状
    B, H, M, N = mask_tensor.shape
    q_idx = torch.arange(M).unsqueeze(-1)  # 创建一个列向量
    kv_idx = torch.arange(N).view(1,-1)  # 创建一个行向量
    # 创建一个下三角掩码
    mask = (q_idx+1)*block_M > kv_idx*block_N
    # 判断是否相等
    return torch.all(mask_tensor.bool() == mask.to(mask_tensor.device))

def is_less_causal_mask(mask_tensor, block_M, block_N):
    """
    mask_tensor: (B,H,seqlen//BLOCKM,seqlen//BLOCKN)
    
    return:
    True: mask_tensor is a less causal mask
    False: mask_tensor is not a less causal mask
    """
    # 获取张量的形状
    B, H, M, N = mask_tensor.shape
    q_idx = torch.arange(M).unsqueeze(-1)  # 创建一个列向量
    kv_idx = torch.arange(N).view(1,-1)  # 创建一个行向量
    # 创建一个上三角掩码
    mask = (q_idx+1)*block_M-1 < (kv_idx)*block_N
    filter_tensor = mask_tensor[...,mask]
    is_all_zero = torch.all(filter_tensor == 0)
    return is_all_zero


BLOCK_SIZE = 128
def create_block_mask(mask_mod, B, H, QLen, KVLen, device, Q_BLOCK_SIZE=None, KV_BLOCK_SIZE=None):
    if Q_BLOCK_SIZE is None:
        Q_BLOCK_SIZE = BLOCK_SIZE
    if KV_BLOCK_SIZE is None:
        KV_BLOCK_SIZE = BLOCK_SIZE
    mask_tensor = create_mask(mask_mod, B, H, QLen, KVLen, device)
    partial_block_mask, full_block_mask = _convert_mask_to_block_mask(
        mask_tensor,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        separate_full_blocks=False,
    )
    # TODO: sparse block
    
    return partial_block_mask

def create_block_idx(mask_mod, B, H, QLen, KVLen, device, Q_BLOCK_SIZE=None, KV_BLOCK_SIZE=None):
    block_mask = create_block_mask(mask_mod, B, H, QLen, KVLen, device, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
    block_idx = torch.nonzero(block_mask, as_tuple=False)
    # TODO
    return block_idx
