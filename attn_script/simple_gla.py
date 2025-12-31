from attn_engine import LinearAttentionEngine
from core import SymbolicTensor
from core import CustomIO
from core.utils import meta_tensor

import torch
import torch.nn.functional as F 
import math

"""
Example of simple gla/gated retnet

fwd:
input:
Q: [B, H, T, D]
K: [B, H, T, D]
V: [B, H, T, DV]
decay: [B, H, T]
...custom_inputs

output: 
O: [B, H, T, DV]
"""

D = 256
scale = 1 / D**0.5
def q_mod(q, custom_io):
    return q * scale

def eval():
    import itertools
    BHSDDVs = itertools.product(
        # [64,],# [1, 16, 64],
        # [8,16],
        # [1024,2048,4096],
        # [64,],
        # [64,]
        
        # [64],
        # [24],
        # [1024,2048,4096],
        # [128,],
        # [128,]
        
        # [8], # 64
        # [40],
        # [1024,2048,4096],
        # [256,],
        # [256,]
        
        [1,8],
        [32],
        [2048,4096],
        [256,],
        [512,]
        
    )
    
    for B,H,S,D,DV in BHSDDVs:
        dtype = torch.bfloat16
        qkv_meta = (
            meta_tensor(B, H, S, D, dtype=dtype),
            meta_tensor(B, H, S, D, dtype=dtype),
            meta_tensor(B, H, S, DV, dtype=dtype),
        )
        custom_io = CustomIO(
            {
            }
        )
        mod = LinearAttentionEngine(qkv_meta, q_mod=q_mod,
                                custom_io = custom_io,
                                tune=False, tune_filename="simple_gla",
                                tune_bwd=True)

        from benchmark.bench_utils import do_bench_simple_gla
        print(f"eval B={B}, H={H}, S={S}, D={D}, DV={DV}")
        # try:
        do_bench_simple_gla(mod, B, H, S, D, DV, BT=64, requires_grad=True)
        # except Exception as e:
        #     print("bench failed", e)
        #     continue
        
if __name__ == "__main__":
    B, H, S, D, DV = 1, 32, 2048, D, 512
    dtype = torch.bfloat16
    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, DV, dtype=dtype),
    )
    custom_io = CustomIO(
        {
        }
    )
    mod = LinearAttentionEngine(qkv_meta, q_mod=q_mod,
                            custom_io = custom_io,
                            tune=True, tune_filename="simple_gla",
                            tune_bwd=False)

    from benchmark.bench_utils import do_bench_simple_gla
    do_bench_simple_gla(mod, B, H, S, D, DV, BT=64, requires_grad=True)
    # eval()
