from attn_engine import LinearAttentionEngine
from core import SymbolicTensor
from core import CustomIO
from core import meta_tensor

import torch
import torch.nn.functional as F 
import math

"""
Example of retention

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

def decay_mod(decay, custom_io): # (B,H,seqlen)
    return decay.log()

D = 256
scale = 1 / D**0.5
def q_mod(q, custom_io):
    return q * scale

def eval():
    import itertools
    BHSDDVs = itertools.product(
        [1,8], # 64
        [32],
        [2048,4096,8192],
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
        mod = LinearAttentionEngine(qkv_meta, q_mod=q_mod, decay_mod=decay_mod,
                                custom_io = custom_io,
                                tune=False, tune_filename="retention_linear",
                                tune_bwd=True)

        from benchmark.bench_utils import do_bench_retention_linear
        print(f"eval B={B}, H={H}, S={S}, D={D}, DV={DV}")
        try:
            do_bench_retention_linear(mod, B, H, S, D, DV, requires_grad=True)
        except Exception as e:
            print("bench failed", e)
            
if __name__ == "__main__":
    B, H, T, D, DV = 1, 20, 1024, D, 512 # bug 16384
    qkv_meta = (
        meta_tensor(B, H, T, D, dtype=torch.bfloat16),
        meta_tensor(B, H, T, D, dtype=torch.bfloat16),
        meta_tensor(B, H, T, DV, dtype=torch.bfloat16),
    )
    custom_io = CustomIO(
        {
        }
    )
    mod = LinearAttentionEngine(
        qkv_meta,
        q_mod=q_mod, decay_mod=decay_mod,
                            custom_io = custom_io,
                            tune=True, tune_filename="retention_linear",
                            tune_bwd=True)

    from benchmark.bench_utils import do_bench_retention_linear
    do_bench_retention_linear(mod, B, H, T, D, DV, requires_grad=True)
    # eval()