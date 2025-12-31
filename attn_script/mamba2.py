from attn_engine import LinearAttentionEngine
from core import SymbolicTensor
from core import CustomIO
from core import meta_tensor

import torch
import torch.nn.functional as F 
import math

"""
Example of mamba2 SSD

fwd:
input:
Q: [B, H, T, D]
K: [B, H, T, D]
V: [B, H, T, DV]
decay: [B, H, T] or [B, H, T, D] (TODO) float32
...custom_inputs

output: 
O: [B, H, T, DV]
"""

# A = SymbolicTensor("A", shape=(1, H))
def decay_mod(decay, custom_io): # (B,H,seqlen)
    A = custom_io.input_tensors["A"]
    return (decay*A)

# dt = SymbolicTensor("dt", shape=(B, H, T))
def k_mod(k, custom_io): # (B,H,seqlen, D)
    dt = custom_io.input_tensors["dt"]
    return k * dt

if __name__ == "__main__":
    B, H, T, D, DV = 1, 80, 2048, 128, 64 # bug 16384
    dtype = torch.bfloat16
    qkv_meta = (
        meta_tensor(B, H, T, D, dtype=dtype),
        meta_tensor(B, H, T, D, dtype=dtype),
        meta_tensor(B, H, T, DV, dtype=dtype),
    )
    custom_io = CustomIO(
        {
            "A": (1, H),
            "dt": (B, H, T)
        }
    )
    mod = LinearAttentionEngine(
        qkv_meta,
        decay_mod=decay_mod, k_mod=k_mod,
                            custom_io = custom_io)

    from benchmark.bench_utils import do_bench_mamba
    do_bench_mamba(mod, B, H,H,H, T, D, DV, BT=64)
