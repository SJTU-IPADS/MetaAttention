from attn_engine import LinearAttentionEngine
from core import SymbolicTensor
from core import CustomIO
from core import meta_tensor
from autotuner.arch import get_attn_device

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

def mamba2(B, HQ, S, D, DV, HK=None, HV=None, dtype=torch.bfloat16, tune=False):
    if HK is None:
        HK = HQ
    if HV is None:
        HV = HQ
    def decay_mod(decay, custom_io): # (B,H,seqlen)
        A = custom_io.input_tensors["A"]
        return (decay*A)

    def v_mod(v, custom_io): # (B,H,seqlen, D)
        dt = custom_io.input_tensors["dt"]
        return v * dt


    qkv_meta = (
        meta_tensor(B, HQ, S, D, dtype=dtype),
        meta_tensor(B, HK, S, D, dtype=dtype),
        meta_tensor(B, HV, S, DV, dtype=dtype),
    )
    custom_io = CustomIO(
        {
            "A": (1, "heads"),
            "dt": ("batch", "heads", "seq_len")
        }
    )
    attn_device = get_attn_device()
    mod = LinearAttentionEngine(qkv_meta,
        decay_mod=decay_mod, v_mod=v_mod,
                                custom_io = custom_io,
                                tune=tune, tune_filename=f"tuned_config/{attn_device.name}/mamba2",
                                tune_bwd=tune)
    return mod
