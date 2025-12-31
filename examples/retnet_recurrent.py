from attn_engine import LinearAttentionEngine
from core import SymbolicTensor
from core import CustomIO
from core import meta_tensor
from autotuner.arch import get_attn_device

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

def retnet_recurrent(B, H, S, D, DV, dtype=torch.bfloat16, tune=False):

    def decay_mod(decay, custom_io): # (B,H,seqlen)
        return decay.log()

    scale = 1 / D**0.5
    def q_mod(q, custom_io):
        return q * scale

    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, DV, dtype=dtype),
    )
    custom_io = CustomIO(
        {
        }
    )
    attn_device = get_attn_device()
    mod = LinearAttentionEngine(qkv_meta, q_mod=q_mod, decay_mod=decay_mod,
                            custom_io = custom_io,
                            tune=tune, tune_filename=f"tuned_config/{attn_device.name}/retention_linear",
                            tune_bwd=tune)

    return mod
 