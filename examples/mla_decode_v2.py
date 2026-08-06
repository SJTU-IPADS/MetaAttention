from attn_engine import AttentionEngine
import torch
import math
from attn_engine import OnlineFunc
from core import CustomIO
from core import SymbolScalar
from core import Var
from core import meta_tensor

"""
Example of mla attention decode with online softmax
"""

def mla_decode(B, H, SKV, D, DV, HK, HV, SQ=1, dtype=torch.bfloat16):
    
    softmax_scale = 1/D ** 0.5
    # elementwise on attention scores
    def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
        return score * softmax_scale

    class OnlineSoftmax(OnlineFunc):
        def __init__(self):
            """
            define online_rowscales and final_rowscales
            """
            online_rowscales = {
                "m": SymbolScalar("m", Var("-inf")),
                "r": SymbolScalar("r", Var("0.0")),
            }
            final_rowscales = {
                "lse": SymbolScalar("lse", Var("0.0")),
            }
            external_fwd_inputs = CustomIO()
            super().__init__(online_rowscales, final_rowscales,
                        external_fwd_inputs)
        

        # scan
        @staticmethod
        def online_fwd(scores, online_rowscales, b, h, q_idx):

            m , r = online_rowscales["m"], online_rowscales["r"]
            m_new = m.max(scores.get_reduce("max"))
            scale_tmp = (m - m_new).exp()
            r = r * scale_tmp
            
            scores = (scores - m_new).exp()
            r = r + scores.get_reduce("sum")

            new_online_rowscales = {
                "m": m_new,
                "r": r,
            }
            o_scale = scale_tmp
            return scores, new_online_rowscales, o_scale
        
        @staticmethod
        def combine(final_rowscales, ):
            lse = final_rowscales["lse"]
            lse_max = lse.get_reduce("max")
            row_sum = (lse - lse_max).exp()
            row_sum_sum = row_sum.get_reduce("sum")
            lse_sum = row_sum_sum.log() + lse_max
            o_scale = (lse - lse_sum).exp()
            return o_scale

        @staticmethod
        def online_fwd_epilogue(o, online_rowscales, b, h, q_idx):
            o_new = o / online_rowscales["r"]
            lse = (online_rowscales["r"]).log() + online_rowscales["m"]
            final_rowscales = {
                "lse": lse,
            }
            return o_new, final_rowscales

        @staticmethod
        def forward(scores, final_rowscales, b, h, q_idx, kv_idx):
            lse = final_rowscales["lse"]
            scores_new = (scores-lse).exp()
            return scores_new
        
        @staticmethod
        def backward(dp, scores, final_rowscales, doosum_rowscales, b, h, q_idx, kv_idx):
            dppsum = doosum_rowscales
            dscores = (dp - dppsum)*scores 
            return dscores

    qkv_meta = (
        meta_tensor(B, H, SQ, D, dtype=dtype),
        meta_tensor(B, HK, SKV, D, dtype=dtype),
        meta_tensor(B, HV, SKV, DV, dtype=dtype),
        )

    custom_fwd_inputs = CustomIO({
        
    })

    online = OnlineSoftmax()
    mod = AttentionEngine(
        qkv_meta,
        custom_fwd_inputs, score_mod=score_mod, mask_mod=None,
        online_func=online,
        kv_shared=True,
        backend="cute",
    )
    
    return mod
