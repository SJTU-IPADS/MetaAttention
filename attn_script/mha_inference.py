from attn_engine import AttentionEngine
import torch
import math
from attn_engine import OnlineFunc
from core import CustomIO
from core import SymbolicArray, SymbolScalar, SymbolicTensor
from core import Var
from core.utils import meta_tensor

"""
Example of attention decode with online softmax
"""


D = 192
softmax_scale = 1/D ** 0.5
# elementwise on attention scores
def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
    # softmax_scale = custom_fwd_inputs.input_tensors["softmax_scale"]
    return score * softmax_scale

class OnlineSoftmax(OnlineFunc):
    def __init__(self):
        """
        define online_rowscales and final_rowscales
        """
        online_rowscales = {
            "m": SymbolScalar("m", Var("-inf")), # -inf 
            "r": SymbolScalar("r", Var("0.0")),
        }
        final_rowscales = {
            "lse": SymbolScalar("lse", Var("0.0")), # not used in codegen
        }
        external_fwd_inputs = CustomIO()
        # external_bwd_inputs = CustomIO({
        #     "dppsum": (B,H,S),
        # })
        super().__init__(online_rowscales, final_rowscales,
                    external_fwd_inputs) # , external_bwd_inputs)
    

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

    # reduce
    # @staticmethod
    # def combine(online_rowscales, ):
    #     m = online_rowscales["m"]
    #     r = online_rowscales["r"]
    #     m_max = m.get_reduce("max")
    #     scale_tmp = (m - m_max).exp()
    #     r_new = r * scale_tmp
    #     r_sum = r_new.get_reduce("sum")
    #     o_scale = scale_tmp / r_sum
    #     return o_scale
    
    @staticmethod
    def combine(final_rowscales, ):
        lse = final_rowscales["lse"]
        lse_max = lse.get_reduce("max")
        row_sum = (lse - lse_max).exp2()
        row_sum_sum = row_sum.get_reduce("sum")
        lse_sum = row_sum_sum.log2() + lse_max
        o_scale = (lse - lse_sum).exp2()
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
        dppsum = doosum_rowscales # external_bwd_tensor.input_tensors["dppsum"]
        dscores = (dp - dppsum)*scores # TODO: bug if swap
        return dscores

if __name__ == "__main__":
    B, H ,S, D, DV = 8, 16, 8192, D, 128
    dtype = torch.float16 # performance regression for bfloat16
    qkv_meta = (
        meta_tensor(B, H, 128, D, dtype=dtype),
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, DV, dtype=dtype),
    )

    custom_fwd_inputs = CustomIO({
        # "softmax_scale": (1,),
    })

    online = OnlineSoftmax()
    mod = AttentionEngine(
        qkv_meta,
        custom_fwd_inputs, score_mod=score_mod, mask_mod=None,
        online_func=online,
        # tune=False, tune_file="mha_tune.json"
    )

    from benchmark.bench_utils import do_bench_attention
    do_bench_attention(mod, B, H, S, D, DV, dtype=dtype, causal=False, seqlenq=128)
