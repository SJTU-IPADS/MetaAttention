from attn_engine import AttentionEngine
import torch
import math
from attn_engine import OnlineFunc
from core import CustomIO
from core import SymbolicArray, SymbolScalar, SymbolicTensor
from core import Var
from core import meta_tensor
from autotuner.arch import get_attn_device


def relu_attention(B, H, S, D, DV, dtype=torch.float16, tune=False):
    
    scores_scale = 1/D**0.5
    def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
        score = score * scores_scale
        score = score.max(0)
        return score


    class OnlineIdentity(OnlineFunc):
        def __init__(self):
            online_rowscales = {
            }
            final_rowscales = {
            }
            external_fwd_inputs = CustomIO()
            super().__init__(online_rowscales, final_rowscales,
                        external_fwd_inputs)

        @staticmethod
        def online_fwd(scores, online_rowscales, b, h, q_idx):
            o_scale = SymbolScalar("o_scale", Var("1"))
            return scores, online_rowscales, o_scale
        
        @staticmethod
        def online_fwd_epilogue(o, online_rowscales, b, h, q_idx):
            return o, {}
        
        @staticmethod
        def forward(scores, final_rowscales, b, h, q_idx, kv_idx):
            return scores

        @staticmethod
        def backward(dp, scores, final_rowscales, doosum_rowscales, b, h, q_idx, kv_idx):
            return dp

    custom_fwd_inputs = CustomIO({
    })
    
    
    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, DV, dtype=dtype),
    )
    attn_device = get_attn_device()

    mod = AttentionEngine(
        qkv_meta,
        custom_fwd_inputs, score_mod=score_mod, mask_mod=None,
        online_func=OnlineIdentity(),
        tune=tune, tune_file =f"tuned_config/{attn_device.name}/reluattn_tune.json",
        tune_bwd=tune, tune_file_bwd=f"tuned_config/{attn_device.name}/reluattn_bwd.json"
    )

    return mod