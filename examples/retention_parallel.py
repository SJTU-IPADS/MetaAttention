from attn_engine import AttentionEngine
import torch
import math
from attn_engine import OnlineFunc
from core import CustomIO
from core import SymbolicArray, SymbolScalar, SymbolicTensor
from core import Var
from core import meta_tensor
from autotuner.arch import get_attn_device

"""
Example of retnet Attention

# query: [B, H, q_len, DK]
# key: [B, H, kv_len, DK]
# value: [B, H, kv_len, DV]

def retnet_attention(query, key, value):
    scores = query @ key
    scores = scores * mask
    p = scores / scores.abs().sum().clamp(min=1)
    o = p @ value
"""

def retention_parallel(B, H, S, D, DV, dtype=torch.float16, tune=False):
    
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    softmax_scale = D**0.5
    def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
        mask = custom_fwd_inputs.input_tensors["mask"]
        return score * mask

    class OnlineRetention(OnlineFunc):
        def __init__(self):
            online_rowscales = {
                "r_wo_clamp": SymbolScalar("r_wo_clamp", Var("0.0")), # 0.0
                "r": SymbolScalar("r", Var("0.0")), # 0.0
            }
            final_rowscales = {
                "r": SymbolScalar("r", Var("0.0")), # 0.0
            }
            external_fwd_inputs = CustomIO()
            super().__init__(online_rowscales, final_rowscales,
                            external_fwd_inputs)
        
        @staticmethod
        def online_fwd(scores,online_rowscales, b, h, q_idx):
            r_wo_clamp = online_rowscales["r_wo_clamp"]
            r = online_rowscales["r"]
            # r_wo_clamp = r_wo_clamp + scores.abs().get_reduce("sum")
            r_wo_clamp = r_wo_clamp + scores.get_reduce("abssum")
            r_new = r_wo_clamp.max(1.0)
            o_scale = r / r_new

            scores = scores / r_new

            new_online_rowscales = {
                "r_wo_clamp": r_wo_clamp,
                "r": r_new,
            }

            return scores, new_online_rowscales, o_scale
        
        @staticmethod
        def online_fwd_epilogue(o, online_rowscales, b, h, q_idx):
            r = online_rowscales["r"]
            final_rowscales = {
                "r": r,
            }
            return o, final_rowscales


        @staticmethod
        def forward(scores, final_rowscales, b, h, q_idx, kv_idx):
            r = final_rowscales["r"]
            scores_new = scores / r
            return scores_new
        
        @staticmethod
        def backward(dp, scores, final_rowscales, doosum_rowscales, b, h, q_idx, kv_idx):

            dscores = dp / final_rowscales["r"]

            return dscores


    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=torch.float16),
        meta_tensor(B, H, S, D, dtype=torch.float16),
        meta_tensor(B, H, S, DV, dtype=torch.float16),
    )


    custom_fwd_inputs = CustomIO({
        "mask": (1, "heads", "seq_len", "seq_len_kv"), # (1, H, S, S),
    })

    online = OnlineRetention()
    attn_device = get_attn_device()
    mod = AttentionEngine(
        qkv_meta,
        custom_fwd_inputs, score_mod=score_mod, mask_mod=causal_mask,
        online_func=online,
        mask_value="0",
        tune=tune, tune_file=f"tuned_config/{attn_device.name}/retention_parallel_fwd.json",
        # tune_bwd = True, tune_file_bwd = "retention_parallel_bwd.json"
    )
    
    return mod

if __name__ == "__main__":
    # Example usage
    B, H, S, D, DV = 1, 32, 2048, 256, 512
    mod = retention_parallel(B, H, S, D, DV, tune=True)

    print(mod)
    print(f"AttentionEngine Succuessfully created.")
