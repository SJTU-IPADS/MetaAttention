from attn_engine import AttentionEngine
import torch
import math
from attn_engine import OnlineFunc
from core import CustomIO
from core import SymbolicArray, SymbolScalar, SymbolicTensor
from core import Var
from core.utils import meta_tensor



D = 64
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
if __name__ == "__main__":
    B, H ,S, D = 64,6,1024,D
    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=torch.float16),
        meta_tensor(B, H, S, D, dtype=torch.float16),
        meta_tensor(B, H, S, D, dtype=torch.float16),
    )

    mod = AttentionEngine(
        qkv_meta,
        custom_fwd_inputs, score_mod=score_mod, mask_mod=None,
        online_func=None, backend="cute"
    )
    from benchmark.bench_utils import do_bench_reluattn
    do_bench_reluattn(mod, B, H, S, D, D)

