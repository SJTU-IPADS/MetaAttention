from attn_engine import AttentionEngine
import torch
import math
from attn_engine import OnlineFunc
from core import CustomIO
from core import SymbolicArray, SymbolScalar, SymbolicTensor
from core import Var
from core import meta_tensor

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
    softmax_bias = custom_fwd_inputs.input_tensors["softmax_bias"]
    score = score + softmax_bias
    score = ((score*0.5).tanh() + 1) * 0.5
    return score

# def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
#     return 1 / (1 + (-score).exp())


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
    "softmax_bias": (1,),
})

def eval():
    import itertools
    BHSD = list(
        itertools.product(
            (1,8,),
            (32,),
            (2048,4096,8192),
            (128,)#,128)
        )
    )
    for B, H, S, D in BHSD:
        qkv_meta = (
            meta_tensor(B, H, S, D, dtype=torch.bfloat16),
            meta_tensor(B, H, S, D, dtype=torch.bfloat16),
            meta_tensor(B, H, S, D, dtype=torch.bfloat16),
        )


        mod = AttentionEngine(
            qkv_meta,
            custom_fwd_inputs, score_mod=score_mod, mask_mod=causal_mask,
            online_func=None, backend="cute"
        )

        from benchmark.bench_utils import do_bench_sigmoidattn_cute
        print(f"B={B}, H={H}, S={S}, D={D}")
        do_bench_sigmoidattn_cute(mod, B, H, S, D, D, dtype=torch.bfloat16)

if __name__ == "__main__":
    # B, H ,S, D = 1,32,2048,128 
    # qkv_meta = (
    #     meta_tensor(B, H, S, D, dtype=torch.bfloat16),
    #     meta_tensor(B, H, S, D, dtype=torch.bfloat16),
    #     meta_tensor(B, H, S, D, dtype=torch.bfloat16),
    # )


    # mod = AttentionEngine(
    #     qkv_meta,
    #     custom_fwd_inputs, score_mod=score_mod, mask_mod=causal_mask,
    #     online_func=None, backend="cute"
    # )

    # from benchmark.bench_utils import do_bench_sigmoidattn_cute
    # do_bench_sigmoidattn_cute(mod, B, H, S, D, D, dtype=torch.bfloat16)
    eval()

