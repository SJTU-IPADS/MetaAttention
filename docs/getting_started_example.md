# Example for Attention with Online Softmax

The Attention mechnism has the form in pseudo code:
```py
# query: [B, q_len, H, DK]
# key: [B, kv_len, H, DK]
# value: [B, kv_len, H, DV]
scores = query @ key
scores = mask_mod(scores)
scores = score_mod(scores)
p = online_func(scores)
o = p @ value
```

Below is a example for causal softmax Attention forward & backward.
```py
from attn_engine import AttentionEngine
import torch
from attn_engine import OnlineFunc
from core import CustomIO
from core import create_block_mask
from core.utils import meta_tensor

D = 128
softmax_scale = 1/D ** 0.5
# define custom inputs(except Q, K, V) for Attention, here none
custom_fwd_inputs = CustomIO({
})

# define elementwise modification on attention scores
def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
    """
    input: score, custom_fwd_inputs, b, h, q_idx, kv_idx
    output: new score
    """
    return score * softmax_scale


# define online_func: OnlineSoftmax as a subclass of OnlineFunc
class OnlineSoftmax(OnlineFunc):
    def __init__(self):
        """
        define and initialize  online_rowscales and final_rowscales
        """
        online_rowscales = {
            "m": SymbolScalar("m", Var("-inf")),
            "r": SymbolScalar("r", Var("0.0")),
        }
        final_rowscales = {
            "lse": SymbolScalar("lse", Var("0.0")),
        }
        super().__init__(online_rowscales, final_rowscales)
    

    @staticmethod
    def online_fwd(scores, online_rowscales, b, h, q_idx):
        """
        define online forward computing logic.
        input: scores, online_rowscales, b, h, q_idx
        output: new_scores, new_online_rowscales, o_scale
        """

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
    def online_fwd_epilogue(o, online_rowscales, b, h, q_idx):
        """
        define the computation for final output and final_rowscales
        input: o, online_rowscales, b, h, q_idx
        output: o_new, final_rowscales
        """
        o_new = o / online_rowscales["r"]
        lse = (online_rowscales["r"]).log() + online_rowscales["m"]
        final_rowscales = {
            "lse": lse,
        }
        return o_new, final_rowscales

    @staticmethod
    def forward(scores, final_rowscales, b, h, q_idx, kv_idx):
        """
        define the forward computation logic for backward recomputation
        input: scores, final_rowscales, b, h, q_idx, kv_idx
        output: scores_new
        """
        lse = final_rowscales["lse"]
        scores_new = (scores-lse).exp()
        return scores_new
    
    @staticmethod
    def backward(dp, scores, final_rowscales, doosum, b, h, q_idx, kv_idx):
        """
        define the backward computation logic for backward gradient computation
        input: dp, scores, final_rowscales, doosum, b, h, q_idx, kv_idx
        output: dscores
        """
        dppsum = doosum_rowscales
        dscores = (dp - dppsum)*scores
        return dscores

# define mask pattern on attention score
def causal_mask(b, h, q_idx, kv_idx):
    """
    input: the index of scores: b, h, q_idx, kv_idx
    output: Boolean value for mask
    """
    return q_idx >= kv_idx

if __name__ == "__main__":
    # define input shape&type
    B, H ,S, D, DV = 1,128,32768,D, 128
    dtype = torch.float16
    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, DV, dtype=dtype),
    )

    # generate runtime attention op
    mod = AttentionEngine(
        qkv_meta,
        custom_fwd_inputs, score_mod=score_mod, mask_mod=causal_mask,
        online_func=OnlineSoftmax(),
        tune=False, tune_file="mha_tune.json"
    )

    # pytorch call
    q = torch.randn(B, S, H, D, dtype=dtype, device="cuda")
    k = torch.randn(B, S, H, D, dtype=dtype, device="cuda")
    v = torch.randn(B, S, H, DV, dtype=dtype, device="cuda")
    out = mod(q, k, v)
    out.backward(torch.ones_like(out))


    
```

