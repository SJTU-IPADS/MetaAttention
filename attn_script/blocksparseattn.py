from attn_engine import AttentionEngine
import torch
import math
from attn_engine import OnlineFunc
from core import CustomIO
from core import SymbolicArray, SymbolScalar, SymbolicTensor
from core import Var
from core import meta_tensor

"""
Example of causal attention with online softmax
"""



BLOCK = 128
def generate_block_mask_and_mask():
    def mask_mod(b, h, q_idx, kv_idx):
        return torch.logical_and(q_idx//BLOCK >= kv_idx//BLOCK, q_idx//BLOCK < kv_idx//BLOCK + 2)
    from core.transform.core import create_mask, create_block_mask
    block_mask = create_block_mask(mask_mod, B, H, S, S, device="cuda", Q_BLOCK_SIZE=BLOCK, KV_BLOCK_SIZE=BLOCK)
    mask = create_mask(mask_mod, B, H, S, S, device="cuda")
    return block_mask, mask

def true_mask_mod(b, h, q_idx, kv_idx):
    return True

D = 128
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
    B, H ,S, D, DV = 1,32,2048,D, 128
    dtype = torch.float16 # performance regression for bfloat16
    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=dtype),
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
        tune=False, tune_file="mha_tune.json",
        extern_block_mask=True,
    )

    # test sliding window attention
    test_bwd = True
    torch.cuda.manual_seed(0)
    q = torch.randn(B, S, H, D, dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn(B, S, H, D, dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn(B, S, H, DV, dtype=dtype, device="cuda", requires_grad=True)
    block_mask, mask = generate_block_mask_and_mask()
    o = mod(q,k,v, block_mask=block_mask)
    do = torch.randn_like(o, dtype=dtype, device="cuda")
    if test_bwd:
        o.backward(do, retain_graph=True)
    
    def ref_attn(q,k,v):
        attn = torch.einsum("bqhd,bkhd->bhqk", q, k)
        attn = (attn * (1/D ** 0.5)).float()
        # set to -inf for not masked positions
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        return torch.einsum("bhqk,bkhd->bqhd", attn.to(dtype), v)
    
    ref_o = ref_attn(q,k,v)
    print(torch.allclose(o, ref_o, atol=1e-2, rtol=1e-2))
    from benchmark.bench_utils import print_debug
    # print_debug(o[:,:,:,:], ref_o[:,:,:,:], atol=1e-2, rtol=1e-2)
    
    if test_bwd:
        q_grad, k_grad, v_grad = q.grad.clone(), k.grad.clone(), v.grad.clone()
        q.grad.zero_()
        k.grad.zero_()
        v.grad.zero_()
        ref_o.backward(do)
        print(torch.allclose(q.grad, q_grad, atol=1e-2, rtol=1e-2))
        print(torch.allclose(k.grad, k_grad, atol=1e-2, rtol=1e-2))
        print(torch.allclose(v.grad, v_grad, atol=1e-2, rtol=1e-2))
