"""Dense/causal MHA and GQA expressed with the MetaAttention online-softmax DSL.

Metadata is BHSD; runtime tensors are contiguous BSHD. Set ``kv_heads=heads``
for MHA, or fewer KV heads for GQA. Both forward and backward are generated.
"""

import torch

from attn_engine import AttentionEngine, OnlineFunc
from core import CustomIO, SymbolScalar, Var, meta_tensor


def softmax_attention(
    batch, heads, seqlen, dim, *, kv_heads=None, causal=False, dtype=torch.float16
):
    """Construct fixed-length self-attention with scale ``dim**-0.5``."""
    kv_heads = heads if kv_heads is None else kv_heads
    if kv_heads <= 0 or heads % kv_heads:
        raise ValueError("heads must be a positive multiple of kv_heads")

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
        return score * dim**-0.5

    class OnlineSoftmax(OnlineFunc):
        def __init__(self):
            super().__init__(
                {
                    "m": SymbolScalar("m", Var("-inf")),
                    "r": SymbolScalar("r", Var("0.0")),
                },
                {"lse": SymbolScalar("lse", Var("0.0"))},
                CustomIO(),
            )

        @staticmethod
        def online_fwd(scores, online_rowscales, b, h, q_idx):
            m, r = online_rowscales["m"], online_rowscales["r"]
            m_new = m.max(scores.get_reduce("max"))
            o_scale = (m - m_new).exp()
            scores = (scores - m_new).exp()
            r_new = r * o_scale + scores.get_reduce("sum")
            return scores, {"m": m_new, "r": r_new}, o_scale

        @staticmethod
        def online_fwd_epilogue(o, online_rowscales, b, h, q_idx):
            r, m = online_rowscales["r"], online_rowscales["m"]
            return o / r, {"lse": r.log() + m}

        @staticmethod
        def forward(scores, final_rowscales, b, h, q_idx, kv_idx):
            return (scores - final_rowscales["lse"]).exp()

        @staticmethod
        def backward(
            dp, scores, final_rowscales, doosum_rowscales, b, h, q_idx, kv_idx
        ):
            return (dp - doosum_rowscales) * scores

    return AttentionEngine(
        (
            meta_tensor(batch, heads, seqlen, dim, dtype=dtype),
            meta_tensor(batch, kv_heads, seqlen, dim, dtype=dtype),
            meta_tensor(batch, kv_heads, seqlen, dim, dtype=dtype),
        ),
        CustomIO({}),
        score_mod=score_mod,
        mask_mod=causal_mask if causal else None,
        online_func=OnlineSoftmax(),
        backend="tl",
    )
