# AttentionEngine Frontend

AttentionEngine is a frontend for customized attention mechanism. It generates high-performant kernels comparable to flashAttention. 

## supported customized attention mechanism 

**Attention(sigmoid attn, relu attn) and Linear attention(mamba2, simple gla, retention_linear)**

Any function on attention scores that can be represented as elementwise & row_reduce, such as softmax, retention, sigmoid, etc.

## Detailed usage

```py
AttentionEngine: o = attention_engine(query: Tensor, key: Tensor, value: Tensor,
                            query_mod, key_mod, value_mod,
                            score_mod, block_mask, 
                            online_func)

# It basically does:

scores = query @ key
p = online_func(scores)
o = p @ value

# A more detailed version is:

scores = query_mod(query) @ key_mod(key)
scores = block_mask(scores)
scores = score_mod(scores)
p = online_func(scores)
o = p @ value_mod(value)


# Parameters:

*_mod is user-defined element-wise function, default is identity function.
block_mask is a mask that applied to scores. the block_mask can be implemented with score_mod, but with block_mask, better performance can be achieved.
online_func is a class that has **row-online algorithm** for scores. We have predefined OnlineSoftmax, online_retention, Identity. User can also define their own online algorithm.


# How to define your own online algorithm:

## the online algorithm template:

### online forward: 

online_rowscales = {
    "r0": INIT_VALUE0,
    "r1": INIT_VALUE1,
} 
final_rowscales = {
    "m0": INIT_VALUE,
    "m1": INIT_VALUE,
}
o = torch.zeros(B, H, q_len, D, device="cuda", dtype=torch.float16)
for i in range(kv_len//BLOCK_N):
    scores = q @ k
    scores = block_mask(scores)
    scores = score_mod(scores)
    scores, online_rowscales, o_scale = online_func.online_fwd(scores, online_rowscales)
    o = o*o_scale
    o += scores @ v
    k.advance(BLOCK_N)
    v.advance(BLOCK_N)
o, final_rowscales = online_func.online_fwd_epilogue(o, online_rowscales)

### backward:

scores = q @ k
scores = block_mask(scores)
scores = score_mod(scores)
scores = online_func.forward(scores, final_rowscales)
dv = scores^T @ do
dp = do @ v^T
dscores = online_func.backward(dp, scores)
dk = dscores^T @ q
dq = dscores @ k



## User need to do:
1. Define a class that inherits from OnlineFunc.
2. Implement __init__, online_fwd, online_fwd_epilogue, forward, backward methods.

```
To implement a customized online_func on scores, please go to next section. 

## customized online_func language

- implicit broadcast
- reduce_sum & reduce_max in online_fwd
- dppsum for bwd

## Limitation
- Online_func does not support autodiff, so user need to define the fwd and bwd for online function.
- for backward, only the grad of q, k, v is computed, not including custom input tensor.

## TODO
- block_mask
- b,h,q_idx,kv_idx

