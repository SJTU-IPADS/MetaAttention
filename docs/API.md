# AttentionEngine API


## Custom Attention Level API
This level API is designed for users to define their own attention mechanism. Users needs to define the following components:
- `qkv_meta`: meta information for Q, K, V, such as shape, dtype
- `score_mod`: elementwise modification on attention scores
- `mask_mod`: mask modification on attention scores
- `online_func`: online function for attention scores
- `custom_fwd_inputs`: custom inputs for attention mechanism
```py
mod = AttentionEngine(
    qkv_meta: Tuple[MetaTensor, MetaTensor, MetaTensor],
    score_mod: Callable[[Tensor, CustomIO, int, int, int, int], Tensor],
    custom_fwd_inputs: CustomIO,
    online_func: OnlineFunc,
    mask_mod: Callable[[int, int, int, int], Bool]
)
```

For compiled Attention module, users can use pytorch-compatible API to interact with the module.
```py
output = mod(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    custom_inputs: Optional[List[torch.Tensor]]
)
output.backward(do)
```

### OnlineFunc

OnlineFunc is a class that defines the online function for attention scores, such as online softmax and retention.
```py
class OnlineFunc:
    def __init__(self):
        pass
    def online_fwd(scores, online_rowscales, b, h, q_idx) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
        pass
    def online_fwd_epilogue(o, online_rowscales, b, h, q_idx) -> Tuple[Tensor, Dict[str, Tensor]]:
        pass
    def forward(scores, online_rowscales, b, h, q_idx, kv_idx) -> Tensor:
        pass
    def backward(dp, scores, final_rowscales, doosum, b, h, q_idx, kv_idx) -> Tensor:
        pass
```
Examples can be found in the [Getting-started Example](./getting_started_example.md).

### score_mod
`score_mod` takes the following inputs:
- `score`: attention scores
- `custom_fwd_inputs`: custom input tensors
- `b`: batch index
- `h`: head index
- `q_idx`: query index
- `kv_idx`: key index

`score_mod` returns the modified attention scores.
```
def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx) -> Tensor:
    return new_score
```

Examples can be found in the [Getting-started Example](./getting_started_example.md).

### mask_mod
`mask_mod` takes the following inputs:
- `b`: batch index
- `h`: head index
- `q_idx`: query index
- `kv_idx`: key index

`mask_mod` returns Bool value to indicate whether the attention score should be masked.
```
def mask_mod(b, h, q_idx, kv_idx) -> Bool:
    return True
```


### Customized Linear Attention API

AttentionEngine also support customized linear attention mechanism. Users can define their own linear attention mechanism by defining the following components:
- `qkv_meta`: meta information for Q, K, V, such as shape, dtype
- `q_mod`: elementwise modification on Q
- `k_mod`: elementwise modification on K
- `v_mod`: elementwise modification on V
- `decay_mod`: elementwise modification on decay
- `custom_io`: custom inputs for attention mechanism
```py
mod = LinearAttentionEngine(
    qkv_meta: Tuple[MetaTensor, MetaTensor, MetaTensor],
    q_mod: Callable[[Tensor, CustomIO], Tensor],
    k_mod: Callable[[Tensor, CustomIO], Tensor],
    v_mod: Callable[[Tensor, CustomIO], Tensor],
    decay_mod: Callable[[Tensor, CustomIO], Tensor],
    custom_io: CustomIO
)
output = mod(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    custom_io: Optional[List[torch.Tensor]]
)
output.backward(do)
```



## Gated Delta Rule API

`GDNEngine` is the dedicated H20 training API for Gated Delta Rule. It is not
an algorithm mode of `LinearAttentionEngine`.

```py
from attn_engine import GDNEngine

engine = GDNEngine("cuda")
output, final_state = engine(
    query,               # [B, Hk, T, 128], bfloat16
    key,                 # [B, Hk, T, 128], bfloat16
    value,               # [B, Hv, T, 128], bfloat16
    gate,                # [B, Hv, T], float32, log-space decay
    beta,                # [B, Hv, T], float32
    scale=0.125,         # optional Python float; default 128**-0.5
    initial_state=state, # optional [B, Hv, 128, 128], float32
    output_final_state=True,
)
```

`Hv` must be a positive multiple of `Hk`; each query/key head serves
`Hv/Hk` value and state heads. `T` must be positive and divisible by the fixed
chunk size 64. Inputs are contiguous and head-first. Output is BF16; requested
final state and a provided initial-state gradient are FP32. Losses may consume
output, final state, or both, and backward covers query, key, value, gate,
beta, and a provided initial state through the differentiable GDN reference
path while preserving the same public H20 contract.

The first release supports only the repository's validated NVIDIA H20 target, fixed key/value dimensions 128, BF16 query/key/value, and FP32 gate/beta/state. It rejects other devices, dtypes, layouts, dimensions, and unaligned lengths before kernel execution.
Q/K normalization, variable lengths, padding, and decode caches are outside this API. A candidate FlashQLA-derived fused schedule is an internal implementation choice; it does not change this public contract.

# Upcoming Features 

## Attention Library Level API
This level API is designed for users to use the existing attention mechanism in the library. 
```py
mod = AttentionLibrary(
    attn_type: str="SoftmaxAttention",
    mask_type: str="Causal",
    use_types: str="Train",
)
```

## Custom Attention Level API
- Support for varlen, block-sparse mask and block-sparse indices
```py
mod = AttentionEngine(
    ...
)
mod(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_len_q: torch.Tensor=None,
    cu_len_kv: torch.Tensor=None,
    block_sparse_mask: torch.Tensor=None,
    block_sparse_indices: torch.Tensor=None,
    block_num: torch.Tensor=None,
)
```
- Support OnlineFunc for decoding
```py
class OnlineFunc:
    ...
    def combine(final_rowscales)-> Tensor:
        """Compute logic for the combine kernel"""
        return o_scale

```
- Support mask_mod for decoding

```py
def mask_mod(b, h, q_idx, kv_idx, custom_fwd_inputs) -> Bool:
    """The offset of q need to be passed by custom_fwd_inputs"""
    return True
```


