from core.transform.core import create_block_mask, is_causal_mask, is_less_causal_mask
# mask on attention score
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def causal_mask_1(b, h, q_idx, kv_idx):
    return q_idx+1 >= kv_idx

def causal_mask_2(b, h, q_idx, kv_idx):
    return q_idx-128 >= kv_idx

B, H, S = 2, 4, 512

Q_BLOCK_SIZE = 128
K_BLOCK_SIZE = 64
def test_mask(mask_mod):
    mask_tensor = create_block_mask(mask_mod, B, H, S, S, "cpu", Q_BLOCK_SIZE, K_BLOCK_SIZE)
    print(mask_tensor.shape)
    print(mask_tensor)

    print(is_causal_mask(mask_tensor, Q_BLOCK_SIZE, K_BLOCK_SIZE))
    print(is_less_causal_mask(mask_tensor, Q_BLOCK_SIZE, K_BLOCK_SIZE))

test_mask(causal_mask)
test_mask(causal_mask_1)
test_mask(causal_mask_2)

