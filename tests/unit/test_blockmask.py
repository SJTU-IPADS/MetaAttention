import pytest


from core.transform.core import create_block_mask, is_causal_mask, is_less_causal_mask

pytestmark = pytest.mark.unit


# mask on attention score
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def causal_mask_1(b, h, q_idx, kv_idx):
    return q_idx + 1 >= kv_idx


def causal_mask_2(b, h, q_idx, kv_idx):
    return q_idx - 128 >= kv_idx


B, H, S = 2, 4, 512

Q_BLOCK_SIZE = 128
K_BLOCK_SIZE = 64


def test_causal_mask_classification():
    mask_tensor = create_block_mask(
        causal_mask, B, H, S, S, "cpu", Q_BLOCK_SIZE, K_BLOCK_SIZE
    )
    assert bool(is_causal_mask(mask_tensor, Q_BLOCK_SIZE, K_BLOCK_SIZE)) is True
    assert bool(is_less_causal_mask(mask_tensor, Q_BLOCK_SIZE, K_BLOCK_SIZE)) is True


def test_causal_mask_plus_one_classification():
    mask_tensor = create_block_mask(
        causal_mask_1, B, H, S, S, "cpu", Q_BLOCK_SIZE, K_BLOCK_SIZE
    )
    assert bool(is_causal_mask(mask_tensor, Q_BLOCK_SIZE, K_BLOCK_SIZE)) is False
    assert bool(is_less_causal_mask(mask_tensor, Q_BLOCK_SIZE, K_BLOCK_SIZE)) is False


def test_strictly_past_mask_classification():
    mask_tensor = create_block_mask(
        causal_mask_2, B, H, S, S, "cpu", Q_BLOCK_SIZE, K_BLOCK_SIZE
    )
    assert bool(is_causal_mask(mask_tensor, Q_BLOCK_SIZE, K_BLOCK_SIZE)) is False
    assert bool(is_less_causal_mask(mask_tensor, Q_BLOCK_SIZE, K_BLOCK_SIZE)) is True
