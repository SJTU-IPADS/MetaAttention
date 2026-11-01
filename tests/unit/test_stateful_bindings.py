from __future__ import annotations

import torch

from attn_engine import Batch, FeatureRole, HeadRole, Sequence, TensorInput
from attn_engine.stateful.runtime.bindings import build_binding


def test_dense_binding_preserves_strides_and_grouped_head_mapping_without_expansion():
    head = HeadRole("query")
    feature = FeatureRole("feature")
    # Deliberately non-contiguous BF16 storage: the descriptor must use its
    # physical strides rather than asking for a normalized copy.
    raw = (
        torch.arange(2 * 2 * 3 * 4, dtype=torch.float32)
        .reshape(2, 2, 3, 4)
        .to(torch.bfloat16)
    )
    raw = raw.transpose(1, 2)
    spec = TensorInput("query", (Batch, Sequence, head, feature), torch.bfloat16)
    binding = build_binding(
        spec,
        raw,
        role_sizes={head: 4, feature: 4},
        mode="dense",
        state_heads=4,
        feature_roles=(feature,),
    )

    assert binding.raw is raw
    assert binding.dtype == torch.bfloat16
    assert binding.source_heads == 2
    assert binding.target_heads == 4
    assert binding.head_group_size == 2
    assert binding.source_strides == tuple(raw.stride())
    assert binding.tensor.data_ptr() == raw.data_ptr()
    assert binding.load(1, 3, 2).dtype == torch.float32
    # target heads 2 and 3 map to source head 1; no repeated storage exists.
    assert binding.offset(1, 2, 2) == binding.offset(1, 3, 2)
    expected = raw[0, 1, 1, 2].float()
    torch.testing.assert_close(binding.load(1, 2, 2), expected)


def test_packed_binding_uses_offsets_and_supports_empty_sequences():
    head = HeadRole("state")
    feature = FeatureRole("feature")
    offsets = torch.tensor([0, 0, 2, 2, 5], dtype=torch.int32)
    raw = (
        torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3).to(torch.bfloat16)
    )
    spec = TensorInput("key", (Batch, head, Sequence, feature), torch.bfloat16)
    binding = build_binding(
        spec,
        raw,
        role_sizes={head: 2, feature: 3},
        mode="packed",
        state_heads=2,
        sequence_offsets=offsets,
        feature_roles=(feature,),
    )

    assert binding.sequence_count == 4
    assert binding.token_count == 5
    assert binding.sequence_for_token(0) == 1
    assert binding.sequence_for_token(1) == 1
    assert binding.sequence_for_token(2) == 3
    assert binding.token_in_sequence(4) == 2
    torch.testing.assert_close(binding.load(4, 1, 2), raw[4, 1, 2].float())
    assert binding.offset(4, 1, 2) == raw.storage_offset() + 4 * raw.stride(
        0
    ) + raw.stride(1) + 2 * raw.stride(2)


def test_broadcast_binding_replays_batch_value_across_dense_tokens():
    feature = FeatureRole("feature")
    raw = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    spec = TensorInput("bias", (Batch, feature), torch.float32)
    binding = build_binding(
        spec,
        raw,
        role_sizes={feature: 3},
        mode="dense",
        sequence_count=2,
        uniform_length=4,
        state_heads=3,
        feature_roles=(feature,),
    )
    assert binding.broadcast_axes == ("head",)
    assert binding.load(0, 2, 1).item() == 2.0
    assert binding.load(7, 1, 1).item() == 2.0
    assert binding.offset(0, 0, 1) == binding.offset(3, 2, 1)
