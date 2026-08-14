from __future__ import annotations

import importlib

import pytest


pytestmark = [pytest.mark.unit, pytest.mark.examples]


EXAMPLE_FACTORIES = (
    ("examples.gated_retention", "gated_retention"),
    ("examples.gated_delta_rule", "gated_delta_rule"),
    ("examples.mamba2", "mamba2"),
    ("examples.mha", "causal_softmax_attention"),
    ("examples.mha_decode", "softmax_attention_decode"),
    ("examples.mha_v2", "causal_softmax_attention"),
    ("examples.mla_decode", "mla_decode"),
    ("examples.mla_decode_v2", "mla_decode"),
    ("examples.reluattn", "relu_attention"),
    ("examples.reluattn_v2", "relu_attention"),
    ("examples.retention_parallel", "retention_parallel"),
    ("examples.retnet_recurrent", "retnet_recurrent"),
    ("examples.sigmoid_attn", "sigmoid_attention"),
    ("examples.sigmoid_attn_v2", "sigmoid_attention"),
    ("examples.sparse_gqa_decode", "sparse_gqa_decode"),
)


@pytest.mark.parametrize(
    ("module_name", "factory_name"),
    EXAMPLE_FACTORIES,
    ids=[
        f"{module.rsplit('.', 1)[-1]}.{factory}"
        for module, factory in EXAMPLE_FACTORIES
    ],
)
def test_official_example_factory_imports(module_name: str, factory_name: str):
    module = importlib.import_module(module_name)
    assert callable(getattr(module, factory_name))
