from __future__ import annotations

import importlib

import pytest


pytestmark = [pytest.mark.unit, pytest.mark.examples]


EXAMPLE_FACTORIES = (
    ("examples.abc_attention", "abc_attention"),
    ("examples.based_attention", "based_attention"),
    ("examples.comba", "comba"),
    ("examples.delta_rule", "delta_rule_attention"),
    ("examples.gated_delta_product", "gated_delta_product"),
    ("examples.gated_oja_rule", "gated_oja_rule"),
    ("examples.gdn2", "gdn2"),
    ("examples.generalized_delta_rule", "iplr_delta_rule"),
    ("examples.generalized_delta_rule", "dplr_delta_rule"),
    ("examples.kda", "kda"),
    ("examples.momentum_delta_rule", "momentum_delta_rule"),
    ("examples.preconditioned_delta", "preconditioned_delta"),
    ("examples.lightnet_stable", "lightnet_stable"),
    ("examples.gated_delta_rule", "gated_delta_rule"),
    ("examples.gated_retention", "gated_retention"),
    ("examples.gated_slot_attention", "gated_slot_attention"),
    ("examples.hgrn", "hgrn"),
    ("examples.linear_attention", "linear_attention"),
    ("examples.linear_attention", "simple_gla_attention"),
    ("examples.linear_attention", "gla_attention"),
    ("examples.linear_attention", "lightning_attention"),
    ("examples.mamba2", "mamba2"),
    ("examples.linear_attention", "delta_rule_attention"),
    ("examples.linear_attention", "recurrent_linear_attention"),
    ("examples.mha", "causal_softmax_attention"),
    ("examples.mha_decode", "softmax_attention_decode"),
    ("examples.mha_v2", "causal_softmax_attention"),
    ("examples.mla_decode", "mla_decode"),
    ("examples.normalized_linear_attention", "normalized_linear_attention"),
    ("examples.rebased_attention", "rebased_attention"),
    ("examples.reluattn", "relu_attention"),
    ("examples.reluattn_v2", "relu_attention"),
    ("examples.retention_attention", "retention_attention"),
    ("examples.retention_parallel", "retention_parallel"),
    ("examples.retnet_recurrent", "retnet_recurrent"),
    ("examples.rwkv6_attention", "rwkv6_attention"),
    ("examples.rwkv7_state", "rwkv7_state"),
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


def test_stateful_profile_factories_return_stateful_operators():
    from attn_engine import StatefulOperator
    from examples.gated_delta_rule import gated_delta_rule
    from examples.gated_retention import gated_retention
    from examples.mamba2 import mamba2
    from examples.retnet_recurrent import retnet_recurrent

    factories = (
        gated_retention(1, 2, 4, 64, 64),
        retnet_recurrent(1, 2, 4, 64, 64),
        mamba2(1, 2, 4, 64, 64),
        gated_delta_rule(1, 2, 4, 64, 64),
    )
    assert all(isinstance(operator, StatefulOperator) for operator in factories)
