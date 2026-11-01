"""Native softmax proof is structural, temporal, and resource bounded."""

from dataclasses import replace

import torch

from attn_engine import (
    AlgorithmIR,
    Readout,
    Softmax,
    StateTerm,
    Target,
)
from attn_engine.stateful.lowering.chunked import compile_chunked_recipe
from attn_engine.stateful.lowering.diagonal.diagonal_native import (
    native_diagonal_resources,
)
from examples import abc_attention as abc


def _sizes(dim=64, slots=16, value=64):
    return {
        abc.KEY: dim,
        abc.VALUE: value,
        abc.SLOT: slots,
        abc.QUERY_HEADS: 1,
        abc.KEY_HEADS: 1,
        abc.VALUE_HEADS: 1,
        abc.STATE_HEADS: 1,
    }


def test_abc_default_keeps_gemm_and_accounts_for_vector_cut():
    op = abc.abc_attention(1, 1, 65, 64, 16, 64)
    sizes = _sizes()
    compiled = op.compile(role_sizes=sizes, target=Target(), mode="dense")
    executable = op._plans[compiled.specialization]
    recipe = executable.chunked_recipe.native_diagonal
    assert executable.plan.candidate.lowerer.endswith("readout-cut")
    assert recipe.fused_softmax is False
    assert recipe.region.vectors == (StateTerm("normalizer", 1, None),)
    assert next(s for s in recipe.states if s.name == "normalizer").vector
    resources = native_diagonal_resources(recipe, sizes, 1)
    assert resources.cut_bytes == 12 * 16
    assert resources.cut_launches == 3
    assert (
        executable.plan.workspace.bytes_per_token
        >= resources.output_bytes + resources.cut_bytes
    )
    candidates = executable.plan.lowered_candidates
    assert any(c.lowerer == "tilelang-native-diagonal" for c in candidates)
    assert not any(c.lowerer.endswith("shared-scores") for c in candidates)


def test_softmax_region_uses_structure_after_state_renaming_and_input_pointwise():
    base = abc.abc_attention(1, 1, 65, 13, 19, 11, dtype=torch.float32).algorithm
    names = {
        "key_slot": "left_cache",
        "slot_value": "right_cache",
        "normalizer": "mass",
    }
    first = StateTerm(
        names["key_slot"],
        base.readouts[0].expression.operand.left.operand.left.operand,
        abc.KEY,
    )
    z = StateTerm("mass", 1, None)
    out = StateTerm(
        "right_cache",
        Softmax((first + 0.07) / (z + 0.03), abc.SLOT) / (z + 0.02),
        abc.SLOT,
    )
    ir = AlgorithmIR(
        inputs=base.inputs,
        states=tuple(replace(s, name=names[s.name]) for s in base.states),
        transition=tuple(replace(t, state=names[t.state]) for t in base.transitions),
        readouts=(Readout("out", out, torch.float32),),
        head_mapping=base.head_mapping,
    )
    recipe = compile_chunked_recipe(ir, _sizes(13, 19, 11)).native_diagonal
    assert recipe is not None and recipe.region.vectors == (z,)
    assert recipe.softmax[1].state == "left_cache"
    assert recipe.softmax[2].state == "right_cache"


def test_oversized_region_keeps_nonfused_execution_candidate():
    op = abc.abc_attention(1, 1, 7, 96, 96, 33, dtype=torch.float32)
    compiled = op.compile(role_sizes=_sizes(96, 96, 33), target=Target(), mode="dense")
    executable = op._plans[compiled.specialization]
    assert executable.chunked_recipe.native_diagonal is not None
    assert executable.chunked_recipe.native_diagonal.fused_softmax is False
    assert not any(
        c.lowerer.endswith("readout-cut") for c in executable.plan.lowered_candidates
    )


def test_unbound_vector_operand_preserves_generic_chunked_path():
    base = abc.abc_attention(1, 1, 7, 13, 19, 11, dtype=torch.float32).algorithm
    # Scalar 2 requires a vector-State multiplication; the first implementation
    # only binds the raw vector snapshot. Reject the native proof, not the IR.
    z = StateTerm("normalizer", 2, None)
    first = base.readouts[0].expression.operand.left.operand.left
    out = StateTerm("slot_value", Softmax(first / z, abc.SLOT) / z, abc.SLOT)
    ir = AlgorithmIR(
        inputs=base.inputs,
        states=base.states,
        transition=base.transitions,
        readouts=(Readout("out", out, torch.float32),),
        head_mapping=base.head_mapping,
    )
    recipe = compile_chunked_recipe(ir, _sizes(13, 19, 11))
    assert recipe is not None and recipe.native_diagonal is None
