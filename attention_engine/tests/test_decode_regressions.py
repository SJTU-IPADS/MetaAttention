from core import Var
from core.transform.core import SymbolicColReduceArray
from core.codegen.tl_gen import generate_tl_from_dag
from core.template.attn_template import TlAttnTemplate


class _OnlineSoftmax:
    @staticmethod
    def combine(final_rowscales):
        lse = final_rowscales["lse"]
        lse_max = lse.get_reduce("max")
        row_sum = (lse - lse_max).exp()
        row_sum_sum = row_sum.get_reduce("sum")
        lse_sum = row_sum_sum.log() + lse_max
        return (lse - lse_sum).exp()


def test_online_softmax_combine_uses_natural_log_codegen():
    final_rowscales = {
        "lse": SymbolicColReduceArray(
            "lse", Var("lse"), shape_idx=["num_split", "block_M2"]
        )
    }
    combine_scale = _OnlineSoftmax.combine(final_rowscales)
    tl_code, _ = generate_tl_from_dag([combine_scale])
    tl_code_str = str(tl_code)
    assert "T.log2(lse_1_0_sum_0[i0]) * 0.69314718" in tl_code_str
    assert "T.exp2(lse[i0,i1]*1.442695)" in tl_code_str
    assert "T.log2(lse_1_0_sum_0[i0]) + lse_max_0[i0]" not in tl_code_str
    assert "T.exp2(lse[i0,i1])" not in tl_code_str


def test_sparse_decode_template_renders_seq_len_kv_scalar():
    rendered = TlAttnTemplate(
        "attention_engine/core/template/tl_template/attn/blockattn_decode_varlen_tl.py",
        BATCH="1",
        HEADS="4",
        GROUPS="2",
        DIM="64",
        DIMV="64",
        infer_mask_block_N="32",
        SEQ_LEN_KV="128",
    )()
    assert "max_cache_seqlen=128" in rendered
    assert "{SEQ_LEN_KV}" not in rendered
