from dataclasses import dataclass
import os.path as osp

from ..template.attn_template import TlAttnTemplate

THIS_FILE_PATH = osp.dirname(osp.abspath(__file__))
TEMPLATE_PATH = osp.join(
    THIS_FILE_PATH,
    "../template/tl_template/attn/mla_decode_tl.py")

@dataclass
class lowerOutput:
    tl_dtype: str = "float16"
    
    BATCH: int = "0"
    HEADS: int = "0"
    KV_HEAD_NUM: int = "0"
    KV_CTX: int = "0"
    DIM: int = "0"
    PE_DIM: int = "0"
    
    
    
    
def lower_tl(score_mod, block_mask, online_func,
             custom_fwd_inputs,
             Batch, headq, head, seqlenkv,
             dimqk, dimv, tl_dtype, mask_value, tuned_config=None):
    lower_output = lowerOutput(tl_dtype=tl_dtype, BATCH=str(Batch),
                               HEADS=str(headq), KV_HEAD_NUM=str(head), 
                               KV_CTX=str(seqlenkv), DIM=str(dimv), 
                               PE_DIM=str(dimqk-dimv))
    
    
    return TlAttnTemplate(
        template_dir=TEMPLATE_PATH,
        **lower_output.__dict__,
    )()
    
    