# from ..attn_engine import OnlineFunc
from ..transform.core import SymbolScalar, SymbolicArray, CustomIO, SymbolicColReduceArray
from ..transform.graph import Var, Const
from ..utils import IndentedCode
from ..codegen.tl_gen import generate_tl_from_dag
from ..template.attn_template import TlAttnTemplate
from dataclasses import dataclass

import os
import os.path as osp

from ..codegen.common import *
from copy import copy, deepcopy
from sympy import symbols
from typing import Callable
import sympy as sp

THIS_FILE_PATH = osp.dirname(osp.abspath(__file__))
TEMPLATE_PATH = osp.join(
    THIS_FILE_PATH,
    "../template/tl_template/attn/attn_decode_tl.py")

# TODO: bwd map
shape_idx_map = {
    "batch": "bid",
    "heads": "hid",
    "seq_len": "mid*block_M:(mid+1)*block_M",
    "seq_len_kv": "(seq_len_kv // num_split) * sid + k * block_N : (seq_len_kv // num_split) * sid + (k + 1) * block_N",
    "1": "0"
    # others: ":" -> ":"
}
shape_idx_map_sp = {
    "batch": sp.simplify("bid"),
    "heads": sp.simplify("hid"),
    "seq_len": sp.simplify("mid*block_M"),
    "seq_len_kv": sp.simplify("(seq_len_kv // num_split) * sid + k * block_N"),
    "1": sp.simplify("0")
    # others: ":" -> ":"
}
shape_idx_onchip_map = {
    "batch": "",
    "heads": "",
    "seq_len": "block_M",
    "seq_len_kv": "block_N",
    "1": ""
}
shape_idx_onchip_step_map_sp = {
    "batch": sp.simplify("0"),
    "heads": sp.simplify("0"),
    "seq_len": sp.simplify("block_M"),
    "seq_len_kv": sp.simplify("block_N"),
    "1": sp.simplify("0")
}
shape_idx_onchip_dim_map = [
    "seq_len", "seq_len_kv"
]

RECURRENT_DIM = "block_N"

from .lower import CopyMap, KernelOptionsBase, AttnFwdKernelOption, lower_kernel, AttnBwdKernelOption

@dataclass
class lowerOutput:
    swizzle_shared: str = ""
    tl_dtype: str = "float16"
    is_inf_mask: str = "True"

    # problem shape
    BATCH: str = "1"
    HEADS: str = "1"
    SEQ_LEN: str = "1"
    SEQ_LEN_KV: str = "1"
    DIM: str = "1"
    DIMV: str = "1"
    
    # score_mod name&code
    scores: str = "scores"
    
    # online_func name&code
    scores_online: str = "scores"
    acc_o: str = "acc_o"
    qkT: str = "qkT"
    dsT: str = "dsT"
    doosum_shared: str = "doosum_shared"
    
    # combine kernel
    combinekernel_scale_o_varname: str = "scale_o"
    combinekernel_combine: str = ""

from .lower import lowerKernelBaseOutput



@dataclass
class lowerOnlineFuncOutput:
    online_rowscales_initvalue: str
    online_func_def: str
    call_online_func: str
    online_func_epilogue: str
    online_rowscales_update: str

    isused_doosum: bool
    final_rowscales_length: int
    final_rowscales_load: str
    online_func_fwd: str
    custom_bwd_inputs_load: str
    custom_bwd_body: str
    final_rowscales_shared_init: str
    custom_bwd_inputs: str
    custom_bwd_inputs_init: str
    o_scale_varname: str
    
    torch_alloc_final_rowscales: str
    final_rowscales_list: str
  
@dataclass
class customInputOutput:
    custom_fwd_inputs_load_shared: str = ""
    custom_fwd_inputs_load_s2r: str = ""
    custom_fwd_inputs_load_shared_bwd: str = ""

from .lower import TunnerOutput, lowerScoreModOutput


def lower_online_func(online_func, lower_output: lowerOutput,
                      kernel_options: AttnFwdKernelOption=None,
                      bwd_kernel_options: AttnBwdKernelOption=None):  
    online_fwd = online_func.online_fwd
    # 1. init input vars
    scores = SymbolicArray(
        lower_output.scores_online,
        Var(lower_output.scores_online),
        shape_idx=[
            kernel_options.tile_M,
            kernel_options.tile_N])
    online_rowscales = online_func.online_rowscales
    # TODO: support online_rowscales b h q_idx kv_idx
    b = SymbolScalar("b", Var("b"))
    h = SymbolScalar("h", Var("h"))
    q_idx = SymbolScalar("q_idx", Var("q_idx"))
    
    acco = SymbolicArray(lower_output.acc_o, Var(lower_output.acc_o), shape_idx=[kernel_options.tile_M, kernel_options.dimv])
    
    online_fwd_func_name = "online_func"

    # online&epilogue
    input_vars = {}

    # 2. fill op for online_rowscales
    online_rowscales_initvalue = IndentedCode()
    for k, v in online_rowscales.items():  # v.code is Var
        tl_init_value = v.code.name
        online_rowscales_initvalue.add_line(fill_op(v, tl_init_value))

    # 3. online_fwd func op def&call
    scores_new, new_online_rowscales, o_scalevar = online_fwd(
        scores, online_rowscales, b, h, q_idx)
    for k, v in new_online_rowscales.items():
        new_online_rowscales[k].set_allow_reuse(False)
    o_scalevar.set_allow_reuse(False)
    tl_code, input_vars_online = generate_tl_from_dag(
        list(new_online_rowscales.values()) + [scores_new, o_scalevar])
    online_func_def = func_block(
        online_fwd_func_name, input_vars_online.values(), tl_code
    )
    call_online_func = call_op(online_fwd_func_name, input_vars_online.values())
    input_vars.update(input_vars_online)
    
    # 4. o_scale update
    o_scale_varname = o_scalevar.varname

    # 4. online_rowscales update
    online_rowscales_update = IndentedCode()
    for k, v in new_online_rowscales.items():
        if v.varname == k:
            continue
        online_rowscales_update.add_line(
            copy_op(v, online_rowscales[k])
        )

    # 5. final_rowscales block
    for k, v in online_rowscales.items():
        online_rowscales[k].clear_codegen()
    acco_new, new_final_rowscales\
        = online_func.online_fwd_epilogue(acco, online_rowscales, b, h, q_idx)
    tl_code, input_vars_final = generate_tl_from_dag(
        [acco_new] + list(new_final_rowscales.values()))
    online_func_epilogue = str(tl_code)
    input_vars.update(input_vars_final)
    
    # 6. add kernel output tensor & intermediate tensor
    torch_alloc_final_rowscales = ""
    final_rowscales_list = ""
    for k, v in new_final_rowscales.items():
        kernel_options.add_output_tensor(
            v.varname, v.shape, False, 
            ["batch", "heads", "num_split", "seq_len"], v.dtype,
            f"g_{k}",
            [sp.simplify(ii) for ii in ["bid", "hid", "sid", "mid * block_M"]],
            [3,]
        )
        torch_alloc_final_rowscales += f"g_{k} = torch.empty([BATCH, H, num_split, N_CTXQ], dtype=torch.float, device=q.device)\n"
        final_rowscales_list += f"g_{k}, "
            
    for _, input_var in input_vars.items():
        if input_var.varname == scores.varname:
            continue
        kernel_options.add_intermediate_tensor(
            input_var.varname, input_var.shape_idx, False, input_var.dtype
        )


    # 7. bwd: TODO
    isused_doosum = False
    online_func_fwd = ""
    custom_bwd_inputs = ""
    custom_bwd_inputs_load = ""
    custom_bwd_inputs_init = ""
    custom_bwd_body = ""
    final_rowscales_load = ""
    final_rowscales_bwd = {}
    final_rowscales_length = 0
    final_rowscales_shared_init = ""
    if bwd_kernel_options is not None:
        qkT = SymbolScalar(lower_output.qkT, Var(lower_output.qkT), shape_idx=[str(bwd_kernel_options.tile_M), str(bwd_kernel_options.tile_N)])
        kv_idx = SymbolScalar("kv_idx", Var("kv_idx"))
        dsT = SymbolScalar(lower_output.dsT, Var(lower_output.dsT), shape_idx=[str(bwd_kernel_options.tile_M), str(bwd_kernel_options.tile_N)])
        doosum_shared = SymbolScalar(lower_output.doosum_shared, Var(lower_output.doosum_shared), shape_idx=["1", str(bwd_kernel_options.tile_N)])
            
        
        for k, v in online_func.final_rowscales.items():
            final_rowscales_bwd[k] = SymbolScalar(
                f"{k}_shared", Var(f"{k}"), shape_idx=[
                    "1", "block_N"])
        scores_2 = online_func.forward(
            qkT,
            final_rowscales_bwd,
            b,
            h,
            q_idx,
            kv_idx)

        tl_code, input_vars_fwd = generate_tl_from_dag([scores_2])
        online_func_fwd = str(tl_code)

        qkT.clear_codegen()
        dscores = online_func.backward(
            dsT, qkT, final_rowscales_bwd, doosum_shared, b, h, q_idx, kv_idx)

        tl_code, input_vars_bwd = generate_tl_from_dag([dscores])
        custom_bwd_body = str(tl_code)

        if lower_output.doosum_shared in input_vars_bwd:
            isused_doosum = True

        custom_bwd_inputs = f"g_doosum: T.Buffer([batch, heads, seq_len], accum_dtype), \n" if isused_doosum else ""
        for k, v in final_rowscales_bwd.items():
            final_rowscales_shared_init += f"{v.varname} = T.alloc_shared([{', '.join(v.shape_idx)}], accum_dtype, scope='shared')\n"
        custom_bwd_inputs_init = "doosum_shared = T.alloc_shared([1, block_N], accum_dtype, scope='shared')" if isused_doosum else ""

        for k, v in final_rowscales_bwd.items():
            final_rowscales_load += f"T.copy(g_{k}[bz, bx, k * block_N : (k + 1) * block_N], {v.varname})\n"
        custom_bwd_inputs_load = "T.copy(g_doosum[bz, bx, k * block_N : (k + 1) * block_N], doosum_shared)" if isused_doosum else ""
        final_rowscales_length = len(final_rowscales_bwd)

    return lowerOnlineFuncOutput(
        online_rowscales_initvalue=str(online_rowscales_initvalue),
        online_func_def=str(online_func_def),
        call_online_func=call_online_func,
        online_func_epilogue=online_func_epilogue,
        online_rowscales_update=str(online_rowscales_update),

        isused_doosum=isused_doosum,
        final_rowscales_length=final_rowscales_length,
        final_rowscales_load=final_rowscales_load,
        online_func_fwd=online_func_fwd,
        custom_bwd_inputs_load=custom_bwd_inputs_load,
        custom_bwd_body=custom_bwd_body,
        final_rowscales_shared_init=final_rowscales_shared_init,
        custom_bwd_inputs=custom_bwd_inputs,
        custom_bwd_inputs_init=custom_bwd_inputs_init,
        o_scale_varname=o_scale_varname,
        torch_alloc_final_rowscales=torch_alloc_final_rowscales,
        final_rowscales_list=final_rowscales_list
    )

def lower_combine(online_func, lower_output: lowerOutput,
                      kernel_options: AttnFwdKernelOption=None,
                      bwd_kernel_options: AttnBwdKernelOption=None):
    combine = online_func.combine
    final_rowscales = {}
    for k, v in online_func.final_rowscales.items():
        final_rowscales[k] = SymbolicColReduceArray(
            v.varname, v.code, shape_idx=[
                "num_split", "block_M2"
            ]
        )
    combine_o_scale = combine(final_rowscales)
    tl_code, input_vars = generate_tl_from_dag([combine_o_scale], )
    # print(tl_code)
    lower_output.combinekernel_combine = str(tl_code)
    lower_output.combinekernel_scale_o_varname = combine_o_scale.varname
    for input_var in input_vars.values():
        kernel_options.add_intermediate_tensor(
            input_var.varname, input_var.shape_idx, False, input_var.dtype
        )
    for k, v in final_rowscales.items():
        kernel_options.add_input_tensor(
            v.varname, v.shape_idx, False, ["batch", "heads", "num_split", "seq_len"], 
            v.dtype,
            [sp.simplify(ii) for ii in ["bz", "by", "0", "bx * block_M2"]],
            [2, 3]
        )
        
    

from .lower import lower_score_mod

def lower_custom_inputs(custom_fwd_inputs, lower_output: lowerOutput, kernel_options: KernelOptionsBase):
    # TODO!!!!
    # deal with custom inputs tensors
    
    custom_fwd_inputs_load_shared = ""
    custom_fwd_inputs_load_s2r = ""
    for k, v in custom_fwd_inputs.input_tensors.items():
        # modify shape
        shape_idx_copy_sp = [(shape_idx_map_sp[shape] if shape in shape_idx_map_sp.keys(
        ) else sp.simplify("0")) for shape in v.shape_idx]
        shape_idx_block = [(shape_idx_onchip_map[shape] if shape in shape_idx_onchip_map.keys(
        ) else shape) for shape in v.shape_idx]
        # remove "" in list
        # TODO:bug[block_N] -> [1,block_N]
        shape_idx_block = [shape for shape in shape_idx_block if shape != ""]
        if shape_idx_block == []:
            shape_idx_block = ["1"]
        shape_idx_block_step_sp = [(shape_idx_onchip_step_map_sp[shape] if shape in shape_idx_onchip_step_map_sp.keys(
        ) else sp.simplify(shape)) for shape in v.shape_idx]
        shape_idx_dim_map = [idx for idx, shape in enumerate(v.shape_idx) if shape in shape_idx_onchip_dim_map]
        custom_input_dtype = "accum_dtype"
        
        kernel_options.fragment_tensors[k] = (SymbolScalar(k, Var(k), shape_idx=shape_idx_block, dtype=custom_input_dtype))
        kernel_options.global_tensors_input[f"g_{k}"] = (SymbolScalar(f"g_{k}", Var(f"g_{k}"), shape_idx=v.shape_idx, dtype=custom_input_dtype))
        
        # tl copy bug when "1"
        if not (RECURRENT_DIM in shape_idx_block):
            kernel_options.copy_maps.append(
                CopyMap(kernel_options.global_tensors_input[f"g_{k}"], kernel_options.fragment_tensors[k], shape_idx_copy_sp, shape_idx_dim_map)
            )
        elif len(shape_idx_block) > 1 and shape_idx_block[0] != "1":
            custom_input_dtype = "dtype"
            kernel_options.shared_tensors[f"{k}_shared"] = (SymbolScalar(f"{k}_shared", Var(f"{k}_shared"), shape_idx=shape_idx_block, dtype=custom_input_dtype))
            custom_fwd_inputs_load_shared += str(
                load_op(kernel_options.global_tensors_input[f"g_{k}"], kernel_options.shared_tensors[f"{k}_shared"], shape_idx_dim_map, src_dim_list=list(range(len(shape_idx_copy_sp))), src_idx_list=shape_idx_copy_sp) + "\n"
            )
            custom_fwd_inputs_load_s2r += copy_op(kernel_options.shared_tensors[f"{k}_shared"], kernel_options.fragment_tensors[k]) + "\n"
            lower_output.swizzle_shared += f"{k}_shared: tl.layout.make_swizzled_layout({k}_shared), \n"
        else:
            # custom_fwd_inputs_load_shared += f"T.copy(g_{k}[{', '.join(shape_idx_copy)}], {k})\n"
            custom_fwd_inputs_load_shared += str(
                load_op(kernel_options.global_tensors_input[f"g_{k}"], kernel_options.fragment_tensors[k], shape_idx_dim_map, src_dim_list=list(range(len(shape_idx_copy_sp))), src_idx_list=shape_idx_copy_sp) + "\n"
            )
        # TODO: dtype of custom_fwd_inputs
        
        kernel_options.fragment_tensors[k].dtype = custom_input_dtype
        kernel_options.fragment_tensors[k].shape_idx = shape_idx_block
        kernel_options.global_tensors_input[f"g_{k}"].dtype = custom_input_dtype
        custom_fwd_inputs.input_tensors[k].shape_idx = shape_idx_block
        
    return customInputOutput(
        custom_fwd_inputs_load_shared=custom_fwd_inputs_load_shared,
        custom_fwd_inputs_load_s2r=custom_fwd_inputs_load_s2r,
        custom_fwd_inputs_load_shared_bwd="",
    )

def lower_tl(score_mod, block_mask, online_func,
             custom_fwd_inputs,
             batch, heads, seq_len, seq_len_kv,
             dimqk, dimv, tl_dtype, mask_value, tuned_config=None):

    lower_output = lowerOutput(BATCH=batch, HEADS=heads, SEQ_LEN=seq_len,
                              SEQ_LEN_KV=seq_len_kv, DIM=dimqk, DIMV=dimv)
    lower_output.tl_dtype = tl_dtype
    # TODO: mask_value: 0 or -inf
    lower_output.is_inf_mask = "True" if block_mask is not None and mask_value == "-inf" else "False"

    # tune
    if tuned_config is None:
        tune_output = TunnerOutput(block_M="128", block_N="64", stages="2", thread_num="128", shared_fuse="False")
    else:
        tune_output = TunnerOutput(**tuned_config)
    # FWD split config
    # TODO
    if dimv > 256:
        tune_output.block_M = "64"
        tune_output.block_N = "64"
        tune_output.stages = "1"
        tune_output.shared_fuse = "True"
    # TODO: ugly
    if tune_output.shared_fuse == "True":
        lower_output.scores_online = "scores_1"
        
    # 2. kernel config options 
    kernel_options = AttnFwdKernelOption(tile_M=sp.simplify("block_M"), tile_N=sp.simplify("block_N"), 
                                         dim=sp.simplify("dim"), dimv=sp.simplify("dimv"))
    kernel_code_template = lowerKernelBaseOutput("main_split")
    
    
    lower_custom_inputs_output = lower_custom_inputs(
        custom_fwd_inputs, lower_output, kernel_options)
    
    lower_score_mod_output = lower_score_mod(
        score_mod, custom_fwd_inputs, lower_output, kernel_options, None)
    lower_online_func_output = lower_online_func(
        online_func, lower_output, kernel_options, None)
    output_idx_list = [i for i in range(4 +
                                        len(custom_fwd_inputs.input_tensors) +
                                        len(online_func.final_rowscales), 4 +
                                        1 +
                                        len(custom_fwd_inputs.input_tensors) +
                                        len(online_func.final_rowscales))]

    lower_kernel(kernel_options, kernel_code_template)
    # combine kernel
    kernel_options2 = KernelOptionsBase()
    kernel_code_template2 = lowerKernelBaseOutput("combine")
    lower_combine(online_func, lower_output, kernel_options2, None)
    lower_kernel(kernel_options2, kernel_code_template2)

    custom_fwd_inputs_list = (",".join(kernel_options.global_tensors_input.keys()) + ",") if len(kernel_options.global_tensors_input) > 0 else ""
    return TlAttnTemplate(
        TEMPLATE_PATH,
        custom_fwd_inputs=kernel_code_template.input_args,
        custom_fwd_inputs_list=custom_fwd_inputs_list,
        custom_fwd_inputs_init=kernel_code_template.alloc,
        custom_fwd_inputs_load_prolog=kernel_code_template.input_args_copy_prologue,
        final_rowscales_output=kernel_code_template.output_args,
        final_rowscales_save=kernel_code_template.output_args_copy_epilogue,
        
        combinekernel_input_init=kernel_code_template2.alloc,
        combinekernel_input_load=kernel_code_template2.input_args_copy_prologue,
        **lower_custom_inputs_output.__dict__,
        **lower_online_func_output.__dict__,
        **lower_score_mod_output.__dict__,

        **lower_output.__dict__,
        **tune_output.__dict__,

        output_idx_list=str(output_idx_list),
    )()
