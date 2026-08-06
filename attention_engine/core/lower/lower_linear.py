from ..transform.core import SymbolScalar, SymbolicArray, CustomIO
from ..transform.graph import Var, Const
from ..utils import IndentedCode
from ..codegen.tl_gen import generate_tl_from_dag
from ..template.linear_attn_template import TlLinearAttnTemplate
import copy

from dataclasses import dataclass

shape_idx_map_o = {
    "batch": "bb",
    "heads": "bh",
    "seq_len": "by*BT:(by+1)*BT",
    "dimqk": "ik*BK:(ik+1)*BK",
    "dimv": "bx*BV:(bx+1)*BV",
    "1": "0"
    # others: ":" -> ":"
}
shape_idx_onchip_map_o = {
    "batch": "",
    "heads": "",
    "seq_len": "BT",
    "dimqk": "BK",
    "dimv": "BV",
    "1": ""
}

shape_idx_map_h = {
    "batch": "bb",
    "heads": "bhead",
    "seq_len": "i_t*BT:(i_t+1)*BT",
    "dimqk": "bx*BK:(bx+1)*BK",
    "dimv": "by*BV:(by+1)*BV",
    "1": "0"
    # others: ":" -> ":"
}
shape_idx_onchip_map_h = {
    "batch": "",
    "heads": "",
    "seq_len": "BT",
    "dimqk": "BK",
    "dimv": "BV",
    "1": ""
}

# TODO: bug, backward use_count


@dataclass
class lowerOutput:
    # problem shape
    BATCH: str = ""
    HQ: str = ""
    HK: str = ""
    H: str = ""
    N_CTX: str = ""
    D_HEAD: str = ""
    D_HEADV: str = ""
    
    k_mod_expr: str = ""
    v_mod_expr: str = ""
    decay_mod_expr: str = ""
    q_mod_expr: str = ""
    custom_inputs_list: str = ""
    chunk_h_custom_inputs_list: str = ""
    k_mod_expr_fused_h: str = ""
    chunk_o_custom_inputs_list: str = ""
    v_mod_expr_fused_o: str = ""
    output_idx_list_h: str = "[3,]"
    output_idx_list_o: str = "[5,]"
    h_alloc_buffer_list: str = ""
    o_alloc_buffer_list: str = ""
    custom_inputs_list_o: str = ""
    custom_inputs_list_h: str = ""
    k_mod_expr_2: str = ""
    v_mod_expr_2: str = ""
    v_mod_bwd_expr: str = ""
    k_mod_bwd_expr: str = ""
    decay_mod_bwd_expr: str = ""
    q_mod_bwd_expr: str = ""
    custom_inputs_grad_list: str = ""
    q_name: str = "q"
    k_name: str = "k"
    v_name: str = "v"
    decay_name: str = "decay"
    dq_name: str = "dq"
    dk_name: str = "dk"
    dv_name: str = "dv"
    ddecay_name: str = "dg2"
    q_name1: str = "q"
    k_name1: str = "k"
    v_name1: str = "v"
    decay_name1: str = "decay"
    k_mod_expr1: str = ""
    v_mod_expr1: str = ""
    decay_mod_expr1: str = ""
    q_mod_expr1: str = ""
    k_name2: str = "k"
    v_name2: str = "v"
    decay_name2: str = "decay"


@dataclass
class TunnerOutput:
    TUNE:str = "False"
    TUNE_FILE:str=""
    TUNE_BWD:str = "False"
    TUNE_FILE_BWD:str = ""
    
    BT: str = "64"
    BK_h: str = "64"
    BV_h: str = "64"
    num_stages_h: str = "2"
    num_threads_h: str = "128"
    BK_o: str = "64"
    BV_o: str = "64"
    num_stages_o: str = "2"
    num_threads_o: str = "128"

    BT_BWD: str = "64"
    BK_dh: str = "64"
    BV_dh: str = "64"
    num_stages_dh: str = "2"
    num_threads_dh: str = "128"
    BK_dqkg: str = "64"
    BV_dqkg: str = "64"
    num_stages_dqkg: str = "1"
    num_threads_dqkg: str = "128"
    BK_dv: str = "64"
    BV_dv: str = "64"
    num_stages_dv: str = "1"
    num_threads_dv: str = "128"


bwd_custom_output_dict = {}


def lowerKmod(k_mod, custom_io, lower_output: lowerOutput):
    k = SymbolicArray("k", Var("k"), shape_idx=["B", "H", "T", "D"])
    custom_io1 = copy.deepcopy(custom_io)
    k.count += 1
    for kname, v in custom_io1.input_tensors.items():
        custom_io1.input_tensors[kname].count += 1
    new_k = k_mod(k, custom_io1)
    pytorch_code, input_vars = generate_tl_from_dag([new_k], to_tl=False)
    lower_output.k_mod_expr = str(pytorch_code)
    lower_output.k_name = new_k.varname

    # bwd
    k = SymbolicArray("k", Var("k"), shape_idx=["B", "H", "T", "D"])
    custom_io1 = copy.deepcopy(custom_io)
    k.count += 1
    for kname, v in custom_io1.input_tensors.items():
        custom_io1.input_tensors[kname].count += 1
    new_k = k_mod(k, custom_io1)
    dnew_k = SymbolicArray("dk", Var("dk"), shape_idx=["B", "H", "T", "D"])
    new_k.backward(dnew_k)
    pytorch_code, input_vars, inputs = generate_tl_from_dag(
        [new_k], to_tl=False, return_inputs=True)
    lower_output.k_mod_expr1 = str(pytorch_code)
    lower_output.k_name1 = new_k.varname
    lower_output.k_name2 = new_k.varname
    input_vars_with_grad = {k: v for k, v in inputs.items() if v.require_grad}
    pytorch_code, input_vars_grad = generate_tl_from_dag(
        [ii.grad for ii in input_vars_with_grad.values()], to_tl=False)
    lower_output.k_mod_bwd_expr = str(pytorch_code)
    lower_output.dk_name = k.grad.varname
    bwd_custom_output_dict.update({
        k: v.grad.varname for k, v in input_vars_with_grad.items()
    })


def lowerVmod(v_mod, custom_io, lower_output: lowerOutput, bwd_only=False):
    if not bwd_only:
        vv = SymbolicArray("v", Var("v"), shape_idx=["B", "H", "T", "D"])
        custom_io1 = copy.deepcopy(custom_io)
        vv.count += 1
        for kname, _v in custom_io1.input_tensors.items():
            custom_io1.input_tensors[kname].count += 1
        new_v = v_mod(vv, custom_io1)
        pytorch_code, input_vars = generate_tl_from_dag([new_v], to_tl=False)
        lower_output.v_mod_expr = str(pytorch_code)
        lower_output.v_name = new_v.varname

    # bwd
    vv = SymbolicArray("v", Var("v"), shape_idx=["B", "H", "T", "D"])
    custom_io1 = copy.deepcopy(custom_io)
    vv.count += 1
    for kname, _v in custom_io1.input_tensors.items():
        custom_io1.input_tensors[kname].count += 1
    new_v = v_mod(vv, custom_io1)
    dnew_v = SymbolicArray("dv", Var("dv"), shape_idx=["B", "H", "T", "DV"])
    new_v.backward(dnew_v)
    pytorch_code, input_vars, inputs = generate_tl_from_dag(
        [new_v], to_tl=False, return_inputs=True)
    lower_output.v_mod_expr1 = str(pytorch_code)
    lower_output.v_name1 = new_v.varname
    lower_output.v_name2 = new_v.varname
    input_vars_with_grad = {k: v for k, v in inputs.items() if v.require_grad}
    pytorch_code, input_vars_grad = generate_tl_from_dag(
        [ii.grad for ii in input_vars_with_grad.values()], to_tl=False)
    lower_output.v_mod_bwd_expr = str(pytorch_code)
    lower_output.dv_name = vv.grad.varname
    bwd_custom_output_dict.update({
        k: v.grad.varname for k, v in input_vars_with_grad.items()
    })

# TODO: tmp solution


def lowerFusedVmod(v_mod, custom_io, lower_output: lowerOutput):
    vv = SymbolicArray("bs", Var("bs"), shape_idx=["BT", "BT2"])
    custom_io1 = copy.deepcopy(custom_io)
    new_v = v_mod(vv, custom_io1)

    # TODO: change here to avoid warning
    tl_code, input_vars = generate_tl_from_dag([new_v])
    input_vars.pop("bs")

    new_custom_io = CustomIO()
    for k, v in input_vars.items():
        new_custom_io.input_tensors[k] = copy.deepcopy(v)
        new_custom_io.input_tensors[k].clear_codegen()
        new_custom_io.input_tensors[k].varname = f"{k}_local"
        if len(v.shape_idx) == 4:
            raise ("Not support shape_idx with 4")
        # else:
        #     lower_output.v_mod_expr_fused_o += \
        #     f"T.copy({k}[{','.join([shape_idx_map_o[i] for i in v.shape_idx])}], {k}_shared)\n"
        #     lower_output.v_mod_expr_fused_o += \
        #     f"T.copy({k}_shared, {k}_local)\n"
        #     new_custom_io.input_tensors[k].shape_idx = [shape_idx_onchip_map_o[ sidx] for sidx in v.shape_idx if shape_idx_onchip_map_o[ sidx]!=""]
        #     lower_output.o_alloc_buffer_list += f"{k}_shared = T.alloc_shared(({','.join(new_custom_io.input_tensors[k].shape_idx)},), dtype=dtype, scope='shared')\n"
        #     lower_output.o_alloc_buffer_list += f"{k}_local = T.alloc_fragment(({','.join(new_custom_io.input_tensors[k].shape_idx)},), dtype=accum_dtype)\n"
        elif v.shape_idx == ["batch", "heads", "seq_len"]:
            new_custom_io.input_tensors[k].shape_idx = ["1", "BT2"]
            lower_output.o_alloc_buffer_list += f"{k}_shared = T.alloc_shared(({','.join(new_custom_io.input_tensors[k].shape_idx)},), dtype=dtype, scope='shared')\n"
            lower_output.o_alloc_buffer_list += f"{k}_local = T.alloc_fragment(({','.join(new_custom_io.input_tensors[k].shape_idx)},), dtype=accum_dtype)\n"
            lower_output.v_mod_expr_fused_o += \
                f"T.copy({k}[{','.join([shape_idx_map_o[i] for i in v.shape_idx])}], {k}_shared)\n"
            lower_output.v_mod_expr_fused_o += \
                f"T.copy({k}_shared, {k}_local)\n"
            lower_output.chunk_o_custom_inputs_list += f"{k}: T.Buffer(({','.join(v.shape_idx)}), dtype),\n"
            lower_output.custom_inputs_list_o += f"{k},"
        else:
            raise ("Not support shape_idx")

    lower_output.output_idx_list_o = f"[{len(input_vars)+5},]"
    vv = SymbolicArray("bs", Var("bs"), shape_idx=["BT", "BT2"])
    new_v = v_mod(vv, new_custom_io)
    tl_code, input_vars = generate_tl_from_dag([new_v])
    lower_output.v_mod_expr_fused_o += str(tl_code)

    ############################

    vv = SymbolicArray("b_k", Var("b_k"), shape_idx=["BT", "BK"])
    custom_io1 = copy.deepcopy(custom_io)
    new_v = v_mod(vv, custom_io1)

    tl_code, input_vars = generate_tl_from_dag([new_v])
    input_vars.pop("b_k")

    new_custom_io = CustomIO()
    for k, v in input_vars.items():
        new_custom_io.input_tensors[k] = copy.deepcopy(v)
        new_custom_io.input_tensors[k].clear_codegen()
        new_custom_io.input_tensors[k].varname = f"{k}_local"
        if len(v.shape_idx) == 4:
            raise ("Not support shape_idx with 4")
        elif v.shape_idx == ["batch", "heads", "seq_len"]:
            new_custom_io.input_tensors[k].shape_idx = ["BT"]
            lower_output.h_alloc_buffer_list += f"{k}_shared = T.alloc_shared(({','.join(new_custom_io.input_tensors[k].shape_idx)},), dtype=dtype, scope='shared')\n"
            lower_output.h_alloc_buffer_list += f"{k}_local = T.alloc_fragment(({','.join(new_custom_io.input_tensors[k].shape_idx)},), dtype=accum_dtype)\n"
            lower_output.k_mod_expr_fused_h += \
                f"T.copy({k}[{','.join([shape_idx_map_h[i] for i in v.shape_idx])}], {k}_shared)\n"
            lower_output.k_mod_expr_fused_h += \
                f"T.copy({k}_shared, {k}_local)\n"
            lower_output.chunk_h_custom_inputs_list += f"{k}: T.Buffer(({','.join(v.shape_idx)}), dtype),\n"
            lower_output.custom_inputs_list_h += f"{k},"
        else:
            raise ("Not support shape_idx")

    lower_output.output_idx_list_h = f"[{len(input_vars)+3},]"
    vv = SymbolicArray("b_k", Var("b_k"), shape_idx=["BT", "BK"])
    new_v = v_mod(vv, new_custom_io)
    tl_code, input_vars = generate_tl_from_dag([new_v])
    lower_output.k_mod_expr_fused_h += str(tl_code)


def lowerDecaymod(decay_mod, custom_io, lower_output: lowerOutput):
    decay = SymbolicArray("decay", Var("decay"), shape_idx=["B", "H", "T"])
    custom_io1 = copy.deepcopy(custom_io)
    decay.count += 1
    for kname, v in custom_io1.input_tensors.items():
        custom_io1.input_tensors[kname].count += 1
    new_decay = decay_mod(decay, custom_io1)
    pytorch_code, input_vars = generate_tl_from_dag([new_decay], to_tl=False)
    lower_output.decay_mod_expr = str(pytorch_code)
    lower_output.decay_name = new_decay.varname

    # bwd
    decay = SymbolicArray("decay", Var("decay"), shape_idx=["B", "H", "T"])
    custom_io1 = copy.deepcopy(custom_io)
    decay.count += 1
    for kname, v in custom_io1.input_tensors.items():
        custom_io1.input_tensors[kname].count += 1
    new_decay = decay_mod(decay, custom_io1)
    dnew_decay = SymbolicArray("dg2", Var("dg2"), shape_idx=["B", "H", "T"])
    new_decay.backward(dnew_decay)
    pytorch_code, input_vars, inputs = generate_tl_from_dag(
        [new_decay], to_tl=False, return_inputs=True)
    lower_output.decay_mod_expr1 = str(pytorch_code)
    lower_output.decay_name1 = new_decay.varname
    lower_output.decay_name2 = new_decay.varname
    input_vars_with_grad = {k: v for k, v in inputs.items() if v.require_grad}
    pytorch_code, input_vars_grad = generate_tl_from_dag(
        [ii.grad for ii in input_vars_with_grad.values()], to_tl=False)
    lower_output.decay_mod_bwd_expr = str(pytorch_code)
    lower_output.ddecay_name = decay.grad.varname
    bwd_custom_output_dict.update({
        k: v.grad.varname for k, v in input_vars_with_grad.items()
    })


def lowerQmod(q_mod, custom_io, lower_output: lowerOutput):
    # Qmod fused
    # q = SymbolicArray("q", Var("q"), shape_idx=["B", "H", "T", "D"])
    # custom_io1 = copy.deepcopy(custom_io)
    # q.count += 1
    # for kname, v in custom_io1.input_tensors.items():
    #     custom_io1.input_tensors[kname].count += 1
    # new_q = q_mod(q, custom_io1)
    # pytorch_code, input_vars = generate_tl_from_dag([new_q], to_tl=False)
    # lower_output.q_mod_expr = str(pytorch_code)
    # lower_output.q_name = new_q.varname

    # bwd
    q = SymbolicArray("q", Var("q"), shape_idx=["B", "H", "T", "D"])
    custom_io1 = copy.deepcopy(custom_io)
    q.count += 1
    for kname, v in custom_io1.input_tensors.items():
        custom_io1.input_tensors[kname].count += 1
    new_q = q_mod(q, custom_io1)
    dq = SymbolicArray("dq", Var("dq"), shape_idx=["B", "H", "T", "D"])
    new_q.backward(dq)
    pytorch_code, input_vars, inputs = generate_tl_from_dag(
        [new_q], to_tl=False, return_inputs=True)
    lower_output.q_mod_expr1 = str(pytorch_code)
    lower_output.q_name1 = new_q.varname
    input_vars_with_grad = {k: v for k, v in inputs.items() if v.require_grad}
    pytorch_code, input_vars_grad = generate_tl_from_dag(
        [ii.grad for ii in input_vars_with_grad.values()], to_tl=False)
    lower_output.q_mod_bwd_expr = str(pytorch_code)
    lower_output.dq_name = q.grad.varname
    bwd_custom_output_dict.update({
        k: v.grad.varname for k, v in input_vars_with_grad.items()
    })


def lowerQmodFused(q_mod, custom_io, lower_output: lowerOutput):
    bq = SymbolScalar("bq", Var("bq"), shape_idx=["BT", "BK"])
    new_q = q_mod(bq, custom_io)
    tl_code, input_vars = generate_tl_from_dag([new_q])
    lower_output.q_mod_expr = str(tl_code)


def lower_tl(qkv_meta, q_mod, k_mod, v_mod, decay_mod, custom_io, tuned_config=None,
             tune=False, tune_filename="", tune_bwd=False):

    if tuned_config is None:
        tune_output = TunnerOutput(TUNE=tune, TUNE_FILE=tune_filename, TUNE_BWD=tune_bwd, TUNE_FILE_BWD=tune_filename)
    else:
        tune_output = TunnerOutput(**tuned_config)

    BATCH, HQ, N_CTX, D_HEAD = qkv_meta[0].shape
    DV = qkv_meta[2].shape[3]
    HK = qkv_meta[1].shape[1]
    H = qkv_meta[2].shape[1]
    
    lower_output = lowerOutput(BATCH=BATCH, HQ=HQ, HK=HK, H=H, N_CTX=N_CTX,D_HEAD=D_HEAD,D_HEADV=DV)
    if k_mod:
        lowerKmod(k_mod, custom_io, lower_output)
    if v_mod:
        try:
            lowerFusedVmod(v_mod, custom_io, lower_output)
            # bwd
            lowerVmod(v_mod, custom_io, lower_output, bwd_only=True)
            lower_output.v_mod_expr_2 = lower_output.v_mod_expr1
            lower_output.v_mod_expr1 = ""
            lower_output.k_name2 = lower_output.k_name
            lower_output.v_name2 = lower_output.v_name
        except BaseException:
            lowerVmod(v_mod, custom_io, lower_output)
    if decay_mod:
        lowerDecaymod(decay_mod, custom_io, lower_output)
    if q_mod:
        lowerQmodFused(q_mod, custom_io, lower_output)
        # Qmod bwd only
        lowerQmod(q_mod, custom_io, lower_output)
    lower_output.custom_inputs_list = ", ".join(
        [f"{varname}" for varname in custom_io.input_tensors.keys()])
    lower_output.custom_inputs_list += "," if lower_output.custom_inputs_list else ""
    lower_output.custom_inputs_grad_list = ",".join(
        [f"{bwd_custom_output_dict[k] if k in bwd_custom_output_dict.keys() else None}" for k in custom_io.input_tensors.keys()])
    lower_output.custom_inputs_grad_list += "," if lower_output.custom_inputs_grad_list else ""
    return TlLinearAttnTemplate(
        **(lower_output.__dict__),
        **(tune_output.__dict__)
    )()
