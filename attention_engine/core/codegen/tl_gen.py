from ..transform.core import SymbolScalar, SymbolicArray, SymbolicColReduceArray
from ..transform.graph import Var, Const
from ..utils import IndentedCode
from typing import Tuple


def to_tl_op(type: str, *args: SymbolScalar):
    code = IndentedCode()
    if type == "ReduceSum":
        code.add_line(
            # , clear=True)"
            f"T.reduce_sum({args[1].varname}, {args[0].varname},dim=1)"
        )
    elif type == "ReduceMax":
        code.add_line(
            f"T.reduce_max({args[1].varname}, {args[0].varname},dim=1, clear=True)"
        )
    elif type == "ReduceAbsSum":
        code.add_line(
            f"T.reduce_abssum({args[1].varname}, {args[0].varname},dim=1)"
        )
    elif type == "ColReduceSum":
        code.add_line(
            f"T.reduce_sum({args[1].varname}, {args[0].varname},dim=0)"
        )
    elif type == "ColReduceMax":
        code.add_line(
            f"T.reduce_max({args[1].varname}, {args[0].varname},dim=0, clear=True)"
        )
    elif type == "ColReduceAbsSum":
        code.add_line(
            f"T.reduce_abssum({args[1].varname}, {args[0].varname},dim=0)"
        )
    elif type == "Sub" or type == "Add" or type == "Mul" or type == "Div" or type == "Neg" or type == "Exp" or type == "Log" or type == "Abs" or type == "Max" or type == "Tanh" or type == "MaxBwd" or type == "Exp2" or type == "Log2":
        # args idx
        # note: assume input shape is validate: ["1",...] or [arg0[0], ...]
        # assume not square: ["block_M", "block_M"]
        idx_strs = []
        idx_str = [
            f"i{i}" if idx != "1" else f"0" for i,
            idx in enumerate(args[0].shape_idx)]
        idx_str = ",".join(idx_str)
        idx_strs.append(idx_str)
        for _, arg in enumerate(args[1:]):
            input_idx = arg.shape_idx
            idx_str_t = []
            target_i = 0
            for i, idx in enumerate(input_idx):
                if idx == "1":
                    idx_str_t.append(f"0")
                    continue
                while target_i < len(args[0].shape_idx) and idx != args[0].shape_idx[target_i]:
                    target_i += 1
                if target_i >= len(args[0].shape_idx):
                    print(
                        f"Warning: {args[0].varname} {args[0].shape_idx} {arg.varname} {arg.shape_idx}")
                    idx_str_t.append(f"i{i}")
                else:
                    idx_str_t.append(f"i{target_i}")
                    
            idx_str_t = ",".join(idx_str_t)
            idx_strs.append(idx_str_t)
        # [block_M,block_N]
        loop_str = ",".join(args[0].shape_idx)

        # for loop
        code.add_line(
            f"for {idx_str} in T.Parallel({loop_str}):"
        )
        code.more_indent()

        # remove [] for scalar
        for i, tmp_idx_str in enumerate(idx_strs):
            idx_strs[i] = f"[{tmp_idx_str}]" if len(
                tmp_idx_str) > 0 else tmp_idx_str

        if type == "Sub":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = {args[1].varname}{idx_strs[1]} - {args[2].varname}{idx_strs[2]}"
            )
        elif type == "Add":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = {args[1].varname}{idx_strs[1]} + {args[2].varname}{idx_strs[2]}"
            )
        elif type == "Max":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = T.max({args[1].varname}{idx_strs[1]}, {args[2].varname}{idx_strs[2]})"
            )
        elif type == "Exp":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = T.exp2({args[1].varname}{idx_strs[1]}*1.442695)"
            )
        elif type == "Exp2":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = T.exp2({args[1].varname}{idx_strs[1]})"
            )
        elif type == "Mul":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = {args[1].varname}{idx_strs[1]} * {args[2].varname}{idx_strs[2]}"
            )
        elif type == "Div":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = {args[1].varname}{idx_strs[1]} / {args[2].varname}{idx_strs[2]}"
            )
        elif type == "Log":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = T.log2({args[1].varname}{idx_strs[1]}) * 0.69314718"
            )
        elif type == "Log2":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = T.log2({args[1].varname}{idx_strs[1]})"
            )
        elif type == "Tanh":
            code.add_line(
                # f"{args[0].varname}[{idx_str}] = T.tanh({args[1].varname}{idx_strs[1]})"
                f"fast_tanh({args[1].varname}{idx_strs[1]}, {args[0].varname}[{idx_str}])"
            )
        elif type == "Abs":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = T.abs({args[1].varname}{idx_strs[1]})"
            )
        elif type == "MaxBwd":
            code.add_line(
                f"{args[0].varname}[{idx_str}] = T.if_then_else({args[2].varname}{idx_strs[2]} > {args[3].varname}{idx_strs[3]}, {args[1].varname}{idx_strs[1]}, float(0))"
            )
        else:  # TODO
            raise NotImplementedError(str(type))

        code.less_indent()
    else:
        raise NotImplementedError
    return code


def to_cute_op(type: str, *args: SymbolScalar):
    code = IndentedCode()
    if type == "ReduceSum":
        code.add_line(
            f"flash::reduce_sum</*zero_init=*/true, /*warp_reduce=*/true>({args[1].varname}, {args[0].varname});"
        )
    elif type == "ReduceMax":
        code.add_line(
            f"flash::template reduce_max</*zero_init=*/true>({args[1].varname}, {args[0].varname});"
        )
    elif type == "Sub" or type == "Add" or type == "Mul" or type == "Div" or type == "Neg" or type == "Exp" or type == "Exp2" or type == "Log" or type == "Abs" or type == "Max" or type == "Tanh":
        # args idx
        # note: assume input shape is validate: ["1",...] or [arg0[0], ...]
        idx_strs = []
        idx_lists = []
        for _, arg in enumerate(args):
            input_idx = arg.shape_idx
            idx_list = [
                f"i{i}" if idx != "1" else f"0" for i,
                idx in enumerate(input_idx)]
            idx_list = [] if idx_list == ["0"] else idx_list
            idx_str = ",".join(idx_list)
            idx_strs.append(idx_str)
            idx_lists.append(idx_list)
        # [block_M,block_N]
        loop_str = ",".join(args[0].shape_idx)
        idx_str = idx_strs[0]
        idx_list = idx_lists[0]

        # for loop
        for ii, idx in enumerate(idx_list):
            code.add_line(
                "#pragma unroll"
            )
            code.add_line(
                f"for (int {idx}=0; {idx} < size<{ii}>({args[0].varname}); {idx}++) {{"
            )
            code.more_indent()

        # remove [] for scalar
        for i, tmp_idx_str in enumerate(idx_strs):
            idx_strs[i] = f"({tmp_idx_str})" if len(
                tmp_idx_str) > 0 else tmp_idx_str

        if type == "Sub":
            code.add_line(
                f"{args[0].varname}({idx_str}) = {args[1].varname}{idx_strs[1]} - {args[2].varname}{idx_strs[2]};"
            )
        elif type == "Add":
            code.add_line(
                f"{args[0].varname}({idx_str}) = {args[1].varname}{idx_strs[1]} + {args[2].varname}{idx_strs[2]};"
            )
        elif type == "Max":
            code.add_line(
                f"{args[0].varname}({idx_str}) = cute::max({args[1].varname}{idx_strs[1]}, {args[2].varname}{idx_strs[2]});"
            )
        elif type == "Exp":
            code.add_line(
                f"{args[0].varname}({idx_str}) = exp2f({args[1].varname}{idx_strs[1]}*1.442695);"
            )
        elif type == "Exp2":
            code.add_line(
                f"{args[0].varname}({idx_str}) = exp2f({args[1].varname}{idx_strs[1]});"
            )
        elif type == "Mul":
            code.add_line(
                f"{args[0].varname}({idx_str}) = {args[1].varname}{idx_strs[1]} * {args[2].varname}{idx_strs[2]};"
            )
        elif type == "Div":
            code.add_line(
                f"{args[0].varname}({idx_str}) = {args[1].varname}{idx_strs[1]} / {args[2].varname}{idx_strs[2]};"
            )
        elif type == "Log":
            code.add_line(
                f"{args[0].varname}({idx_str}) = __logf({args[1].varname}{idx_strs[1]});"
            )
        elif type == "Tanh":
            code.add_line(
                f"{args[0].varname}({idx_str}) = cutlass::fast_tanh({args[1].varname}{idx_strs[1]});"
            )
        elif type == "Abs":
            code.add_line(
                f"{args[0].varname}({idx_str}) = cute::abs({args[1].varname}{idx_strs[1]});"
            )
        else:  # TODO
            raise NotImplementedError(str(type))

        for ii, idx in enumerate(idx_list):
            code.less_indent()
            code.add_line(
                "}"
            )
    else:
        raise NotImplementedError
    # debug
    code.add_line(
        f"// used: {args[0].count}, use_list: {[usea.varname for usea in args[0].use_list]}"
    )
    return code


def to_pytorch_op(type: str, *args: SymbolScalar):
    code = IndentedCode()
    if type == "ReduceSum":
        code.add_line(
            f"{args[0].varname} = torch.sum({args[1].varname}, dim=-1)"
        )
    elif type == "ReduceMax":
        code.add_line(
            f"{args[0].varname} = torch.max({args[1].varname}, dim=-1)"
        )
    elif type == "Sub" or type == "Add" or type == "Mul" or type == "Div" or type == "Neg" or type == "Exp" or type == "Log" or type == "Abs" or type == "Max":
        # args idx
        # assume outputn dim max
        output_idx = args[0].shape_idx
        argnames = [arg.varname for arg in args]

        # for bwd grad
        max_len = max(len(arg.shape_idx) for arg in args)
        if len(output_idx) < max_len:
            suffix_code = f"{argnames[0]} = {argnames[0]}.sum(dim=({','.join([str(-i) for i in range(1,1+max_len-len(output_idx))])}))"
            output_idx += [f"_{i}" for i in range(max_len - len(output_idx))]
        else:
            suffix_code = ""

        for i, arg in enumerate(args):
            assert (len(arg.shape_idx) <= len(output_idx))
            if len(arg.shape_idx) == 0:
                continue
            if len(arg.shape_idx) < len(output_idx):
                # a -> a[...,None]
                argnames[i] = f"{arg.varname}[..." + ",None" * \
                    (len(output_idx) - len(arg.shape_idx)) + "]"

        if type == "Sub":
            code.add_line(
                f"{argnames[0]} = {argnames[1]} - {argnames[2]}"
            )
        elif type == "Add":
            code.add_line(
                f"{argnames[0]} = {argnames[1]} + {argnames[2]}"
            )
        elif type == "Max":
            code.add_line(
                f"{argnames[0]} = torch.maximum({argnames[1]}, {argnames[2]})"
            )
        elif type == "Exp":
            code.add_line(
                f"{argnames[0]} = torch.exp({argnames[1]})"
            )
        elif type == "Mul":
            code.add_line(
                f"{argnames[0]} = {argnames[1]} * {argnames[2]}"
            )
        elif type == "Div":
            code.add_line(
                f"{argnames[0]} = {argnames[1]} / {argnames[2]}"
            )
        elif type == "Log":
            code.add_line(
                f"{argnames[0]} = torch.log({argnames[1]})"
            )
        else:  # TODO
            raise NotImplementedError(str(type))

        code.add_line(suffix_code)
    else:
        raise NotImplementedError
    return code


def generate_tl_from_dag(x_list: list[SymbolScalar], to_tl: bool = True, to_cute: bool = False,
                         output_var_name_list=None, return_inputs=False) -> Tuple[IndentedCode, dict]:
    # global var
    input_vars = {}
    inputs = {}

    def generate_tl(x: SymbolScalar, varname: str = None):
        tl_code = IndentedCode()
        if x.lowered:
            return tl_code
        if isinstance(x.code, Var):
            # tl_code.add_line(f"{x.varname} = {x.code.name}")
            # add input_var
            input_vars[x.varname] = x
            inputs[x.varname] = x
            return tl_code
        if isinstance(x.code, Const):
            return tl_code
        # for i, input_item in enumerate(x.code.inputs):
        #     tl_code += generate_tl(SymbolScalar(f"{x.varname}_i{i}", input_item))
        for i, input_item in enumerate(x.prev):
            tl_code += generate_tl(input_item)
            # if input_item.varname == "dsT_0":
            #     print("dsT_0::",input_item.varname)
            #     print("x:", x.varname)
            #     print(input_item.visit_count)
        # all previous node be generated before visit_count+1
        for i, input_item in enumerate(x.prev):
            input_item.visit_count += 1

        # optimize tl performance by inplace operation
        # print(x.varname)
        # print([x.varname for x in x.prev])
        # print("count:",[x.count for x in x.prev])
        # print("visit:",[x.visit_count for x in x.prev])
        # print("use_list:",[[usea.varname for usea in x.use_list] for x in x.prev])
        for i, input_item in enumerate(x.prev):
            if input_item.shape_idx == x.shape_idx:
                if (input_item.count == 1 or input_item.visit_count == input_item.count) and input_item.allow_reuse:
                    x.varname = input_item.varname
                    break
        if varname is not None:  # overwrite varname
            x.varname = varname
        # add input_var
        if x.varname not in input_vars.keys():
            input_vars[x.varname] = x
        # tl_code += to_tl_op(x.code.type, x, *[SymbolScalar(f"{x.varname}_i{i}", input_item) for i, input_item in enumerate(x.code.inputs)])
        if to_tl:
            tl_code += to_tl_op(x.code.type, x, *x.prev)
        elif to_cute:
            tl_code += to_cute_op(x.code.type, x, *x.prev)
        else:
            tl_code += to_pytorch_op(x.code.type, x, *x.prev)
        x.lowered = True
        return tl_code

    tl_code = IndentedCode()
    for idx, x in enumerate(x_list):
        tl_code += generate_tl(
            x,
            varname=output_var_name_list[idx] if output_var_name_list is not None else None)
    if not return_inputs:
        return tl_code, input_vars
    else:
        return tl_code, input_vars, inputs
