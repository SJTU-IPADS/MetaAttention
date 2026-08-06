from core.lower.lower_linear import lower_tl


import importlib.util
import tempfile
import os
import hashlib
import sys


class LinearAttentionEngine:
    def __init__(self, qkv_meta, q_mod=None, k_mod=None, v_mod=None, decay_mod=None, custom_io=None,
                 tune=False, tune_filename="tune_result", tune_bwd=False):
        self._compile_tl(qkv_meta, q_mod, k_mod, v_mod, decay_mod, custom_io, tune=tune, tune_filename=tune_filename, tune_bwd=tune_bwd)


    def __call__(self, *args, **kargs):

        o = self.attention(*args, **kargs)
        return o

    def _compile_tl(self, qkv_meta, q_mod, k_mod, v_mod, decay_mod,
                    custom_io, tuned_config=None,
                    tune=False, tune_filename="", tune_bwd=False):
        tl_code = lower_tl(
            qkv_meta,
            q_mod,
            k_mod,
            v_mod,
            decay_mod,
            custom_io,
            tuned_config,
            tune=tune,
            tune_filename=tune_filename,
            tune_bwd=tune_bwd,)
        self.tl_code = tl_code  # for debug
        # local_vars = {}
        # exec(tl_code, globals(), local_vars)
        # globals().update(local_vars)
        # self.attention = local_vars["attention"]
        code_hash = hashlib.md5(tl_code.encode()).hexdigest()
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        file_path = os.path.join(cache_dir, f"{code_hash}.py")
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(tl_code)
                f.flush()
        # file_path = "/home/aiscuser/cfy/AttentionEngine/attn_script/retention_linear_tlcode1.py"
        if cache_dir not in sys.path:
            sys.path.append(cache_dir)
        module_name = code_hash 

        if module_name in sys.modules:
            # 如果已经加载过，直接复用，避免重复编译/初始化
            tl_attn = sys.modules[module_name]
        else:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            tl_attn = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = tl_attn 
            spec.loader.exec_module(tl_attn)
        # spec = importlib.util.spec_from_file_location("tl_attn", file_path)
        # tl_attn = importlib.util.module_from_spec(spec)
        # spec.loader.exec_module(tl_attn)
        self.attention = tl_attn.linear_attention
