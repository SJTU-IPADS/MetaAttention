# Copyright (c) 2023, Tri Dao.

from typing import Optional, Union

import torch
import torch.nn as nn

# isort: off
# We need to import the CUDA kernels after importing torch
# import flash_attn_3._C # Registers operators with PyTorch

# isort: on

# flash_attn_3_cuda = torch.ops.flash_attn_3
import shutil
import os
import sys
import subprocess
import stat
from pathlib import Path
from packaging.version import parse, Version
import warnings
import platform
import sysconfig
import tarfile
import itertools

import urllib.request
import urllib.error

import torch.utils.cpp_extension
from torch.utils.cpp_extension import CUDA_HOME

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"

include_path = [

]

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "flash_attn_3"

BASE_WHEEL_URL = "https://github.com/Dao-AILab/flash-attention/releases/download/{tag_name}/{wheel_name}"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FLASH_ATTENTION_FORCE_CXX11_ABI", "FALSE") == "TRUE"

DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"
DISABLE_SPLIT = True # os.getenv("FLASH_ATTENTION_DISABLE_SPLIT", "FALSE") == "TRUE"
DISABLE_PAGEDKV = True # os.getenv("FLASH_ATTENTION_DISABLE_PAGEDKV", "FALSE") == "TRUE"
DISABLE_APPENDKV = True # os.getenv("FLASH_ATTENTION_DISABLE_APPENDKV", "FALSE") == "TRUE"
DISABLE_LOCAL = True # os.getenv("FLASH_ATTENTION_DISABLE_LOCAL", "FALSE") == "TRUE"
DISABLE_SOFTCAP = True # os.getenv("FLASH_ATTENTION_DISABLE_SOFTCAP", "FALSE") == "TRUE"
DISABLE_PACKGQA = True # os.getenv("FLASH_ATTENTION_DISABLE_PACKGQA", "FALSE") == "TRUE"
DISABLE_FP16 = os.getenv("FLASH_ATTENTION_DISABLE_FP16", "FALSE") == "TRUE"
DISABLE_FP8 = True # os.getenv("FLASH_ATTENTION_DISABLE_FP8", "FALSE") == "TRUE"
DISABLE_VARLEN = True # os.getenv("FLASH_ATTENTION_DISABLE_VARLEN", "FALSE") == "TRUE"
DISABLE_CLUSTER = os.getenv("FLASH_ATTENTION_DISABLE_CLUSTER", "FALSE") == "TRUE"
DISABLE_HDIM64 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM64", "FALSE") == "TRUE"
DISABLE_HDIM96 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM96", "FALSE") == "TRUE"
DISABLE_HDIM128 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM128", "FALSE") == "TRUE"
DISABLE_HDIM192 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM192", "FALSE") == "TRUE"
DISABLE_HDIM256 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM256", "FALSE") == "TRUE"
DISABLE_SM8x = True # os.getenv("FLASH_ATTENTION_DISABLE_SM80", "FALSE") == "TRUE"

ENABLE_VCOLMAJOR = False # os.getenv("FLASH_ATTENTION_ENABLE_VCOLMAJOR", "FALSE") == "TRUE"


# HACK: we monkey patch pytorch's _write_ninja_file to pass
# "-gencode arch=compute_sm90a,code=sm_90a" to files ending in '_sm90.cu',
# and pass "-gencode arch=compute_sm80,code=sm_80" to files ending in '_sm80.cu'
from torch.utils.cpp_extension import (
    IS_HIP_EXTENSION,
    COMMON_HIP_FLAGS,
    SUBPROCESS_DECODE_ARGS,
    IS_WINDOWS,
    get_cxx_compiler,
    _join_rocm_home,
    _join_cuda_home,
    _is_cuda_file,
    _maybe_write,
)

def _write_ninja_file(path,
                      cflags,
                      post_cflags,
                      cuda_cflags,
                      cuda_post_cflags,
                      cuda_dlink_post_cflags,
                      sources,
                      objects,
                      ldflags,
                      library_target,
                      with_cuda,
                      **kwargs,  # kwargs (ignored) to absorb new flags in torch.utils.cpp_extension
                      ) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `cuda_cflags`: list of flags to pass to $nvcc. Can be None.
    `cuda_postflags`: list of flags to append to the $nvcc invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_cuda`: If we should be compiling with CUDA.
    """
    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = sanitize_flags(cuda_dlink_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    assert len(sources) == len(objects)
    assert len(sources) > 0

    compiler = get_cxx_compiler()

    # Version 1.3 is required for the `deps` directive.
    config = ['ninja_required_version = 1.3']
    config.append(f'cxx = {compiler}')
    if with_cuda or cuda_dlink_post_cflags:
        if IS_HIP_EXTENSION:
            nvcc = _join_rocm_home('bin', 'hipcc')
        else:
            nvcc = _join_cuda_home('bin', 'nvcc')
        if "PYTORCH_NVCC" in os.environ:
            nvcc_from_env = os.getenv("PYTORCH_NVCC")    # user can set nvcc compiler with ccache using the environment variable here
        else:
            nvcc_from_env = nvcc
        config.append(f'nvcc_from_env = {nvcc_from_env}')
        config.append(f'nvcc = {nvcc}')

    if IS_HIP_EXTENSION:
        post_cflags = COMMON_HIP_FLAGS + post_cflags
    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
        cuda_post_cflags_sm80 = [s if s != 'arch=compute_90a,code=sm_90a' else 'arch=compute_80,code=sm_80' for s in cuda_post_cflags]
        flags.append(f'cuda_post_cflags_sm80 = {" ".join(cuda_post_cflags_sm80)}')
        cuda_post_cflags_sm80_sm90 = cuda_post_cflags + ['-gencode', 'arch=compute_80,code=sm_80']
        flags.append(f'cuda_post_cflags_sm80_sm90 = {" ".join(cuda_post_cflags_sm80_sm90)}')
    flags.append(f'cuda_dlink_post_cflags = {" ".join(cuda_dlink_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ['rule compile']
    if IS_WINDOWS:
        compile_rule.append(
            '  command = cl /showIncludes $cflags -c $in /Fo$out $post_cflags')
        compile_rule.append('  deps = msvc')
    else:
        compile_rule.append(
            '  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags')
        compile_rule.append('  depfile = $out.d')
        compile_rule.append('  deps = gcc')

    if with_cuda:
        cuda_compile_rule = ['rule cuda_compile']
        nvcc_gendeps = ''
        # --generate-dependencies-with-compile is not supported by ROCm
        # Nvcc flag `--generate-dependencies-with-compile` is not supported by sccache, which may increase build time.
        if torch.version.cuda is not None and os.getenv('TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES', '0') != '1':
            cuda_compile_rule.append('  depfile = $out.d')
            cuda_compile_rule.append('  deps = gcc')
            # Note: non-system deps with nvcc are only supported
            # on Linux so use --generate-dependencies-with-compile
            # to make this work on Windows too.
            nvcc_gendeps = '--generate-dependencies-with-compile --dependency-output $out.d'
        cuda_compile_rule_sm80 = ['rule cuda_compile_sm80'] + cuda_compile_rule[1:] + [
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm80'
        ]
        cuda_compile_rule_sm80_sm90 = ['rule cuda_compile_sm80_sm90'] + cuda_compile_rule[1:] + [
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm80_sm90'
        ]
        cuda_compile_rule.append(
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags')

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        if is_cuda_source:
            if source_file.endswith('_sm90.cu'):
                rule = 'cuda_compile'
            elif source_file.endswith('_sm80.cu'):
                rule = 'cuda_compile_sm80'
            else:
                rule = 'cuda_compile_sm80_sm90'
        else:
            rule = 'compile'
        if IS_WINDOWS:
            source_file = source_file.replace(':', '$:')
            object_file = object_file.replace(':', '$:')
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f'build {object_file}: {rule} {source_file}')

    if cuda_dlink_post_cflags:
        devlink_out = os.path.join(os.path.dirname(objects[0]), 'dlink.o')
        devlink_rule = ['rule cuda_devlink']
        devlink_rule.append('  command = $nvcc $in -o $out $cuda_dlink_post_cflags')
        devlink = [f'build {devlink_out}: cuda_devlink {" ".join(objects)}']
        objects += [devlink_out]
    else:
        devlink_rule, devlink = [], []

    if library_target is not None:
        link_rule = ['rule link']
        if IS_WINDOWS:
            cl_paths = subprocess.check_output(['where',
                                                'cl']).decode(*SUBPROCESS_DECODE_ARGS).split('\r\n')
            if len(cl_paths) >= 1:
                cl_path = os.path.dirname(cl_paths[0]).replace(':', '$:')
            else:
                raise RuntimeError("MSVC is required to load C++ extensions")
            link_rule.append(f'  command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out')
        else:
            link_rule.append('  command = $cxx $in $ldflags -o $out')

        link = [f'build {library_target}: link {" ".join(objects)}']

        default = [f'default {library_target}']
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm80)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm80_sm90)  # type: ignore[possibly-undefined]
    blocks += [devlink_rule, link_rule, build, devlink, link, default]
    content = "\n\n".join("\n".join(b) for b in blocks)
    # Ninja requires a new lines at the end of the .ninja file
    content += "\n"
    _maybe_write(path, content)


# Monkey patching
# torch.utils.cpp_extension._write_ninja_file = _write_ninja_file


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


# Taken from https://github.com/pytorch/pytorch/blob/master/tools/setup_helpers/env.py
def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


# Copied from https://github.com/triton-lang/triton/blob/main/python/setup.py
def is_offline_build() -> bool:
    """
    Downstream projects and distributions which bootstrap their own dependencies from scratch
    and run builds in offline sandboxes
    may set `FLASH_ATTENTION_OFFLINE_BUILD` in the build environment to prevent any attempts at downloading
    pinned dependencies from the internet or at using dependencies vendored in-tree.

    Dependencies must be defined using respective search paths (cf. `syspath_var_name` in `Package`).
    Missing dependencies lead to an early abortion.
    Dependencies' compatibility is not verified.

    Note that this flag isn't tested by the CI and does not provide any guarantees.
    """
    return check_env_flag("FLASH_ATTENTION_OFFLINE_BUILD", "")


# Copied from https://github.com/triton-lang/triton/blob/main/python/setup.py
def get_flashattn_cache_path():
    user_home = os.getenv("FLASH_ATTENTION_HOME")
    if not user_home:
        user_home = os.getenv("HOME") or os.getenv("USERPROFILE") or os.getenv("HOMEPATH") or None
    if not user_home:
        raise RuntimeError("Could not find user home directory")
    return os.path.join(user_home, ".flashattn")


def open_url(url):
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0'
    headers = {
        'User-Agent': user_agent,
    }
    request = urllib.request.Request(url, None, headers)
    # Set timeout to 300 seconds to prevent the request from hanging forever.
    return urllib.request.urlopen(request, timeout=300)


def download_and_copy(name, src_func, dst_path, version, url_func):
    if is_offline_build():
        return
    flashattn_cache_path = get_flashattn_cache_path()
    base_dir = os.path.dirname(__file__)
    system = platform.system()
    arch = platform.machine()
    arch = {"arm64": "aarch64"}.get(arch, arch)
    supported = {"Linux": "linux", "Darwin": "linux"}
    url = url_func(supported[system], arch, version)
    src_path = src_func(supported[system], arch, version)
    tmp_path = os.path.join(flashattn_cache_path, "nvidia", name)  # path to cache the download
    dst_path = os.path.join(base_dir, os.pardir, "third_party", "nvidia", "backend", dst_path)  # final binary path
    src_path = os.path.join(tmp_path, src_path)
    download = not os.path.exists(src_path)
    if download:
        print(f'downloading and extracting {url} ...')
        file = tarfile.open(fileobj=open_url(url), mode="r|*")
        file.extractall(path=tmp_path)
    os.makedirs(os.path.split(dst_path)[0], exist_ok=True)
    print(f'copy {src_path} to {dst_path} ...')
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    else:
        shutil.copy(src_path, dst_path)


def nvcc_threads_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "2"
    return ["--threads", nvcc_threads]


# NVIDIA_TOOLCHAIN_VERSION = {"nvcc": "12.3.107"}
NVIDIA_TOOLCHAIN_VERSION = {"nvcc": "12.6.85", "ptxas": "12.8.93"}

exe_extension = sysconfig.get_config_var("EXE")


cmdclass = {}
ext_modules = []

# We want this even if SKIP_CUDA_BUILD because when we run python setup.py sdist we want the .hpp
# files included in the source distribution, in case the user compiles from source.
# subprocess.run(["git", "submodule", "update", "--init", "../csrc/cutlass"])

if not SKIP_CUDA_BUILD:
    # print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    check_if_cuda_home_none(PACKAGE_NAME)
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.3"):
        raise RuntimeError("FlashAttention-3 is only supported on CUDA 12.3 and above")

    # TODO: avoid download nvcc
    # ptxas 12.8 gives the best perf currently
    # We want to use the nvcc front end from 12.6 however, since if we use nvcc 12.8
    # Cutlass 3.8 will expect the new data types in cuda.h from CTK 12.8, which we don't have.
    if bare_metal_version != Version("12.8"):
        download_and_copy(
            name="nvcc",
            src_func=lambda system, arch, version: f"cuda_nvcc-{system}-{arch}-{version}-archive/bin",
            dst_path="bin",
            version=NVIDIA_TOOLCHAIN_VERSION["nvcc"],
            url_func=lambda system, arch, version:
            f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/{system}-{arch}/cuda_nvcc-{system}-{arch}-{version}-archive.tar.xz",
        )
        download_and_copy(
            name="ptxas",
            src_func=lambda system, arch, version: f"cuda_nvcc-{system}-{arch}-{version}-archive/bin/ptxas",
            dst_path="bin",
            version=NVIDIA_TOOLCHAIN_VERSION["ptxas"],
            url_func=lambda system, arch, version:
            f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/{system}-{arch}/cuda_nvcc-{system}-{arch}-{version}-archive.tar.xz",
        )
        download_and_copy(
            name="ptxas",
            src_func=lambda system, arch, version: f"cuda_nvcc-{system}-{arch}-{version}-archive/nvvm/bin",
            dst_path="nvvm/bin",
            version=NVIDIA_TOOLCHAIN_VERSION["ptxas"],
            url_func=lambda system, arch, version:
            f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/{system}-{arch}/cuda_nvcc-{system}-{arch}-{version}-archive.tar.xz",
        )
        base_dir = os.path.dirname(__file__)
        ctk_path_new = os.path.abspath(os.path.join(base_dir, os.pardir, "third_party", "nvidia", "backend", "bin"))
        nvcc_path_new = os.path.join(ctk_path_new, f"nvcc{exe_extension}")
        # Need to append to path otherwise nvcc can't find cicc in nvvm/bin/cicc
        # nvcc 12.8 seems to hard-code looking for cicc in ../nvvm/bin/cicc
        os.environ["PATH"] = ctk_path_new + os.pathsep + os.environ["PATH"]
        os.environ["PYTORCH_NVCC"] = nvcc_path_new
        # Make nvcc executable, sometimes after the copy it loses its permissions
        os.chmod(nvcc_path_new, os.stat(nvcc_path_new).st_mode | stat.S_IEXEC)

    cc_flag = []
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_90a,code=sm_90a")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    repo_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../"))
    cutlass_dir = repo_dir / "3rd_parties" / "cutlass_dc4817"

    feature_args = (
        []
        + (["-DFLASHATTENTION_DISABLE_BACKWARD"] if DISABLE_BACKWARD else [])
        + (["-DFLASHATTENTION_DISABLE_PAGEDKV"] if DISABLE_PAGEDKV else [])
        + (["-DFLASHATTENTION_DISABLE_SPLIT"] if DISABLE_SPLIT else [])
        + (["-DFLASHATTENTION_DISABLE_APPENDKV"] if DISABLE_APPENDKV else [])
        + (["-DFLASHATTENTION_DISABLE_LOCAL"] if DISABLE_LOCAL else [])
        + (["-DFLASHATTENTION_DISABLE_SOFTCAP"] if DISABLE_SOFTCAP else [])
        + (["-DFLASHATTENTION_DISABLE_PACKGQA"] if DISABLE_PACKGQA else [])
        + (["-DFLASHATTENTION_DISABLE_FP16"] if DISABLE_FP16 else [])
        + (["-DFLASHATTENTION_DISABLE_FP8"] if DISABLE_FP8 else [])
        + (["-DFLASHATTENTION_DISABLE_VARLEN"] if DISABLE_VARLEN else [])
        + (["-DFLASHATTENTION_DISABLE_CLUSTER"] if DISABLE_CLUSTER else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM64"] if DISABLE_HDIM64 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM96"] if DISABLE_HDIM96 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM128"] if DISABLE_HDIM128 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM192"] if DISABLE_HDIM192 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM256"] if DISABLE_HDIM256 else [])
        + (["-DFLASHATTENTION_DISABLE_SM8x"] if DISABLE_SM8x else [])
        + (["-DFLASHATTENTION_ENABLE_VCOLMAJOR"] if ENABLE_VCOLMAJOR else [])
    )

    # we define sources here
    sources = [
        os.path.join(os.path.dirname(__file__), "flash_api.cpp"),
        os.path.join(os.path.dirname(__file__), "flash_fwd_sm90.cu"),
        os.path.join(os.path.dirname(__file__), "flash_bwd_sm90.cu"),
        os.path.join(os.path.dirname(__file__), "flash_prepare_scheduler.cu"),
    ]
    # DTYPE_FWD_SM80 = ["bf16"] + (["fp16"] if not DISABLE_FP16 else [])
    # DTYPE_FWD_SM90 = ["bf16"] + (["fp16"] if not DISABLE_FP16 else []) + (["e4m3"] if not DISABLE_FP8 else [])
    # DTYPE_BWD = ["bf16"] + (["fp16"] if not DISABLE_FP16 else [])
    # HEAD_DIMENSIONS_BWD = (
    #     []
    #     + ([64] if not DISABLE_HDIM64 else [])
    #     + ([96] if not DISABLE_HDIM96 else [])
    #     + ([128] if not DISABLE_HDIM128 else [])
    #     + ([192] if not DISABLE_HDIM192 else [])
    #     + ([256] if not DISABLE_HDIM256 else [])
    # )
    # HEAD_DIMENSIONS_FWD = ["all", "diff"]
    # HEAD_DIMENSIONS_FWD_SM80 = HEAD_DIMENSIONS_BWD
    # SPLIT = [""] + (["_split"] if not DISABLE_SPLIT else [])
    # PAGEDKV = [""] + (["_paged"] if not DISABLE_PAGEDKV else [])
    # SOFTCAP = [""] + (["_softcap"] if not DISABLE_SOFTCAP else [])
    # SOFTCAP_ALL = [""] if DISABLE_SOFTCAP else ["_softcapall"]
    # PACKGQA = [""] + (["_packgqa"] if not DISABLE_PACKGQA else [])
    # # We already always hard-code PackGQA=true for Sm8x
    # sources_fwd_sm80 = [f"instantiations/flash_fwd_hdim{hdim}_{dtype}{paged}{split}{softcap}_sm80.cu"
    #                     for hdim, dtype, split, paged, softcap in itertools.product(HEAD_DIMENSIONS_FWD_SM80, DTYPE_FWD_SM80, SPLIT, PAGEDKV, SOFTCAP_ALL)]
    # # We already always hard-code PackGQA=true for Sm9x if PagedKV or Split
    # sources_fwd_sm90 = [f"instantiations/flash_fwd_hdim{hdim}_{dtype}{paged}{split}{softcap}{packgqa}_sm90.cu"
    #                     for hdim, dtype, split, paged, softcap, packgqa in itertools.product(HEAD_DIMENSIONS_FWD, DTYPE_FWD_SM90, SPLIT, PAGEDKV, SOFTCAP, PACKGQA)
    #                     if not (packgqa and (paged or split))]
    # sources_bwd_sm80 = [f"instantiations/flash_bwd_hdim{hdim}_{dtype}{softcap}_sm80.cu"
    #                     for hdim, dtype, softcap in itertools.product(HEAD_DIMENSIONS_BWD, DTYPE_BWD, SOFTCAP)]
    # sources_bwd_sm90 = [f"instantiations/flash_bwd_hdim{hdim}_{dtype}{softcap}_sm90.cu"
    #                     for hdim, dtype, softcap in itertools.product(HEAD_DIMENSIONS_BWD, DTYPE_BWD, SOFTCAP_ALL)]
    # if DISABLE_BACKWARD:
    #     sources_bwd_sm90 = []
    #     sources_bwd_sm80 = []
    # sources = (
    #     ["flash_api.cpp"]
    #     + (sources_fwd_sm80 if not DISABLE_SM8x else []) + sources_fwd_sm90
    #     + (sources_bwd_sm80 if not DISABLE_SM8x else []) + sources_bwd_sm90
    # )
    # if not DISABLE_SPLIT:
    #     sources += ["flash_fwd_combine.cu"]
    # sources += ["flash_prepare_scheduler.cu"]
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "--ftemplate-backtrace-limit=0",  # To debug template code
        "--use_fast_math",
        # "--keep",
        # "--ptxas-options=--verbose,--register-usage-level=5,--warn-on-local-memory-usage",  # printing out number of registers
        "--resource-usage",  # printing out number of registers
        # f"--split-compile={os.getenv('NVCC_THREADS', '4')}",  # split-compile is faster
        "-lineinfo",
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",  # Necessary for the WGMMA shapes that we use
        "-DCUTLASS_ENABLE_GDC_FOR_SM90",  # For PDL
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
        "-DNDEBUG",  # Important, otherwise performance is severely impacted
    ]
    if get_platform() == "win_amd64":
        nvcc_flags.extend(
            [
                "-D_USE_MATH_DEFINES",  # for M_LN2
                "-Xcompiler=/Zc:__cplusplus",  # sets __cplusplus correctly, CUTLASS_CONSTEXPR_IF_CXX17 needed for cutlass::gcd
            ]
        )
    include_dirs = [
        str(Path(this_dir)),
        str(cutlass_dir / "include"),
    ]
    
    flash_attn_3_cuda = torch.utils.cpp_extension.load(
        name="flash_attn_3_hopper_cuda"+"{{dimqk}}_{{dimv}}_{{cutlass_dtype}}".replace("::", "_").replace(" ", "_"),
        sources=sources,
        extra_cflags=[
            "-O3", "-std=c++17",
        ] + feature_args,
        extra_cuda_cflags=nvcc_threads_args() + nvcc_flags + cc_flag + feature_args,
        extra_include_paths=include_dirs,
        with_cuda=True,
    )


# isort: on


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _flash_attn_forward(
        q,
        k,
        v,
        k_new,
        v_new,
        qv,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        kv_batch_idx,
        leftpad_k,
        rotary_cos,
        rotary_sin,
        seqlens_rotary,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        rotary_interleaved=True,
        scheduler_metadata=None,
        num_splits=1,
        pack_gqa=None,
        sm_margin=0):
    q, k, k_new, v_new = [maybe_contiguous(x) for x in (q, k, k_new, v_new)]
    v = v.contiguous() if v.stride(-1) != 1 and v.stride(-3) != 1 else v
    cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new = [
        maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new)
    ]
    seqused_q, seqused_k = [maybe_contiguous(x) for x in (seqused_q, seqused_k)]
    page_table, kv_batch_idx, leftpad_k = [
        maybe_contiguous(x) for x in (page_table, kv_batch_idx, leftpad_k)
    ]
    rotary_cos, rotary_sin = [maybe_contiguous(x) for x in (rotary_cos, rotary_sin)]
    seqlens_rotary = maybe_contiguous(seqlens_rotary)
    out, softmax_lse, *rest = flash_attn_3_cuda.fwd(
        q,
        k,
        v,
        k_new,
        v_new,
        qv,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        kv_batch_idx,
        leftpad_k,
        rotary_cos,
        rotary_sin,
        seqlens_rotary,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        attention_chunk,
        softcap,
        rotary_interleaved,
        scheduler_metadata,
        num_splits,
        pack_gqa,
        sm_margin,
    )
    return out, softmax_lse, *rest


def _flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        sequed_q,
        sequed_k,
        max_seqlen_q,
        max_seqlen_k,
        dq,
        dk,
        dv,
        softmax_scale,
        causal,
        window_size=(-1, -1),
        softcap=0.0,
        deterministic=False,
        sm_margin=0,
):
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    dq, dk, dv, softmax_d, *rest = flash_attn_3_cuda.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        sequed_q,
        sequed_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        deterministic,
        sm_margin,
    )
    return dq, dk, dv, softmax_d


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        softmax_scale,
        causal,
        q_descale=None, k_descale=None, v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        deterministic=False,
        num_heads_q=None,
        sm_margin=0,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        if qkv.dim() == 5:
            assert qkv.shape[-3] == 3
            q, k, v = qkv.unbind(dim=-3)
        else:
            assert qkv.dim() == 4
            assert num_heads_q is not None
            num_heads_k = (qkv.shape[2] - num_heads_q) // 2
            assert num_heads_k * 2 + num_heads_q == qkv.shape[2]
            q, k, v = qkv.split([num_heads_q, num_heads_k, num_heads_k], dim=-2)
        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None, None,  # k_new, v_new
            None,  # qv
            None,  # out
            None, None, None,   # cu_seqlens_q/k/k_new
            None, None,   # seqused_q/k
            None, None,   # max_seqlen_q/k
            None, None, None,   # page_table, kv_batch_idx, leftpad_k,
            None, None, None,  # rotary_cos/sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            sm_margin=sm_margin,
        )
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.attention_chunk = attention_chunk
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.ndim = qkv.dim()
        ctx.sm_margin = sm_margin
        # return out, softmax_lse
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"
        if ctx.ndim == 5:
            qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
            dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
            dq, dk, dv = dqkv.unbind(dim=-3)
        else:
            num_heads_q = q.shape[2]
            num_heads_k = k.shape[2]
            qkv_shape = q.shape[:-2] + (num_heads_q + num_heads_k * 2, *q.shape[-1:])
            dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
            dq, dk, dv = dqkv.split([num_heads_q, num_heads_k, num_heads_k], dim=-2)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None, None, # cu_seqlens_q, cu_seqlens_k,
            None, None, # sequed_q, sequed_k,
            None, None, # max_seqlen_q, max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
        )
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv=None,
        q_descale=None, k_descale=None, v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
    ):
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
        # out, q, k, v, out_padded, softmax_lse = _flash_attn_forward(
        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None, None,  # k_new, v_new
            qv,  # qv
            None,  # out
            None, None, None,   # cu_seqlens_q/k/k_new
            None, None,   # seqused_q/k
            None, None,   # max_seqlen_q/k
            None, None, None,   # page_table, kv_batch_idx, leftpad_k,
            None, None, None,  # rotary_cos/sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
        )
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.attention_chunk = attention_chunk
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        return out# , softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None, None, # cu_seqlens_q, cu_seqlens_k,
            None, None, # sequed_q, sequed_k,
            None, None, # max_seqlen_q, max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
        )
        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        qv=None,
        q_descale=None, k_descale=None, v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
    ):
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
        # out, q, k, v, out_padded, softmax_lse = _flash_attn_varlen_forward(
        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None, None,  # k_new, v_new
            qv,  # qv
            None,  # out
            cu_seqlens_q,
            cu_seqlens_k,
            None,   # cu_seqlens_k_new
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            None, None, None,   # page_table, kv_batch_idx, leftpad_k,
            None, None, None,  # rotary_cos/sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
        )
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.attention_chunk = attention_chunk
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        return out, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = ctx.saved_tensors
        assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
        )
        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def flash_attn_qkvpacked_func(
    qkv,
    softmax_scale=None,
    causal=False,
    q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    deterministic=False,
    num_heads_q=None,
    sm_margin=0,
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_kvpacked_func and flash_attn_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to
            the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnQKVPackedFunc.apply(
        qkv,
        softmax_scale,
        causal,
        q_descale, k_descale, v_descale,
        window_size,
        attention_chunk,
        softcap,
        deterministic,
        num_heads_q,
        sm_margin,
    )


def flash_attn_func(
    q,
    k,
    v,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv,
        q_descale, k_descale, v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
    )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqused_q=None,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
):
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        qv,
        q_descale, k_descale, v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
    )


def flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None):
    return flash_attn_3_cuda.fwd_combine(out_partial, lse_partial, out, out_dtype)


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk=0,
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,    # Can be tuned for speed
    pack_gqa=None,   # Can be tuned for speed
    sm_margin=0,     # Can be tuned if some SMs are used for communication
    return_softmax_lse=False,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no page_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a page_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim_v) if there's no page_table,
            or (num_blocks, page_block_size, nheads_k, headdim_v) if there's a page_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim_v). Similar to k.
        qv [optional]: (batch_size, seqlen, nheads, headdim_v)
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
        page_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    out, softmax_lse, *rest = _flash_attn_forward(
        q,
        k_cache,
        v_cache,
        k,
        v,
        qv,
        None,  # out
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_seqlens,
        max_seqlen_q,
        None,  # max_seqlen_k
        page_table,
        cache_batch_idx,
        cache_leftpad,
        rotary_cos,
        rotary_sin,
        rotary_seqlens,
        q_descale, k_descale, v_descale,
        softmax_scale,
        causal=causal,
        window_size=window_size,
        attention_chunk=attention_chunk,
        softcap=softcap,
        rotary_interleaved=rotary_interleaved,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
    )
    # return (out, softmax_lse) if return_softmax_lse else out
    return (out, softmax_lse, *rest) if return_softmax_lse else out


def get_scheduler_metadata(
    batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_kv, headdim,
    cache_seqlens: torch.Tensor,
    qkv_dtype=torch.bfloat16,
    headdim_v=None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    max_seqlen_k_new=0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk=0,
    has_softcap=False,
    num_splits=0,    # Can be tuned for speed
    pack_gqa=None,   # Can be tuned for speed
    sm_margin=0,     # Can be tuned if some SMs are used for communication
):
    cache_seqlens = maybe_contiguous(cache_seqlens)
    if headdim_v is None:
        headdim_v = headdim
    scheduler_metadata = flash_attn_3_cuda.get_scheduler_metadata(
        batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_kv, headdim, headdim_v,
        qkv_dtype,
        cache_seqlens,
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_leftpad,
        page_size,
        max_seqlen_k_new,
        causal,
        window_size[0], window_size[1],
        attention_chunk,
        has_softcap,
        num_splits,
        pack_gqa,
        sm_margin,
    )
    return scheduler_metadata
