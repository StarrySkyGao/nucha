import os
import re
import subprocess
import sys

import setuptools
import torch
from packaging import version as packaging_version
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension


class CustomBuildExtension(BuildExtension):
    # 重写build_extensions方法
    def build_extensions(self):
        # 遍历所有的扩展
        for ext in self.extensions:
            # 如果扩展的extra_compile_args中没有cxx，则添加一个空的列表
            if not "cxx" in ext.extra_compile_args:
                ext.extra_compile_args["cxx"] = []
            # 如果扩展的extra_compile_args中没有nvcc，则添加一个空的列表
            if not "nvcc" in ext.extra_compile_args:
                ext.extra_compile_args["nvcc"] = []
            # 如果编译器类型是msvc
            if self.compiler.compiler_type == "msvc":
                # 将扩展的extra_compile_args中的msvc添加到cxx中
                ext.extra_compile_args["cxx"] += ext.extra_compile_args["msvc"]
                # 将扩展的extra_compile_args中的nvcc_msvc添加到nvcc中
                ext.extra_compile_args["nvcc"] += ext.extra_compile_args["nvcc_msvc"]
            # 否则
            else:
                # 将扩展的extra_compile_args中的gcc添加到cxx中
                ext.extra_compile_args["cxx"] += ext.extra_compile_args["gcc"]
        # 调用父类的build_extensions方法
        super().build_extensions()


def get_sm_targets() -> list[str]:
    # 获取CUDA_HOME路径，如果没有设置CUDA_HOME，则使用nvcc
    nvcc_path = os.path.join(CUDA_HOME, "bin/nvcc") if CUDA_HOME else "nvcc"
    try:
        # 执行nvcc --version命令，获取nvcc版本信息
        nvcc_output = subprocess.check_output([nvcc_path, "--version"]).decode()
        # 使用正则表达式匹配nvcc版本信息
        match = re.search(r"release (\d+\.\d+), V(\d+\.\d+\.\d+)", nvcc_output)
        if match:
            # 获取nvcc版本号
            nvcc_version = match.group(2)
        else:
            # 如果没有匹配到nvcc版本信息，则抛出异常
            raise Exception("nvcc version not found")
        print(f"Found nvcc version: {nvcc_version}")
    except:
        # 如果nvcc没有找到，则抛出异常
        raise Exception("nvcc not found")

    # 判断nvcc版本是否支持sm120
    support_sm120 = packaging_version.parse(nvcc_version) >= packaging_version.parse("12.8")

    # 获取NUNCHAKU_INSTALL_MODE环境变量，如果没有设置，则默认为FAST
    install_mode = os.getenv("NUNCHAKU_INSTALL_MODE", "FAST")
    if install_mode == "FAST":
        # 如果NUNCHAKU_INSTALL_MODE为FAST，则获取支持的sm版本
        ret = []
        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            sm = f"{capability[0]}{capability[1]}"
            # 如果sm为120且nvcc版本支持sm120，则将sm改为120a
            if sm == "120" and support_sm120:
                sm = "120a"
            # 判断sm是否在支持的版本中，如果不是，则抛出异常
            assert sm in ["75", "80", "86", "89", "120a"], f"Unsupported SM {sm}"
            # 如果sm不在ret中，则添加到ret中
            if sm not in ret:
                ret.append(sm)
    else:
        # 如果NUNCHAKU_INSTALL_MODE不为FAST，则判断是否为ALL
        assert install_mode == "ALL"
        # 如果为ALL，则支持的sm版本为75，80，86，89，120a
        ret = ["75", "80", "86", "89"]
        # 如果nvcc版本支持sm120，则添加120a到ret中
        if support_sm120:
            ret.append("120a")
    # 返回支持的sm版本
    return ret


if __name__ == "__main__":
    fp = open("nunchaku/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])  # eval就是为了去掉双引号，将字符串字面量转换为字符串对象

    torch_version = torch.__version__.split("+")[0]
    torch_major_minor_version = ".".join(torch_version.split(".")[:2])
    version = version + "+torch" + torch_major_minor_version # 版本号拼接 eg：0.3.0dev1+torch2.6

    ROOT_DIR = os.path.dirname(__file__) # 获取当前文件的目录

    INCLUDE_DIRS = [
        "src",
        "third_party/cutlass/include",
        "third_party/json/include",
        "third_party/mio/include",
        "third_party/spdlog/include",
        "third_party/Block-Sparse-Attention/csrc/block_sparse_attn",
    ]

    INCLUDE_DIRS = [os.path.join(ROOT_DIR, dir) for dir in INCLUDE_DIRS]

    DEBUG = False

    def ncond(s) -> list:
        if DEBUG:
            return []
        else:
            return [s]

    def cond(s) -> list:
        if DEBUG:
            return [s]
        else:
            return []

    sm_targets = get_sm_targets()
    print(f"Detected SM targets: {sm_targets}", file=sys.stderr)

    assert len(sm_targets) > 0, "No SM targets found"

    GCC_FLAGS = ["-DENABLE_BF16=1", "-DBUILD_NUNCHAKU=1", "-fvisibility=hidden", "-g", "-std=c++20", "-UNDEBUG", "-Og"]
    MSVC_FLAGS = ["/DENABLE_BF16=1", "/DBUILD_NUNCHAKU=1", "/std:c++20", "/UNDEBUG", "/Zc:__cplusplus", "/FS"]
    NVCC_FLAGS = [
        "-DENABLE_BF16=1",
        "-DBUILD_NUNCHAKU=1",
        "-g",
        "-std=c++20",
        "-UNDEBUG",
        "-Xcudafe",
        "--diag_suppress=20208",  # spdlog: 'long double' is treated as 'double' in device code
        *cond("-G"),
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_HALF2_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        f"--threads={len(sm_targets)}",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=--allow-expensive-optimizations=true",
    ]

    if os.getenv("NUNCHAKU_BUILD_WHEELS", "0") == "0":
        NVCC_FLAGS.append("--generate-line-info")

    for target in sm_targets:
        NVCC_FLAGS += ["-gencode", f"arch=compute_{target},code=sm_{target}"]

    NVCC_MSVC_FLAGS = ["-Xcompiler", "/Zc:__cplusplus", "-Xcompiler", "/FS", "-Xcompiler", "/bigobj"]

    nunchaku_extension = CUDAExtension(
        name="nunchaku._C",
        sources=[
            # "nunchaku/csrc/pybind.cpp",
            # "src/interop/torch.cpp",
            # "src/activation.cpp",
            # "src/layernorm.cpp",
            # "src/Linear.cpp",
            # *ncond("src/FluxModel.cpp"),
            # *ncond("src/SanaModel.cpp"),
            # "src/Serialization.cpp",
            # "src/Module.cpp",
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_hdim64_fp16_sm80.cu"),
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_hdim64_bf16_sm80.cu"),
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_hdim128_fp16_sm80.cu"),
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_hdim128_bf16_sm80.cu"),
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_block_hdim64_fp16_sm80.cu"),
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_block_hdim64_bf16_sm80.cu"),
            *ncond(
                "third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_block_hdim128_fp16_sm80.cu"
            ),
            *ncond(
                "third_party/Block-Sparse-Attention/csrc/block_sparse_attn/src/flash_fwd_block_hdim128_bf16_sm80.cu"
            ),
            # "src/kernels/activation_kernels.cu",
            # "src/kernels/layernorm_kernels.cu",
            # "src/kernels/misc_kernels.cu",
            # "src/kernels/zgemm/gemm_w4a4.cu",
            # "src/kernels/zgemm/gemm_w4a4_test.cu",
            # "src/kernels/zgemm/gemm_w4a4_launch_fp16_int4.cu",
            # "src/kernels/zgemm/gemm_w4a4_launch_fp16_int4_fasteri2f.cu",
            # "src/kernels/zgemm/gemm_w4a4_launch_fp16_fp4.cu",
            # "src/kernels/zgemm/gemm_w4a4_launch_bf16_int4.cu",
            # "src/kernels/zgemm/gemm_w4a4_launch_bf16_fp4.cu",
            # "src/kernels/zgemm/gemm_w8a8.cu",
            # "src/kernels/zgemm/attention.cu",
            # "src/kernels/dwconv.cu",
            # "src/kernels/gemm_batched.cu",
            # "src/kernels/gemm_f16.cu",
            # "src/kernels/awq/gemm_awq.cu",
            # "src/kernels/awq/gemv_awq.cu",
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/flash_api.cpp"),
            *ncond("third_party/Block-Sparse-Attention/csrc/block_sparse_attn/flash_api_adapter.cpp"),
        ],
        extra_compile_args={"gcc": GCC_FLAGS, "msvc": MSVC_FLAGS, "nvcc": NVCC_FLAGS, "nvcc_msvc": NVCC_MSVC_FLAGS},
        include_dirs=INCLUDE_DIRS,
    )

    setuptools.setup(
        name="nunchaku_gmtest",
        version=version,
        packages=setuptools.find_packages(),
        ext_modules=[nunchaku_extension],
        cmdclass={"build_ext": CustomBuildExtension},
    )
