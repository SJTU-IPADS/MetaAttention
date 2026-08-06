from .arch_base import Arch


class A100(Arch):
    def __init__(self):
        self.name = "A100"
        self.reg_cap = 65536  # 32768
        self.register_per_thread = 255
        self.smem_cap = 163 * 1024  # 164*1024
        self.mma_primitive = [16, 8, 16]
        self.threads_per_mma = 32
        self.threads_cap = 1024

        self.compute_max_core = 108
        self.warp_size = 32

        self.sm_partition = 4
        self.transaction_size = [32, 128]   # in bytes
        self.bandwidth = [1319, 16308]
        self.platform = "CUDA"
        self.compute_capability = "80"
