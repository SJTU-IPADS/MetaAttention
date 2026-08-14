# 01 — H20 单 chunk GDN 前向

**What to build:** 让 H20 用户通过 GDN Engine 的公开接口运行一个完整的 64-token Gated Delta Rule 前向。输入使用 head-first BF16 query/key/value 和 FP32 log-space gate/beta，采用零初始状态与默认 scale，返回 BF16 output；实际生成的 sm89 TileLang callable 必须覆盖 gate cumsum、因果 KKT 求解和 state/output 计算，并与项目内 FP64 reference 对齐。

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] GDN Engine 在 CUDA H20/sm89 上接受 `q/k:[B,Hk,64,128]`、`v:[B,Hv,64,128]`、`g/beta:[B,Hv,64]`，并返回 `[B,Hv,64,128]` BF16 output。
- [ ] 首个 tracer 场景要求 `Hk == Hv`、零初始状态和默认 `128**-0.5` scale，并执行真实生成的 TileLang kernel，而非 PyTorch fallback。
- [ ] 单 chunk output 相对 FP64 reference 的 L2 误差不超过 0.02。
- [ ] Engine 创建或调用前明确拒绝非 sm89 设备、错误 dtype、非 128 head dimension、错误布局或不匹配的 batch/sequence shape。
- [ ] CPU-safe reference、输入契约和 import 测试可在无 H20 环境运行；H20 功能测试使用现有 GPU marker/fixture 约定。
- [ ] 直接采用的 FlashQLA 数学参考或测试行为保留 MIT attribution。
