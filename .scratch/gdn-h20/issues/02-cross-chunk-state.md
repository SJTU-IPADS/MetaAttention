# 02 — 跨 chunk 状态传递

**What to build:** 让 H20 用户通过同一 GDN Engine 运行多个 64-token chunk，选择性提供 FP32 initial state、请求 FP32 final state，并覆盖自定义 query scale。完成后，用户可以验证完整跨 chunk recurrence，而不是只能运行孤立的单块计算。

**Blocked by:** 01 — H20 单 chunk GDN 前向.

**Status:** ready-for-agent

- [ ] 对任意正且 64 对齐的固定长度 T，output 与项目内 FP64 reference 的相对 L2 误差不超过 0.02。
- [ ] 缺少 initial state 时使用零状态；提供时只接受 `[B,Hv,128,128]` FP32 state。
- [ ] `output_final_state=False` 时不返回或物化公开 final state；开启时返回 `[B,Hv,128,128]` FP32 final state。
- [ ] 零状态与非零 initial state 的 output/final state 均对齐 reference。
- [ ] 默认 scale 与显式自定义 Python float scale 均有行为测试。
- [ ] `T == 0` 或 `T % 64 != 0` 时显式抛出契约错误，不截断、不隐式 padding。
