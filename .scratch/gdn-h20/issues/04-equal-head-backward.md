# 04 — 等头 GDN 完整反向

**What to build:** 让训练用户在 `Hk == Hv` 下从 GDN output 反向传播到 query、key、value、log-space gate 和 beta。实现必须在 forward 保存 gate cumsum 与 KKT 结果，在 backward 重建 chunk states，避免重复 KKT 求解或保存所有前向 state。

**Blocked by:** 02 — 跨 chunk 状态传递.

**Status:** ready-for-agent

- [ ] output-only loss 产生 `dq`、`dk`、`dv`、`dg` 和 `dbeta`，每项相对 FP64 reference 的 L2 误差不超过 0.02。
- [ ] 单 chunk和多 chunk、零 initial state 和非零 initial state 均覆盖 backward parity。
- [ ] gate gradient 包含正确的 reverse chunk-local cumsum 语义。
- [ ] forward autograd context 保存 gate cumsum 与 KKT 结果；backward 重建所需 chunk states，不保存 `w/u`、corrected value 或全量 forward states。
- [ ] beta 接近 0 和 1、gate 接近 0 和显著负值的确定性场景均通过梯度检查。
- [ ] 现有 scalar-decay LinearAttentionEngine 的 CPU-safe 和可运行 GPU 测试保持不变。
