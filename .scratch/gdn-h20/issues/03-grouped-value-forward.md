# 03 — Grouped Value Attention 前向

**What to build:** 让 GDN Engine 支持 `Hv % Hk == 0` 的 GDN Grouped Value Attention。每个 query/key head 必须稳定映射到 `Hv/Hk` 个 value/state heads，用户可以在单 chunk 和多 chunk 固定长度输入上获得与 FP64 reference 一致的 output 和可选 final state。

**Blocked by:** 01 — H20 单 chunk GDN 前向.

**Status:** ready-for-agent

- [ ] 支持至少一个 `Hv > Hk` 的单 chunk GVA 场景，output 相对 FP64 reference 的 L2 误差不超过 0.02。
- [ ] 支持至少一个 `Hv > Hk` 的多 chunk GVA 场景，包括非零 initial state 和 final state 请求。
- [ ] query/key 到 value/state head 的映射与 `Hv/Hk` 分组契约一致，不引入运行时 transpose 或复制型 layout adapter。
- [ ] `Hv % Hk != 0`、零 head count 或头维度不一致时在 kernel 执行前明确拒绝。
- [ ] `Hk == Hv` 的既有前向行为保持不变。
