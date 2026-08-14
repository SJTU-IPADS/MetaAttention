# 05 — 状态与 GVA 梯度

**What to build:** 完成首期训练契约：final state 可参与 loss，provided initial state 可接收梯度，GVA 路径将复制的 query/key 梯度按 head group 正确归约。用户可以对 output、final state 或二者联合求导，并在 `Hv > Hk` 下获得完整梯度。

**Blocked by:** 03 — Grouped Value Attention 前向; 04 — 等头 GDN 完整反向.

**Status:** ready-for-agent

- [ ] final-state-only loss 产生正确的 `dq/dk/dv/dg/dbeta/dinitial_state`，每项相对 FP64 reference 的 L2 误差不超过 0.02。
- [ ] output-only、final-state-only 和联合 loss 的梯度行为分别验证。
- [ ] provided initial state 返回 FP32 `dinitial_state`；缺少 initial state 时不返回伪造的 state gradient。
- [ ] 至少一个 `Hv > Hk` 场景验证 query/key gradients 沿 `Hv/Hk` 映射正确求和。
- [ ] equal-head 与 GVA 的 output、final state 和所有支持梯度均满足 0.02 relative-L2 标准。
- [ ] final state 未请求时不改变 output-only autograd 行为。
