# 06 — H20 基线与发布收口

**What to build:** 将完整 H20 GDN Engine 作为可发现、可复现、可维护的首期能力交付。用户获得公开接口说明和可运行示例；维护者获得代表 shape 的性能/显存/编译基线、完整验证命令和明确的支持边界。

**Blocked by:** 05 — 状态与 GVA 梯度.

**Status:** ready-for-agent

- [ ] 公开文档准确描述 head-first layout、BF16/FP32 精度、`K=V=128`、`chunk=64`、`T % 64 == 0`、GVA、state、scale 和 H20/sm89 限制。
- [ ] 示例通过实际 GDN Engine 展示固定长度 forward/backward、可选 initial state 和 final state，不使用 mock 或 PyTorch fallback。
- [ ] 在冻结的代表 shape 上记录 kernel latency、端到端 latency、峰值显存和首次编译时间，并记录 warmup、重复次数、dtype、硬件与软件版本。
- [ ] 性能数据只作为后续融合基线，不以 Hopper FlashQLA 或现有 retention 延迟作为 pass/fail 门槛。
- [ ] MIT attribution 和 GDN/QLA 术语与领域词汇表、ADR、规格一致。
- [ ] CPU-safe 单元套件和 H20 GDN functional suite 通过；验证覆盖 output、final state、全部支持梯度和拒绝路径。
- [ ] 现有 LinearAttentionEngine、RetNet、gated retention 和 Mamba2 的公开行为保持不变。
