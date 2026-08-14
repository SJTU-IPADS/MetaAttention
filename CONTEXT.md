# MetaAttention

## Language

**Gated Delta Rule (GDN)**:
The QLA operator targeted by this repository. It is distinct from scalar-decay gated retention: GDN takes query, key, value, log-space gate, and beta, and uses a delta-corrected recurrent state update.
_Avoid_: Gated Attention, GLA

**QLA**:
The upstream FlashQLA implementation of Gated Delta Rule chunked prefill. In this repository, “QLA adaptation” means adding a dedicated GDN path, not changing the existing scalar-decay linear-attention path.

**First Release**:
The initial GDN contract: H20/sm89 BF16 fixed-length training with forward and backward execution, GVA, and initial/final recurrent state. Variable-length and automatic intra-card context parallelism are excluded.

**GDN Engine**:
The dedicated module that owns GDN lowering and kernel execution. It is separate from the scalar-decay LinearAttentionEngine.

**Head-first Layout**:
The GDN Engine public tensor layout: query and key are `[B,Hk,T,K]`, value is `[B,Hv,T,V]`, and gate and beta are `[B,Hv,T]`. This is the native MetaAttention layout.
_Avoid_: Token-first Layout

**GDN State**:
The optional fp32 recurrent state with layout `[B,Hv,128,128]`. A missing initial state means zero state; final state is returned only when explicitly requested.

**GDN Grouped Value Attention (GVA)**:
The first-release head mapping where `Hv % Hk == 0`. Each query/key head serves `Hv/Hk` value and recurrent-state heads.

**GDN Chunk Alignment**:
The first release requires `T % 64 == 0`. Unaligned input is rejected explicitly; the engine never truncates or pads sequences implicitly.

**H20 GDN Adapter**:
The sm89 TileLang implementation of GDN. It follows FlashQLA's mathematical reference and test behavior but does not reuse Hopper-only warp-specialized kernels.

**GDN Precision Contract**:
Query, key, value, and output use bfloat16. Log-space gate, beta, initial state, final state, and state accumulation use float32.
