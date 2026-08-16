# FlashQLA-derived fused GDN adaptation

**Status: proposed**

The repository will evaluate a FlashQLA-derived single-card fused GDN forward/backward schedule behind the existing head-first `GDNEngine` API, rather than adding a FlashQLA runtime dependency or changing the public layout. The implementation must first pass an isolated fixed-length probe and then the full existing behavioral contract: output, final state, and all supported gradients within 2% relative L2, no implicit tail padding, and peak allocation no greater than the staged baseline; no runtime fallback will be published. The upstream MIT copyright and license notice must be retained for copied or adapted source, while the adaptation remains limited to fused single-card execution and excludes context parallelism.

## Considered options

- Keep the staged path only: lower integration risk, but leaves the measured state/output stage at about 631 ms for the baseline shape.
- Add runtime shape fallback: improves experimental coverage, but creates two production paths and hides incomplete fused validation.
- Change the public API to FlashQLA's token-first interface: avoids an adapter seam, but breaks the repository's established head-first contract.

## Verification boundary

GPU compilation, numerical validation, and performance measurements require an exclusive GPU. The current shared GPU occupancy is not evidence for the fused implementation's correctness or performance.

An initial probe compiled and ran the upstream fixed-length forward on the recorded environment, but it ran before shared-GPU occupancy was checked. It is only a feasibility observation, not accepted correctness or performance evidence; the probe must be repeated on an exclusive GPU.
