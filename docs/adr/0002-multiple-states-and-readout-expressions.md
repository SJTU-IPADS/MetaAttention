---
status: accepted
---

# Express Several States and Readout Expressions in the Algorithm IR

## Decision

Algorithm IR schema v2 keeps every v1 form and adds the vocabulary needed by the supported Stateful Operator profiles, without giving up the Scanable guarantee:

- **Several independent States.** `AlgorithmIR` accepts one `StateTransition`
  per declared `StateSpec`. States share one Head role and use at most two
  distinct Feature roles. A State Transition is driven by Inputs only, so every
  State keeps an associative affine summary whose storage is bounded by Feature
  dimensions.
- **Rank-1 States.** `StateSpec(name, (Batch, Head, Feature))` declares a vector
  State, which is how a linear-attention denominator or a HGRN hidden vector is
  represented. Internally it is promoted to a matrix with a singleton second
  Feature, so the existing per-State summary and scan kernels apply unchanged.
- **Expression readouts.** `StateContraction` stays the canonical single-role
  readout. `Readout(name, expression, output_dtype)` adds the general form:
  `StateTerm` nodes reference post-transition States, `Divide` expresses a
  denominator, `Clamp` and the transcendental nodes express normalizers, and an
  Input term expresses a non-State additive contribution.
- **Richer Input Expressions.** `Divide`, `Log`, `Sigmoid`, `Sqrt`,
  `Reciprocal`, `Clamp`, and `Shift(expr, lag)` join `Input`, `Constant`, `Add`,
  `Multiply`, `Negate`, and `Exp`. Transcendental and reciprocal forms compute in
  FP32, exactly as `Exp` already did.
- **Token shift.** `Shift` reads the value of an expression `lag` tokens earlier
  inside the same sequence and zero-fills the tokens before a sequence start, in
  both dense and packed execution.

Schema-v2 Programs are lowered by one reference lowering that composes the same
parallel segmented scans the v1 specialized plans use: per-State affine
summaries are built from the typed IR, scanned, and the readouts evaluate the
authored expression against the post-transition States. Gradients flow through
the same lowering. V1-shaped Programs keep their existing specialized TileLang
plans unchanged, so schema v2 cannot regress a v1 workload.

## Considered Options
A dedicated lowering per model family was rejected: it reintroduces the model catalog this compiler exists to avoid, and each new family would need its own analysis, backward, and cache identity. Extending the language and lowering each Program through the shared summary/scan machinery keeps one semantic source.

State Terms in Transition operands were rejected for this schema, preserving
independent input-driven affine summaries. This is a language restriction, not
proof that every State reference destroys an affine summary: a linear action on
Prior State, including cross-State coupling, can remain jointly affine. The
TTT and Titans references use nonlinear State-derived normalization in their
updates and need a separate execution decision. A nonlinear transformation
between an auxiliary scan and a downstream scan is another distinct case.

Product Features (Based, Rebased: a `[K*K, V]` numerator) were deferred in this
schema because it lacked a typed product Feature expression. ADR 0003 adds
Product Feature roles, `Outer`, `Contract`, `Softmax`, and prior-State readouts.
A product occupies one State axis, so these examples keep a two-Feature-axis
matrix State. They do not require a three-factor Injection: `Outer(k, k)` is
one factor of the two-factor matrix Injection.

## Consequences

- Every supported per-token linear recurrence is expressed by a Program in `examples/`, so coverage remains at the generic Stateful Operator seam.
- The v1 fast paths are unchanged, so the workspace envelope is now charged per
  State of the Program rather than from the first State alone.
- Programs that use v2 vocabulary select the reference plan, which is a
  correctness-first lowering: it is not yet a competitive kernel for long
  sequences, and no autotuned kernel configuration exists for it. Making the
  reference lowering fast (chunked kernels for rank-1 States and diagonal decay,
  chunk-parallel output passes) is the next performance decision, not a
  semantics change.
