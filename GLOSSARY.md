# MetaAttention

MetaAttention describes attention and stateful sequence computations through typed mathematical programs. This glossary defines the language used to discuss their semantics and coverage.

## Language

### Programs and state evolution

**Stateful Operator**:
A sequence computation with explicit mathematical State carried between steps and exposed for continuation.
_Avoid_: Model family, recurrent model

**Algorithm IR**:
The typed mathematical description of a computation's Inputs, States, Transitions, and Readouts, independent of execution schedule and physical representation.
_Avoid_: Model template, kernel description

**State**:
A named per-sequence mathematical value carried between transitions. A computation may maintain several States with different roles.
_Avoid_: Matrix memory as a synonym for every State, layer cache

**Transition**:
The rule producing the next State values from prior State values and Inputs; any intermediate dependencies and update order are part of its semantics.
_Avoid_: Propagation as a synonym for the entire update

**Prior State**:
The State value immediately before the current token's Transition.
_Avoid_: Previous invocation State

**Current State**:
The State value after the current token's complete Transition; intermediate values require a separately named version.
_Avoid_: Latest State when the update stage is unspecified

**Propagation**:
The action transforming an existing State contribution within a Transition.
_Avoid_: Complete Transition

**Injection**:
An additive contribution to a State within a Transition.
_Avoid_: Input projection

**Readout**:
A named output derived from Inputs and explicitly selected State values. It may combine several States and may be nonlinear even when their Transitions are affine.
_Avoid_: State Contraction as a synonym for every Readout

**State Contraction**:
A reduction of a State against an operand along a named Feature Role.
_Avoid_: Readout as a synonym for this specific operation

**Continuation**:
Processing a subsequent sequence segment using the mathematical state retained from the preceding segment, with results consistent with processing the combined sequence.
_Avoid_: Token shift

### Roles and dependency structure

**Feature Role**:
A named logical axis participating in state algebra; its position determines the direction of an axis action.
_Avoid_: Physical dimension

**Product Feature Role**:
A logical Feature Role formed from the Cartesian product of two Feature Roles.
_Avoid_: Higher-rank State when the product occupies one State axis

**Head Mapping**:
The declared sharing relationship between logical Head Roles, including how contributions to a shared head combine.
_Avoid_: Physical head replication

**Independent States**:
States whose Transitions do not consume other States' values. Their Readouts may still combine them.
_Avoid_: Independent Readouts

**Coupled Affine Transition**:
A Transition affine in the joint prior State, including dependencies between States, with coefficients determined independently of that joint prior State.
_Avoid_: Nonlinear recurrence merely because several States interact

**Scan Stage**:
A sequence computation with a compact associative transition summary, producing State trajectories or final State for downstream computation.
_Avoid_: Kernel chunk, token-local update stage

**Scan Stage DAG**:
An acyclic dependency graph of Scan Stages and transformations between their results; a downstream Stage may use an upstream State trajectory to determine its coefficients.
_Avoid_: Joint Affine Summary when the inter-stage transformation is nonlinear

**Stage Binding**:
A typed connection from an upstream Scan Stage's token Readout to a downstream Stage's Input, preserving sequence identity and the selected State timing.
_Avoid_: Final-State substitution for a trajectory

**Scan Identity**:
The State value representing an empty sequence for a particular scan algebra.
_Avoid_: Zero State as a universal identity

**Nonlinear State Feedback**:
A dependency in which a nonlinear function of the State being updated determines its own subsequent evolution.
_Avoid_: Nonlinear Readout

### Scanability and coverage

**Affine Summary**:
A representation of a sequence segment's State transformation as a linear action plus an additive contribution; summaries compose associatively.
_Avoid_: Per-token State history

**Scanable Program**:
A mathematical Program with a proven associative transition summary whose storage is independent of sequence length at fixed State dimensions.
_Avoid_: Every affine expression accepted by the current schema, fast kernel

**Recurrent Program**:
A mathematical Program for which a compact associative transition summary has not been established.
_Avoid_: Proven impossible to parallelize, invalid Program

**Mathematical Coverage**:
The ability to represent a specified computation and its outputs under stated assumptions, including an equivalent change of State representation where declared.
_Avoid_: Complete model support

**Numerical Coverage**:
Evidence that an implementation preserves the specified computation within a stated error budget and numerical range.
_Avoid_: Algebraic equivalence

**State Compatibility**:
Agreement on the meaning and representation of initial and final State, allowing continuation or exchange under an explicit conversion contract.
_Avoid_: Output parity

**Layer Coverage**:
Coverage of a complete model layer's computation and continuation contract, including its preprocessing, output transformations, and caches.
_Avoid_: Core recurrence coverage

**Performance Coverage**:
Measured time and memory behavior for specified workloads and execution contracts against an identified baseline.
_Avoid_: Mathematical Coverage, correctness alone
