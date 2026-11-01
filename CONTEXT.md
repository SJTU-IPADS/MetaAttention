# Stateful Compiler Context

Domain terms are defined in [GLOSSARY.md](GLOSSARY.md). This document records
compiler behavior and implementation constraints.

## Algorithm IR compatibility
Compatibility means accepting previously unseen compositions expressible in the existing Algorithm IR and satisfying its validation and Scanable rules. It does not mean recognizing arbitrary frontend or backend IR directly.
## Canonical IR
The validated and canonicalized form consumed by structural analysis. It removes representation-only differences while preserving logical role order, authored noncommutative propagation order, and floating-point association.

## Simplified IR
An optimization view derived from Canonical IR through registered semantics-preserving rewrites. It exposes reusable algebraic structure without changing the public Algorithm IR identity.

## Mathematical plan
A compiler-internal algorithm structure derived after mandatory safe simplification. It records the chosen mathematical representation and optional equivalent structural rewrites, but excludes lowering implementation, concrete kernel parameters, schedules, tuning results, and hardware details.

## Lowered candidate
A compilable backend implementation of one Mathematical Plan. Lowered Candidates are benchmarked against one another; only the selected candidate proceeds to kernel-parameter autotuning.
## Compiled operator
The result of compilation is a callable Forward Operator. When backward generation is requested, the operator is wrapped with compiler-generated autograd integration; users invoke the forward result and PyTorch invokes the generated backward kernel. The compiled result does not expose a separate public backward handle. The selected executable binds its execution adapter at compilation, including native/chunked dispatch and legal reference fallbacks; the public operator validates each invocation without interpreting lowering families or recipes. Replacing a candidate during tuning rebuilds this binding.

## Candidate binding
Formal compilation and tuning trials use one candidate-binding operation to derive the Mathematical Plan, kernel configuration, workspace policy, schedule, backward context, composition, and executable identities. Binding retains unselected structural recipes, so probing a cut or factored implementation cannot constrain a later candidate. Tuning invokes an unpublished compiled handle through ordinary validation and execution; it does not construct a placeholder executable, register a trial in the operator's specialization cache, or publish trial artifacts. Only the final successful selected executable is published. Candidate ranking, measurement-cache policy, and workload resource limits remain separate from binding.

## Invocation validation
Affine scans, LogAddExp scans, and Scan Stage DAGs share a compiled invocation-validation contract for input layouts, fixed logical roles, initial State, target, aliasing, and gradients. Each State carries its own default Scan Identity (zero for affine, `-inf` for LogAddExp); dense-empty legality is explicit and a DAG requires every child to allow it. Stage declaration order determines public State order. Validation does not borrow a child operator or reinterpret dense empty inputs as packed sequences; resource gates and execution remain separate.

## Tuning workload
A compiler-chosen representative set of sequence workloads used when the caller does not know future sequence lengths. Its measurements select performance strategy but do not change Algorithm IR semantics.
## Dynamic workload dimensions
Batch size, dense Sequence length, packed logical-sequence count, packed token count, and packed offsets are runtime dimensions. They do not select a new Mathematical Plan or compiled kernel. Head and Feature dimensions remain compile-time specialization dimensions.


## Current Algorithm IR capability
Schema v5 retains schema-v4 support for several named States, each with one or two ordered Feature
roles and one Transition. States may be independent or coupled by typed affine
State Injections on one shared Feature axis. A Product Feature occupies one axis;
it does not increase State rank. The reference executor requires all States to
share one Head role.

The legacy Transition spelling applies ordered Propagations, then adds all
Injections. `steps` instead preserves interleaved propagation and writes.
`StateInjection(StateReference(...))` adds an input-driven linear action on
Prior State or another State's completed Current State. Current references
induce an acyclic Transition dependency checked before lowering. Coefficients
and Product Injection expressions remain Input-driven; State Terms are
reserved for Readouts. Product Injections may contain several factors, subject
to typing and lowering constraints. Readouts may combine multiple States,
nonlinear arithmetic and nested contractions; `prior=True` selects Prior State.

`Shift` zero-fills sequence starts within an invocation. It does not retain
input history across invocations; passing final State alone does not supply
that history.

## Scanability and language acceptance
A compact associative summary is a mathematical property, separate from schema
acceptance, available lowering, and concrete resource feasibility. Current
Transitions admit input-driven affine maps. Coupled States with matching roles
and actions on one shared Feature axis lower to a joint block-affine summary,
scanned in parallel. Mixed-axis coupling and heterogeneous State roles are
rejected by this first coupled implementation. Nonlinear Readouts do not change
the scanability of State evolution.

A Scan Stage DAG may apply a nonlinear Readout transformation between an
upstream auxiliary scan and downstream memory scan without a joint Affine
Summary. Schema v5 implements explicit typed stage bindings: each stage proves
its own algebra, and gradients flow through the bound trajectories. The finite
LogAddExp algebra uses `-inf` identity and a safe identity VJP. Stages may have
different State Head roles; downstream Head Mapping preserves auxiliary head
sharing. Optional checkpoint recomputes child intermediates; the resource
budget still charges the conservative saved-scan baseline. See
[ADR 0005](docs/adr/0005-scan-stage-dag-and-logaddexp.md). General Nonlinear State
Feedback still has no compact-summary proof or sequential recurrence fallback.

## ExecutionResult
The immutable public result of every Stateful Operator invocation. It contains named outputs and, only when requested, named final State; the result shape never changes with the number of outputs or states.

## Target capability
The concrete hardware, dtype, layout, and resource envelope accepted by a compiler specialization. A capability failure does not change a program's Scanable classification.

## Structural optimization
A compiler-selected equivalent representation or lowering justified by typed Algorithm IR structure and specialization metadata. It is shared by every matching program and never depends on a model name or complete known profile identity.
_Avoid_: Algorithm-specific optimization, profile-specific kernel, model dispatch
