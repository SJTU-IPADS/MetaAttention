---
status: accepted
---

# Product Feature Roles, Feature Reductions, and Prior-State Readouts

## Decision

The Algorithm IR expression language gains the vocabulary needed by the supported product-Feature profiles, without changing the Scanable guarantee or the lowering pipeline:

- **Several distinct Feature roles, including products.** `ProductFeatureRole(left,
  right)` declares a Feature whose axis is the flattened product of two Features,
  and a State may use that product as one of its axes. `Outer(left, right, role)`
  builds the product axis of two expressions. This is how Based and Rebased keep a
  `[K*K, V]` numerator: the second-order Feature is one axis of a matrix State,
  not a new State rank, so the per-State affine summary and the segmented scan are
  unchanged.
- **`Contract(operand, coefficient, role)`.** A Feature-axis reduction of
  `operand * coefficient` that keeps a singleton axis, so the result broadcasts
  against a Feature-carrying term. `coefficient` may be feature-free, which makes
  a pure reduction expressible. A Contract operand either carries exactly the
  contracted Feature or no Feature at all.
- **`Softmax(operand, role)`.** Normalizes one Feature axis of a value that
  already carries that axis. A Program joins two States by contracting the first
  into a slot score, softmaxing the slot axis, and contracting the second State
  with the result; that is how the library's Gated Slot Attention composes its
  `hk` and `hv` States.
- **Prior-State readouts.** `StateTerm(..., prior=True)` reads the State as it was
  before the current token's transition. RWKV6's documented recurrence reads the
  previous State and adds a current-token bonus, so a post-transition-only readout
  cannot express it.
- **Feature-free Injection factors.** An Injection factor may declare a Feature
  role that its expression does not carry; the value then broadcasts along the
  declared axis. A constant `+1` Injection on a one-Feature State is the token
  count that Based and Rebased divide by, and it is exactly the broadcast case the
  schema already documented.

Every addition is a typed node with an analysis rule, a canonical-form entry, a
reference-lowering evaluation, and an H20 resource check. No addition introduces
a model name, a per-family kernel, a state-dependent Transition, or a
sequence-length-sized summary, so Scanability and the lowering contract are
unchanged.

## Considered Options

A general einsum or arbitrary-contraction node was rejected: it would make
readout typing depend on complete shape data instead of Feature roles, and the
observed library families need exactly a Feature-axis reduction plus a product
Feature axis. Keeping the two-Feature-role rule from schema v2 was rejected
because a `[K*K, V]` State cannot be typed without a product Feature. A dedicated
inner-loop construct was rejected again: Based, Rebased, GSA, and RWKV6 are all
expressible as ordered affine Transitions, so they need no new execution model.
Reading a State mid-readout after applying its own transition was rejected as an
implicit second transition; `prior=True` names the pre-transition State instead.

## Consequences

- Based, ReBased, Gated Slot Attention, and RWKV6 are authored as Stateful Operator Programs in `examples/`; their coverage uses independent references at that seam.
- The reference lowering's workspace is charged per State axis, so a product
  Feature multiplies the State size: Rebased at `K = 64` (`K*K = 4096`) exceeds the
  H20 reference-plan envelope and is rejected with `RUNTIME_RESOURCE` rather than
  silently allocating. Making that plan chunk its output pass is a performance
  decision, not a language one.
- ADR 0002's rejection of product Features is superseded by this ADR: Feature
  roles may include typed products, while each State still has one or two
  Feature axes. Based and Rebased use two-factor Injections whose second-order
  factor is a product-Feature operand; this does not impose a general two-factor
  limit. Current Product Injections admit several typed factors, including
  Feature-free weights. The restriction on State Terms in Transitions still
  stands. Cross-State affine coupling, nonlinear transformations between scans,
  and nonlinear State Feedback require different summary proofs.
