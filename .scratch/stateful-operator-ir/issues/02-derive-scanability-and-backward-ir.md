# Derive Scanability and Backward IR

Labels: `ready-for-agent`

## Goal
Derive target-independent affine scan summaries and compiler-owned backward semantics from the typed public IR.

## Scope
- Define dense affine State maps as the canonical associative summary.
- Prove closure for ordered Axis Scale, general Rank-One Propagation, and additive Product Injections.
- Classify valid programs as Scanable or Recurrent without consulting Target resources.
- Derive canonical internal Backward IR from node VJPs, including output/final-State cotangents, reverse propagation, Head Mapping reductions, initial-State gradients, and alias accumulation.

## Non-goals
No PyTorch eager production backward, hand-written profile backward, public analysis API, or schedule selection.

## Dependencies
Issue 01 public schema and errors.

## Acceptance
Public invocations of previously unseen legal compositions produce gradients matching independent PyTorch recurrences. Missing cotangents are zero; final-State-only and joint losses work. `left != right` rank-one propagation and noncommuting order are covered. A valid non-Scanable program receives a stable Recurrent error before Target analysis.
