# MF2 State-Conditioned Return Executor V4.5

Status: frozen engineering interface; not a paper result.

## Responsibility

The executor implements the deterministic lifecycle after an external policy
has selected `CHECKPOINTED_EXCURSION(branch)` or `BACKTRACK(checkpoint)`. It
stores a public return-controller reference and enforces:

`AT_CHECKPOINT -> EXPLORING -> RETURNING -> AT_CHECKPOINT -> COMMITTED`.

A failed return enters `RETURN_FAILED`. No other branch can be explored or
committed until the same public controller succeeds on an explicit retry. A
successful return marks the explored branch `EXHAUSTED`; the external OPP may
then select among remaining `UNTRIED` branches.

## Scientific boundary

This component does not infer when to return, choose a branch, predict cost,
or consume observations. It therefore closes execution semantics but does not
claim learned state-conditioned `BACKTRACK`. That claim remains gated on new
train-only post-excursion observations and counterfactual labels.

The executor must not silently teleport, substitute a different checkpoint,
or mark a failed return as successful. Oracle and frozen public controllers
remain separately reportable as required by `FROZEN_SPEC.md`.
