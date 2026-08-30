# Method-Freeze 3ZG: Core-Preserving Hierarchical Return Gating

Status: frozen after MF3ZF failed the fixed RxR val_seen gate with two
metric-neutral interventions, and before any MF3ZG task-metric run or new
val_unseen episode selection.

## Fixed method

MF3ZG preserves the complete MF3ZE core proposal path and adds a disjoint
expansion tier instead of replacing it. A score in the original MF3V interval
`(q0.985, q0.995]` is evaluated by the frozen MF3ZE action-aligned return gate.
A score in the new interval `(q0.970, q0.985]` is evaluated independently by
the MF3ZF return model with a semantic non-negative expected-return boundary
and the frozen MF3ZF harm boundary. A rejected expansion proposal does not
consume the one core-tier proposal opportunity. Either tier may execute at
most one native-to-frozen-runner-up switch per episode.

The zero expected-return boundary is fixed because an optional intervention
must predict non-negative benefit; it is not selected from a metric grid.
MF3ZF's val_seen failure is used only to motivate core preservation and this
semantic boundary. No MF3ZG val_seen or val_unseen metric selects a threshold.

## Required gates

The expansion rule must authorize at least 24 scene-disjoint OOF RxR-train
proposals, have positive total final-return utility, reduce catastrophic harms
relative to the ungated expanded proposer, and remain positive after removing
any one selected training scene. MF3ZG then must pass the unchanged 57-scene
MF3V val_seen task-metric protocol before a fresh val_unseen power holdout is
sealed. Every previously consumed unseen episode is excluded.

## Online boundary

Inference uses only the current instruction, recurrent checkpoint state,
current native and runner-up visual embeddings, and current controller
statistics. It uses no future frame, teacher action, ground-truth distance,
counterfactual return, branch exploration, or physical backtracking.
