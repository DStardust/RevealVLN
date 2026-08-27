# MF2 Correctness Revision CR7: Tie-Aware Listwise Action-Q

Status: train-only engineering candidate. This document does not modify
`FROZEN_SPEC.md` and is not a paper result.

## Triggering evidence

The sealed V4.3 diagnosis reproduced seed instability on the 83-event,
scene-disjoint internal development partition. All-three branch agreement was
0.8313, while the median predicted top-1/top-2 gap was only 0.16--0.28 despite
a median teacher gap to the next distinct cost of 4.0. Mean/median/rank
ensembles did not reliably improve both regret and exact oracle equivalence.

The train-only label audit also establishes a structural tie: all 424 target
branches have equal `COMMIT` and `CHECKPOINTED_EXCURSION` costs. A loss that
arbitrarily chooses one member of this optimal set would manufacture an action
distinction absent from the counterfactual target.

## Minimal revision

CR7 preserves the V4 architecture, causal inputs, Huber regressions, paired-gap
loss, and margin ranking. It adds one tie-aware listwise cross-entropy over all
candidate/action pairs. Every action at the event's minimum teacher cost shares
the target mass uniformly. The predeclared auxiliary weight is 1.0.

This directly trains the global decision that the online controller executes,
while keeping predicted Q values as the sole runtime selector. It adds no
runtime module, threshold, ensemble, simulator input, or future observation.

## Scope and acceptance

Training and selection use only the existing 341-event train partition and
83-event scene-disjoint internal development partition. The inspected RxR
`val_unseen` controller result is forbidden from checkpoint or hyperparameter
selection. CR7 is accepted only if three fresh seeds jointly improve branch
agreement, mean teacher-cost regret, and exact oracle equivalence relative to
the sealed V4.3 development diagnosis. A future confirmatory evaluation must
use data not used to motivate or select CR7.

CR7 still does not establish state-conditioned `BACKTRACK`; that remains a
separate post-excursion observation and executor gate.
