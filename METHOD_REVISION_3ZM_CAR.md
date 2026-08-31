# MF3ZM-CAR v1 — criterion-aligned risk-constrained switch policy

`mf3zm_car_v1` is one bounded revision after the sealed MF3ZL-RCSP v1r1
train-development failure.  The earlier result remains immutable and is not
overwritten.

## Scientific question

Does RCSP fail because its optimization estimand differs from its declared
hard deployment criterion?  CAR tests this question without changing the
frozen ETP-R1 policy, MF3V proposal ranker, MF3ZG hierarchy, exact one-switch
labels, utility, semantic representation, or rank-4 architecture.

## Frozen data and boundaries

CAR consumes the existing audited 1,540 canonical exact events over 39 MP3D
development scenes (997 RxR and 543 R2R).  It does not collect new rollouts,
reuse the consumed confirmation cohort for selection, or access `val_seen`,
`val_unseen`, `test`, or `test_challenge`.  The 39-scene outer assignment and
the exact identity/prefix contract are unchanged.

## Objective alignment

For a joint fit, each domain receives total mass 1/2 and events within a
domain are equally weighted:

\[
w_i = 1/(D N_d).
\]

The model still predicts a switch logit `s(x)` and the fixed deployment rule
is `switch iff s(x) > 0`.  During fitting the forward value is exactly the
hard mask `1[s>0]`; a straight-through sigmoid is used only to provide a
gradient.  No threshold or calibration grid is searched.

Catastrophe is the frozen event label
`1[delta_utility <= -0.10]`.  The hard event-level selected catastrophic rate
is constrained not to exceed the corresponding ungated event rate.  The
selected event-level utility and every domain leave-one-scene-out utility are
also constrained to be non-negative during optimization.  A zero-selected
domain is explicitly infeasible; no epsilon denominator is used.

The only searched hyperparameter is the pre-registered weight-decay grid
`{1e-4, 1e-3, 1e-2}`.  Scene-disjoint nested fitting remains 5 outer folds and
4 inner folds with common initialization seeds.  Inner candidate choice uses
only inner-OOF magnitude-weighted preference loss after scientific feasibility
checks; outer targets never choose a candidate.

## Independent controls

Controls run even if CAR-full fails, but a control can never authorize a
checkpoint or public evaluation:

- CAR without leave-one-scene constraints;
- CAR with soft (probability) risk constraint;
- CAR with the frozen 28D engineered representation;
- CAR with policy-side scalars only;
- CAR without risk constraint;
- RxR-only and R2R-only CAR-full;
- frozen DSR v1 on the expanded 1,540-row data;
- target-free fold/domain-matched margin, proposal-score, and random baselines.

## Pre-registered failure and authorization rules

CAR development passes only if all five outer folds complete and every domain
has non-zero intervention, positive event-level utility, positive
leave-one-scene-out minimum, no higher catastrophic rate than ungated, and no
negative outer-fold/domain utility.  It must also beat both simple matched
budget baselines in utility without exceeding their catastrophic rate.

Any failed condition writes a complete diagnostic with
`TRAIN_DEVELOPMENT_FAIL` and no checkpoint.  A pass still does not authorize a
public split; public evaluation requires a separate sealed authorization
artifact and a later decision.

If CAR-full and its representation/objective controls fail this protocol, the
frozen-proposal learned-gate family is stopped.  A future richer-causal-state
proposal would be a new revision, not an unlogged CAR patch.
