# MF3ZK-DSR v1 — Policy-Anchored Distributional Switch Critic

Status: train-development method revision, sealed before DSR audit/training.

This revision does not amend `FROZEN_SPEC.md`.  The ETP-R1 frontend, MF3V
proposal ranker, MF3ZG core-preserving hierarchy, frozen runner-up identity,
one-switch budget, and utility definition remain unchanged.  The historical
MF3ZK confirmation and MF3ZK-NP v9 train-development result are consumed
negative evidence and may not be used for DSR selection.

## Intervention target

For a causal state `x`, native action `a_n`, and frozen runner-up `a_r`, the
label is the exact same-episode, same-prefix one-switch utility difference

```text
delta_u = U(trajectory with a_r at the decision)
        - U(trajectory with frozen a_n)
U = 0.50*nDTW + 0.25*SDTW + 0.25*SPL
```

Non-exact or cross-episode pairs are prohibited.  No future observation,
benchmark ID, tier ID, scene ID, task metric, or public-split outcome is a
model input.

## Model and decision

The model consumes the existing 28 action-aligned causal features.  A
24-hidden-unit GELU residual network predicts ordered return quantiles at
`{0.20, 0.50, 0.80}`.  Its median contains a non-positive native-margin
anchor, `softplus(beta) * -log1p(native_margin)`.  Non-negative softplus spans
on either side of the median prevent quantile crossing.

Training minimizes weighted multi-quantile pinball loss.  Joint training gives
RxR and R2R equal total weight, gives every MP3D scene equal weight within its
benchmark, and divides a scene's weight across its events.  There is no
separate harm classifier.

The only operating rule is fixed before training:

```text
switch iff predicted q20(delta_u | x) > 0
otherwise execute the frozen native action
```

No utility or harm threshold is searched.  Hidden size, quantile levels,
optimizer, training steps, seeds, and learning rate are fixed.  Four inner
whole-scene folds may select only weight decay from `{1e-4, 1e-3, 1e-2}` by
inner-OOF quantile loss among scientifically feasible candidates.  All
candidates use the same folds, initialization seeds, and full-batch order.
The fixed optimizer is full-batch AdamW for 300 steps at learning rate 0.01,
with seeds `{20260830, 20260831, 20260832}`.  Output heads begin at zero median
residual and a symmetric 0.05 quantile span; these constants are not selected
from outcomes.

## Development protocol

Five outer folds assign complete raw MP3D scene IDs; a scene shared by RxR and
R2R remains one cluster.  Every outer prediction is produced by a model and
weight decay selected without that outer scene.  The final weight decay is the
outer-fold mode (smallest on a tie), after which three fixed-seed models are
fit on all authorized train-development rows.

Before any model is fit, a target-independent proposal-support audit reports
per-domain prevalence, positive-scene support, fixed 5/10/20-percent oracle
upper bounds, outer-fold support, and low-native-margin deciles.  Oracle
outcomes are diagnostics only and never select architecture, loss, features,
or an operating threshold.  The audit fails if both 10% and 20% oracle totals
are non-positive or if positive switches occur in fewer than
`max(5, ceil(20% of domain scenes))` scenes.

Development fails closed if a joint outer fold has no legal prediction, an
eligible outer-fold/domain has zero intervention, a domain has non-positive
deployed utility, leave-one-selected-scene utility is non-positive,
catastrophic rate exceeds its ungated or fold/domain-matched low-margin
baseline, or DSR does not exceed that low-margin baseline at the identical
fold/domain intervention budget.

## Provenance and evaluation boundary

`MF3ZK_DSR_PROTOCOL.json` is created and committed before audit/training.  It
binds source manifest bytes/hashes, every selected feature file, canonical
label-content and source-provenance hashes, consumed scenes, fold assignment,
model/loss constants, audit rules, and failure criteria.  Training begins by
rebuilding and comparing that inventory; drift aborts before optimization.
The 50 overlapping RxR decisions are required to have identical full
label-defining content, while their distinct collection traces remain recorded
as separate source provenance.  This is deliberately not described as
byte-identical full source records.

The trainer cannot authorize task metrics, fresh confirmation, or a public
split.  This revision never runs the consumed 52-episode confirmation and
never reads `val_seen`, `val_unseen`, `test`, or `test_challenge`.  Any later
authorization requires a separate versioned artifact after all methods,
baselines, and ablations are frozen.
