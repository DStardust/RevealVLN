# MF3ZK-DSR v1 train-development result

Revision: `mf3zk_dsr_v1`

Pre-seal commit: `a227011cdfe13e130dc68a3a78deba5076e14bc7`

Final status: `TRAIN_DEVELOPMENT_FAIL`

## Boundary

This run used only the 249 pre-sealed unique exact-one-switch development
events from 39 MP3D scenes.  It did not run Habitat, collect a new rollout,
reuse the old confirmation cohort, read `val_seen`, `val_unseen`, `test`, or
`test_challenge`, or modify `FROZEN_SPEC.md`.  No deployable DSR gate was
written.

## Proposal-support audit

The target-aware upper-bound audit passed and was not used for model or
threshold selection.

| Domain | Rows | Scenes | Positive events | Positive scenes | Required positive scenes | Oracle utility at 10% | Oracle utility at 20% |
|---|---:|---:|---:|---:|---:|---:|---:|
| R2R | 102 | 33 | 54 | 26 | 7 | 2.606029 | 3.242481 |
| RxR | 147 | 38 | 63 | 32 | 8 | 4.328344 | 5.033100 |

The frozen runner-up proposal population therefore contains positive support
in both domains and is not rejected by the pre-registered support stop rule.

## Nested DSR failure

All three arms stopped at outer fold 0 because none of the three pre-sealed
weight-decay candidates passed the inner-OOF scientific feasibility screen.
No outer held-out target was used to make this decision.

For the joint arm, weight decay `1e-4` had the best scientifically relevant
case: inner-OOF total utility was positive (`0.789468`), including positive
RxR (`0.760734`) and R2R (`0.028734`) utility.  It nevertheless failed because
the selected R2R catastrophic rate was `6.25%`, above the ungated R2R rate of
`4.55%`.  Weight decays `1e-3` and `1e-2` additionally produced non-positive
domain utility.

The single-domain controls did not rescue the result:

- RxR-only: every candidate had negative inner-OOF utility and a catastrophic
  rate above ungated.
- R2R-only: all candidates had positive inner-OOF utility, but catastrophic
  rates of `6.00%` to `6.25%`, above the ungated `4.55%` rate.

Because no inner candidate was feasible, the run correctly did not produce a
complete outer-OOF score, equal-budget comparison, checkpoint, confirmation
authorization, or public-evaluation authorization.  Changing the quantile,
zero threshold, optimizer constants, architecture, or feasibility rule after
this result would define a new consumed-evidence revision and is not part of
DSR v1.

## Immutable evidence

- Protocol SHA-256:
  `6d16095c568ddab820fea34a04422cde692c66687beeed8d4a58717c76bae8ac`
- Proposal-support audit SHA-256:
  `ff69aea3a39e4ed30ceebde5d7e417b72180bd1d541278f6053ae5793c912f3c`
- Train-development result SHA-256:
  `76aed000e4f8992fc8e953a1fb745d371ca6a7675bd38f0afc9affdc3cef79ff`

Post-run source and implementation verification rebuilt the pre-sealed
inventory successfully.  The next algorithmic step is intentionally not
chosen in this revision.
