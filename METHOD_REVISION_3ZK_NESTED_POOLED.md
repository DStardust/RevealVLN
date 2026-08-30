# MF3ZK-NP — Nested pooled action-aligned gate

Status: versioned train-development revision (2026-08-30)

This document records a correctness revision of MF3ZK.  It does not amend the
claims or gates in `FROZEN_SPEC.md`, and it does not authorize a public split.
The previous 52-episode confirmation is consumed evidence of a failed
generalization attempt.  It may be reported retrospectively, but it is never
used to choose a feature, threshold, weight, or model in this revision.

## Fixed method boundary

The ETP-R1 frontend, MF3V proposal ranker, and MF3ZG core-preserving proposal
hierarchy remain frozen.  A proposal is still a single native-to-frozen-
runner-up action switch.  Core and expansion rows are pooled into one
action-aligned return estimator and one harm estimator; the proposal tier is
not supplied as a feature.  Tier routing remains an inference-time property
of the frozen hierarchy.

Historical core and expansion collectors contain some byte-identical
episode/step counterfactuals.  The pooled loader verifies scene, decision,
target, and feature hash equality and counts each exact label once.  A
conflicting duplicate fails closed.  The retained row's core/expansion tier is
derived from the frozen hierarchy thresholds rather than the source filename.

## Selection protocol

1. Five deterministic outer folds assign complete MP3D scenes; a scene cannot
   occur in both the fit and evaluation side of a fold.
2. Within each outer-fit portion, four complete-scene inner folds select the
   regularization value and one operating rule from a fixed grid.  The grid is
   scanned only on inner out-of-fold predictions.  The inner-fold salt is
   fixed independently of code/output version names.
3. The selected rule is applied to the outer-held-out rows with that fold's
   thresholds.  These fold-specific decisions are the only development
   estimate used for status checks and coverage reports.
4. The final deployment rule is fixed before any confirmation run: modal L2
   (smallest on a tie) and the median return/harm thresholds from the five
   inner selections.  Applying this aggregate rule to outer predictions is
   retained only as a diagnostic, never as a selection result.
5. Bagged fits sample a shared union of MP3D scene clusters once across RxR
   and R2R.  Benchmark-balanced observation weights are used for the joint
   fit, so a scene present in both benchmarks is not treated as independent
   evidence.

Feasibility requires positive utility, positive leave-one-selected-scene
utility, a minimum intervention count, and a catastrophic **rate** no greater
than the ungated candidate population.  A raw catastrophic count is not used
as the risk constraint because it is confounded by coverage.

## Required diagnostics

Every arm (joint, RxR-only, and R2R-only) records the complete supervised
coverage funnel: eligible rows, proposal candidates, core/expansion rows,
actually changed rows, return-safe rows, harm-safe rows, authorized rows, and
positive/negative/catastrophic realized deltas.  The denominator is kept
explicit: exact one-switch supervised rows are not silently described as all
collection routes.

A fixed risk--coverage curve and exact-budget baselines (low native-margin,
high proposal score, and deterministic uniform random) are reported.  Each
baseline receives exactly the nested gate's outer-fold intervention count;
none is tuned on its realized target values.

All action-aligned records carry identity checks before feature construction
and before execution.  Global action IDs must be unique, the native and
alternative IDs must round-trip to the declared global indices, and the
alternative must be a current unvisited candidate.

## Authorization boundary

`model_fit_status`, `scientific_control_status`,
`confirmation_authorization_status`, and `public_eval_authorization_status`
are separate fields.  A fitted model is not thereby authorized for task
metrics.  This revision requires a newly sealed, previously untouched
train-development confirmation after the method is frozen.  It does not
authorize R2R/RxR `val_seen`, `val_unseen`, `test`, or `test_challenge` runs.

If the nested train-development control fails, the result is recorded as a
failure; thresholds must not be relaxed to force a pass.  If a fresh
confirmation fails, the failure is retained as evidence and no public claim
is made.
