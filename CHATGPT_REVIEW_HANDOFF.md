# RevealVLN — Review Handoff Snapshot

Snapshot date: 2026-08-30
Repository: `DStardust/RevealVLN`
Purpose: provide a bounded, reproducible context for an independent technical
review of the current method and its next revision.

## Current method under review

`MF3ZK` is a train-only joint revision built on the frozen `MF3ZG` hierarchy:

1. The ETP-R1 VLN policy, MF3V proposal ranker, and proposal/backbone weights
   remain frozen.
2. A shared action-aligned return/harm gate is fitted on RxR-train and
   R2R-train with effective benchmark weight 1:1.
3. The gate uses instruction, checkpoint/history, native-action, and proposed
   alternative embeddings together with policy-side features. It does not
   receive a benchmark identifier.
4. Core and expansion proposal tiers are disjoint and retain the one-switch
   hierarchy. RxR-only and R2R-only fits use the same code and scene-fold
   protocol as controls.
5. Model selection uses scene-disjoint OOF data. Confirmation scenes are
   excluded across the two benchmarks.

The intended scientific claim is conditional and safety-oriented: an
action-aligned return/harm screen can decide when a frozen policy's proposed
alternative is worth executing. It is not a claim that every episode should
be changed, and it is not yet a public benchmark result.

## Evidence available in this snapshot

- 1,200 deterministic R2R-train treatment routes were collected with the
  frozen proposal controller; no public split was read.
- 778 missing same-episode native baselines were completed with a separate
  train-only worker. Pairing is by exact episode ID; cross-episode pairing is
  prohibited.
- Strict assembly retained 126 exact one-switch pairs (46 core, 80
  expansion). The other 1,074 routes produced no exact one-switch pair and
  were retained only as documented exclusions.
- Joint fitting used 299 rows over 39 fit scenes. The two joint tiers and the
  RxR-only/R2R-only controls all passed the internal train-return gate.
- A sealed confirmation cohort contains 52 R2R-train episodes over 13 scenes
  not used for fitting. All 156 paired runs completed without runtime errors.
- The confirmation gate **failed**: MF3ZK made one action change versus three
  for MF3ZG; MF3ZK minus MF3ZG mean utility was -0.00086519, with a scene
  bootstrap interval [-0.00208637, 0]. This is a real generalization failure
  signal, not a successful benchmark result.
- Consequently, no MF3ZK public `val_unseen` or `test` evaluation is
  authorized in this snapshot.

The machine-readable evidence is in:

- `artifacts/design/METHOD_REVISION_3ZK_JOINT_TRAINING.md`
- `artifacts/training/mf3zk_joint_v1/MF3ZK_JOINT_PROTOCOL.json`
- `artifacts/training/mf3zk_joint_v1/r2r_collection/MF3ZK_R2R_DIRECT_SWITCH_MANIFEST.json`
- `artifacts/training/mf3zk_joint_v1/gates/MF3ZK_JOINT_TRAINING_RESULT.json`
- `artifacts/training/mf3zk_joint_v1/confirmation/MF3ZK_TRAIN_CONFIRMATION_RESULT.json`

## Questions for an independent reviewer

Please inspect the source and evidence and answer these questions without
assuming a positive result:

1. Is the action-aligned return/harm gate a sufficiently distinct and
   defensible contribution over a standard policy residual, reranker, or
   confidence threshold?
2. Do the collection, exact-episode pairing, scene holdout, OOF selection,
   and confirmation procedures prevent leakage and post-hoc selection?
3. Is the low activation rate a calibration/coverage problem, a proposal
   problem, or evidence that the method is not useful for this task?
4. What is the smallest scientifically valid revision that could improve
   held-out performance without tuning on the confirmation or public unseen
   split?
5. Which ablations and baselines are mandatory before making any CVPR-level
   claim? Please distinguish engineering checks, exploratory results, and
   publishable evidence.
6. Audit the implementation for silent action-index drift, incorrect metric
   pairing, scene-fold errors, or overly permissive gate criteria.

## Scope and exclusions

This Git snapshot intentionally does **not** contain Matterport/R2R/RxR
payloads, checkpoints, virtual environments, caches, raw per-episode traces,
large logs, or any secret file. Those assets remain project-local under the
workspace and are not needed to review the design logic. No API key or private
credential is part of the repository.

The result above is a research diagnostic. It must not be described as SOTA,
a public benchmark improvement, or evidence of acceptance probability.
