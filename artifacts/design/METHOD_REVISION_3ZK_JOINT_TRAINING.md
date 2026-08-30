# Method Revision 3ZK — Joint RxR/R2R Return-Risk Training

Status: `FROZEN_BEFORE_3ZK_TRAINING`

This is a versioned revision of the experimental training protocol.  It does
not modify `FROZEN_SPEC.md`, the released ETP-R1 checkpoint, or any earlier
MF3V/MF3ZG artifact.

## Motivation

The intended deployed model is one cross-benchmark VLN controller.  Training
the new safety module on RxR and R2R train data jointly is therefore the
mainline experiment.  RxR-only and R2R-only fits remain mandatory controls;
they are not alternative test-set selections.

## What is shared and what is frozen

The ETP-R1 language/vision/navigation front end and the MF3V proposal ranker
remain frozen.  3ZK jointly fits the action-aligned return/harm gate on exact
one-switch counterfactual returns from both benchmarks.  The gate has the same
28-dimensional feature definition and the same core/expansion hierarchy as
MF3ZG.  This isolates the cross-benchmark calibration effect and avoids
claiming a backbone fine-tuning gain that is not measured.

The deployed 3ZK controller is one shared model and does not receive a
benchmark identifier.  Benchmark identifiers are used only for sampling,
accounting, and audit reports.

## Data and split rules

* Only `train` payloads may be opened for constructing 3ZK labels.
* Every counterfactual switch is action-aligned and has exactly one changed
  action.  Inputs are causal (instruction, current history, native and
  runner-up embeddings); route geometry is used only to form the train label.
* RxR and R2R examples are balanced by effective dataset weight, so the larger
  corpus cannot dominate the fit.
* Cross-validation folds are assigned by raw MP3D scene ID.  A scene shared by
  RxR and R2R therefore remains in one fold, preventing cross-benchmark scene
  leakage.
* The R2R `val_seen` split has already been consumed by earlier engineering
  evaluations (including a full 778-episode run).  It must not be described as
  a fresh confirmation set.  3ZK uses a deterministic scene-held-out portion
  of R2R **train** for development/confirmation and proceeds to the sealed
  `val_unseen` protocol only after all training choices are fixed.
* No R2R `val_unseen`, RxR unseen, test, or test-challenge payload may be read
  during collection, fitting, threshold search, or model selection.

## Mainline and controls

The report must contain three fits with the same code and fixed folds:

1. RxR-only action-aligned gate;
2. R2R-only action-aligned gate;
3. 3ZK joint, dataset-balanced gate (mainline).

The joint fit is accepted only if its train-scene OOF checks pass and its
sealed per-benchmark confirmation is non-inferior to the frozen MF3ZG control
on both benchmarks.  A regression on either benchmark is reported rather than
hidden by averaging the two datasets.

## Acceptance gates

Before any public unseen evaluation, all of the following must hold:

* source manifests and feature files are regular, project-local, and
  hash-verified;
* all rows are `train` and have no future-frame or task-metric leakage;
* scene-disjoint OOF folds have zero scene overlap;
* each domain has finite predictions, a positive aggregate utility for the
  authorized switches, fewer catastrophic switches than the ungated cohort,
  and a positive leave-one-scene-out total;
* the joint model has enough authorized training events in both core and
  expansion tiers;
* the train-scene confirmation cohort is sealed before fitting and is not used
  to choose thresholds;
* RxR and R2R seen/unseen results are reported separately with SR, SPL, nDTW,
  SDTW, utility, intervention count, and scene-cluster bootstrap intervals.

## Scientific claim boundary

3ZK can support the claim “one frozen-front-end controller learns a
cross-benchmark action-aligned return-risk gate.”  It cannot support a claim
of universal navigation, SOTA, or positive unseen transfer until the sealed
unseen evaluations pass.  The earlier MF3ZG artifacts remain the
reproducibility baseline and are never overwritten.
