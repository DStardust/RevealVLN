# Method-Freeze 3ZE: Action-Aligned Counterfactual Return Gate

Status: design-frozen before the complete direct-switch return cohort was
assembled. This revision does not alter `FROZEN_SPEC.md` or the frozen MF3V
result.

## Problem isolated by MF3ZC/MF3ZD

MF3V proposes a single switch from the frozen ETP-R1 native action to its
frozen runner-up. Its ranker is trained against a three-step teacher-index
proxy. That proxy can be correct locally while the executed switch reduces
final navigation return. A pre-decision critic trained on a different V6
reversible-excursion action also failed out-of-fold, so action semantics must
match deployment.

## Revision

MF3ZE leaves the MF3V proposer, score band, one-intervention budget, frozen
ETP-R1 checkpoint, and runner-up action unchanged. It adds an
Action-Aligned Counterfactual Return Gate (ACRG):

1. On RxR train only, identify every episode on which frozen MF3V would
   switch.
2. Reuse the already recorded native-policy final metrics and execute exactly
   one matched rollout in which the actual MF3V runner-up switch is applied.
3. Label the pre-decision state with the paired final-return difference
   `0.50*nDTW + 0.25*SDTW + 0.25*SPL`.
4. Fit only from features available before the action: pooled instruction,
   current checkpoint/history, native and runner-up embeddings, the decision
   step, native margin, and the frozen MF3V ensemble statistics.
5. At deployment, MF3V may switch only when ACRG's cross-fitted safety rule
   authorizes it. Rejected proposals delegate exactly to frozen ETP-R1.

The gate cannot choose a new action, inspect a future observation, use a
geodesic/oracle field, read an evaluation label, execute a probe action, or
backtrack.

## Capacity and validation controls

- The primary representation is a fixed low-dimensional set of normalized
  scalar and cosine-relation features. No scene identifier or episode
  identifier is an input.
- Candidate predictors are regularized linear return regression and
  regularized logistic harm classification. A small neural critic may be
  reported only as a predeclared comparator, not silently substituted after
  evaluation.
- Every training prediction used to select the authorization rule is
  out-of-fold under the existing deterministic scene partition. No scene may
  occur in both fit and evaluation sides of a fold.
- Rule selection is train-only. It maximizes aggregate paired utility subject
  to at least 12 authorized out-of-fold proposals, positive mean deployed
  utility over the whole cohort, and fewer catastrophic harms
  (`delta utility <= -0.10`) than ungated MF3V. Ties prefer fewer parameters,
  fewer interventions, then the larger safety margin.
- After selection, the model is refit on all train scenes. RxR val_seen must
  retain the pre-existing task-metric gate (positive utility with a paired
  scene-bootstrap 95% lower bound above zero and non-negative SR/SPL/nDTW).
  Failure is preserved and is not sent to unseen.
- A new RxR val_unseen holdout must exclude every episode used by earlier
  MF3V/MF3ZA/MF3ZC analyses. Earlier public-unseen results are development
  evidence, not pristine final benchmark claims.

## Claims allowed if all gates pass

MF3ZE is a deployment-aligned, causal safety refinement of UAD: it connects
uncertain-action discovery to the final navigation objective using paired
train-only counterfactuals while remaining an online, instruction-guided VLN
controller. It is not branch exploration, physical backtracking, a second
topology, or an oracle lookahead.
