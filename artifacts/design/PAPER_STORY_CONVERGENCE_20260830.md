# RevealNav Paper-Story Convergence — 2026-08-30

## Decision

The claim-bearing instruction-guided VLN candidate remains the frozen MF3ZG
method.  MF3ZI and MF3ZJ are rejected fallback extensions.  They do not enter
the final architecture, figures, method equation, or headline result.

This decision preserves every previously frozen artifact and status.  In
particular, the MF3ZG unseen artifact remains
`FRESH_HOLDOUT_ADVANTAGE_FAIL` because its preregistered auxiliary requirement
to beat the native-margin controller was not met.  The same artifact also
contains a successful primary paired endpoint: utility `+0.006493` with 95%
scene-cluster bootstrap interval `[+0.000153, +0.012677]`, with positive nDTW,
SPL, and SR point estimates.  The comparison to the uncertainty controller is
inconclusive rather than significantly negative: `-0.001621`, interval
`[-0.011816, +0.009304]`.

## Final story in one paragraph

A frozen VLN policy can commit to a locally plausible branch before its
instruction evidence is temporally decisive.  RevealNav learns a
prefix-causal, horizon-consistent UAD score for the frozen policy's exact
native-to-runner-up choice, then screens that exact action with paired
train-only counterfactual return evidence.  A core-preserving hierarchy widens
coverage without replacing the high-confidence path, and executes at most one
residual switch while otherwise delegating bit-exactly to the frozen policy.

## Three claim-bearing pieces

1. **Horizon-consistent UAD proposal.**  The instruction, causal history,
   current candidates, and native scores predict whether the runner-up is a
   near-horizon rescue or harm.  Future teacher indices are training labels,
   never online inputs.
2. **Action-aligned counterfactual safety.**  Paired RxR-train direct-switch
   rollouts supervise the exact action used at deployment, closing the mismatch
   between local teacher labels and final navigation return.
3. **Core-preserving selective execution.**  High-confidence core and
   coverage expansion proposals have disjoint opportunities and one shared
   executed-switch budget.  A rejected expansion never removes the proven core
   path.

The hierarchy is an inference rule within selective execution, not a fourth
module.  The dataset construction protocol is a supporting contribution, not
a second method story.

## Boundaries that keep the claim defensible

- Frozen ETP-R1 frontend and action set; no newly invented navigation action.
- No future observation, teacher index, geodesic metric, or counterfactual
  return online.
- No second topology, checkpoint memory, physical rollback, or branch
  exploration in RxR/R2R.
- Multi-branch topology remains a separate open-vocabulary object-search
  extension and cannot support the instruction-guided benchmark claim.
- MF3ZI's 17-dimensional critic failed train-only fitting.  MF3ZJ passed its
  ordinary train OOF screen but failed the nested-scene audit and the locked
  val-seen lower-confidence gate.  Neither may be presented as a successful
  method or silently folded into MF3ZG.

## Evidence already earned

- RxR val-seen, 57 paired scenes: utility `+0.009552`, 95% interval
  `[+0.000043, +0.025740]`; SR, SPL, and nDTW point estimates non-negative;
  utility over native-margin control `+0.010482`.
- Fresh non-overlapping RxR val-unseen, 387 episodes: primary utility
  `+0.006493`, 95% interval `[+0.000153, +0.012677]`; 38 executed changes;
  positive nDTW, SPL, and SR point estimates.

These are promising method signals, not yet a complete CVPR benchmark table.

## Required evidence before paper lock

1. Run the frozen MF3ZG settings unchanged on R2R val-seen and val-unseen.
2. Run official full-split RxR evaluation and, if available, a blind challenge
   submission; do not tune after opening either result.
3. Compare against the frozen ETP-R1 baseline and current public VLN methods
   under matched sensors/checkpoints, reporting SR, SPL, nDTW, SDTW, runtime,
   parameters, and intervention coverage.
4. Report ablations for horizon consistency, action-aligned return gating, and
   core-preserving expansion.  MF3ZI/MF3ZJ belong only in an optional negative
   diagnostic table.
5. Finish the delayed 600-item three-reviewer Gold audit before making dataset
   quality claims; it is not required to run the navigation model itself.

No acceptance probability or all-positive-review claim is warranted until
these external and ablation results exist.
