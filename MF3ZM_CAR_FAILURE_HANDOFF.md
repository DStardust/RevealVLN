# MF3ZM-CAR v1 failure handoff

## Decision

MF3ZM-CAR v1 is a completed **train-development failure**.  It must not be
described as a near-pass, used to authorize confirmation/public unseen, or
rescued by post-hoc threshold tuning.  No checkpoint was produced.

The pre-registered stop rule is now triggered: the project should stop
swapping learned gates over the same frozen single-decision proposal features.
The next defensible method question is whether richer, strictly causal
temporal/UAD evidence can identify beneficial irreversible interventions
across scenes.

## What was tested

- 1,540 canonical exact-one-switch train-development events over 39 MP3D
  scenes: 997 RxR and 543 R2R.
- Five whole-scene outer folds and four whole-scene inner folds.
- Fixed weight-decay grid `{1e-4, 1e-3, 1e-2}`; three fixed seeds; 800 steps.
- Fixed deployment rule `switch_logit > 0`.
- Event-equal/domain-balanced preference objective, hard-forward catastrophic
  constraint, and explicit leave-one-scene utility constraints.
- Equal-budget low-native-margin and high-MF3V-score baselines.
- Independent controls: no-scene constraint, soft risk, 28D representation,
  policy-only representation, no-risk, RxR-only, R2R-only, and frozen DSR v1
  retrained on the expanded data.

No confirmation scene or public split was consumed.

## Result

- Mainline: `NESTED_CAR_FAIL`.
- All five outer folds: no feasible inner candidate for any allowed weight
  decay.
- Outer OOF prediction therefore remained incomplete.
- Every CAR control failed; all five folds were infeasible for the 28D,
  policy-only, RxR-only, and R2R-only controls as well.
- Frozen DSR v1 on the expanded data passed outer folds 0--2 but failed fold 3:
  every candidate had non-positive RxR utility; two also exceeded the ungated
  RxR catastrophic rate.
- All nine jobs completed without execution errors in 17 minutes 39 seconds.
- `checkpoint_created=false`; `public_unseen_authorized=false`.

The compact per-fold/per-candidate evidence is in
`MF3ZM_CAR_REVIEW_RESULT.json`.  It records the SHA-256 and byte size of the
full local result without adding the 27 MB raw artifact to Git.

## Evidence-backed failure pattern

1. **Criterion alignment did not restore scene generalization.**  Some inner
   candidates had positive aggregate utility in both domains, but still failed
   fold-level utility, leave-one-scene robustness, risk, or equal-budget
   baseline criteria.  No outer fold had a deployable candidate.
2. **The problem is not isolated to the semantic rank-4 representation.**
   Both the old 28D representation and policy-only control failed every outer
   fold.
3. **Removing individual constraints does not rescue the method.**  The
   no-scene, soft-risk, and no-risk variants also failed every outer fold.
4. **Pooling is not the sole cause.**  RxR-only and R2R-only CAR both failed all
   five folds.
5. **More exact labels alone are insufficient.**  Frozen DSR v1 still failed
   after expansion from 249 to 1,540 events.
6. **Simple proposal-side rules remain hard to beat.**  Learned candidates
   repeatedly had lower utility and/or higher catastrophic rate than matched
   low-margin or high-proposal-score baselines.

## What the evidence does and does not imply

Supported conclusion:

> Under the frozen runner-up proposal support, the current single-decision
> causal observables and tested gate families do not reliably identify
> beneficial interventions across MP3D scenes while satisfying the declared
> risk and equal-budget criteria.

Not established:

- that all frozen runner-up interventions are useless;
- that exact counterfactual supervision is invalid;
- that no richer strictly causal state can solve the problem;
- that RxR or R2R switching is fundamentally impossible;
- any claim about public unseen performance.

## Correctness and execution notes

The original implementation was too slow because it rebuilt scene masks,
recomputed the same hard gate, and synchronized once per scene at every step.
`revealnav_mf3/car_fast.py` hoists constants and batches the mathematically
identical leave-one-scene constraints.  Independent pre-registered arms run in
parallel CPU processes because this small head benchmarked faster on CPU than
CUDA.  The execution revision was sealed before the result was observed.

The reference/accelerated equivalence gate passed with identical
initialization hashes, hard authorization counts, zero-selection diagnostics,
and zero maximum parameter difference in the sealed synthetic check.  The
scientific data, model architecture, objective, thresholds, folds, seeds,
training steps, and failure rules were unchanged.

## Requested external review

Please audit the code and compact evidence, then answer:

1. Is the pre-registered stop rule now justified, or is there a specific
   correctness flaw that invalidates the failure?
2. If the stop rule is justified, what is the smallest versioned method that
   adds richer **strictly causal temporal/UAD state** without touching public
   splits or tuning on consumed outcomes?
3. Which minimal oracle/identifiability audit should precede that method?
4. What baselines and stop conditions prevent another cycle of gate-family
   meta-overfitting?

Do not recommend changing thresholds, utility weights, split membership, or
the declared risk criterion merely to make this result pass.
