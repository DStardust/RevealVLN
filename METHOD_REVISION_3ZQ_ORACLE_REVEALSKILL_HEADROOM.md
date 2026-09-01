# MF3ZQ Exploratory Oracle RevealSkill Headroom v1

Revision identifier: `mf3zq_oracle_revealskill_headroom_v1`

This is a one-shot exploratory feasibility check for the original
evidence-preserving RevealSkill formulation.  It is not a deployment gate and
does not alter the sealed MF3ZP formal protocol.  The experiment asks whether
perfect decision-specific cognitive state (DEC/UAD, option memory, and
control-backed Reveal/Expiry) would have headroom over the frozen ETP-R1
native policy.

## Frozen boundary

ETP-R1, candidate generation, low-level controller, simulator configuration,
episode budget, utility (`0.50 nDTW + 0.25 SDTW + 0.25 SPL`), option memory
budget (`M=8`), return horizon (`H=8`), UAD stability (`K=3`), and deterministic
option ordering are fixed.  No Q model, policy learner, threshold search, or
public-split evaluation is part of this revision.  Historical CAR, RCSP, DSR,
MF3ZN, MF3ZO, and MF3ZP artifacts are read only.

## Oracle information boundary

The oracle may read the independent visual DEC/factor annotations, current
option identity, option birth/persistence, and returnability computed by the
frozen controller.  It may not read future observations/actions, route truth as
an action answer, final metrics, reward, delta utility, CAR/REE predictions, or
teleport/pose shortcuts.  A missing option-specific binding or missing
control-backed returnability is an unsupported episode, not an imputed label.

Prerequisite constraints record historical satisfaction and do not require the
three-prefix stability rule.  DEC constraints independently derive U/A/D with
exactly three consecutive complete prefixes.  Future, redundant, and incorrect
constraints do not enter current option readiness.

## Fixed arms

* **A**: frozen ETP-R1 native behavior.
* **B**: oracle DEC readiness; no option memory or expiry-aware backtracking.
* **C**: B plus bounded option preservation/backtracking (`M=8`), without
  expiry-aware logic.
* **D**: C plus true option-specific Reveal/Expiry and safe returnability.

All movement is delegated to the same frozen ETP-R1 executor.  The policy code
only emits one of `FOLLOW`, `INSPECT`, `EXPLORE`, `BACKTRACK`, `COMMIT`, or
`STOP`; it never edits simulator state.

## Population and stop rule

The population is the fixed 80-event independent visual-review set (40 R2R,
40 RxR, 39 raw MP3D scenes, one event per episode).  It is not formal human
gold.  If any episode cannot be mapped to an option-specific DEC chain and a
legal frozen-controller continuation, the run records it as unsupported and
does not replace it.  The full exploratory PASS requires, separately for R2R
and RxR, positive paired utility with a positive raw-scene bootstrap lower
bound, at least 25% PCR relative reduction with a positive bootstrap lower
bound, and no higher catastrophe rate than the matched baseline.  Any failure
or unsupported population triggers `MF3ZQ_EXPLORATORY_ORACLE_HEADROOM_FAIL` and
stops downstream learning/public evaluation.

The protocol is sealed before any rollout result is read.  A failed run never
writes a deployment checkpoint and never changes `formal MF3ZP oracle_headroom`.
