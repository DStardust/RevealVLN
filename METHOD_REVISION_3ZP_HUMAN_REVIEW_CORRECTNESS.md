# MF3ZP Human Review Correctness Erratum

The machine-readable sealed MF3ZP formal protocol is authoritative for future
formal label validation: exactly three distinct blinded reviewers and one
distinct adjudicator are required. Every non-unanimous DEC/factor item must be
adjudicated before an immutable gold artifact is written. Passing kappa with
incomplete adjudication is `MF3ZP_LABEL_VALIDITY_PENDING_ADJUDICATION`, never a
formal pass.

The single-expert scout is different: one expert completes a first pass and a
blind test-retest calibration. It may establish annotation readiness only. It
cannot create gold labels or authorize Oracle Headroom, REE, skill rollouts,
checkpoints, or public evaluation.
