# MF3ZP Qwen-provisional exploratory training v1

This is an explicitly exploratory branch authorized after the user chose to
skip human review temporarily.  It does not change, replace, or reopen the
sealed MF3ZP RevealSkill protocol.  Qwen `qwen3.8-max` responses remain
provisional machine annotations, never gold labels.

The run uses only the sealed 300-event development pilot and its strictly
causal prefix requests.  A fixed ridge probe compares snapshot-only causal
features with temporal causal summaries and with a Qwen-derived semantic-state
augmentation.  Exact intervention utility is opened only after all inference
features have been built, and only for development events whose
`(dataset, scene, episode, decision_step)` identity has an exact canonical
match.  No public split, confirmation result, or deployment checkpoint is
used.

This branch is a feasibility diagnostic, not formal label-validity evidence,
Oracle Headroom evidence, or a TUAD/RevealSkill deployment result.
