# MF3ZP Single-Expert DEC Calibration Scout v1

Revision: `mf3zp_single_expert_dec_scout_v1`.

This is an annotation/data-readiness calibration, not a scientific RevealSkill
gate. It neither replaces the sealed three-reviewer requirement nor authorizes
Oracle Headroom, REE, skill rollout, checkpoint generation, or public-split
evaluation.

The scout audits only the decision-specific Decisive Evidence Chain (DEC). The
existing event-agnostic Qwen decomposition contains no reliable event-specific
DEC marker, so this revision predeclares its proposed DEC as the complete frozen
Qwen graph. The expert independently marks each atom as `DEC_REQUIRED`,
`PREREQUISITE_ONLY`, `FUTURE_NOT_RELEVANT`, `REDUNDANT`, or `INCORRECT`, and may
add separately stored missing DEC atoms. Human DEC consists of the first two
roles plus missing atoms. Frozen Qwen graph records are never edited.
Across the blind retest, added atoms align only by exact normalized text plus
the expert's explicit Qwen mapping/match type; no embedding similarity or
outcome-aware matching is permitted. Nonmatches count as DEC-membership
disagreement rather than being silently dropped.

The outcome-blind selection is fixed at 80 events (R2R 40, RxR 40) by per-scene
hash ordering and scene round-robin. A 20-event retest subset (10/10) is sealed
with an independent salt. The first review uses at most prefixes
`[max(prefix_start, t-4), t]`; optional older evidence must still be at or before
the decision. S/G/E are reviewed without Qwen factors or rationales and U/A/D is
mechanically derived with exactly `K=3`.

Readiness thresholds are frozen before labels: intra-expert UAD kappa >= .75,
intra-expert E kappa >= .80, DEC precision >= .80, DEC recall >= .90,
Qwen/expert UAD accuracy >= .80, and false-decisive rate <= .10. At least 20
Qwen-D comparisons are required; fewer yields `INSUFFICIENT_QWEN_D_SUPPORT`.
Passing produces only `READY_FOR_FORMAL_MULTI_REVIEW`. A failure cannot be
repaired inside this version by changing labels, prompts, model, events,
semantics, K, metrics, or thresholds.

No utility or outcome file is read by selection or review preparation. No Qwen
request is made. The retest event IDs are sealed now, while its blank package is
materialized only after the first review has independently passed structural
validation.
