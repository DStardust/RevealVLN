# MF3ZP Codex Independent Visual Review v1

Revision: `mf3zp_codex_visual_review_v1`

This revision corrects the scope mismatch in `mf3zp_codex_proxy_ree_v1`.
The earlier artifact transformed cached Qwen S/G/E predictions and therefore
did not satisfy the user's request for an independent, image-by-image review.
It remains immutable historical evidence but is inadmissible as the requested
80-event visual review or as evidence about independently reviewed labels.

## Fixed review procedure

- Population and order are the already frozen 80-event scout selection.
- The reviewer is Codex, recorded as an AI visual reviewer—not a human expert.
- Each event is opened separately.  Every causal panorama in the review window
  is inspected, in chronological order, together with the instruction and the
  candidate constraint graph.
- The constraint graph is an object being audited, not accepted as truth.
  Constraints may be marked required, prerequisite, future, redundant, or
  incorrect; missing decisive atoms may be added explicitly.
- DEC roles and every per-prefix S/G/E value are entered by the reviewer.
- No cached Qwen S/G/E, U/A/D, rationale, utility, outcome, prior prediction,
  CAR match, or public-split record may be read by the labeling path.
- U/A/D is derived mechanically with the existing fixed `K=3` rule.

The resulting labels are `CODEX_INDEPENDENT_VISUAL_REVIEW`, not human gold and
not formal label-validity evidence.  A later training probe is exploratory,
uses raw-scene-disjoint evaluation, produces no deployment checkpoint, and
cannot authorize Oracle Headroom, formal REE, skill policy, or public splits.

## One-shot boundary

The 80 labels are completed before training.  After any training result is
read, labels, DEC semantics, model constants, and evaluation rules are frozen.
No threshold, architecture, regularization, seed, or label revision is allowed
inside this version.
