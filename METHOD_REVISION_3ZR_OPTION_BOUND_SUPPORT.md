# MF3ZR — Option-Bound Evidence & Recoverability Support v1

`mf3zr_option_bound_support_v1` is a data-and-observation-support revision.
It does not introduce a learner, policy, oracle rollout, or public evaluation.
It preserves the fixed 80-event MF3ZQ population and asks only whether the
objects needed by a future option-specific Reveal/Expiry experiment can be
represented without guessing.

## Frozen boundary

MF3ZM-CAR, RCSP, DSR, MF3ZN, MF3ZO, MF3ZP, and MF3ZQ are historical immutable
records.  MF3ZQ ended at `FAIL_AT_POPULATION_SUPPORT`; no numerical oracle arm
was run.  The single-decision learned-gate family remains permanently stopped.
MF3ZR does not alter any of those artifacts and never accesses public splits.

The source population is exactly 80 unique episodes (40 R2R and 40 RxR) over
39 raw MP3D scenes.  Unsupported rows are retained; there is no replacement
sampling or outcome-adaptive selection.

## Support objects

For each causal prefix, the revision records an explicit edge

\[
B_{t,k,b}\in\{\mathrm{SUPPORTS},\mathrm{CONTRADICTS},
\mathrm{UNRESOLVED},\mathrm{SHARED\_CONTEXT},\mathrm{NOT\_APPLICABLE}\}.
\]

Opaque ETP candidate aliases are the only candidate identity source.  A
candidate rank is display metadata, never semantic truth.  Contextual and
discriminative flags are mutually exclusive, and shared context may bind to
multiple options.  A syntactically valid edge is not usable until an
independent option-binding review verifies it.  The current frozen visual
labels contain no such option-specific verification, so the materialized
edges remain `UNRESOLVED` and `verified=false`.

Option IDs are deterministic:

```text
sha256({event_id, first_seen_step, candidate_id})
```

Anchor references are derived from the first causal observation.  An identity
gap is marked unresolved rather than repaired with hindsight.

## Recoverability

Expiry support requires a real callback bound to the frozen ETP-R1 controller,
with `RETURN_HORIZON=8`.  A geometry-only distance, navmesh shortcut, pose
reset, teleport, or snapshot restore cannot count as a return.  The current
80-event source has no callable sealed callback for this audit, therefore each
option receives `EXECUTION_UNAVAILABLE`.  Reveal/Expiry are consequently
`*_NOT_COMPUTABLE`; no step is imputed.

## Fixed support gate

`K=3`, memory budget `M=8`, and return horizon `8` are fixed.  Support would
require at least 64/80 joint events, at least 30/40 per domain, and at least 30
supported unique episodes in each domain.  The audit reports outcome-blind
descriptive differences between supported and unsupported rows only.

No Qwen call/read, outcome read, checkpoint, oracle arm, or public split access
is permitted in this revision.  If the gate fails, the only valid state is
`MF3ZR_OPTION_BOUND_SUPPORT_FAIL`; a future numerical experiment must be a
separately sealed MF3ZS revision.
