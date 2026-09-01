# MF3ZP Qwen3.8-Max provisional annotation pilot

This is a small, exploratory annotator-compatibility pilot requested after
the sealed MF3ZP v2/v2r1 batches.  It is not a new navigation method and it
does not amend either sealed protocol.

## Fixed boundary

The pilot uses a deterministic, outcome-blind subset of the same 150 MF3ZO
event identities, balanced 10 R2R/10 RxR, and retains only request prefixes
with `prefix_step <= event.decision_step`.  Event selection is based only on
sealed identities and a fixed hash order.  It never reads exact returns,
catastrophe labels, fold outcomes, or public splits.

Qwen3.8-Max is queried with the unchanged role-blinded semantic questions
and the original response validator.  A fixed formatting addendum asks for a
short non-empty rationale.  Transport retries and concurrency are bounded;
provider responses and credentials are not logged.

The output is provisional machine annotation only.  It may report whether
the fixed U/A/D projection becomes complete, but it cannot be called human
verified oracle supervision, cannot authorize Probe A/TUAD, and cannot create
a checkpoint or access val/test splits.

If the endpoint/model is unavailable or the fixed pilot fails schema
validation, the result is recorded as a data-readiness failure.  No prompt,
model, event subset, or threshold is changed in response to the result.
